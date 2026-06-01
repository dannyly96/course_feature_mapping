"""Tiled YOLO inference + SAM 2.1 boundary refinement.

Runs the trained model over the target course in overlapping tiles, stitches
instances across seams, refines boundaries with SAM 2.1, and returns a
GeoDataFrame of per-class polygon predictions in EPSG:4326.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import MultiPolygon, Polygon, box as bbox_polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from .config import GolfMapperConfig
from .geometry import clip_to_boundary, drop_small_polygons, exg_canopy_mask, make_valid_gdf
from .utils import get_logger

log = get_logger(__name__)

# ── ultralytics / samgeo import guards ───────────────────────────────────────

try:
    from ultralytics import YOLO as _YOLO
    _HAS_ULTRALYTICS = True
except ImportError:
    _YOLO = None  # type: ignore[assignment,misc]
    _HAS_ULTRALYTICS = False

try:
    from samgeo import SamGeo as _SamGeo
    _HAS_SAMGEO = True
except ImportError:
    _SamGeo = None  # type: ignore[assignment]
    _HAS_SAMGEO = False


# ── Public entry point ────────────────────────────────────────────────────────

def run_inference(
    tif_path: Path,
    boundary: Any,    # shapely geometry in EPSG:4326
    model_path: Path,
    cfg: GolfMapperConfig,
) -> gpd.GeoDataFrame:
    """Run full inference pipeline; return GeoDataFrame of predictions in EPSG:4326.

    Steps:
      1. Tile the GeoTIFF and run the YOLO model on each tile.
      2. Stitch instances across tile seams (merge overlapping same-class masks).
      3. Refine boundaries with SAM 2.1 (if available).
      4. Add ExG canopy layer for the 'woods' class.
      5. Clip all predictions to the course boundary; drop tiny artifacts.

    Raises:
        ImportError: ultralytics not installed.
    """
    if not _HAS_ULTRALYTICS:
        raise ImportError(
            "ultralytics is required for inference. pip install ultralytics>=8.3.0"
        )
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = _YOLO(str(model_path))
    log.info("Loaded model: %s", model_path.name)

    # 1. Tiled inference → raw instance list
    log.info("Running tiled YOLO inference on %s…", tif_path.name)
    raw_instances = _tiled_inference(model, tif_path, cfg)
    log.info("Raw instances before stitch: %d", len(raw_instances))

    # 2. Stitch overlapping instances
    stitched = stitch_instances(raw_instances, cfg.inference.iou_threshold)
    log.info("Instances after stitch: %d", len(stitched))

    # 3. SAM 2.1 refinement
    if _HAS_SAMGEO and not cfg.model.use_sam3:
        stitched = _refine_with_sam2(stitched, tif_path, cfg)
    elif cfg.model.use_sam3:
        log.info("SAM 3 requested — checking HF access…")
        stitched = _refine_with_sam3_or_fallback(stitched, tif_path, cfg)
    else:
        log.warning("samgeo not installed; skipping SAM boundary refinement.")

    # 4. Convert instances to GeoDataFrame (mask → polygon in imagery CRS → EPSG:4326)
    gdf = _instances_to_geodataframe(stitched, tif_path, cfg)

    # 5. Add ExG canopy layer
    canopy_gdf = detect_canopy_exg(tif_path, cfg, boundary)
    if not canopy_gdf.empty:
        gdf = gpd.GeoDataFrame(
            gpd.pd.concat([gdf, canopy_gdf], ignore_index=True),
            crs=gdf.crs,
        )

    # 6. Clip to boundary + drop small artifacts
    gdf = clip_to_boundary(gdf, boundary)
    gdf = drop_small_polygons(gdf, cfg.inference.min_area_m2)
    gdf = make_valid_gdf(gdf)

    log.info("Final predictions: %d polygons", len(gdf))
    return gdf


# ── Tiled inference ───────────────────────────────────────────────────────────

def _tiled_inference(
    model: Any,
    tif_path: Path,
    cfg: GolfMapperConfig,
) -> list[dict]:
    """Run the model over overlapping tiles; return list of raw instance dicts."""
    tile_size = cfg.labels.tile_size
    overlap = cfg.inference.tile_overlap
    step = tile_size - overlap
    conf = cfg.inference.conf_threshold
    instances: list[dict] = []

    with rasterio.open(tif_path) as src:
        h, w = src.height, src.width
        transform = src.transform
        crs = src.crs

        for row_off in range(0, h, step):
            for col_off in range(0, w, step):
                th = min(tile_size, h - row_off)
                tw = min(tile_size, w - col_off)
                if th < tile_size // 4 or tw < tile_size // 4:
                    continue

                window = Window(col_off, row_off, tw, th)
                tile_transform = src.window_transform(window)
                img = src.read(window=window)[:3]   # (3, H, W) → (H, W, 3) for YOLO

                # Pad to tile_size if at image edge
                if th < tile_size or tw < tile_size:
                    padded = np.zeros((3, tile_size, tile_size), dtype=img.dtype)
                    padded[:, :th, :tw] = img
                    img = padded

                img_hwc = np.moveaxis(img, 0, -1)   # (H, W, 3)
                results = model.predict(img_hwc, conf=conf, verbose=False)

                for r in results:
                    if r.masks is None:
                        continue
                    for mask_arr, box, cls in zip(
                        r.masks.data.cpu().numpy(),
                        r.boxes,
                        r.boxes.cls.cpu().numpy().astype(int),
                    ):
                        conf_val = float(box.conf.cpu().numpy()[0])
                        # Convert mask pixels to geo polygon via tile transform
                        poly = _mask_to_geo_polygon(mask_arr, tile_transform)
                        if poly is None:
                            continue
                        instances.append({
                            "class_id": int(cls),
                            "conf": conf_val,
                            "polygon": poly,
                            "crs": str(crs),
                        })
    return instances


def _mask_to_geo_polygon(mask: np.ndarray, transform: Any) -> Polygon | None:
    """Convert a binary mask to a geo-referenced Shapely polygon."""
    from rasterio.features import shapes as _shapes

    mask_u8 = (mask > 0.5).astype(np.uint8)
    polys = []
    for geom_dict, val in _shapes(mask_u8, transform=transform):
        if int(val) == 1:
            try:
                polys.append(make_valid(Polygon(geom_dict["coordinates"][0])))
            except Exception:
                continue
    if not polys:
        return None
    merged = unary_union(polys)
    if isinstance(merged, MultiPolygon):
        merged = max(merged.geoms, key=lambda p: p.area)
    return merged if isinstance(merged, Polygon) and not merged.is_empty else None


# ── Cross-tile stitching ──────────────────────────────────────────────────────

def stitch_instances(
    instances: list[dict],
    iou_threshold: float = 0.45,
) -> list[dict]:
    """Merge overlapping same-class instances produced by adjacent tiles.

    Instances with IoU > iou_threshold and the same class_id are merged via
    union; the higher-confidence instance's class_id and conf are kept.

    This is an O(n²) greedy merge — acceptable for the instance counts typical
    of a single golf course.
    """
    if not instances:
        return []

    merged: list[dict] = list(instances)
    changed = True
    while changed:
        changed = False
        out: list[dict] = []
        used = [False] * len(merged)
        for i, inst_a in enumerate(merged):
            if used[i]:
                continue
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                inst_b = merged[j]
                if inst_a["class_id"] != inst_b["class_id"]:
                    continue
                iou = _polygon_iou(inst_a["polygon"], inst_b["polygon"])
                if iou > iou_threshold:
                    union_poly = make_valid(inst_a["polygon"].union(inst_b["polygon"]))
                    best_conf = max(inst_a["conf"], inst_b["conf"])
                    inst_a = {**inst_a, "polygon": union_poly, "conf": best_conf}
                    used[j] = True
                    changed = True
            out.append(inst_a)
            used[i] = True
        merged = out

    return merged


def _polygon_iou(a: Polygon, b: Polygon) -> float:
    try:
        inter = a.intersection(b).area
        union = a.union(b).area
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


# ── SAM refinement ────────────────────────────────────────────────────────────

def _refine_with_sam2(
    instances: list[dict],
    tif_path: Path,
    cfg: GolfMapperConfig,
) -> list[dict]:
    """Refine instance boundaries using SAM 2.1 box-prompted segmentation.

    Verify the SamGeo API (model_version, predict_box kwargs) against the
    installed samgeo version before deploying.
    """
    if not _HAS_SAMGEO:
        return instances
    try:
        sam = _SamGeo(
            model_type=cfg.model.sam_model_type,
            model_version=cfg.model.sam_model_version,
            automatic=False,
        )
        sam.set_image(str(tif_path))
    except Exception as exc:
        log.warning("SAM 2.1 init failed (%s) — skipping refinement.", exc)
        return instances

    refined: list[dict] = []
    for inst in instances:
        try:
            bbox = inst["polygon"].bounds   # (minx, miny, maxx, maxy)
            # SAM expects [x1, y1, x2, y2] box in pixel coordinates
            with rasterio.open(tif_path) as src:
                inv = ~src.transform
            px_box = [
                *(inv * (bbox[0], bbox[1])),   # col, row of SW corner
                *(inv * (bbox[2], bbox[3])),   # col, row of NE corner
            ]
            masks, _, _ = sam.predict(boxes=np.array([px_box]))
            if masks is not None and len(masks) > 0:
                mask_geo = _mask_to_geo_polygon(masks[0], src.transform)
                if mask_geo is not None:
                    inst = {**inst, "polygon": mask_geo}
        except Exception as exc:
            log.debug("SAM refinement failed for one instance: %s", exc)
        refined.append(inst)
    return refined


def _refine_with_sam3_or_fallback(
    instances: list[dict],
    tif_path: Path,
    cfg: GolfMapperConfig,
) -> list[dict]:
    """Try SAM 3 text-prompted refinement; fall back to SAM 2.1 on failure."""
    try:
        from samgeo import SamGeo3  # type: ignore[import]
        log.info("SAM 3 available — using text-prompted refinement.")
        # SAM 3 text-prompting is class-specific; grouping by class_id
        # Full implementation follows samgeo.SamGeo3 API (verify in docs).
        # For now, fall through to SAM 2.1 as the stable path.
        raise NotImplementedError("SAM 3 text-prompted refinement — see samgeo docs.")
    except Exception as exc:
        log.warning("SAM 3 unavailable (%s) — falling back to SAM 2.1.", exc)
        return _refine_with_sam2(instances, tif_path, cfg)


# ── ExG canopy ────────────────────────────────────────────────────────────────

def detect_canopy_exg(
    tif_path: Path,
    cfg: GolfMapperConfig,
    boundary: Any,
) -> gpd.GeoDataFrame:
    """Detect woodland canopy using Excess Green index + morphological cleanup.

    Produces natural=wood area polygons (OSM convention — not individual trees).
    The ExG mask is combined with the model's 'woods' class detections in export.
    """
    woods_cls = cfg.class_by_name("woods")

    with rasterio.open(tif_path) as src:
        rgb = np.moveaxis(src.read()[:3], 0, -1).astype(np.float32) / 255.0
        transform = src.transform
        crs = src.crs

    mask = exg_canopy_mask(rgb, threshold=0.08)

    # Morphological cleanup: remove noise, fill small holes
    try:
        from scipy.ndimage import binary_closing, binary_opening
        mask = binary_opening(mask, iterations=3).astype(np.uint8)
        mask = binary_closing(mask, iterations=5).astype(np.uint8)
    except ImportError:
        mask = mask.astype(np.uint8)

    # Vectorize
    from rasterio.features import shapes as _shapes
    records = []
    for geom_dict, val in _shapes(mask, transform=transform):
        if int(val) != 1:
            continue
        try:
            geom = make_valid(Polygon(geom_dict["coordinates"][0]))
            if not geom.is_empty:
                records.append({"class_id": woods_cls.id, "class_name": "woods", "geometry": geom})
        except Exception:
            continue

    if not records:
        return gpd.GeoDataFrame({"class_id": [], "class_name": [], "geometry": []}, crs=str(crs))

    gdf = gpd.GeoDataFrame(records, crs=str(crs))
    if str(crs) != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


# ── Instance → GeoDataFrame ───────────────────────────────────────────────────

def _instances_to_geodataframe(
    instances: list[dict],
    tif_path: Path,
    cfg: GolfMapperConfig,
) -> gpd.GeoDataFrame:
    """Convert a list of instance dicts to a GeoDataFrame in EPSG:4326."""
    if not instances:
        return gpd.GeoDataFrame(
            {"class_id": [], "class_name": [], "conf": [], "geometry": []},
            crs="EPSG:4326",
        )

    with rasterio.open(tif_path) as src:
        tif_crs = src.crs

    records = []
    for inst in instances:
        cls_id = inst["class_id"]
        try:
            cls_def = cfg.class_by_id(cls_id)
        except KeyError:
            continue
        records.append({
            "class_id":   cls_id,
            "class_name": cls_def.name,
            "conf":       inst["conf"],
            "geometry":   inst["polygon"],
        })

    gdf = gpd.GeoDataFrame(records, crs=str(tif_crs))
    if str(tif_crs) != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    return gdf
