"""OSM → raster semantic mask + YOLO instance label dataset builder.

Produces an Ultralytics-format segmentation dataset split by course (no tile
from the same course appears in both train and val, preventing data leakage).
"""
from __future__ import annotations

import hashlib
import logging
import random
import shutil
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
import yaml
from rasterio.windows import Window
from shapely.geometry import box as bbox_polygon

from .config import GolfMapperConfig
from .geometry import (
    clip_to_boundary,
    exg_canopy_mask,
    geo_polygon_to_pixel,
    make_valid_gdf,
    polygon_to_yolo_seg,
    rasterize_geodataframe,
)
from .osm import (
    OverpassClient,
    build_feature_geometry_query,
    classify_tags,
    osm_element_to_area_id,
    parse_osm_ref,
)
from .utils import ensure_dir, get_logger

log = get_logger(__name__)

try:
    import osm2geojson as _osm2geojson
    _HAS_OSM2GEOJSON = True
except ImportError:
    _osm2geojson = None  # type: ignore[assignment]
    _HAS_OSM2GEOJSON = False


# ── Public entry point ────────────────────────────────────────────────────────

def build_dataset(
    manifest_gdf: gpd.GeoDataFrame,
    cfg: GolfMapperConfig,
    imagery_paths: dict[str, Path] | None = None,
    boundary_geoms: dict[str, Any] | None = None,
) -> Path:
    """Build a tiled Ultralytics segmentation dataset from training courses.

    For each qualifying training course the pipeline:
      1. Downloads the Esri World Imagery GeoTIFF (cached on disk).
      2. Fetches the OSM feature geometries from Overpass (cached on disk).
      3. Rasterizes OSM features onto the imagery to produce pixel-aligned labels.
      4. Tiles imagery + labels into TILE×TILE patches with OVERLAP-pixel overlap.
      5. Writes images/ and labels/ directories in Ultralytics format.

    The train/val/test split is performed *by course* (not by tile) so no tile
    from the same course can appear in more than one split.

    Args:
        manifest_gdf:   Course manifest (from discovery.discover_courses).
        cfg:            Pipeline configuration.
        imagery_paths:  Optional pre-downloaded {course_id: tif_path} dict.
                        If None, imagery is downloaded automatically for every
                        training course using fetch_all_training_imagery().
        boundary_geoms: Optional {course_id: boundary geometry} dict.
                        If None, boundaries are taken from manifest_gdf.geometry.

    Returns:
        Path to the dataset root (containing images/, labels/, data.yaml).
    """
    if imagery_paths is None or boundary_geoms is None:
        from .imagery import fetch_all_training_imagery
        log.info(
            "imagery_paths not supplied — downloading imagery for all %d training course(s).",
            manifest_gdf["qualifies_as_training"].sum(),
        )
        imagery_paths, boundary_geoms = fetch_all_training_imagery(manifest_gdf, cfg)
    dataset_dir = cfg.data.dataset_dir
    ensure_dir(dataset_dir)

    client = OverpassClient(cfg.overpass, cache_dir=cfg.data.overpass_cache)

    training_courses = manifest_gdf[manifest_gdf["qualifies_as_training"]]["course_id"].tolist()
    if not training_courses:
        raise ValueError("No training courses in manifest — cannot build dataset.")

    # Split courses first (by course, not tile)
    splits = split_courses(training_courses, cfg.training.val_split, cfg.training.test_split, cfg.training.seed)
    log.info(
        "Dataset split: train=%d, val=%d, test=%d courses",
        len(splits["train"]), len(splits["val"]), len(splits["test"]),
    )

    tile_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}

    for course_id in training_courses:
        tif_path = imagery_paths.get(course_id)
        boundary = boundary_geoms.get(course_id)
        if tif_path is None or not tif_path.exists() or boundary is None:
            log.warning("Skipping %s — missing imagery or boundary.", course_id)
            continue

        # Determine which split this course belongs to
        for split_name, course_list in splits.items():
            if course_id in course_list:
                split = split_name
                break
        else:
            split = "train"

        osm_type, osm_id = parse_osm_ref(course_id)
        area_id = osm_element_to_area_id(osm_type, osm_id)

        log.info("Building tiles for %s [%s]…", course_id, split)
        features_gdf = fetch_course_features(client, area_id, course_id, cfg)
        if features_gdf.empty:
            log.warning("No features fetched for %s — skipping.", course_id)
            continue

        n = _tile_course(
            tif_path, features_gdf, boundary, course_id, split, dataset_dir, cfg
        )
        tile_counts[split] += n
        log.info("  → %d tiles written to %s/", n, split)

    write_data_yaml(cfg, dataset_dir)
    log.info(
        "Dataset complete: %s | train=%d val=%d test=%d tiles",
        dataset_dir, tile_counts["train"], tile_counts["val"], tile_counts["test"],
    )
    return dataset_dir


def fetch_course_features(
    client: OverpassClient,
    area_id: int,
    course_id: str,
    cfg: GolfMapperConfig,
) -> gpd.GeoDataFrame:
    """Fetch and classify OSM feature geometries for a course as a GeoDataFrame."""
    if not _HAS_OSM2GEOJSON:
        raise ImportError("osm2geojson required — pip install osm2geojson")

    ql = build_feature_geometry_query(area_id)
    data = client.query(ql, cache_key=f"feat_geom:{course_id}")

    try:
        geojson = _osm2geojson.json2geojson(data)
    except Exception as exc:
        log.error("osm2geojson failed for %s: %s", course_id, exc)
        return _empty_features_gdf()

    records = []
    for feat in geojson.get("features", []):
        geom_dict = feat.get("geometry")
        if geom_dict is None:
            continue
        props = feat.get("properties", {})
        if props.get("type") not in ("way", "relation"):
            continue
        tags = props.get("tags") or {}
        cls = classify_tags(tags, cfg.classes, skip_background=True)
        if cls is None:
            continue
        try:
            from shapely.geometry import shape
            from shapely.validation import make_valid
            geom = make_valid(shape(geom_dict))
            if not geom.is_empty:
                records.append({"class_id": cls.id, "class_name": cls.name, "geometry": geom})
        except Exception:
            continue

    if not records:
        return _empty_features_gdf()

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    log.info("Fetched %d feature geometries for %s", len(gdf), course_id)
    return gdf


def split_courses(
    course_ids: list[str],
    val_split: float,
    test_split: float,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Assign course IDs to train / val / test splits.

    Split is performed at course level — tiles from the same course NEVER
    appear in more than one split (prevents label leakage).

    Args:
        course_ids: List of course ID strings.
        val_split:  Fraction of courses for validation (e.g. 0.15).
        test_split: Fraction of courses for test (e.g. 0.10).

    Returns:
        {'train': [...], 'val': [...], 'test': [...]}
    """
    ids = list(course_ids)
    random.seed(seed)
    random.shuffle(ids)
    n = len(ids)
    n_test = max(1, round(n * test_split))
    n_val = max(1, round(n * val_split))
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train, n_val, n_test = max(1, n - 2), 1, 1
    return {
        "train": ids[:n_train],
        "val":   ids[n_train: n_train + n_val],
        "test":  ids[n_train + n_val:],
    }


def write_data_yaml(cfg: GolfMapperConfig, dataset_dir: Path) -> Path:
    """Write the Ultralytics data.yaml for this dataset."""
    data = {
        "path": str(dataset_dir),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc": cfg.num_classes,
        "names": cfg.class_names,
    }
    out = dataset_dir / "data.yaml"
    with out.open("w") as fh:
        yaml.dump(data, fh, default_flow_style=False)
    log.info("Wrote %s (nc=%d, classes=%s)", out, data["nc"], data["names"])
    return out


# ── Tiling ────────────────────────────────────────────────────────────────────

def _tile_course(
    tif_path: Path,
    features_gdf: gpd.GeoDataFrame,
    boundary: Any,
    course_id: str,
    split: str,
    dataset_dir: Path,
    cfg: GolfMapperConfig,
) -> int:
    """Tile one course's imagery + labels and write to the dataset directory."""
    tile_size = cfg.labels.tile_size
    overlap = cfg.labels.overlap
    step = tile_size - overlap

    img_out = dataset_dir / "images" / split
    lbl_out = dataset_dir / "labels" / split
    ensure_dir(img_out)
    ensure_dir(lbl_out)

    safe_id = course_id.replace("/", "_")
    n_written = 0
    n_empty_skipped = 0

    with rasterio.open(tif_path) as src:
        full_h, full_w = src.height, src.width
        transform = src.transform
        crs = src.crs

        # Reproject features to imagery CRS for rasterization
        if str(crs) != str(features_gdf.crs):
            feat = make_valid_gdf(features_gdf.to_crs(crs))
        else:
            feat = make_valid_gdf(features_gdf)

        n_tiles_total = 0
        n_empty = 0

        for row_off in range(0, full_h, step):
            for col_off in range(0, full_w, step):
                h = min(tile_size, full_h - row_off)
                w = min(tile_size, full_w - col_off)
                if h < tile_size // 2 or w < tile_size // 2:
                    continue  # skip tiny edge tiles

                window = Window(col_off, row_off, w, h)
                tile_transform = src.window_transform(window)
                tile_bounds = rasterio.transform.array_bounds(h, w, tile_transform)
                tile_bbox = bbox_polygon(*tile_bounds)

                # Clip features to tile
                feat_tile = feat[feat.geometry.intersects(tile_bbox)].copy()
                if not feat_tile.empty:
                    feat_tile = clip_to_boundary(feat_tile, tile_bbox)

                # Semantic mask for empty-tile detection
                sem_mask = rasterize_geodataframe(
                    feat_tile, "class_id", tile_transform, (h, w)
                ) if not feat_tile.empty else np.zeros((h, w), dtype=np.uint8)

                labeled_frac = float((sem_mask > 0).sum()) / (h * w)
                n_tiles_total += 1

                if labeled_frac == 0:
                    n_empty += 1
                    # Sub-sample empty tiles
                    if n_empty / max(1, n_tiles_total) > cfg.labels.empty_tile_max_ratio:
                        n_empty_skipped += 1
                        continue

                # Read RGB tile
                img_data = src.read(window=window)[:3]   # (3, H, W)

                # Pad to tile_size if at image edge
                if h < tile_size or w < tile_size:
                    padded = np.zeros((3, tile_size, tile_size), dtype=img_data.dtype)
                    padded[:, :h, :w] = img_data
                    img_data = padded
                    pad_h, pad_w = tile_size, tile_size
                else:
                    pad_h, pad_w = h, w

                # Build YOLO label lines
                label_lines = []
                for _, row in feat_tile.iterrows():
                    from shapely.geometry import Polygon as _Poly
                    geom = row.geometry
                    if isinstance(geom, _Poly):
                        polys = [geom]
                    elif hasattr(geom, "geoms"):
                        polys = list(geom.geoms)
                    else:
                        continue
                    for poly in polys:
                        px_poly = geo_polygon_to_pixel(poly, tile_transform)
                        if px_poly is None:
                            continue
                        line = polygon_to_yolo_seg(px_poly, int(row["class_id"]), pad_w, pad_h)
                        if line:
                            label_lines.append(line)

                # Write tile image (PNG for lossless)
                stem = f"{safe_id}_r{row_off}_c{col_off}"
                img_path = img_out / f"{stem}.png"
                _write_png(img_data, img_path)

                # Write label file (even if empty — required by Ultralytics)
                lbl_path = lbl_out / f"{stem}.txt"
                lbl_path.write_text("\n".join(label_lines))

                n_written += 1

    if n_empty_skipped:
        log.debug("  Skipped %d empty tiles (ratio cap).", n_empty_skipped)
    return n_written


def _write_png(img: np.ndarray, path: Path) -> None:
    """Write a (3, H, W) uint8 array as a PNG via rasterio."""
    _, h, w = img.shape
    with rasterio.open(
        path, "w",
        driver="PNG",
        height=h, width=w,
        count=3, dtype="uint8",
    ) as dst:
        dst.write(img)


def _empty_features_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"class_id": [], "class_name": [], "geometry": []}, crs="EPSG:4326")
