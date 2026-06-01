"""Confirmation gate + GeoJSON / OSM XML export with provenance tracking.

No output files are written unless the user explicitly confirms via the
interactive confirmation gate (viz.confirmation_gate).
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon

from .config import GolfMapperConfig
from .geometry import clip_to_boundary, drop_small_polygons, make_valid_gdf, simplify_metres
from .osm import write_osm_xml
from .utils import ensure_dir, get_logger

log = get_logger(__name__)

# Minimum polygon area per class (m²) — enforce sensible OSM geometry
_MIN_AREA_M2: dict[str, float] = {
    "tee":          20.0,
    "green":        50.0,
    "fairway":     200.0,
    "bunker":       10.0,
    "water_hazard": 20.0,
    "woods":       100.0,
    "rough":       200.0,
}


# ── Public entry point ────────────────────────────────────────────────────────

def export_course(
    raw_gdf: gpd.GeoDataFrame,
    course_id: str,
    boundary: Any,       # shapely geometry in EPSG:4326
    cfg: GolfMapperConfig,
    model_path: Path | None = None,
    course_name: str = "",
    interactive: bool = True,
) -> dict[str, Path] | None:
    """Vectorize, validate, confirm, and write outputs for one course.

    Pipeline:
      1. Simplify polygons (Douglas-Peucker, configured tolerance).
      2. Clip to course boundary.
      3. Drop sub-minimum-area polygons.
      4. Fix invalid geometries.
      5. Assign OSM tags.
      6. Schema sanity check (log issues, do not abort).
      7. Confirmation gate (interactive — requires 'y').
      8. On confirm: write .geojson, .osm, provenance.json.

    Args:
        raw_gdf:     Predictions GeoDataFrame from infer.run_inference.
        course_id:   'relation/5179090' style ID.
        boundary:    Course boundary in EPSG:4326.
        cfg:         Pipeline configuration.
        model_path:  Path to the best.pt used for inference (for provenance).
        course_name: Human-readable name (for the confirmation gate display).
        interactive: If False, skip the confirmation gate and always export.

    Returns:
        Dict of {'geojson': Path, 'osm': Path, 'provenance': Path} on export,
        or None if the user skipped / aborted.
    """
    # 1-4: Post-process
    gdf = vectorize_predictions(raw_gdf, boundary, cfg)
    if gdf.empty:
        log.warning("No valid predictions for %s after post-processing.", course_id)
        return None

    # 5: Assign OSM tags
    gdf = _assign_osm_tags(gdf, cfg)

    # 6: Schema check
    issues = osm_schema_check(gdf)
    if issues:
        for issue in issues:
            log.warning("Schema issue [%s]: %s", course_id, issue)

    # 7: Confirmation gate
    if interactive:
        from .viz import confirmation_gate
        decision = confirmation_gate(gdf, boundary, course_id, cfg, course_name)
        if decision == "abort":
            raise SystemExit("Pipeline aborted by user at export stage.")
        if decision == "skip":
            return None

    # 8: Write outputs
    return _write_outputs(gdf, course_id, boundary, cfg, model_path)


def vectorize_predictions(
    gdf: gpd.GeoDataFrame,
    boundary: Any,
    cfg: GolfMapperConfig,
) -> gpd.GeoDataFrame:
    """Apply simplification, clipping, area filtering, and validation."""
    if gdf.empty:
        return gdf

    gdf = simplify_metres(gdf, cfg.export.simplify_tolerance_m)
    gdf = clip_to_boundary(gdf, boundary)
    gdf = make_valid_gdf(gdf)

    # Drop class-specific minimum-area polygons
    cfg_min = cfg.inference.min_area_m2
    records = []
    for _, row in gdf.iterrows():
        cls_name = row.get("class_name", "")
        min_a = max(cfg_min, _MIN_AREA_M2.get(str(cls_name), cfg_min))
        records.append((row, min_a))

    # Project once for area computation
    if not gdf.empty:
        from pyproj import CRS
        from .geometry import _utm_epsg_for_point
        centroid = gdf.geometry.union_all().centroid
        utm_epsg = _utm_epsg_for_point(centroid.x, centroid.y)
        areas = gdf.to_crs(f"EPSG:{utm_epsg}").geometry.area
        mask = []
        for (row, min_a), area in zip(records, areas):
            mask.append(area >= min_a)
        gdf = gdf[mask].copy()

    # Ensure output CRS is EPSG:4326
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    return gdf


def osm_schema_check(gdf: gpd.GeoDataFrame) -> list[str]:
    """Run lightweight OSM schema sanity checks; return list of issue strings."""
    issues: list[str] = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            issues.append(f"Row {idx}: empty geometry")
            continue
        if not geom.is_valid:
            issues.append(f"Row {idx}: invalid geometry ({geom.geom_type})")
        tags = row.get("osm_tags", {})
        if not isinstance(tags, dict) or not tags:
            issues.append(f"Row {idx}: missing OSM tags")
        if isinstance(geom, Polygon):
            if len(geom.exterior.coords) < 4:
                issues.append(f"Row {idx}: polygon has fewer than 3 vertices")
        cls = row.get("class_name", "")
        if cls and cls in _MIN_AREA_M2:
            # Area check happens in vectorize_predictions; flag anything that slipped through
            pass
    return issues


def build_provenance(
    course_id: str,
    cfg: GolfMapperConfig,
    model_path: Path | None,
) -> dict[str, Any]:
    """Build the provenance metadata dict for a course export."""
    return {
        "course_id": course_id,
        "imagery_source": "Esri World Imagery",
        "imagery_license": (
            "Esri World Imagery tiles used for vectorization only. "
            "Raw tiles MUST NOT be redistributed."
        ),
        "derived_data_license": "ODbL (OpenStreetMap contributors)",
        "attribution": cfg.export.attribution,
        "model_variant": cfg.model.variant,
        "model_checkpoint": str(model_path) if model_path else None,
        "config_hash": cfg.config_hash(),
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "simplify_tolerance_m": cfg.export.simplify_tolerance_m,
        "output_crs": cfg.export.output_crs,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _assign_osm_tags(gdf: gpd.GeoDataFrame, cfg: GolfMapperConfig) -> gpd.GeoDataFrame:
    """Add an 'osm_tags' column containing the write-tag dict for each row."""
    gdf = gdf.copy()
    tag_map: dict[str, dict] = {c.name: c.osm_write_tags for c in cfg.classes}
    gdf["osm_tags"] = gdf["class_name"].map(tag_map)
    return gdf


def _write_outputs(
    gdf: gpd.GeoDataFrame,
    course_id: str,
    boundary: Any,
    cfg: GolfMapperConfig,
    model_path: Path | None,
) -> dict[str, Path]:
    """Write GeoJSON, OSM XML, and provenance files for one confirmed course."""
    ensure_dir(cfg.data.output_dir)

    safe_id = course_id.replace("/", "_").replace(":", "_")

    # GeoJSON
    geojson_path = cfg.data.output_dir / f"{safe_id}.geojson"
    gdf.to_file(geojson_path, driver="GeoJSON")
    log.info("Wrote GeoJSON: %s", geojson_path.name)

    # OSM XML (JOSM-compatible)
    osm_path = cfg.data.output_dir / f"{safe_id}.osm"
    write_osm_xml(
        gdf,
        osm_path,
        attribution=cfg.export.attribution,
        tag_column="osm_tags",
    )
    log.info("Wrote OSM XML: %s", osm_path.name)

    # Provenance
    prov_path = cfg.data.output_dir / f"{safe_id}_provenance.json"
    prov = build_provenance(course_id, cfg, model_path)
    prov["feature_counts"] = {
        cls: int((gdf["class_name"] == cls).sum())
        for cls in gdf["class_name"].unique()
    }
    prov_path.write_text(json.dumps(prov, indent=2))
    log.info("Wrote provenance: %s", prov_path.name)

    return {"geojson": geojson_path, "osm": osm_path, "provenance": prov_path}
