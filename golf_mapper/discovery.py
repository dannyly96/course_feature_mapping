"""Course discovery and eligibility filtering via the Overpass API.

Produces a GeoDataFrame manifest with one row per discovered course, including
boundary geometry, per-class feature counts, and a training eligibility flag.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
from shapely.validation import make_valid

from .config import GolfMapperConfig
from .osm import (
    OverpassClient,
    build_boundary_query,
    build_discovery_query,
    build_feature_count_query,
    compute_eligibility,
    count_features_from_elements,
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
    log.warning(
        "osm2geojson not installed — boundary geometry extraction will fail. "
        "Install with: pip install osm2geojson"
    )

try:
    from shapely.geometry import shape as _shape
except ImportError:
    _shape = None  # type: ignore[assignment]


_MANIFEST_COLUMNS = [
    "course_id", "name", "osm_type", "osm_id",
    "qualifies_as_training", "feature_counts", "total_features",
    "centroid_lon", "centroid_lat", "geometry",
]


# ── Public entry point ────────────────────────────────────────────────────────

def discover_courses(cfg: GolfMapperConfig) -> gpd.GeoDataFrame:
    """Discover golf courses for the configured AOI and return a manifest GeoDataFrame.

    Each row represents one course:
      - geometry:               boundary polygon (EPSG:4326)
      - course_id:              '<type>/<id>'  e.g. 'relation/5179090'
      - name:                   OSM name tag (or fallback)
      - osm_type / osm_id:      OSM element type and numeric ID
      - feature_counts:         JSON string of {class_name: count}
      - total_features:         sum of all feature counts
      - qualifies_as_training:  True when total_features >= cfg.labels.min_features
      - centroid_lon/lat:       centroid of the boundary in EPSG:4326

    The manifest is also written to cfg.data.output_dir / 'course_manifest.gpkg'.

    Overpass responses are cached in cfg.data.overpass_cache to avoid redundant calls.
    """
    client = OverpassClient(cfg.overpass, cache_dir=cfg.data.overpass_cache)

    if cfg.aoi.type == "osm_id":
        rows = _discover_single_course(cfg, client)
    elif cfg.aoi.type == "bbox":
        rows = _discover_bbox_courses(cfg, client)
    else:
        raise NotImplementedError(
            f"AOI type {cfg.aoi.type!r} is not implemented. Use 'osm_id' or 'bbox'."
        )

    if not rows:
        log.warning("No courses found for the configured AOI.")
        return gpd.GeoDataFrame(columns=_MANIFEST_COLUMNS, geometry="geometry", crs="EPSG:4326")

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

    n_training = int(gdf["qualifies_as_training"].sum())
    log.info(
        "Discovered %d course(s); %d qualify as training (min_features=%d).",
        len(gdf), n_training, cfg.labels.min_features,
    )

    _write_manifest(gdf, cfg.data.output_dir)
    return gdf


# ── Internal helpers ──────────────────────────────────────────────────────────

def _discover_single_course(
    cfg: GolfMapperConfig,
    client: OverpassClient,
) -> list[dict[str, Any]]:
    """Fetch boundary + feature counts for the single course in cfg.aoi.osm_id."""
    osm_type, osm_id = parse_osm_ref(cfg.aoi.osm_id)
    course_id = f"{osm_type}/{osm_id}"

    log.info("Fetching boundary for %s (%s)…", cfg.aoi.name, course_id)
    boundary_ql = build_boundary_query(osm_type, osm_id)
    boundary_data = client.query(boundary_ql, cache_key=f"boundary:{course_id}")
    boundary_geom, name = _extract_boundary(boundary_data, osm_type, osm_id, cfg.aoi.name)

    if boundary_geom is None:
        log.error("Could not extract boundary geometry for %s — skipping.", course_id)
        return []

    area_id = osm_element_to_area_id(osm_type, osm_id)
    feature_counts = _fetch_feature_counts(client, area_id, course_id, cfg)
    total = sum(feature_counts.values())
    qualifies = compute_eligibility(feature_counts, cfg.labels.min_features)

    centroid = boundary_geom.centroid
    return [
        {
            "course_id": course_id,
            "name": name,
            "osm_type": osm_type,
            "osm_id": osm_id,
            "qualifies_as_training": qualifies,
            "feature_counts": json.dumps(feature_counts),
            "total_features": total,
            "centroid_lon": round(centroid.x, 6),
            "centroid_lat": round(centroid.y, 6),
            "geometry": boundary_geom,
        }
    ]


def _discover_bbox_courses(
    cfg: GolfMapperConfig,
    client: OverpassClient,
) -> list[dict[str, Any]]:
    """Discover all golf courses within the configured bounding box."""
    assert cfg.aoi.bbox is not None, "aoi.bbox must be set when aoi.type='bbox'"
    bbox = cfg.aoi.bbox  # [min_lon, min_lat, max_lon, max_lat]

    log.info("Discovering courses in bbox %s…", bbox)
    ql = build_discovery_query(bbox=bbox)
    data = client.query(ql, cache_key=f"discovery:bbox:{bbox}")

    courses = _parse_courses_from_response(data)
    log.info("Found %d leisure=golf_course element(s) in bbox.", len(courses))

    rows: list[dict[str, Any]] = []
    for osm_type, osm_id, name in courses:
        course_id = f"{osm_type}/{osm_id}"
        try:
            boundary_ql = build_boundary_query(osm_type, osm_id)
            boundary_data = client.query(boundary_ql, cache_key=f"boundary:{course_id}")
            boundary_geom, resolved_name = _extract_boundary(
                boundary_data, osm_type, osm_id, name or course_id
            )
            if boundary_geom is None:
                log.warning("No valid boundary for %s — skipping.", course_id)
                continue

            area_id = osm_element_to_area_id(osm_type, osm_id)
            feature_counts = _fetch_feature_counts(client, area_id, course_id, cfg)
            total = sum(feature_counts.values())
            qualifies = compute_eligibility(feature_counts, cfg.labels.min_features)

            centroid = boundary_geom.centroid
            rows.append({
                "course_id": course_id,
                "name": resolved_name,
                "osm_type": osm_type,
                "osm_id": osm_id,
                "qualifies_as_training": qualifies,
                "feature_counts": json.dumps(feature_counts),
                "total_features": total,
                "centroid_lon": round(centroid.x, 6),
                "centroid_lat": round(centroid.y, 6),
                "geometry": boundary_geom,
            })
        except Exception as exc:
            log.warning("Failed to process %s: %s", course_id, exc)

    return rows


def _fetch_feature_counts(
    client: OverpassClient,
    area_id: int,
    course_id: str,
    cfg: GolfMapperConfig,
) -> dict[str, int]:
    ql = build_feature_count_query(area_id)
    data = client.query(ql, cache_key=f"features:{course_id}")
    counts = count_features_from_elements(data["elements"], cfg.classes)
    log.info(
        "Feature counts for %s: %s  (total=%d)",
        course_id, counts, sum(counts.values()),
    )
    return counts


def _extract_boundary(
    data: dict[str, Any],
    osm_type: str,
    osm_id: int,
    fallback_name: str,
) -> tuple[Any, str]:
    """Convert an Overpass boundary response to a Shapely geometry + course name.

    Returns (geometry, name). geometry is None if conversion fails.
    """
    if not _HAS_OSM2GEOJSON:
        raise ImportError(
            "osm2geojson is required for geometry extraction. "
            "Install with: pip install osm2geojson"
        )

    try:
        geojson = _osm2geojson.json2geojson(data)
    except Exception as exc:
        log.error("osm2geojson conversion failed for %s/%d: %s", osm_type, osm_id, exc)
        return None, fallback_name

    target_geom = None
    name = fallback_name

    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        if (
            props.get("type") == osm_type
            and props.get("id") == osm_id
            and feature.get("geometry") is not None
        ):
            try:
                geom = _shape(feature["geometry"])
                geom = make_valid(geom)
                if not geom.is_empty:
                    target_geom = geom
            except Exception as exc:
                log.warning(
                    "Invalid geometry for %s/%d: %s — skipping this feature.",
                    osm_type, osm_id, exc,
                )
                continue
            name = (props.get("tags") or {}).get("name", fallback_name)
            break

    if target_geom is None:
        log.warning(
            "No %s/%d geometry in osm2geojson output (%d features total).",
            osm_type, osm_id, len(geojson.get("features", [])),
        )

    return target_geom, name


def _parse_courses_from_response(data: dict[str, Any]) -> list[tuple[str, int, str]]:
    """Extract (osm_type, osm_id, name) tuples from a discovery Overpass response."""
    results = []
    for elem in data.get("elements", []):
        etype = elem.get("type")
        if etype not in ("way", "relation"):
            continue
        tags = elem.get("tags", {})
        if tags.get("leisure") != "golf_course":
            continue
        results.append((etype, int(elem["id"]), tags.get("name", "")))
    return results


def _write_manifest(gdf: gpd.GeoDataFrame, output_dir: Path | str) -> None:
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    out = output_dir / "course_manifest.gpkg"
    gdf.to_file(out, driver="GPKG")
    log.info("Manifest written: %s (%d course(s))", out, len(gdf))


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from .config import load_config
    from .utils import setup_logging

    parser = argparse.ArgumentParser(description="Discover golf courses via Overpass")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    cfg.data.make_dirs()

    gdf = discover_courses(cfg)
    if not gdf.empty:
        print(gdf[["course_id", "name", "qualifies_as_training", "total_features"]].to_string())
