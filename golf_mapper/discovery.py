"""Course discovery and eligibility filtering via the Overpass API.

Bbox mode uses a single Overpass query that fetches all course boundaries and all
golf features together, then counts features per course with a geopandas spatial
join in Python — replacing the previous N+1 query-per-course pattern.
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
    build_feature_count_query,
    build_single_bbox_query,
    classify_tags,
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

    bbox mode makes ONE Overpass request and counts features via spatial join.
    osm_id mode makes TWO requests (boundary + feature count) for a single course.

    The manifest is written to cfg.data.output_dir / 'course_manifest.gpkg'.
    Overpass responses are cached in cfg.data.overpass_cache.
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


# ── osm_id mode (2 queries for one course) ────────────────────────────────────

def _discover_single_course(
    cfg: GolfMapperConfig,
    client: OverpassClient,
) -> list[dict[str, Any]]:
    """Fetch boundary + feature counts for the single course in cfg.aoi.osm_id."""
    osm_type, osm_id = parse_osm_ref(cfg.aoi.osm_id)
    course_id = f"{osm_type}/{osm_id}"

    log.info("Fetching boundary for %s (%s)…", cfg.aoi.name, course_id)
    boundary_data = client.query(
        build_boundary_query(osm_type, osm_id),
        cache_key=f"boundary:{course_id}",
    )
    boundary_geom, name = _extract_boundary(boundary_data, osm_type, osm_id, cfg.aoi.name)

    if boundary_geom is None:
        log.error("Could not extract boundary geometry for %s — skipping.", course_id)
        return []

    area_id = osm_element_to_area_id(osm_type, osm_id)
    feat_data = client.query(
        build_feature_count_query(area_id),
        cache_key=f"features:{course_id}",
    )
    feature_counts = count_features_from_elements(feat_data["elements"], cfg.classes)
    total = sum(feature_counts.values())
    log.info("Feature counts for %s: %s (total=%d)", course_id, feature_counts, total)

    centroid = boundary_geom.centroid
    return [{
        "course_id": course_id,
        "name": name,
        "osm_type": osm_type,
        "osm_id": osm_id,
        "qualifies_as_training": compute_eligibility(feature_counts, cfg.labels.min_features),
        "feature_counts": json.dumps(feature_counts),
        "total_features": total,
        "centroid_lon": round(centroid.x, 6),
        "centroid_lat": round(centroid.y, 6),
        "geometry": boundary_geom,
    }]


# ── bbox mode (1 query + Python spatial join) ─────────────────────────────────

def _discover_bbox_courses(
    cfg: GolfMapperConfig,
    client: OverpassClient,
) -> list[dict[str, Any]]:
    """Discover all golf courses in the bbox with a single Overpass request.

    One query fetches every leisure=golf_course boundary AND every golf feature
    in the bbox simultaneously. Feature counts are then computed by spatially
    joining the two sets in Python — no per-course Overpass queries needed.
    """
    assert cfg.aoi.bbox is not None, "aoi.bbox must be set when aoi.type='bbox'"
    bbox = cfg.aoi.bbox

    log.info("Discovering courses in bbox %s (single query)…", bbox)
    data = client.query(
        build_single_bbox_query(bbox),
        cache_key=f"discovery_single:bbox:{bbox}",
    )
    log.info("Single query returned %d elements.", len(data.get("elements", [])))
    return _parse_courses_and_count_features(data, cfg)


def _parse_courses_and_count_features(
    data: dict[str, Any],
    cfg: GolfMapperConfig,
) -> list[dict[str, Any]]:
    """Split a combined Overpass response into courses + features, count via spatial join.

    The response contains both leisure=golf_course boundaries and golf feature
    ways/relations. osm2geojson reconstructs all polygon geometries, then a
    geopandas spatial join assigns each feature to the course it falls within.
    """
    if not _HAS_OSM2GEOJSON:
        raise ImportError(
            "osm2geojson is required for geometry extraction. "
            "Install with: pip install osm2geojson"
        )

    try:
        geojson = _osm2geojson.json2geojson(data)
    except Exception as exc:
        log.error("osm2geojson conversion failed: %s", exc)
        return []

    course_records: list[dict] = []
    feature_records: list[dict] = []

    for feat in geojson.get("features", []):
        props = feat.get("properties", {})
        geom_dict = feat.get("geometry")
        if geom_dict is None:
            continue
        if props.get("type") not in ("way", "relation"):
            continue

        tags = props.get("tags") or {}

        try:
            geom = make_valid(_shape(geom_dict))
            if geom.is_empty:
                continue
        except Exception:
            continue

        if tags.get("leisure") == "golf_course":
            course_records.append({
                "course_id":  f"{props['type']}/{props['id']}",
                "osm_type":   props["type"],
                "osm_id":     int(props["id"]),
                "name":       tags.get("name", ""),
                "geometry":   geom,
            })
        else:
            cls = classify_tags(tags, cfg.classes, skip_background=True)
            if cls is not None:
                feature_records.append({
                    "class_name": cls.name,
                    "geometry":   geom,
                })

    if not course_records:
        log.warning("No leisure=golf_course geometries found in response.")
        return []

    log.info(
        "Parsed %d course boundaries and %d golf features.",
        len(course_records), len(feature_records),
    )

    courses_gdf = gpd.GeoDataFrame(course_records, crs="EPSG:4326")

    # Count features per course via spatial join
    if feature_records:
        features_gdf = gpd.GeoDataFrame(feature_records, crs="EPSG:4326")
        course_feature_counts = _spatial_feature_counts(features_gdf, courses_gdf)
    else:
        log.warning("No golf features found in bbox — all courses will have total_features=0.")
        course_feature_counts = {}

    # Build manifest rows
    rows: list[dict[str, Any]] = []
    for _, row in courses_gdf.iterrows():
        cid = row["course_id"]
        counts = course_feature_counts.get(cid, {})
        total = sum(counts.values())
        centroid = row.geometry.centroid
        rows.append({
            "course_id":             cid,
            "name":                  row["name"],
            "osm_type":              row["osm_type"],
            "osm_id":                row["osm_id"],
            "qualifies_as_training": compute_eligibility(counts, cfg.labels.min_features),
            "feature_counts":        json.dumps(counts),
            "total_features":        total,
            "centroid_lon":          round(centroid.x, 6),
            "centroid_lat":          round(centroid.y, 6),
            "geometry":              row.geometry,
        })

    rows.sort(key=lambda r: r["total_features"], reverse=True)
    return rows


def _spatial_feature_counts(
    features_gdf: gpd.GeoDataFrame,
    courses_gdf: gpd.GeoDataFrame,
) -> dict[str, dict[str, int]]:
    """Return {course_id: {class_name: count}} via geopandas spatial join.

    Uses predicate='within' so only features fully inside a course boundary are
    counted. Features that straddle a boundary (rare mapping edge case) are
    counted toward the course whose boundary they are most contained by via a
    fallback 'intersects' pass.
    """
    result: dict[str, dict[str, int]] = {}

    # Primary pass: features strictly within a course boundary
    joined = gpd.sjoin(
        features_gdf,
        courses_gdf[["course_id", "geometry"]],
        how="inner",
        predicate="within",
    )

    for _, row in joined.iterrows():
        cid = row["course_id"]
        cls = row["class_name"]
        result.setdefault(cid, {})
        result[cid][cls] = result[cid].get(cls, 0) + 1

    # Fallback pass: features that intersect but weren't caught by 'within'
    # (handles features mapped slightly outside the course boundary)
    matched_indices = set(joined.index)
    unmatched = features_gdf[~features_gdf.index.isin(matched_indices)]

    if not unmatched.empty:
        joined2 = gpd.sjoin(
            unmatched,
            courses_gdf[["course_id", "geometry"]],
            how="inner",
            predicate="intersects",
        )
        for _, row in joined2.iterrows():
            cid = row["course_id"]
            cls = row["class_name"]
            result.setdefault(cid, {})
            result[cid][cls] = result[cid].get(cls, 0) + 1

    log.info(
        "Spatial join: %d feature assignments across %d courses.",
        sum(sum(v.values()) for v in result.values()), len(result),
    )
    return result


# ── Shared helpers ────────────────────────────────────────────────────────────

def _extract_boundary(
    data: dict[str, Any],
    osm_type: str,
    osm_id: int,
    fallback_name: str,
) -> tuple[Any, str]:
    """Convert a single-course Overpass response to (geometry, name)."""
    if not _HAS_OSM2GEOJSON:
        raise ImportError("osm2geojson required — pip install osm2geojson")
    try:
        geojson = _osm2geojson.json2geojson(data)
    except Exception as exc:
        log.error("osm2geojson conversion failed for %s/%d: %s", osm_type, osm_id, exc)
        return None, fallback_name

    for feat in geojson.get("features", []):
        props = feat.get("properties", {})
        if (
            props.get("type") == osm_type
            and props.get("id") == osm_id
            and feat.get("geometry") is not None
        ):
            try:
                geom = make_valid(_shape(feat["geometry"]))
                if not geom.is_empty:
                    name = (props.get("tags") or {}).get("name", fallback_name)
                    return geom, name
            except Exception as exc:
                log.warning("Invalid geometry for %s/%d: %s", osm_type, osm_id, exc)

    log.warning(
        "No %s/%d geometry in osm2geojson output (%d features).",
        osm_type, osm_id, len(geojson.get("features", [])),
    )
    return None, fallback_name


def _parse_courses_from_response(data: dict[str, Any]) -> list[tuple[str, int, str]]:
    """Extract (osm_type, osm_id, name) from a discovery response (kept for tests)."""
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
