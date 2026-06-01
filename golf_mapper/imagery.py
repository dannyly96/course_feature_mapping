"""Esri World Imagery acquisition via samgeo.tms_to_geotiff.

Raw imagery tiles are cached locally and MUST NOT be redistributed.
See README — Esri grants permission to trace World Imagery for OSM vectorization,
but the tiles themselves are not open data.
"""
from __future__ import annotations

import importlib
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import rasterio
from pyproj import CRS, Transformer
from shapely.ops import transform as shapely_transform

from .config import GolfMapperConfig
from .utils import ensure_dir, get_logger

log = get_logger(__name__)


# ── Custom exceptions ─────────────────────────────────────────────────────────

class ImageryError(RuntimeError):
    """Base class for imagery acquisition failures."""

class CRSError(ImageryError):
    """GeoTIFF has no CRS or an unexpected CRS."""

class CoverageError(ImageryError):
    """GeoTIFF does not fully cover the required course boundary."""


# ── samgeo / leafmap import detection ─────────────────────────────────────────

# Try top-level samgeo first (>= 0.40.0), then samgeo.common, then leafmap.
# Verify the exact import path against the installed samgeo version — the API
# location has moved between minor releases.
_tms_to_geotiff: Any = None
_SAMGEO_IMPORT_PATH: str | None = None

for _mod_name, _fn_name in [
    ("samgeo", "tms_to_geotiff"),
    ("samgeo.common", "tms_to_geotiff"),
    ("leafmap", "tms_to_geotiff"),
]:
    try:
        _mod = importlib.import_module(_mod_name)
        _fn = getattr(_mod, _fn_name, None)
        if _fn is not None:
            _tms_to_geotiff = _fn
            _SAMGEO_IMPORT_PATH = f"{_mod_name}.{_fn_name}"
            log.debug("tms_to_geotiff resolved at %s", _SAMGEO_IMPORT_PATH)
            break
    except ImportError:
        continue

if _tms_to_geotiff is None:
    log.warning(
        "samgeo / leafmap not installed — imagery download unavailable. "
        "Install with: pip install segment-geospatial"
    )


# ── Constants ─────────────────────────────────────────────────────────────────

_EARTH_RADIUS_M: float = 6_378_137.0   # WGS84 equatorial radius
_TILE_PIXELS: int = 256                 # standard TMS tile dimension


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_course_imagery(
    boundary: Any,          # shapely.Geometry in EPSG:4326
    course_id: str,
    cfg: GolfMapperConfig,
    force: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Download Esri World Imagery for a course; return (tif_path, metadata_dict).

    On cache hit the GeoTIFF is re-validated (CRS + coverage). If validation
    fails or force=True the download is re-run and the cache is overwritten.

    The GeoTIFF is written to:
        cfg.data.imagery_cache / <course_id>_z<zoom>.tif   (gitignored)
    A companion metadata file is written alongside it:
        cfg.data.imagery_cache / <course_id>_z<zoom>_metadata.json

    Raises:
        ImportError:    samgeo / leafmap not installed.
        CRSError:       Downloaded GeoTIFF has no CRS.
        CoverageError:  Downloaded GeoTIFF does not cover the boundary.
        ImageryError:   Download failed with both source string and fallback URL.
    """
    ensure_dir(cfg.data.imagery_cache)
    zoom = cfg.imagery.zoom
    key = imagery_cache_key(course_id, zoom)
    tif_path = cfg.data.imagery_cache / f"{key}.tif"
    meta_path = cfg.data.imagery_cache / f"{key}_metadata.json"

    if not force and tif_path.exists():
        try:
            metadata = validate_geotiff_coverage(tif_path, boundary)
            log.info(
                "Imagery cache hit: %s (%.1f MB)",
                tif_path.name, tif_path.stat().st_size / 1e6,
            )
            return tif_path, metadata
        except (CRSError, CoverageError) as exc:
            log.warning("Cached GeoTIFF invalid (%s) — re-downloading.", exc)

    bbox_wgs84 = compute_course_bbox(boundary, buffer_m=cfg.aoi.buffer_m)
    log.info(
        "Downloading Esri World Imagery: course=%s zoom=%d bbox=[%.4f, %.4f, %.4f, %.4f]",
        course_id, zoom, *bbox_wgs84,
    )

    _download_tiles(
        tif_path,
        bbox_wgs84,
        zoom,
        cfg.imagery.source,
        cfg.imagery.esri_xyz_url,
    )

    metadata = validate_geotiff_coverage(tif_path, boundary)
    metadata.update(
        _build_metadata(course_id, cfg, bbox_wgs84, tif_path, metadata)
    )
    meta_path.write_text(json.dumps(metadata, indent=2))

    log.info(
        "Imagery saved: %s | %.1f MB | %d×%d px | %.2f m/px | CRS=%s",
        tif_path.name,
        metadata["file_size_mb"],
        metadata["width_px"],
        metadata["height_px"],
        metadata["ground_resolution_m"],
        metadata["crs"],
    )
    return tif_path, metadata


def imagery_cache_key(course_id: str, zoom: int) -> str:
    """Return a filesystem-safe string for keying a course + zoom combination.

    Example: 'relation/5179090' + zoom 19  →  'relation_5179090_z19'
    """
    safe = course_id.replace("/", "_").replace(":", "_")
    return f"{safe}_z{zoom}"


def compute_course_bbox(
    boundary: Any,     # shapely.Geometry in EPSG:4326
    buffer_m: float = 50.0,
) -> list[float]:
    """Buffer the boundary in real-world metres and return [west, south, east, north].

    Buffering is performed in a UTM projection centred on the boundary centroid so
    that distances are true metres rather than approximate degree equivalents.
    The buffered envelope is then reprojected to EPSG:4326.

    Args:
        boundary:  Shapely geometry in EPSG:4326 (WGS84).
        buffer_m:  Buffer distance in metres (non-negative).

    Returns:
        [west, south, east, north] in decimal degrees (EPSG:4326).
    """
    assert buffer_m >= 0, f"buffer_m must be >= 0, got {buffer_m}"

    centroid = boundary.centroid
    utm_crs = _utm_crs_for_point(centroid.x, centroid.y)
    wgs84 = CRS.from_epsg(4326)

    to_utm = Transformer.from_crs(wgs84, utm_crs, always_xy=True)
    to_wgs84 = Transformer.from_crs(utm_crs, wgs84, always_xy=True)

    geom_utm = shapely_transform(to_utm.transform, boundary)
    buffered_wgs84 = shapely_transform(to_wgs84.transform, geom_utm.buffer(buffer_m))

    west, south, east, north = buffered_wgs84.bounds
    return [west, south, east, north]


def validate_geotiff_coverage(
    tif_path: Path,
    boundary: Any,    # shapely.Geometry in EPSG:4326
) -> dict[str, Any]:
    """Validate a GeoTIFF's CRS and coverage against a boundary polygon.

    The function reprojects the GeoTIFF bounding box to EPSG:4326 for
    comparison regardless of the GeoTIFF's native CRS (EPSG:3857 is
    typical for TMS outputs from samgeo/leafmap).

    Returns a metadata dict with: crs, bounds_wgs84, width_px, height_px,
    ground_resolution_m.

    Raises:
        CRSError:      GeoTIFF has no CRS (corrupt download).
        CoverageError: GeoTIFF bounds do not fully contain the boundary.
    """
    with rasterio.open(tif_path) as src:
        if src.crs is None:
            raise CRSError(
                f"GeoTIFF {tif_path.name} has no CRS — likely a corrupt download. "
                "Delete the cached file and re-run."
            )
        tif_crs = src.crs
        tif_bounds = src.bounds   # (left, bottom, right, top) in tif_crs units
        width, height = src.width, src.height

    # Reproject tif bounds to WGS84 (EPSG:4326) for comparison with boundary
    if tif_crs.to_epsg() == 4326:
        west, south = tif_bounds.left, tif_bounds.bottom
        east, north = tif_bounds.right, tif_bounds.top
    else:
        t = Transformer.from_crs(tif_crs, CRS.from_epsg(4326), always_xy=True)
        west, south = t.transform(tif_bounds.left, tif_bounds.bottom)
        east, north = t.transform(tif_bounds.right, tif_bounds.top)

    bnd_west, bnd_south, bnd_east, bnd_north = boundary.bounds

    if west > bnd_west or south > bnd_south or east < bnd_east or north < bnd_north:
        raise CoverageError(
            f"{tif_path.name} does not fully cover the boundary.\n"
            f"  GeoTIFF  (WGS84): [{west:.5f}, {south:.5f}, {east:.5f}, {north:.5f}]\n"
            f"  Boundary (WGS84): [{bnd_west:.5f}, {bnd_south:.5f}, {bnd_east:.5f}, {bnd_north:.5f}]"
        )

    # Compute ground resolution from actual pixel spacing (independent of zoom level)
    lat_mid = (south + north) / 2.0
    lon_per_px = (east - west) / width
    m_per_deg_lon = _EARTH_RADIUS_M * math.cos(math.radians(lat_mid)) * math.pi / 180.0
    resolution_m = round(lon_per_px * m_per_deg_lon, 3)

    return {
        "crs": str(tif_crs),
        "bounds_wgs84": [west, south, east, north],
        "width_px": width,
        "height_px": height,
        "ground_resolution_m": resolution_m,
    }


def ground_resolution_m(zoom: int, latitude_deg: float) -> float:
    """Nominal ground resolution (metres/pixel) for a TMS zoom level and latitude.

    Uses the Web Mercator (EPSG:3857) formula:
        GSD = (2π · R_earth · cos(lat)) / (256 · 2^zoom)

    At zoom 19, equator: ≈ 0.30 m/px.  At 33.5°N (Augusta): ≈ 0.25 m/px.
    """
    lat_rad = math.radians(latitude_deg)
    circumference = 2.0 * math.pi * _EARTH_RADIUS_M * math.cos(lat_rad)
    return circumference / (_TILE_PIXELS * (2 ** zoom))


# ── Internal helpers ──────────────────────────────────────────────────────────

def _utm_crs_for_point(lon: float, lat: float) -> CRS:
    """Return the WGS84 UTM CRS (EPSG:326xx / 327xx) for a lon/lat point."""
    zone = int((lon + 180.0) / 6.0) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def _download_tiles(
    tif_path: Path,
    bbox_wgs84: list[float],
    zoom: int,
    source: str,
    fallback_url: str,
) -> None:
    """Invoke tms_to_geotiff; fall back to the explicit XYZ URL if the provider name fails.

    Verified against samgeo >= 0.40.0 and leafmap >= 0.36.0.
    If your installed version has a different signature, check:
        import inspect; print(inspect.signature(_tms_to_geotiff))
    and adjust the kwargs below accordingly.
    """
    if _tms_to_geotiff is None:
        raise ImportError(
            "samgeo (segment-geospatial) or leafmap is required. "
            "Install with: pip install segment-geospatial"
        )
    tif_path.parent.mkdir(parents=True, exist_ok=True)

    def _call(src: str) -> None:
        _tms_to_geotiff(
            output=str(tif_path),
            bbox=bbox_wgs84,
            zoom=zoom,
            source=src,
            overwrite=True,
        )

    try:
        _call(source)
        log.info("tms_to_geotiff OK (source=%r, via %s)", source, _SAMGEO_IMPORT_PATH)
    except Exception as exc_primary:
        log.warning(
            "tms_to_geotiff failed with source=%r (%s); retrying with explicit XYZ URL.",
            source, exc_primary,
        )
        try:
            _call(fallback_url)
            log.info("tms_to_geotiff OK (fallback XYZ URL)")
        except Exception as exc_fallback:
            raise ImageryError(
                f"Imagery download failed.\n"
                f"  source={source!r}: {exc_primary}\n"
                f"  fallback URL:      {exc_fallback}"
            ) from exc_fallback


def fetch_all_training_imagery(
    manifest_gdf: "gpd.GeoDataFrame",  # type: ignore[name-defined]  # noqa: F821
    cfg: GolfMapperConfig,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Download Esri World Imagery for every training course in the manifest.

    The manifest's ``geometry`` column already holds each course's boundary polygon
    (produced by discovery.discover_courses), so no additional Overpass queries are
    needed here.

    Courses whose imagery download fails are logged and skipped — the dataset will
    simply contain fewer training courses for that run.

    Args:
        manifest_gdf: Course manifest GeoDataFrame from discovery.discover_courses.
        cfg:          Pipeline configuration.

    Returns:
        (imagery_paths, boundary_geoms) — both dicts keyed by course_id.
        imagery_paths:  {course_id: Path} to the cached GeoTIFF.
        boundary_geoms: {course_id: shapely geometry} boundary in EPSG:4326.
    """
    import geopandas as gpd  # local import keeps the module importable without geopandas

    training = manifest_gdf[manifest_gdf["qualifies_as_training"]]
    if training.empty:
        log.warning("No training courses in manifest — imagery_paths will be empty.")
        return {}, {}

    log.info(
        "Downloading Esri World Imagery for %d training course(s) at zoom %d…",
        len(training), cfg.imagery.zoom,
    )

    imagery_paths: dict[str, Path] = {}
    boundary_geoms: dict[str, Any] = {}

    for _, row in training.iterrows():
        course_id = str(row["course_id"])
        boundary = row.geometry   # already EPSG:4326 from discovery
        if boundary is None or boundary.is_empty:
            log.warning("Skipping %s — no boundary geometry in manifest.", course_id)
            continue
        try:
            tif_path, meta = fetch_course_imagery(boundary, course_id, cfg)
            imagery_paths[course_id] = tif_path
            boundary_geoms[course_id] = boundary
            log.info(
                "  [%d/%d] %s — %.1f MB, %.3f m/px",
                len(imagery_paths), len(training),
                course_id, meta["file_size_mb"], meta["ground_resolution_m"],
            )
        except Exception as exc:
            log.warning(
                "Imagery download failed for %s: %s — skipping this course.",
                course_id, exc,
            )

    log.info(
        "Imagery ready for %d/%d training course(s).",
        len(imagery_paths), len(training),
    )
    return imagery_paths, boundary_geoms


def _build_metadata(
    course_id: str,
    cfg: GolfMapperConfig,
    bbox_wgs84: list[float],
    tif_path: Path,
    base: dict[str, Any],
) -> dict[str, Any]:
    return {
        "course_id": course_id,
        "source": cfg.imagery.source,
        "source_url": cfg.imagery.esri_xyz_url,
        "zoom": cfg.imagery.zoom,
        "bbox_wgs84": bbox_wgs84,
        "capture_timestamp": datetime.now(timezone.utc).isoformat(),
        "file_size_mb": round(tif_path.stat().st_size / 1e6, 2),
        "samgeo_import_path": _SAMGEO_IMPORT_PATH,
        "attribution": (
            "Esri, Maxar, GeoEye, Earthstar Geographics, CNES/Airbus DS, "
            "USDA, USGS, AeroGRID, IGN, and the GIS User Community"
        ),
        "license_note": (
            "Esri World Imagery tiles are for vectorization only. "
            "Raw tiles MUST NOT be redistributed or re-hosted. "
            "Derived vector data is ODbL (OpenStreetMap contributors)."
        ),
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from shapely.geometry import box as bbox_polygon

    from .config import load_config
    from .utils import setup_logging

    parser = argparse.ArgumentParser(description="Download Esri World Imagery for a course")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--bbox",
        nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Override boundary with a manual bbox",
    )
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    cfg.data.make_dirs()

    if args.bbox:
        boundary = bbox_polygon(*args.bbox)
    else:
        from .osm import OverpassClient, build_boundary_query, parse_osm_ref
        from .discovery import _extract_boundary

        osm_type, osm_id = parse_osm_ref(cfg.aoi.osm_id)
        client = OverpassClient(cfg.overpass, cache_dir=cfg.data.overpass_cache)
        data = client.query(
            build_boundary_query(osm_type, osm_id),
            cache_key=f"boundary:{cfg.aoi.osm_id}",
        )
        boundary, _ = _extract_boundary(data, osm_type, osm_id, cfg.aoi.name)

    tif_path, meta = fetch_course_imagery(boundary, cfg.aoi.osm_id.replace("/", "_"), cfg)
    print(f"GeoTIFF : {tif_path}")
    print(f"Size    : {meta['file_size_mb']:.1f} MB")
    print(f"Dims    : {meta['width_px']} × {meta['height_px']} px")
    print(f"GSD     : {meta['ground_resolution_m']:.3f} m/px")
    print(f"CRS     : {meta['crs']}")
