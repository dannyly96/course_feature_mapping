"""Tests for imagery.py — all pure-logic; no network or samgeo required."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry import Polygon

from golf_mapper.imagery import (
    CoverageError,
    CRSError,
    _utm_crs_for_point,
    compute_course_bbox,
    ground_resolution_m,
    imagery_cache_key,
    validate_geotiff_coverage,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_geotiff(
    path: Path,
    bounds: tuple[float, float, float, float],  # (west, south, east, north) in crs units
    crs_epsg: int = 4326,
    width: int = 100,
    height: int = 100,
) -> None:
    """Write a minimal 3-band uint8 GeoTIFF for testing."""
    west, south, east, north = bounds
    tf = from_bounds(west, south, east, north, width, height)
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=height, width=width,
        count=3, dtype="uint8",
        crs=CRS.from_epsg(crs_epsg),
        transform=tf,
    ) as dst:
        dst.write(np.ones((3, height, width), dtype=np.uint8))


@pytest.fixture
def augusta_poly():
    """Polygon roughly covering Augusta National GC in EPSG:4326."""
    return Polygon([
        (-82.03, 33.49), (-82.01, 33.49),
        (-82.01, 33.52), (-82.03, 33.52),
    ])


# ── imagery_cache_key ─────────────────────────────────────────────────────────

def test_cache_key_deterministic():
    assert imagery_cache_key("relation/5179090", 19) == imagery_cache_key("relation/5179090", 19)


def test_cache_key_differs_by_zoom():
    assert imagery_cache_key("relation/5179090", 18) != imagery_cache_key("relation/5179090", 19)


def test_cache_key_differs_by_course():
    assert imagery_cache_key("relation/5179090", 19) != imagery_cache_key("relation/999", 19)


def test_cache_key_is_filesystem_safe():
    key = imagery_cache_key("relation/5179090", 19)
    assert "/" not in key
    assert ":" not in key


def test_cache_key_contains_zoom_indicator():
    assert "z19" in imagery_cache_key("relation/5179090", 19)
    assert "z18" in imagery_cache_key("relation/5179090", 18)


# ── _utm_crs_for_point ────────────────────────────────────────────────────────

def test_utm_zone_17n_augusta():
    # Augusta, GA: ~33.5°N, -82°W → UTM zone 17N → EPSG:32617
    crs = _utm_crs_for_point(-82.0, 33.5)
    assert crs.to_epsg() == 32617


def test_utm_southern_hemisphere():
    # Sydney, AU: ~-33.9°S, 151.2°E → UTM zone 56S → EPSG:32756
    crs = _utm_crs_for_point(151.2, -33.9)
    assert crs.to_epsg() == 32756


def test_utm_zone_31n_greenwich():
    # Near Greenwich, 3°E, 51°N → zone 31N → EPSG:32631
    crs = _utm_crs_for_point(3.0, 51.0)
    assert crs.to_epsg() == 32631


def test_utm_equator_positive_lon():
    crs = _utm_crs_for_point(10.0, 0.0)
    # lon=10°: zone = int((10+180)/6)+1 = int(31.67)+1 = 32 → 32N = EPSG:32632
    assert crs.to_epsg() == 32632


# ── compute_course_bbox ───────────────────────────────────────────────────────

def test_bbox_is_four_values(augusta_poly):
    assert len(compute_course_bbox(augusta_poly, 0)) == 4


def test_bbox_west_less_than_east(augusta_poly):
    west, south, east, north = compute_course_bbox(augusta_poly, 0)
    assert west < east


def test_bbox_south_less_than_north(augusta_poly):
    west, south, east, north = compute_course_bbox(augusta_poly, 0)
    assert south < north


def test_bbox_zero_buffer_contains_boundary(augusta_poly):
    west, south, east, north = compute_course_bbox(augusta_poly, buffer_m=0)
    bw, bs, be, bn = augusta_poly.bounds   # (minx, miny, maxx, maxy)
    assert west <= bw + 1e-6
    assert south <= bs + 1e-6
    assert east >= be - 1e-6
    assert north >= bn - 1e-6


def test_bbox_buffer_enlarges_bounds(augusta_poly):
    b0 = compute_course_bbox(augusta_poly, buffer_m=0)
    b1 = compute_course_bbox(augusta_poly, buffer_m=200)
    assert b1[0] < b0[0], "west should decrease with buffer"
    assert b1[1] < b0[1], "south should decrease with buffer"
    assert b1[2] > b0[2], "east should increase with buffer"
    assert b1[3] > b0[3], "north should increase with buffer"


def test_bbox_50m_buffer_roughly_correct(augusta_poly):
    # 50m buffer at 33.5°N should shift bounds by roughly 0.0005° lat/lon
    b0 = compute_course_bbox(augusta_poly, 0)
    b50 = compute_course_bbox(augusta_poly, 50)
    delta_lon = b0[0] - b50[0]   # how much west shifted
    delta_lat = b0[1] - b50[1]   # how much south shifted
    assert 1e-4 < delta_lon < 1e-3, f"Unexpected west shift for 50m buffer: {delta_lon:.6f}°"
    assert 1e-4 < delta_lat < 1e-3, f"Unexpected south shift for 50m buffer: {delta_lat:.6f}°"


def test_bbox_negative_buffer_raises(augusta_poly):
    with pytest.raises(AssertionError):
        compute_course_bbox(augusta_poly, buffer_m=-1)


# ── ground_resolution_m ───────────────────────────────────────────────────────

def test_resolution_zoom19_equator_approx_30cm():
    res = ground_resolution_m(zoom=19, latitude_deg=0.0)
    assert 0.25 < res < 0.35, f"Expected ~0.30 m/px at zoom 19 equator, got {res:.4f}"


def test_resolution_zoom19_augusta():
    # At 33.5°N, zoom 19 → ~0.25 m/px
    res = ground_resolution_m(zoom=19, latitude_deg=33.5)
    assert 0.20 < res < 0.30, f"Expected ~0.25 m/px at zoom 19, 33.5°N, got {res:.4f}"


def test_resolution_halves_with_each_zoom_level():
    res_18 = ground_resolution_m(zoom=18, latitude_deg=33.5)
    res_19 = ground_resolution_m(zoom=19, latitude_deg=33.5)
    ratio = res_18 / res_19
    assert 1.9 < ratio < 2.1, f"Resolution should halve each zoom, got ratio {ratio:.3f}"


def test_resolution_decreases_toward_poles():
    res_equator = ground_resolution_m(zoom=19, latitude_deg=0.0)
    res_60n     = ground_resolution_m(zoom=19, latitude_deg=60.0)
    assert res_60n < res_equator


def test_resolution_symmetric_hemispheres():
    north = ground_resolution_m(zoom=19, latitude_deg=33.5)
    south = ground_resolution_m(zoom=19, latitude_deg=-33.5)
    assert math.isclose(north, south, rel_tol=1e-9)


# ── validate_geotiff_coverage ─────────────────────────────────────────────────

def test_validate_coverage_ok_4326(tmp_path, augusta_poly):
    # GeoTIFF covers a larger area than the boundary
    make_geotiff(tmp_path / "ok.tif", (-82.05, 33.47, -81.99, 33.54))
    meta = validate_geotiff_coverage(tmp_path / "ok.tif", augusta_poly)
    assert "crs" in meta
    assert "bounds_wgs84" in meta
    assert meta["width_px"] == 100
    assert meta["height_px"] == 100
    assert meta["ground_resolution_m"] > 0


def test_validate_coverage_boundary_exactly_inside(tmp_path, augusta_poly):
    # Use the boundary's own bounds (zero margin) — should still pass
    bw, bs, be, bn = augusta_poly.bounds
    make_geotiff(tmp_path / "exact.tif", (bw, bs, be, bn))
    meta = validate_geotiff_coverage(tmp_path / "exact.tif", augusta_poly)
    assert meta["width_px"] == 100


def test_validate_coverage_partial_raises(tmp_path, augusta_poly):
    # GeoTIFF only covers part of the boundary
    make_geotiff(tmp_path / "partial.tif", (-82.025, 33.50, -81.99, 33.54))
    with pytest.raises(CoverageError):
        validate_geotiff_coverage(tmp_path / "partial.tif", augusta_poly)


def test_validate_coverage_no_crs_raises(tmp_path, augusta_poly):
    path = tmp_path / "nocrs.tif"
    tf = from_bounds(-82.05, 33.47, -81.99, 33.54, 100, 100)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=100, width=100, count=3, dtype="uint8",
        transform=tf,
        # deliberately omit crs= → src.crs will be None
    ) as dst:
        dst.write(np.ones((3, 100, 100), dtype=np.uint8))
    with pytest.raises(CRSError):
        validate_geotiff_coverage(path, augusta_poly)


def test_validate_coverage_3857_input(tmp_path, augusta_poly):
    """samgeo typically outputs EPSG:3857; coverage check must reproject correctly."""
    t = Transformer.from_crs(4326, 3857, always_xy=True)
    west_m, south_m = t.transform(-82.05, 33.47)
    east_m,  north_m = t.transform(-81.99, 33.54)
    make_geotiff(tmp_path / "merc.tif", (west_m, south_m, east_m, north_m), crs_epsg=3857)
    meta = validate_geotiff_coverage(tmp_path / "merc.tif", augusta_poly)
    assert "3857" in meta["crs"]


def test_validate_coverage_3857_partial_raises(tmp_path, augusta_poly):
    """Coverage check must work correctly for EPSG:3857 inputs."""
    t = Transformer.from_crs(4326, 3857, always_xy=True)
    # Only cover the eastern half of the boundary
    west_m, south_m = t.transform(-82.02, 33.47)
    east_m,  north_m = t.transform(-81.99, 33.54)
    make_geotiff(tmp_path / "partial_merc.tif", (west_m, south_m, east_m, north_m), crs_epsg=3857)
    with pytest.raises(CoverageError):
        validate_geotiff_coverage(tmp_path / "partial_merc.tif", augusta_poly)


def test_validate_coverage_bounds_wgs84_order(tmp_path, augusta_poly):
    make_geotiff(tmp_path / "ok.tif", (-82.05, 33.47, -81.99, 33.54))
    meta = validate_geotiff_coverage(tmp_path / "ok.tif", augusta_poly)
    west, south, east, north = meta["bounds_wgs84"]
    assert west < east
    assert south < north
    assert -180 <= west <= 180
    assert -90 <= south <= 90


def test_validate_coverage_resolution_plausible(tmp_path, augusta_poly):
    # At zoom 19 the resolution should be 0.2-0.5 m/px (our synthetic tif has
    # much coarser pixels, but the function derives resolution from actual pixel size)
    bw, bs, be, bn = -82.05, 33.47, -81.99, 33.54
    make_geotiff(tmp_path / "ok.tif", (bw, bs, be, bn), width=1000, height=700)
    meta = validate_geotiff_coverage(tmp_path / "ok.tif", augusta_poly)
    # 0.06° longitude / 1000px ≈ very coarse but still a positive number
    assert meta["ground_resolution_m"] > 0
