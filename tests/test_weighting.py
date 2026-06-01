"""Tests for weighting.py — proximity weight math and tile sampling."""
from __future__ import annotations

import math
import pytest

from golf_mapper.weighting import (
    compute_course_weights,
    haversine_km,
    proximity_weight,
    weighted_tile_sample,
)


# ── haversine_km ──────────────────────────────────────────────────────────────

def test_haversine_same_point():
    assert haversine_km(0, 0, 0, 0) == pytest.approx(0.0)


def test_haversine_equator_1deg():
    # 1° of longitude at the equator ≈ 111.19 km
    d = haversine_km(0.0, 0.0, 1.0, 0.0)
    assert 110 < d < 112


def test_haversine_pole_to_pole():
    # North pole to south pole ≈ 20,004 km (half of Earth's circumference)
    d = haversine_km(0.0, 90.0, 0.0, -90.0)
    assert 19_900 < d < 20_100


def test_haversine_symmetric():
    d1 = haversine_km(-82.0, 33.5, -74.0, 40.7)   # Augusta → NYC
    d2 = haversine_km(-74.0, 40.7, -82.0, 33.5)
    assert d1 == pytest.approx(d2, rel=1e-9)


def test_haversine_augusta_to_nyc():
    # Augusta, GA → New York City: haversine ≈ 1,071 km
    d = haversine_km(-82.02, 33.50, -74.00, 40.71)
    assert 950 < d < 1_200


# ── proximity_weight ──────────────────────────────────────────────────────────

def test_weight_zero_distance():
    w = proximity_weight(dist_km=0.0, sigma_km=150.0, max_radius_km=500.0)
    assert w == pytest.approx(1.0)


def test_weight_at_sigma_is_1_over_e():
    w = proximity_weight(dist_km=150.0, sigma_km=150.0, max_radius_km=500.0)
    assert w == pytest.approx(math.exp(-1), rel=1e-9)


def test_weight_beyond_max_radius_is_zero():
    w = proximity_weight(dist_km=600.0, sigma_km=150.0, max_radius_km=500.0)
    assert w == 0.0


def test_weight_at_max_radius_is_nonzero():
    w = proximity_weight(dist_km=500.0, sigma_km=150.0, max_radius_km=500.0)
    assert w > 0.0


def test_weight_decreases_with_distance():
    w1 = proximity_weight(100, sigma_km=150, max_radius_km=500)
    w2 = proximity_weight(200, sigma_km=150, max_radius_km=500)
    w3 = proximity_weight(300, sigma_km=150, max_radius_km=500)
    assert w1 > w2 > w3


def test_weight_is_in_unit_interval():
    for d in [0, 50, 150, 300, 499]:
        w = proximity_weight(d, sigma_km=150, max_radius_km=500)
        assert 0.0 <= w <= 1.0


def test_weight_sigma_effect():
    # Wider sigma → higher weight at same distance
    w_narrow = proximity_weight(100, sigma_km=100, max_radius_km=500)
    w_wide   = proximity_weight(100, sigma_km=300, max_radius_km=500)
    assert w_wide > w_narrow


# ── compute_course_weights ────────────────────────────────────────────────────

def test_compute_course_weights_excludes_beyond_radius():
    import geopandas as gpd
    from shapely.geometry import Point

    manifest = gpd.GeoDataFrame({
        "course_id":    ["relation/1", "relation/2"],
        "centroid_lon": [-82.0,       -74.0],      # nearby, far
        "centroid_lat": [33.5,         40.7],
        "geometry":     [Point(-82.0, 33.5), Point(-74.0, 40.7)],
    }, crs="EPSG:4326")

    weights = compute_course_weights(
        target_lon=-82.02, target_lat=33.50,
        manifest_gdf=manifest,
        sigma_km=150, max_radius_km=200,
    )
    # Augusta → Augusta ≈ 0 km → weight ≈ 1
    # Augusta → NYC ≈ 1,400 km > 200 km → excluded
    assert "relation/1" in weights
    assert "relation/2" not in weights


def test_compute_course_weights_target_course_has_high_weight():
    import geopandas as gpd
    from shapely.geometry import Point

    manifest = gpd.GeoDataFrame({
        "course_id":    ["relation/A", "relation/B"],
        "centroid_lon": [-82.02, -90.0],
        "centroid_lat": [33.50,   35.0],
        "geometry":     [Point(-82.02, 33.50), Point(-90.0, 35.0)],
    }, crs="EPSG:4326")

    weights = compute_course_weights(
        -82.02, 33.50, manifest, sigma_km=150, max_radius_km=2000
    )
    # Closest course should have highest weight
    assert weights["relation/A"] > weights["relation/B"]


# ── weighted_tile_sample ──────────────────────────────────────────────────────

def test_weighted_tile_sample_excludes_zero_weight():
    tile_map = {
        "relation/A": ["t1.png", "t2.png"],
        "relation/B": ["t3.png"],
    }
    weights = {"relation/A": 1.0}   # B has no weight → excluded
    result = weighted_tile_sample(tile_map, weights, seed=42)
    assert all("t3" not in p for p in result)
    assert any("t1" in p or "t2" in p for p in result)


def test_weighted_tile_sample_empty_weights_returns_all():
    tile_map = {"relation/A": ["a.png"], "relation/B": ["b.png"]}
    result = weighted_tile_sample(tile_map, {}, seed=0)
    assert len(result) == 2
    assert "a.png" in result and "b.png" in result


def test_weighted_tile_sample_deterministic_seed():
    tile_map = {"relation/A": [f"t{i}.png" for i in range(10)]}
    weights = {"relation/A": 1.0}
    r1 = weighted_tile_sample(tile_map, weights, seed=7)
    r2 = weighted_tile_sample(tile_map, weights, seed=7)
    assert r1 == r2


def test_weighted_tile_sample_high_weight_duplicates():
    tile_map = {
        "relation/high": ["h.png"],
        "relation/low":  ["l.png"],
    }
    weights = {"relation/high": 1.0, "relation/low": 0.01}
    result = weighted_tile_sample(tile_map, weights, seed=0)
    n_high = result.count("h.png")
    n_low  = result.count("l.png")
    # High-weight course should appear more often
    assert n_high >= n_low
