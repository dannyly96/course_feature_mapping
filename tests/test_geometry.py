"""Tests for geometry.py — rasterize/vectorize round-trip, YOLO format, ExG."""
from __future__ import annotations

import numpy as np
import pytest
import geopandas as gpd
from rasterio.transform import from_bounds
from shapely.geometry import Polygon, MultiPolygon, box as bbox_polygon

from golf_mapper.geometry import (
    clip_to_boundary,
    drop_small_polygons,
    exg_canopy_mask,
    geo_polygon_to_pixel,
    make_valid_gdf,
    polygon_to_yolo_seg,
    rasterize_geodataframe,
    simplify_metres,
    vectorize_semantic_mask,
    _utm_epsg_for_point,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def unit_transform(width=100, height=100):
    """Affine transform mapping a (0,0)→(1,1) degree bbox to a 100×100 grid."""
    return from_bounds(0, 0, 1, 1, width, height)


def sample_gdf(polygons, class_ids, crs="EPSG:4326"):
    return gpd.GeoDataFrame(
        {"class_id": class_ids, "geometry": polygons},
        crs=crs,
    )


# ── rasterize_geodataframe ────────────────────────────────────────────────────

def test_rasterize_returns_correct_shape():
    gdf = sample_gdf([bbox_polygon(0.1, 0.1, 0.9, 0.9)], [2])
    tf = unit_transform()
    mask = rasterize_geodataframe(gdf, "class_id", tf, (100, 100))
    assert mask.shape == (100, 100)
    assert mask.dtype == np.uint8


def test_rasterize_background_fill():
    gdf = sample_gdf([bbox_polygon(0.3, 0.3, 0.7, 0.7)], [3])
    tf = unit_transform()
    mask = rasterize_geodataframe(gdf, "class_id", tf, (100, 100), background=0)
    assert mask[0, 0] == 0     # corner is background


def test_rasterize_correct_class_id():
    gdf = sample_gdf([bbox_polygon(0.1, 0.1, 0.9, 0.9)], [5])
    tf = unit_transform()
    mask = rasterize_geodataframe(gdf, "class_id", tf, (100, 100))
    center_val = mask[50, 50]
    assert center_val == 5


def test_rasterize_empty_gdf():
    gdf = gpd.GeoDataFrame({"class_id": [], "geometry": []}, crs="EPSG:4326")
    tf = unit_transform()
    mask = rasterize_geodataframe(gdf, "class_id", tf, (50, 50), background=0)
    assert np.all(mask == 0)


def test_rasterize_multiple_classes():
    g1 = bbox_polygon(0.0, 0.0, 0.5, 1.0)
    g2 = bbox_polygon(0.5, 0.0, 1.0, 1.0)
    gdf = sample_gdf([g1, g2], [1, 2])
    tf = unit_transform()
    mask = rasterize_geodataframe(gdf, "class_id", tf, (100, 100))
    assert mask[50, 25] == 1   # left half → class 1
    assert mask[50, 75] == 2   # right half → class 2


# ── vectorize_semantic_mask ───────────────────────────────────────────────────

def test_vectorize_returns_geodataframe():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:40, 10:40] = 2
    tf = unit_transform()
    gdf = vectorize_semantic_mask(mask, tf, "EPSG:4326")
    assert isinstance(gdf, gpd.GeoDataFrame)


def test_vectorize_skips_background():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:40, 10:40] = 2
    tf = unit_transform()
    gdf = vectorize_semantic_mask(mask, tf, "EPSG:4326")
    assert (gdf["class_id"] == 0).sum() == 0


def test_vectorize_recovers_class_id():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 4
    tf = unit_transform()
    gdf = vectorize_semantic_mask(mask, tf, "EPSG:4326")
    assert 4 in gdf["class_id"].values


def test_vectorize_empty_mask():
    mask = np.zeros((100, 100), dtype=np.uint8)
    tf = unit_transform()
    gdf = vectorize_semantic_mask(mask, tf, "EPSG:4326")
    assert gdf.empty


# ── rasterize ↔ vectorize round-trip ─────────────────────────────────────────

def test_rasterize_vectorize_roundtrip_area():
    """Vectorized polygon should have ~similar area to original."""
    tf = unit_transform(width=200, height=200)
    original = bbox_polygon(0.2, 0.2, 0.8, 0.8)
    gdf = sample_gdf([original], [3])

    mask = rasterize_geodataframe(gdf, "class_id", tf, (200, 200))
    recovered_gdf = vectorize_semantic_mask(mask, tf, "EPSG:4326")

    assert not recovered_gdf.empty
    recovered_area = recovered_gdf.geometry.union_all().area
    # Allow up to 3% difference due to pixel discretisation
    assert abs(recovered_area - original.area) / original.area < 0.03


def test_rasterize_vectorize_roundtrip_overlap():
    """Recovered polygon must substantially overlap original (IoU > 0.9)."""
    tf = unit_transform(width=200, height=200)
    original = bbox_polygon(0.25, 0.25, 0.75, 0.75)
    gdf = sample_gdf([original], [2])

    mask = rasterize_geodataframe(gdf, "class_id", tf, (200, 200))
    recovered_gdf = vectorize_semantic_mask(mask, tf, "EPSG:4326")

    recovered = recovered_gdf.geometry.union_all()
    iou = recovered.intersection(original).area / recovered.union(original).area
    assert iou > 0.90, f"Round-trip IoU too low: {iou:.3f}"


# ── YOLO label format ─────────────────────────────────────────────────────────

def test_polygon_to_yolo_seg_starts_with_class_id():
    poly = Polygon([(10, 20), (90, 20), (90, 80), (10, 80)])
    line = polygon_to_yolo_seg(poly, class_id=3, img_width=100, img_height=100)
    assert line is not None
    assert line.startswith("3 ")


def test_polygon_to_yolo_seg_normalized_coords():
    poly = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    line = polygon_to_yolo_seg(poly, class_id=1, img_width=100, img_height=100)
    parts = line.split()
    xs = [float(parts[i]) for i in range(1, len(parts), 2)]
    ys = [float(parts[i]) for i in range(2, len(parts), 2)]
    assert all(0.0 <= x <= 1.0 for x in xs)
    assert all(0.0 <= y <= 1.0 for y in ys)


def test_polygon_to_yolo_seg_pair_count():
    # 4-vertex square → 4 coord pairs
    poly = Polygon([(10, 10), (90, 10), (90, 90), (10, 90)])
    line = polygon_to_yolo_seg(poly, 2, 100, 100)
    parts = line.split()
    n_coords = len(parts) - 1    # subtract class_id
    assert n_coords % 2 == 0
    assert n_coords // 2 == 4


def test_polygon_to_yolo_seg_coords_clamped_to_unit():
    # Polygon extending beyond the image — coords must be clamped to [0, 1]
    poly = Polygon([(-10, -10), (200, -10), (200, 200), (-10, 200)])
    line = polygon_to_yolo_seg(poly, 0, 100, 100)
    assert line is not None
    parts = line.split()
    xs = [float(parts[i]) for i in range(1, len(parts), 2)]
    ys = [float(parts[i]) for i in range(2, len(parts), 2)]
    assert all(0.0 <= x <= 1.0 for x in xs)
    assert all(0.0 <= y <= 1.0 for y in ys)


def test_geo_polygon_to_pixel_roundtrip():
    tf = unit_transform()
    geo_poly = bbox_polygon(0.1, 0.2, 0.4, 0.8)
    px_poly = geo_polygon_to_pixel(geo_poly, tf)
    assert px_poly is not None
    assert isinstance(px_poly, Polygon)
    # The pixel bounding box should be within [0, 100]
    minx, miny, maxx, maxy = px_poly.bounds
    assert 0 <= minx and maxx <= 100
    assert 0 <= miny and maxy <= 100


# ── ExG canopy mask ───────────────────────────────────────────────────────────

def test_exg_pure_green():
    rgb = np.zeros((10, 10, 3), dtype=np.float32)
    rgb[..., 1] = 1.0   # pure green → ExG = 2(1) - 0 - 0 = 2
    mask = exg_canopy_mask(rgb, threshold=0.1)
    assert mask.all()


def test_exg_pure_sand():
    # Bright sand: R≈G≈B → ExG ≈ 0
    rgb = np.ones((10, 10, 3), dtype=np.float32) * 0.9
    mask = exg_canopy_mask(rgb, threshold=0.1)
    assert not mask.any()


def test_exg_uint8_input():
    rgb = np.zeros((5, 5, 3), dtype=np.uint8)
    rgb[..., 1] = 200   # green channel dominant
    mask = exg_canopy_mask(rgb, threshold=0.1)
    assert mask.any()


def test_exg_output_shape():
    rgb = np.zeros((20, 30, 3), dtype=np.float32)
    mask = exg_canopy_mask(rgb)
    assert mask.shape == (20, 30)


def test_exg_threshold_effect():
    rgb = np.zeros((5, 5, 3), dtype=np.float32)
    rgb[..., 1] = 0.2   # mild green, ExG = 0.4 - 0 - 0 = 0.4
    low_mask  = exg_canopy_mask(rgb, threshold=0.1)
    high_mask = exg_canopy_mask(rgb, threshold=0.5)
    assert low_mask.any()
    assert not high_mask.any()


# ── make_valid_gdf ────────────────────────────────────────────────────────────

def test_make_valid_gdf_drops_none():
    gdf = gpd.GeoDataFrame({"geometry": [None, bbox_polygon(0, 0, 1, 1)]}, crs="EPSG:4326")
    result = make_valid_gdf(gdf)
    assert len(result) == 1


def test_make_valid_gdf_drops_empty():
    from shapely import wkt
    gdf = gpd.GeoDataFrame(
        {"geometry": [Polygon(), bbox_polygon(0, 0, 1, 1)]},
        crs="EPSG:4326",
    )
    result = make_valid_gdf(gdf)
    assert len(result) == 1


# ── clip_to_boundary ──────────────────────────────────────────────────────────

def test_clip_removes_outside():
    boundary = bbox_polygon(0, 0, 1, 1)
    # one polygon inside, one fully outside
    inside  = bbox_polygon(0.1, 0.1, 0.9, 0.9)
    outside = bbox_polygon(2.0, 2.0, 3.0, 3.0)
    gdf = sample_gdf([inside, outside], [1, 2])
    clipped = clip_to_boundary(gdf, boundary)
    assert len(clipped) == 1


def test_clip_empty_gdf():
    gdf = gpd.GeoDataFrame({"class_id": [], "geometry": []}, crs="EPSG:4326")
    boundary = bbox_polygon(0, 0, 1, 1)
    result = clip_to_boundary(gdf, boundary)
    assert result.empty


# ── drop_small_polygons ───────────────────────────────────────────────────────

def test_drop_small_keeps_large():
    # Augusta-scale polygon ~1 km² (huge)
    large = bbox_polygon(-82.03, 33.49, -82.01, 33.52)
    gdf = sample_gdf([large], [1])
    result = drop_small_polygons(gdf, min_area_m2=10.0)
    assert len(result) == 1


def test_drop_small_removes_tiny():
    # Tiny 0.00001° × 0.00001° square ≈ 1 m²
    tiny = bbox_polygon(-82.0, 33.5, -82.0 + 1e-5, 33.5 + 1e-5)
    gdf = sample_gdf([tiny], [1])
    result = drop_small_polygons(gdf, min_area_m2=1000.0)
    assert result.empty
