"""Rasterize / vectorize / simplify / validate geometry helpers.

All functions that accept GeoDataFrames assume EPSG:4326 unless a CRS is
specified. Functions that need metric distances project to UTM internally.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import geopandas as gpd
import numpy as np
from pyproj import CRS, Transformer
from rasterio.features import rasterize as _rasterize
from rasterio.features import shapes as _shapes
from rasterio.transform import Affine
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.validation import make_valid

from .utils import get_logger

log = get_logger(__name__)

_EARTH_RADIUS_M = 6_378_137.0


# ── Rasterize ─────────────────────────────────────────────────────────────────

def rasterize_geodataframe(
    gdf: gpd.GeoDataFrame,
    class_col: str,
    transform: Affine,
    out_shape: tuple[int, int],   # (height, width)
    background: int = 0,
    all_touched: bool = False,
    dtype: str = "uint8",
) -> np.ndarray:
    """Rasterize a GeoDataFrame of features to a per-pixel class-index mask.

    Each feature is painted with its value from class_col. Features are
    rendered in class_col order so that features painted later overwrite
    earlier ones (last-writer-wins).

    Args:
        gdf:         Features with a numeric class_col and valid geometry.
        class_col:   Column name holding the integer class ID.
        transform:   Rasterio Affine transform for the output grid.
        out_shape:   (height, width) of the output array.
        background:  Fill value for pixels not covered by any feature.
        all_touched: If True, burn all pixels touched (not just centres).

    Returns:
        uint8 array of shape out_shape with class indices.
    """
    if gdf.empty:
        return np.full(out_shape, background, dtype=dtype)

    shapes_iter = (
        (mapping(row.geometry), int(row[class_col]))
        for _, row in gdf.iterrows()
        if row.geometry is not None and not row.geometry.is_empty
    )

    return _rasterize(
        shapes_iter,
        out_shape=out_shape,
        fill=background,
        transform=transform,
        all_touched=all_touched,
        dtype=dtype,
    )


# ── Vectorize ─────────────────────────────────────────────────────────────────

def vectorize_semantic_mask(
    mask: np.ndarray,
    transform: Affine,
    crs: Any,
    min_area_px: int = 4,
) -> gpd.GeoDataFrame:
    """Convert a semantic label mask to a GeoDataFrame of class polygons.

    Args:
        mask:        Integer array (uint8/int32) of class indices.
        transform:   Rasterio Affine transform used to georeference the mask.
        crs:         CRS for the output GeoDataFrame.
        min_area_px: Discard polygons whose pixel area is smaller than this.

    Returns:
        GeoDataFrame with columns 'class_id' and 'geometry'.
    """
    src = mask.astype(np.int32)
    records = []
    for geom_dict, value in _shapes(src, transform=transform):
        class_id = int(value)
        if class_id == 0:
            continue  # background
        geom = shape(geom_dict)
        if geom.area < min_area_px * abs(transform.a * transform.e):
            continue
        geom = make_valid(geom)
        if not geom.is_empty:
            records.append({"class_id": class_id, "geometry": geom})

    return gpd.GeoDataFrame(records, crs=crs) if records else gpd.GeoDataFrame(
        {"class_id": [], "geometry": []}, crs=crs
    )


# ── Pixel ↔ geo conversion ────────────────────────────────────────────────────

def geo_polygon_to_pixel(
    polygon: Polygon,
    transform: Affine,
) -> Polygon | None:
    """Convert a geo-referenced polygon to continuous pixel-space coordinates.

    Uses the inverse Affine transform: (col, row) = ~transform * (x_geo, y_geo).

    Returns None if the conversion produces a degenerate polygon.
    """
    inv = ~transform
    px_coords = [inv * (x, y) for x, y in polygon.exterior.coords]
    if len(px_coords) < 3:
        return None
    poly_px = Polygon(px_coords)
    return poly_px if poly_px.is_valid and not poly_px.is_empty else None


def polygon_to_yolo_seg(
    polygon_px: Polygon,
    class_id: int,
    img_width: int,
    img_height: int,
) -> str | None:
    """Convert a pixel-space polygon to a YOLO-seg label line.

    Format: ``<class_id> x1_norm y1_norm x2_norm y2_norm ...``
    where coords are normalised to [0, 1] and (x, y) = (col/W, row/H).

    Returns None if the polygon has fewer than 3 unique points.
    """
    coords = list(polygon_px.exterior.coords[:-1])  # drop closing duplicate
    if len(coords) < 3:
        return None
    norm_parts: list[str] = []
    for col, row in coords:
        x_n = max(0.0, min(1.0, col / img_width))
        y_n = max(0.0, min(1.0, row / img_height))
        norm_parts.append(f"{x_n:.6f} {y_n:.6f}")
    return f"{class_id} " + " ".join(norm_parts)


# ── Geometry cleaning ─────────────────────────────────────────────────────────

def make_valid_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Apply shapely.make_valid to every geometry; drop null/empty results."""
    if gdf.empty:
        return gdf
    gdf = gdf.copy()
    gdf["geometry"] = gdf["geometry"].apply(
        lambda g: make_valid(g) if g is not None else None
    )
    return gdf[~gdf["geometry"].is_empty & gdf["geometry"].notna()].copy()


def simplify_metres(
    gdf: gpd.GeoDataFrame,
    tolerance_m: float,
) -> gpd.GeoDataFrame:
    """Simplify geometries using Douglas-Peucker with a tolerance in metres.

    Projects to UTM for accurate simplification, then reprojects back.
    """
    if gdf.empty or tolerance_m <= 0:
        return gdf
    orig_crs = gdf.crs or CRS.from_epsg(4326)
    centroid = gdf.geometry.union_all().centroid
    utm_epsg = _utm_epsg_for_point(centroid.x, centroid.y)
    gdf_utm = gdf.to_crs(f"EPSG:{utm_epsg}")
    gdf_utm = gdf_utm.copy()
    gdf_utm["geometry"] = gdf_utm["geometry"].simplify(
        tolerance_m, preserve_topology=True
    )
    return gdf_utm.to_crs(orig_crs)


def clip_to_boundary(
    gdf: gpd.GeoDataFrame,
    boundary: Any,   # shapely geometry in the same CRS as gdf
) -> gpd.GeoDataFrame:
    """Clip all features to the course boundary; drop empty results."""
    if gdf.empty:
        return gdf
    clipped = gdf.clip(boundary)
    return clipped[~clipped.geometry.is_empty].copy()


def drop_small_polygons(
    gdf: gpd.GeoDataFrame,
    min_area_m2: float,
) -> gpd.GeoDataFrame:
    """Drop polygons whose area (in m²) is below min_area_m2.

    Projects to UTM for area calculation.
    """
    if gdf.empty or min_area_m2 <= 0:
        return gdf
    orig_crs = gdf.crs or CRS.from_epsg(4326)
    centroid = gdf.geometry.union_all().centroid
    utm_epsg = _utm_epsg_for_point(centroid.x, centroid.y)
    areas = gdf.to_crs(f"EPSG:{utm_epsg}").geometry.area
    return gdf[areas >= min_area_m2].copy()


# ── Vegetation index ──────────────────────────────────────────────────────────

def exg_canopy_mask(
    rgb: np.ndarray,
    threshold: float = 0.1,
) -> np.ndarray:
    """Compute Excess Green index (ExG = 2G − R − B) and threshold to binary.

    Args:
        rgb:       Float or uint8 array of shape (H, W, 3) in RGB channel order.
        threshold: ExG values above this are classified as vegetation.

    Returns:
        Boolean mask of shape (H, W).
    """
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    exg = 2.0 * g - r - b
    return exg > threshold


# ── Internal helpers ──────────────────────────────────────────────────────────

def _utm_epsg_for_point(lon: float, lat: float) -> int:
    zone = int((lon + 180.0) / 6.0) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone
