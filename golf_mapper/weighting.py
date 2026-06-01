"""Proximity-based tile weighting for regional fine-tuning.

Courses closer to the target are assigned higher training weights so that the
model adapts to the regional appearance (e.g. soil colour, turf tone, water
clarity) of the area being mapped.
"""
from __future__ import annotations

import math
import random
from typing import Any

from .utils import get_logger

log = get_logger(__name__)

_EARTH_RADIUS_KM = 6_371.0


# ── Distance ──────────────────────────────────────────────────────────────────

def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance between two WGS84 points in kilometres."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))


# ── Weight functions ──────────────────────────────────────────────────────────

def proximity_weight(
    dist_km: float,
    sigma_km: float,
    max_radius_km: float,
) -> float:
    """Gaussian proximity weight w = exp(−(d/σ)²), clipped to 0 beyond max_radius.

    Args:
        dist_km:       Distance from training course to target course (km).
        sigma_km:      Gaussian decay width (σ). Default config value: 150 km.
        max_radius_km: Courses beyond this radius receive weight 0.

    Returns:
        Weight in [0, 1]. Returns 0.0 for d > max_radius_km.
    """
    if dist_km > max_radius_km:
        return 0.0
    return math.exp(-((dist_km / sigma_km) ** 2))


def compute_course_weights(
    target_lon: float,
    target_lat: float,
    manifest_gdf: Any,   # gpd.GeoDataFrame with centroid_lon / centroid_lat cols
    sigma_km: float,
    max_radius_km: float,
) -> dict[str, float]:
    """Return {course_id: weight} for all training courses in the manifest.

    Courses with weight 0 (beyond max_radius_km) are excluded from the result.

    Args:
        target_lon / target_lat:  Target course centroid in EPSG:4326.
        manifest_gdf:             Course manifest GeoDataFrame (from discovery).
        sigma_km:                 Gaussian σ for proximity_weight.
        max_radius_km:            Hard cutoff distance.

    Returns:
        Dict mapping course_id → raw (unnormalised) weight.
    """
    weights: dict[str, float] = {}
    for _, row in manifest_gdf.iterrows():
        cid = row["course_id"]
        dist = haversine_km(
            target_lon, target_lat,
            float(row["centroid_lon"]), float(row["centroid_lat"]),
        )
        w = proximity_weight(dist, sigma_km, max_radius_km)
        if w > 0:
            weights[cid] = w
        else:
            log.debug("Excluding %s from regional fine-tune (dist=%.0f km > max)", cid, dist)
    log.info(
        "Proximity weights: %d/%d courses within radius %.0f km (σ=%.0f km)",
        len(weights), len(manifest_gdf), max_radius_km, sigma_km,
    )
    return weights


def weighted_tile_sample(
    tile_paths_by_course: dict[str, list[str]],
    course_weights: dict[str, float],
    seed: int = 42,
) -> list[str]:
    """Return a weighted list of tile paths for training, duplicating high-weight tiles.

    Each course's tiles are repeated proportional to its weight (rounded to the
    nearest integer, minimum 1 for any included course).  This implements
    approximate weighted sampling without a custom DataLoader sampler.

    Args:
        tile_paths_by_course: {course_id: [tile_path, ...]} for all tiles.
        course_weights:       {course_id: weight} (0-excluded courses omitted).

    Returns:
        Shuffled list of tile paths with duplication applied.
    """
    if not course_weights:
        # Global mode or no weights — return all tiles once
        all_tiles = [p for paths in tile_paths_by_course.values() for p in paths]
        random.seed(seed)
        random.shuffle(all_tiles)
        return all_tiles

    max_w = max(course_weights.values())
    result: list[str] = []
    for course_id, paths in tile_paths_by_course.items():
        w = course_weights.get(course_id, 0.0)
        if w == 0.0:
            continue
        repeats = max(1, round(paths.__len__() * (w / max_w)))
        result.extend(paths * repeats)

    random.seed(seed)
    random.shuffle(result)
    log.info(
        "Weighted tile sample: %d tiles from %d courses.",
        len(result), len([c for c in tile_paths_by_course if course_weights.get(c, 0) > 0]),
    )
    return result
