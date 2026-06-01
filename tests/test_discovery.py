"""Tests for discovery.py — pure-logic functions (no network)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from golf_mapper.config import DataConfig, GolfMapperConfig
from golf_mapper.discovery import (
    _parse_courses_from_response,
    compute_eligibility,
)
from golf_mapper.osm import count_features_from_elements


def _local_cfg(tmp_path: Path) -> GolfMapperConfig:
    """Return a GolfMapperConfig with all data paths inside tmp_path."""
    data = DataConfig(
        cache_dir=tmp_path / "cache",
        imagery_cache=tmp_path / "cache/imagery",
        overpass_cache=tmp_path / "cache/overpass",
        dataset_dir=tmp_path / "dataset",
        checkpoints_dir=tmp_path / "checkpoints",
        output_dir=tmp_path / "output",
    )
    return GolfMapperConfig().model_copy(update={"data": data})


@pytest.fixture
def classes():
    return GolfMapperConfig().classes


# ── _parse_courses_from_response ──────────────────────────────────────────────

def test_parse_courses_filters_golf_course():
    data = {
        "elements": [
            {"type": "relation", "id": 5179090, "tags": {"leisure": "golf_course", "name": "Test GC"}},
            {"type": "way",      "id": 9999,    "tags": {"leisure": "golf_course", "name": "Other GC"}},
            {"type": "node",     "id": 1,        "tags": {}},                         # nodes skipped
            {"type": "way",      "id": 100,      "tags": {"amenity": "restaurant"}},  # wrong tag
        ]
    }
    results = _parse_courses_from_response(data)
    assert len(results) == 2
    types = {r[0] for r in results}
    assert "relation" in types and "way" in types


def test_parse_courses_extracts_name():
    data = {
        "elements": [
            {
                "type": "relation",
                "id": 5179090,
                "tags": {"leisure": "golf_course", "name": "Augusta National GC"},
            }
        ]
    }
    results = _parse_courses_from_response(data)
    assert results[0][2] == "Augusta National GC"


def test_parse_courses_name_defaults_to_empty():
    data = {
        "elements": [
            {"type": "way", "id": 1, "tags": {"leisure": "golf_course"}}  # no name tag
        ]
    }
    results = _parse_courses_from_response(data)
    assert results[0][2] == ""


def test_parse_courses_empty_response():
    assert _parse_courses_from_response({"elements": []}) == []


def test_parse_courses_no_elements_key():
    assert _parse_courses_from_response({}) == []


# ── Eligibility (also tested in test_osm.py, but discovery owns the concept) ─

def test_eligibility_threshold_exactly_met():
    assert compute_eligibility({"fairway": 50}, min_features=50) is True


def test_eligibility_missing_classes_still_counts():
    counts = {"green": 10, "bunker": 5}
    assert compute_eligibility(counts, min_features=14) is True
    assert compute_eligibility(counts, min_features=16) is False


# ── discover_courses (mocked client) ─────────────────────────────────────────

@patch("golf_mapper.discovery.OverpassClient")
def test_discover_courses_single_osm_id(MockClient, tmp_path):
    """discover_courses with aoi.type='osm_id' should call correct query builders."""
    import shapely.geometry as sg

    fake_count_response = {
        "elements": [
            {"type": "way", "id": i, "tags": {"golf": cls}}
            for i, cls in enumerate(
                ["fairway"] * 18 + ["green"] * 18 + ["tee"] * 18 + ["bunker"] * 10
            )
        ]
    }

    instance = MockClient.return_value

    def query_side_effect(ql, cache_key=None):
        if cache_key and cache_key.startswith("boundary:"):
            return {"elements": []}  # _extract_boundary is patched out
        return fake_count_response

    instance.query.side_effect = query_side_effect

    square = sg.Polygon([(-82.01, 33.50), (-82.00, 33.50), (-82.00, 33.51), (-82.01, 33.51)])
    with patch("golf_mapper.discovery._extract_boundary", return_value=(square, "Augusta National GC")):
        from golf_mapper.discovery import discover_courses
        cfg = _local_cfg(tmp_path)
        cfg.data.make_dirs()
        gdf = discover_courses(cfg)

    assert len(gdf) == 1
    row = gdf.iloc[0]
    assert row["course_id"] == "relation/5179090"
    assert row["qualifies_as_training"] == True  # noqa: E712 — np.True_ != True with `is`
    assert row["total_features"] == 64   # 18+18+18+10
    counts = json.loads(row["feature_counts"])
    assert counts["fairway"] == 18
    assert counts["green"] == 18


@patch("golf_mapper.discovery.OverpassClient")
def test_discover_courses_writes_manifest(MockClient, tmp_path):
    """discover_courses should write course_manifest.gpkg to output_dir."""
    import shapely.geometry as sg

    instance = MockClient.return_value
    instance.query.return_value = {
        "elements": [
            {"type": "way", "id": i, "tags": {"golf": "fairway"}}
            for i in range(60)
        ]
    }

    square = sg.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    with patch("golf_mapper.discovery._extract_boundary", return_value=(square, "Test GC")):
        from golf_mapper.discovery import discover_courses
        cfg = _local_cfg(tmp_path)
        cfg.data.make_dirs()
        discover_courses(cfg)

    assert (tmp_path / "output" / "course_manifest.gpkg").exists()
