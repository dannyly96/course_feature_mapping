"""Tests for evaluate.py — IoU, matching, precision/recall, QA, report output."""
from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Polygon, box as bbox_polygon

from golf_mapper.config import GolfMapperConfig
from golf_mapper.evaluate import (
    class_precision_recall_iou,
    evaluate_course,
    geometry_qa,
    match_predictions,
    polygon_iou,
    write_evaluation_report,
)


# ── polygon_iou ───────────────────────────────────────────────────────────────

def test_iou_identical_polygons():
    p = bbox_polygon(0, 0, 1, 1)
    assert polygon_iou(p, p) == pytest.approx(1.0)


def test_iou_no_overlap():
    p1 = bbox_polygon(0, 0, 1, 1)
    p2 = bbox_polygon(2, 2, 3, 3)
    assert polygon_iou(p1, p2) == pytest.approx(0.0)


def test_iou_half_overlap():
    # Two rectangles sharing half their area
    p1 = bbox_polygon(0, 0, 2, 1)   # area = 2
    p2 = bbox_polygon(1, 0, 3, 1)   # area = 2, overlap = 1
    # IoU = 1 / 3
    assert polygon_iou(p1, p2) == pytest.approx(1 / 3, rel=1e-6)


def test_iou_contained():
    outer = bbox_polygon(0, 0, 4, 4)   # area = 16
    inner = bbox_polygon(1, 1, 3, 3)   # area = 4, contained
    # IoU = 4 / 16 = 0.25
    assert polygon_iou(inner, outer) == pytest.approx(0.25, rel=1e-6)


def test_iou_symmetric():
    p1 = bbox_polygon(0, 0, 2, 1)
    p2 = bbox_polygon(1, 0, 3, 1)
    assert polygon_iou(p1, p2) == pytest.approx(polygon_iou(p2, p1))


# ── match_predictions ─────────────────────────────────────────────────────────

def _make_cls_gdf(polys, cls_name):
    return gpd.GeoDataFrame(
        {"class_name": [cls_name] * len(polys), "geometry": polys},
        crs="EPSG:4326",
    )


def test_match_perfect():
    poly = bbox_polygon(0, 0, 1, 1)
    pred = _make_cls_gdf([poly], "green")
    ref  = _make_cls_gdf([poly], "green")
    matched, unmatched_p, unmatched_r = match_predictions(pred, ref, iou_threshold=0.5)
    assert len(matched) == 1
    assert unmatched_p == []
    assert unmatched_r == []


def test_match_no_overlap():
    pred = _make_cls_gdf([bbox_polygon(0, 0, 1, 1)], "green")
    ref  = _make_cls_gdf([bbox_polygon(5, 5, 6, 6)], "green")
    matched, unmatched_p, unmatched_r = match_predictions(pred, ref, iou_threshold=0.5)
    assert len(matched) == 0
    assert len(unmatched_p) == 1
    assert len(unmatched_r) == 1


def test_match_different_class_not_matched():
    poly = bbox_polygon(0, 0, 1, 1)
    pred = _make_cls_gdf([poly], "fairway")
    ref  = _make_cls_gdf([poly], "green")
    matched, _, _ = match_predictions(pred, ref, iou_threshold=0.5)
    assert len(matched) == 0


def test_match_iou_threshold_respected():
    # IoU ≈ 1/3 — passes at 0.3, fails at 0.5
    p1 = bbox_polygon(0, 0, 2, 1)
    p2 = bbox_polygon(1, 0, 3, 1)
    pred = _make_cls_gdf([p1], "bunker")
    ref  = _make_cls_gdf([p2], "bunker")
    matched_low, _, _ = match_predictions(pred, ref, iou_threshold=0.30)
    matched_high, _, _ = match_predictions(pred, ref, iou_threshold=0.50)
    assert len(matched_low) == 1
    assert len(matched_high) == 0


# ── class_precision_recall_iou ────────────────────────────────────────────────

def test_precision_recall_perfect():
    poly = bbox_polygon(0, 0, 1, 1)
    pred = _make_cls_gdf([poly], "green")
    ref  = _make_cls_gdf([poly], "green")
    m = class_precision_recall_iou(pred, ref, "green", iou_threshold=0.5)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"]    == pytest.approx(1.0)
    assert m["mean_iou"]  == pytest.approx(1.0)
    assert m["n_matched"] == 1


def test_precision_recall_false_positive():
    pred = _make_cls_gdf([bbox_polygon(0,0,1,1), bbox_polygon(3,3,4,4)], "bunker")
    ref  = _make_cls_gdf([bbox_polygon(0,0,1,1)], "bunker")
    m = class_precision_recall_iou(pred, ref, "bunker", 0.5)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"]    == pytest.approx(1.0)


def test_precision_recall_false_negative():
    pred = _make_cls_gdf([bbox_polygon(0,0,1,1)], "tee")
    ref  = _make_cls_gdf([bbox_polygon(0,0,1,1), bbox_polygon(3,3,4,4)], "tee")
    m = class_precision_recall_iou(pred, ref, "tee", 0.5)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"]    == pytest.approx(0.5)


def test_precision_recall_both_empty():
    pred = _make_cls_gdf([], "water_hazard")
    ref  = _make_cls_gdf([], "water_hazard")
    m = class_precision_recall_iou(pred, ref, "water_hazard", 0.5)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"]    == pytest.approx(1.0)
    assert m["n_matched"] == 0


def test_precision_recall_no_pred():
    pred = _make_cls_gdf([], "fairway")
    ref  = _make_cls_gdf([bbox_polygon(0,0,1,1)], "fairway")
    m = class_precision_recall_iou(pred, ref, "fairway", 0.5)
    assert m["precision"] == pytest.approx(0.0)
    assert m["recall"]    == pytest.approx(0.0)


# ── evaluate_course ───────────────────────────────────────────────────────────

def test_evaluate_course_returns_all_classes():
    cfg = GolfMapperConfig()
    pred = gpd.GeoDataFrame({"class_name": [], "geometry": []}, crs="EPSG:4326")
    ref  = gpd.GeoDataFrame({"class_name": [], "geometry": []}, crs="EPSG:4326")
    results = evaluate_course(pred, ref, cfg)
    non_bg_classes = [c.name for c in cfg.classes if not c.is_background]
    for cls in non_bg_classes:
        assert cls in results


# ── geometry_qa ───────────────────────────────────────────────────────────────

def test_geometry_qa_clean():
    gdf = gpd.GeoDataFrame(
        {"geometry": [bbox_polygon(0,0,1,1), bbox_polygon(2,2,3,3)]},
        crs="EPSG:4326",
    )
    qa = geometry_qa(gdf)
    assert qa["total"] == 2
    assert qa["invalid"] == 0
    assert qa["empty"] == 0


def test_geometry_qa_counts_empty():
    gdf = gpd.GeoDataFrame(
        {"geometry": [bbox_polygon(0,0,1,1), Polygon()]},
        crs="EPSG:4326",
    )
    qa = geometry_qa(gdf)
    assert qa["empty"] == 1


# ── write_evaluation_report ───────────────────────────────────────────────────

def test_write_evaluation_report_creates_files(tmp_path):
    metrics = {
        "fairway": {"precision": 0.9, "recall": 0.85, "mean_iou": 0.75,
                    "n_pred": 18, "n_ref": 18, "n_matched": 16},
        "green":   {"precision": 1.0, "recall": 1.0,  "mean_iou": 0.95,
                    "n_pred": 18, "n_ref": 18, "n_matched": 18},
    }
    csv_path, md_path = write_evaluation_report(metrics, tmp_path, course_id="relation/5179090")
    assert csv_path.exists()
    assert md_path.exists()


def test_write_evaluation_report_csv_columns(tmp_path):
    import csv as csv_module
    metrics = {
        "bunker": {"precision": 0.8, "recall": 0.7, "mean_iou": 0.6,
                   "n_pred": 10, "n_ref": 12, "n_matched": 8},
    }
    csv_path, _ = write_evaluation_report(metrics, tmp_path)
    with csv_path.open() as fh:
        reader = csv_module.DictReader(fh)
        rows = list(reader)
    assert rows[0]["class"] == "bunker"
    assert float(rows[0]["precision"]) == pytest.approx(0.8)


def test_write_evaluation_report_markdown_has_table(tmp_path):
    metrics = {
        "tee": {"precision": 1.0, "recall": 1.0, "mean_iou": 1.0,
                "n_pred": 5, "n_ref": 5, "n_matched": 5},
    }
    _, md_path = write_evaluation_report(metrics, tmp_path)
    content = md_path.read_text()
    assert "| Class |" in content
    assert "tee" in content
