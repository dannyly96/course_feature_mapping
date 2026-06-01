"""Held-out evaluation and geometry QA report.

Computes per-class precision, recall, and IoU by matching predicted polygons
to reference OSM polygons on held-out training courses.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

from .config import GolfMapperConfig
from .utils import ensure_dir, get_logger

log = get_logger(__name__)


# ── IoU / matching ────────────────────────────────────────────────────────────

def polygon_iou(pred: Polygon, ref: Polygon) -> float:
    """Intersection-over-Union of two Shapely polygons."""
    try:
        inter = pred.intersection(ref).area
        union = pred.union(ref).area
        return float(inter / union) if union > 0 else 0.0
    except Exception:
        return 0.0


def match_predictions(
    pred_gdf: gpd.GeoDataFrame,
    ref_gdf: gpd.GeoDataFrame,
    iou_threshold: float = 0.5,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedily match predicted polygons to reference polygons by IoU.

    Args:
        pred_gdf:      Predicted polygons with a 'class_name' column.
        ref_gdf:       Reference (OSM) polygons with a 'class_name' column.
        iou_threshold: Minimum IoU to count as a match.

    Returns:
        (matched_pairs, unmatched_pred_ids, unmatched_ref_ids)
        where matched_pairs is a list of (pred_idx, ref_idx) integer index pairs.
    """
    matched: list[tuple[int, int]] = []
    used_pred: set[int] = set()
    used_ref: set[int] = set()

    # Build IoU matrix for same-class pairs
    for pi, pred_row in pred_gdf.iterrows():
        if pi in used_pred:
            continue
        best_iou = 0.0
        best_ri = -1
        for ri, ref_row in ref_gdf.iterrows():
            if ri in used_ref:
                continue
            if pred_row.get("class_name") != ref_row.get("class_name"):
                continue
            iou = polygon_iou(pred_row.geometry, ref_row.geometry)
            if iou > best_iou:
                best_iou = iou
                best_ri = ri
        if best_iou >= iou_threshold and best_ri >= 0:
            matched.append((int(pi), int(best_ri)))
            used_pred.add(pi)
            used_ref.add(best_ri)

    unmatched_pred = [i for i in pred_gdf.index if i not in used_pred]
    unmatched_ref  = [i for i in ref_gdf.index  if i not in used_ref]
    return matched, unmatched_pred, unmatched_ref


# ── Per-class metrics ─────────────────────────────────────────────────────────

def class_precision_recall_iou(
    pred_gdf: gpd.GeoDataFrame,
    ref_gdf: gpd.GeoDataFrame,
    class_name: str,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute precision, recall, and mean IoU for one class.

    Returns:
        {'precision': float, 'recall': float, 'mean_iou': float,
         'n_pred': int, 'n_ref': int, 'n_matched': int}
    """
    pred_cls = pred_gdf[pred_gdf["class_name"] == class_name].copy()
    ref_cls  = ref_gdf[ref_gdf["class_name"]  == class_name].copy()

    n_pred = len(pred_cls)
    n_ref  = len(ref_cls)

    if n_pred == 0 and n_ref == 0:
        return {"precision": 1.0, "recall": 1.0, "mean_iou": 1.0,
                "n_pred": 0, "n_ref": 0, "n_matched": 0}
    if n_pred == 0:
        return {"precision": 0.0, "recall": 0.0, "mean_iou": 0.0,
                "n_pred": 0, "n_ref": n_ref, "n_matched": 0}
    if n_ref == 0:
        return {"precision": 0.0, "recall": 1.0, "mean_iou": 0.0,
                "n_pred": n_pred, "n_ref": 0, "n_matched": 0}

    matched, _, _ = match_predictions(pred_cls, ref_cls, iou_threshold)
    n_matched = len(matched)

    ious = [polygon_iou(pred_cls.loc[pi].geometry, ref_cls.loc[ri].geometry)
            for pi, ri in matched]
    mean_iou = float(np.mean(ious)) if ious else 0.0

    precision = n_matched / n_pred if n_pred else 0.0
    recall    = n_matched / n_ref  if n_ref  else 0.0

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "mean_iou":  round(mean_iou, 4),
        "n_pred":    n_pred,
        "n_ref":     n_ref,
        "n_matched": n_matched,
    }


def evaluate_course(
    pred_gdf: gpd.GeoDataFrame,
    ref_gdf: gpd.GeoDataFrame,
    cfg: GolfMapperConfig,
    iou_threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Compute per-class metrics for one course.

    Args:
        pred_gdf:      Predicted polygons (class_name column required).
        ref_gdf:       Reference OSM polygons (class_name column required).
        cfg:           Pipeline config (used for class list).
        iou_threshold: Match threshold.

    Returns:
        {class_name: {precision, recall, mean_iou, n_pred, n_ref, n_matched}}
    """
    results: dict[str, dict] = {}
    for cls in cfg.classes:
        if cls.is_background:
            continue
        results[cls.name] = class_precision_recall_iou(
            pred_gdf, ref_gdf, cls.name, iou_threshold
        )
    return results


# ── Geometry QA ───────────────────────────────────────────────────────────────

def geometry_qa(gdf: gpd.GeoDataFrame) -> dict[str, int]:
    """Count geometry quality issues in a GeoDataFrame.

    Returns:
        {'total': n, 'invalid': n, 'empty': n, 'repaired': n, 'degenerate': n}
    """
    total = len(gdf)
    n_empty   = int(gdf.geometry.is_empty.sum())
    n_invalid = int((~gdf.geometry.is_valid & ~gdf.geometry.is_empty).sum())
    n_degen   = int(gdf.geometry.apply(
        lambda g: g is not None and not g.is_empty and g.area == 0
    ).sum())
    return {
        "total":      total,
        "invalid":    n_invalid,
        "empty":      n_empty,
        "degenerate": n_degen,
    }


# ── Report writers ────────────────────────────────────────────────────────────

def write_evaluation_report(
    metrics: dict[str, dict[str, float]],
    output_dir: Path,
    course_id: str = "",
    iou_threshold: float = 0.5,
) -> tuple[Path, Path]:
    """Write per-class metrics to CSV and a Markdown summary.

    Returns:
        (csv_path, markdown_path)
    """
    ensure_dir(output_dir)
    safe = (course_id or "eval").replace("/", "_")
    csv_path = output_dir / f"{safe}_metrics.csv"
    md_path  = output_dir / f"{safe}_metrics.md"

    fieldnames = ["class", "precision", "recall", "mean_iou", "n_pred", "n_ref", "n_matched"]
    rows = [
        {"class": cls, **vals}
        for cls, vals in metrics.items()
    ]

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Markdown
    lines = [
        f"# Evaluation Report — {course_id or 'aggregate'}",
        f"\nIoU threshold: {iou_threshold}",
        "\n| Class | Precision | Recall | Mean IoU | Pred | Ref | Matched |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['class']} "
            f"| {row['precision']:.3f} "
            f"| {row['recall']:.3f} "
            f"| {row['mean_iou']:.3f} "
            f"| {row['n_pred']} "
            f"| {row['n_ref']} "
            f"| {row['n_matched']} |"
        )

    # Macro-average across non-zero-ref classes
    valid = [r for r in rows if r["n_ref"] > 0]
    if valid:
        macro_p = np.mean([r["precision"] for r in valid])
        macro_r = np.mean([r["recall"] for r in valid])
        macro_iou = np.mean([r["mean_iou"] for r in valid])
        lines.append(
            f"\n**Macro-average** (n={len(valid)} classes): "
            f"P={macro_p:.3f}, R={macro_r:.3f}, mIoU={macro_iou:.3f}"
        )

    md_path.write_text("\n".join(lines) + "\n")
    log.info("Evaluation report: %s, %s", csv_path.name, md_path.name)
    return csv_path, md_path
