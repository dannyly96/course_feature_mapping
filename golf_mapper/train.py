"""YOLO training loop with geographic proximity weighting.

Supports two modes (config training.weighting_mode):
  global   — one model trained on all qualifying courses, equal tile weights.
  regional — fine-tune global base weights with proximity-weighted sampling.

Model variant is configurable (yolo26-seg or yolo11-seg fallback).
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

from .config import GolfMapperConfig
from .utils import get_logger, set_seeds

log = get_logger(__name__)

# ── ultralytics import guard ──────────────────────────────────────────────────

try:
    from ultralytics import YOLO as _YOLO
    _HAS_ULTRALYTICS = True
except ImportError:
    _YOLO = None  # type: ignore[assignment,misc]
    _HAS_ULTRALYTICS = False
    log.warning(
        "ultralytics not installed — training unavailable. "
        "Install with: pip install ultralytics>=8.3.0"
    )

# ── YOLO26 weight file names ───────────────────────────────────────────────────
# Verify these identifiers against the ultralytics docs / release notes for
# your installed version.  YOLO26 was released Jan 2026; weight filenames
# follow the pattern <variant><size>-seg.pt  (n/s/m/l/x sizes).
_MODEL_WEIGHTS: dict[str, str] = {
    "yolo26-seg": "yolo26n-seg.pt",   # nano — good Colab starting point
    "yolo11-seg": "yolo11n-seg.pt",   # stable fallback
}


# ── Public API ────────────────────────────────────────────────────────────────

def train_model(
    cfg: GolfMapperConfig,
    dataset_dir: Path,
    target_centroid: tuple[float, float] | None = None,
    manifest_gdf: Any | None = None,
) -> Path:
    """Train (or fine-tune) a YOLO segmentation model and return the best checkpoint.

    Args:
        cfg:              Pipeline configuration.
        dataset_dir:      Path to the Ultralytics-format dataset (has data.yaml).
        target_centroid:  (lon, lat) of the target course. Required for regional mode.
        manifest_gdf:     Course manifest GeoDataFrame. Required for regional mode.

    Returns:
        Path to the saved best.pt checkpoint.

    Raises:
        ImportError: ultralytics not installed.
        FileNotFoundError: data.yaml not in dataset_dir.
    """
    _require_ultralytics()
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found in {dataset_dir}")

    set_seeds(cfg.training.seed)
    cfg.data.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    weights_file = _resolve_weights(cfg.model.variant)
    log.info("Loading model: %s (%s)", cfg.model.variant, weights_file)
    model = _YOLO(weights_file)

    train_kwargs = _build_train_kwargs(cfg, data_yaml)

    if cfg.training.weighting_mode == "regional" and target_centroid and manifest_gdf is not None:
        log.info("Regional mode: proximity-weighted fine-tuning (σ=%.0f km)", cfg.training.sigma_km)
        best = _train_regional(model, cfg, train_kwargs, target_centroid, manifest_gdf, dataset_dir)
    else:
        log.info("Global mode: training on all courses with equal weights.")
        results = model.train(**train_kwargs)
        best = _best_checkpoint(results, cfg)

    log.info("Training complete. Best checkpoint: %s", best)
    return best


def _build_train_kwargs(cfg: GolfMapperConfig, data_yaml: Path) -> dict[str, Any]:
    """Assemble the kwargs dict for model.train()."""
    return {
        "data":    str(data_yaml),
        "epochs":  cfg.training.epochs,
        "imgsz":   cfg.training.img_size,
        "batch":   cfg.training.batch_size,
        "seed":    cfg.training.seed,
        "project": str(cfg.data.checkpoints_dir),
        "name":    "golf_mapper",
        "resume":  cfg.training.resume,
        "verbose": True,
    }


def _train_regional(
    model: Any,
    cfg: GolfMapperConfig,
    base_kwargs: dict[str, Any],
    target_centroid: tuple[float, float],
    manifest_gdf: Any,
    dataset_dir: Path,
) -> Path:
    """Run regional fine-tuning: global pre-train → proximity-weighted fine-tune."""
    from .weighting import compute_course_weights

    # Stage 1: short global pre-train to learn basic features
    pre_kwargs = {**base_kwargs, "epochs": max(10, cfg.training.epochs // 5), "name": "golf_global"}
    log.info("Stage 1: global pre-train (%d epochs)…", pre_kwargs["epochs"])
    results = model.train(**pre_kwargs)
    global_checkpoint = _best_checkpoint(results, cfg)

    # Stage 2: reload best global checkpoint and fine-tune with proximity weights
    model2 = _YOLO(str(global_checkpoint))
    target_lon, target_lat = target_centroid
    weights = compute_course_weights(
        target_lon, target_lat, manifest_gdf,
        cfg.training.sigma_km, cfg.training.max_radius_km,
    )

    # Build a proximity-weighted dataset directory (symlink or copy tiles)
    weighted_dir = _build_weighted_dataset(dataset_dir, weights, cfg)

    fine_kwargs = {
        **base_kwargs,
        "data":   str(weighted_dir / "data.yaml"),
        "epochs": cfg.training.epochs,
        "name":   "golf_regional",
    }
    log.info("Stage 2: regional fine-tune (%d epochs)…", fine_kwargs["epochs"])
    results2 = model2.train(**fine_kwargs)
    return _best_checkpoint(results2, cfg)


def _build_weighted_dataset(
    dataset_dir: Path,
    course_weights: dict[str, float],
    cfg: GolfMapperConfig,
) -> Path:
    """Create a weighted dataset directory by duplicating high-weight course tiles.

    Each course's tiles are duplicated proportional to its proximity weight
    (integer rounds, minimum 1 copy).  Returns the path to the weighted dir.
    """
    from .weighting import weighted_tile_sample
    import yaml as _yaml

    weighted_dir = cfg.data.dataset_dir.parent / "dataset_regional"
    if weighted_dir.exists():
        import shutil as _shutil
        _shutil.rmtree(weighted_dir)

    for split in ("train", "val", "test"):
        img_src = dataset_dir / "images" / split
        lbl_src = dataset_dir / "labels" / split
        img_dst = weighted_dir / "images" / split
        lbl_dst = weighted_dir / "labels" / split
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        if split != "train":
            # val/test: copy unchanged
            for p in img_src.glob("*.png"):
                (img_dst / p.name).symlink_to(p.resolve())
            for p in lbl_src.glob("*.txt"):
                (lbl_dst / p.name).symlink_to(p.resolve())
            continue

        # Group tiles by course_id
        by_course: dict[str, list[str]] = {}
        for p in img_src.glob("*.png"):
            # Filename format: <type>_<osm_id>_r<row>_c<col>.png
            parts = p.stem.split("_")
            cid = "_".join(parts[:2])   # e.g. relation_5179090
            by_course.setdefault(cid, []).append(str(p))

        # Remap course_id keys to match weighting dict (uses 'relation/5179090' format)
        def _norm(cid: str) -> str:
            return cid.replace("_", "/", 1)

        norm_weights = {_norm(k): v for k, v in course_weights.items()}
        norm_by_course = {_norm(k): v for k, v in by_course.items()}

        sampled = weighted_tile_sample(norm_by_course, norm_weights, cfg.training.seed)
        for idx, img_path_str in enumerate(sampled):
            img_p = Path(img_path_str)
            lbl_p = lbl_src / (img_p.stem + ".txt")
            dst_stem = f"{img_p.stem}_{idx:05d}"
            (img_dst / f"{dst_stem}.png").symlink_to(img_p.resolve())
            if lbl_p.exists():
                (lbl_dst / f"{dst_stem}.txt").symlink_to(lbl_p.resolve())

    # Write data.yaml for the weighted dataset
    with (dataset_dir / "data.yaml").open() as fh:
        data_meta = _yaml.safe_load(fh)
    data_meta["path"] = str(weighted_dir)
    with (weighted_dir / "data.yaml").open("w") as fh:
        _yaml.dump(data_meta, fh)

    log.info("Weighted dataset written to %s (%d train tiles)", weighted_dir, len(sampled))
    return weighted_dir


def _resolve_weights(variant: str) -> str:
    """Return the weight filename for the given variant; fall back gracefully."""
    name = _MODEL_WEIGHTS.get(variant)
    if name is None:
        fallback = _MODEL_WEIGHTS["yolo11-seg"]
        log.warning("Unknown model variant %r; falling back to %s.", variant, fallback)
        return fallback
    return name


def _best_checkpoint(results: Any, cfg: GolfMapperConfig) -> Path:
    """Extract the best.pt path from YOLO training results."""
    try:
        best = Path(results.save_dir) / "weights" / "best.pt"
        if best.exists():
            return best
    except AttributeError:
        pass
    # Fallback: search checkpoints dir
    candidates = list(cfg.data.checkpoints_dir.rglob("best.pt"))
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    raise FileNotFoundError("Could not locate best.pt after training.")


def _require_ultralytics() -> None:
    if not _HAS_ULTRALYTICS:
        raise ImportError(
            "ultralytics is required for training. "
            "Install with: pip install ultralytics>=8.3.0"
        )
