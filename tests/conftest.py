"""Shared pytest fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def tmp_config(tmp_path: Path) -> Path:
    """Write a minimal valid config.yaml to a temp dir and return its path."""
    cfg: dict = {
        "aoi": {
            "type": "osm_id",
            "osm_id": "relation/5179090",
            "name": "Augusta National",
        },
        "data": {
            "cache_dir":       str(tmp_path / "cache"),
            "imagery_cache":   str(tmp_path / "cache/imagery"),
            "overpass_cache":  str(tmp_path / "cache/overpass"),
            "dataset_dir":     str(tmp_path / "dataset"),
            "checkpoints_dir": str(tmp_path / "checkpoints"),
            "output_dir":      str(tmp_path / "output"),
        },
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(cfg))
    return p
