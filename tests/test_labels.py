"""Tests for labels.py — course splits, YOLO dataset structure, data.yaml."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from golf_mapper.config import GolfMapperConfig
from golf_mapper.labels import split_courses, write_data_yaml


# ── split_courses ─────────────────────────────────────────────────────────────

def test_split_exhaustive():
    ids = [f"relation/{i}" for i in range(20)]
    s = split_courses(ids, val_split=0.15, test_split=0.10, seed=42)
    assert set(s["train"] + s["val"] + s["test"]) == set(ids)
    assert len(set(s["train"] + s["val"] + s["test"])) == 20  # no duplicates


def test_split_no_leakage():
    ids = [f"relation/{i}" for i in range(20)]
    s = split_courses(ids, val_split=0.15, test_split=0.10, seed=42)
    train_set = set(s["train"])
    val_set   = set(s["val"])
    test_set  = set(s["test"])
    assert train_set.isdisjoint(val_set),  "Train and val share a course"
    assert train_set.isdisjoint(test_set), "Train and test share a course"
    assert val_set.isdisjoint(test_set),   "Val and test share a course"


def test_split_ratios_approximately_correct():
    ids = [f"relation/{i}" for i in range(100)]
    s = split_courses(ids, val_split=0.15, test_split=0.10, seed=0)
    assert 10 <= len(s["val"]) <= 20
    assert 7  <= len(s["test"]) <= 15
    assert len(s["train"]) >= 65


def test_split_minimum_sizes():
    # Even with 3 courses, each split must have >= 1 element
    ids = ["relation/1", "relation/2", "relation/3"]
    s = split_courses(ids, val_split=0.15, test_split=0.10, seed=99)
    assert len(s["train"]) >= 1
    assert len(s["val"]) >= 1
    assert len(s["test"]) >= 1


def test_split_deterministic():
    ids = [f"relation/{i}" for i in range(30)]
    s1 = split_courses(ids, 0.15, 0.10, seed=7)
    s2 = split_courses(ids, 0.15, 0.10, seed=7)
    assert s1 == s2


def test_split_different_seeds_differ():
    ids = [f"relation/{i}" for i in range(30)]
    s1 = split_courses(ids, 0.15, 0.10, seed=1)
    s2 = split_courses(ids, 0.15, 0.10, seed=2)
    # It is astronomically unlikely they are identical
    assert s1["train"] != s2["train"]


# ── write_data_yaml ───────────────────────────────────────────────────────────

def test_write_data_yaml_creates_file(tmp_path):
    cfg = GolfMapperConfig()
    out = write_data_yaml(cfg, tmp_path)
    assert out.exists()
    assert out.name == "data.yaml"


def test_write_data_yaml_correct_structure(tmp_path):
    cfg = GolfMapperConfig()
    write_data_yaml(cfg, tmp_path)
    with (tmp_path / "data.yaml").open() as fh:
        data = yaml.safe_load(fh)
    assert "nc" in data
    assert "names" in data
    assert "train" in data
    assert "val" in data
    assert "test" in data
    assert data["nc"] == cfg.num_classes


def test_write_data_yaml_class_count(tmp_path):
    cfg = GolfMapperConfig()
    write_data_yaml(cfg, tmp_path)
    with (tmp_path / "data.yaml").open() as fh:
        data = yaml.safe_load(fh)
    assert data["nc"] == 7
    assert len(data["names"]) == 7


def test_write_data_yaml_class_names_ordered(tmp_path):
    cfg = GolfMapperConfig()
    write_data_yaml(cfg, tmp_path)
    with (tmp_path / "data.yaml").open() as fh:
        data = yaml.safe_load(fh)
    assert data["names"][0] == "rough"   # class_id 0
    assert data["names"][1] == "fairway" # class_id 1


def test_write_data_yaml_paths_contain_split_dirs(tmp_path):
    cfg = GolfMapperConfig()
    write_data_yaml(cfg, tmp_path)
    with (tmp_path / "data.yaml").open() as fh:
        data = yaml.safe_load(fh)
    assert "train" in data["train"]
    assert "val"   in data["val"]
    assert "test"  in data["test"]
