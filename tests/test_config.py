"""Tests for config loading and validation."""
from __future__ import annotations

import pytest

from golf_mapper.config import ClassDef, GolfMapperConfig, load_config


def test_default_config_values():
    cfg = GolfMapperConfig()
    assert cfg.labels.min_features == 50
    assert cfg.imagery.zoom == 19
    assert cfg.labels.tile_size == 1024
    assert cfg.labels.overlap == 128
    assert cfg.training.sigma_km == 150.0
    assert cfg.model.variant == "yolo26-seg"
    assert cfg.training.weighting_mode == "regional"
    assert len(cfg.classes) == 7


def test_load_config_from_yaml(tmp_config):
    cfg = load_config(tmp_config)
    assert cfg.aoi.osm_id == "relation/5179090"
    assert cfg.aoi.name == "Augusta National"


def test_class_by_name():
    cfg = GolfMapperConfig()
    cls = cfg.class_by_name("fairway")
    assert cls.id == 1
    assert cls.osm_write_tags == {"golf": "fairway"}


def test_class_by_id():
    cfg = GolfMapperConfig()
    assert cfg.class_by_id(4).name == "bunker"
    assert cfg.class_by_id(0).is_background is True


def test_class_names_are_id_ordered():
    cfg = GolfMapperConfig()
    names = cfg.class_names
    assert names[0] == "rough"
    assert names[1] == "fairway"
    assert len(names) == cfg.num_classes


def test_class_lookup_raises_on_unknown():
    cfg = GolfMapperConfig()
    with pytest.raises(KeyError):
        cfg.class_by_name("nonexistent")
    with pytest.raises(KeyError):
        cfg.class_by_id(99)


def test_config_hash_is_stable_and_short():
    cfg = GolfMapperConfig()
    assert cfg.config_hash() == cfg.config_hash()
    assert len(cfg.config_hash()) == 16


def test_bbox_required_when_type_is_bbox():
    with pytest.raises(Exception):
        GolfMapperConfig(aoi={"type": "bbox"})  # bbox=None must fail validation


def test_bbox_wrong_length():
    with pytest.raises(Exception):
        GolfMapperConfig(aoi={"type": "bbox", "bbox": [1.0, 2.0]})


def test_missing_config_file():
    with pytest.raises(FileNotFoundError):
        load_config("/definitely/not/there/config.yaml")


def test_water_hazard_has_multiple_read_tags():
    cfg = GolfMapperConfig()
    wh = cfg.class_by_name("water_hazard")
    tag_keys = [list(t.keys())[0] for t in wh.osm_read_tags]
    assert "natural" in tag_keys
    assert "golf" in tag_keys


def test_woods_writes_natural_wood():
    cfg = GolfMapperConfig()
    woods = cfg.class_by_name("woods")
    assert woods.osm_write_tags == {"natural": "wood"}
