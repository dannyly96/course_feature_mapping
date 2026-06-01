"""Tests for osm.py — tag matching, query builders, eligibility, XML writer."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest
from shapely.geometry import MultiPolygon, Polygon

from golf_mapper.config import GolfMapperConfig, OverpassConfig
from golf_mapper.osm import (
    RELATION_AREA_OFFSET,
    OverpassClient,
    build_boundary_query,
    build_discovery_query,
    build_feature_count_query,
    classify_tags,
    compute_eligibility,
    count_features_from_elements,
    osm_element_to_area_id,
    parse_osm_ref,
    tags_match_class,
    write_osm_xml,
)


@pytest.fixture
def classes():
    return GolfMapperConfig().classes


# ── tag matching ──────────────────────────────────────────────────────────────

def test_tags_match_fairway(classes):
    fairway = next(c for c in classes if c.name == "fairway")
    assert tags_match_class({"golf": "fairway"}, fairway)


def test_tags_no_match(classes):
    fairway = next(c for c in classes if c.name == "fairway")
    assert not tags_match_class({"leisure": "park"}, fairway)


def test_tags_match_water_from_natural_tag(classes):
    wh = next(c for c in classes if c.name == "water_hazard")
    assert tags_match_class({"natural": "water"}, wh)


def test_tags_match_water_from_golf_water_hazard(classes):
    wh = next(c for c in classes if c.name == "water_hazard")
    assert tags_match_class({"golf": "water_hazard"}, wh)


def test_tags_match_water_from_lateral_hazard(classes):
    wh = next(c for c in classes if c.name == "water_hazard")
    assert tags_match_class({"golf": "lateral_water_hazard"}, wh)


def test_tags_match_woods_from_natural_wood(classes):
    woods = next(c for c in classes if c.name == "woods")
    assert tags_match_class({"natural": "wood"}, woods)


def test_tags_match_woods_from_landuse_forest(classes):
    woods = next(c for c in classes if c.name == "woods")
    assert tags_match_class({"landuse": "forest"}, woods)


def test_tags_match_requires_exact_value(classes):
    green = next(c for c in classes if c.name == "green")
    assert not tags_match_class({"golf": "fairway"}, green)


def test_classify_tags_fairway(classes):
    cls = classify_tags({"golf": "fairway"}, classes)
    assert cls is not None and cls.name == "fairway"


def test_classify_tags_green(classes):
    cls = classify_tags({"golf": "green"}, classes)
    assert cls is not None and cls.name == "green"


def test_classify_tags_bunker(classes):
    cls = classify_tags({"golf": "bunker"}, classes)
    assert cls is not None and cls.name == "bunker"


def test_classify_tags_tee(classes):
    cls = classify_tags({"golf": "tee"}, classes)
    assert cls is not None and cls.name == "tee"


def test_classify_tags_none_on_unknown(classes):
    assert classify_tags({"amenity": "restaurant"}, classes) is None


def test_classify_tags_skips_background_by_default(classes):
    # rough is background — must not be returned when skip_background=True (default)
    assert classify_tags({"golf": "rough"}, classes, skip_background=True) is None


def test_classify_tags_returns_background_when_allowed(classes):
    cls = classify_tags({"golf": "rough"}, classes, skip_background=False)
    assert cls is not None and cls.name == "rough"


# ── OSM ID / area ID helpers ──────────────────────────────────────────────────

def test_parse_osm_ref_relation():
    assert parse_osm_ref("relation/5179090") == ("relation", 5179090)


def test_parse_osm_ref_way():
    assert parse_osm_ref("way/123456") == ("way", 123456)


def test_parse_osm_ref_node():
    assert parse_osm_ref("node/999") == ("node", 999)


def test_parse_osm_ref_invalid_no_slash():
    with pytest.raises(ValueError, match="Invalid OSM ref"):
        parse_osm_ref("5179090")


def test_parse_osm_ref_invalid_type():
    with pytest.raises(ValueError, match="Unknown OSM element type"):
        parse_osm_ref("area/123")


def test_osm_element_to_area_id_relation():
    assert osm_element_to_area_id("relation", 5179090) == 5179090 + RELATION_AREA_OFFSET


def test_osm_element_to_area_id_way():
    assert osm_element_to_area_id("way", 99999) == 99999


def test_osm_element_to_area_id_node_raises():
    with pytest.raises(ValueError):
        osm_element_to_area_id("node", 1)


# ── Overpass query builders ───────────────────────────────────────────────────

def test_build_boundary_query_contains_id():
    ql = build_boundary_query("relation", 5179090)
    assert "5179090" in ql
    assert "relation" in ql
    assert "out body" in ql
    assert "[out:json]" in ql


def test_build_feature_count_query_area_id_and_out_tags():
    area_id = 5179090 + RELATION_AREA_OFFSET
    ql = build_feature_count_query(area_id)
    assert str(area_id) in ql
    assert "out tags" in ql       # no geometry — more efficient
    assert "golf" in ql
    assert "natural" in ql


def test_build_feature_count_query_does_not_include_out_body():
    area_id = 5179090 + RELATION_AREA_OFFSET
    ql = build_feature_count_query(area_id)
    # Should not pull full geometry
    assert "out body" not in ql


def test_build_discovery_query_bbox():
    ql = build_discovery_query(bbox=[-82.1, 33.4, -81.9, 33.6])
    assert "golf_course" in ql
    assert "33.4" in ql
    assert "-82.1" in ql
    assert "way" in ql
    assert "relation" in ql


# ── Eligibility ───────────────────────────────────────────────────────────────

def test_compute_eligibility_above():
    assert compute_eligibility({"fairway": 30, "green": 21}, 50) is True


def test_compute_eligibility_at_boundary():
    assert compute_eligibility({"fairway": 25, "green": 25}, 50) is True


def test_compute_eligibility_below():
    assert compute_eligibility({"fairway": 20, "green": 15}, 50) is False


def test_compute_eligibility_empty_counts():
    assert compute_eligibility({}, 50) is False


def test_count_features_from_elements_basic(classes):
    elements = [
        {"type": "way", "id": 1, "tags": {"golf": "fairway"}},
        {"type": "way", "id": 2, "tags": {"golf": "green"}},
        {"type": "way", "id": 3, "tags": {"golf": "green"}},
        {"type": "node", "id": 4, "tags": {"golf": "tee"}},       # nodes skipped
        {"type": "way", "id": 5, "tags": {"amenity": "pub"}},     # not a golf class
        {"type": "way", "id": 6, "tags": {"natural": "water"}},   # water_hazard
    ]
    counts = count_features_from_elements(elements, classes)
    assert counts["fairway"] == 1
    assert counts["green"] == 2
    assert counts["water_hazard"] == 1
    assert "tee" not in counts    # nodes are skipped
    assert "rough" not in counts  # background class is not counted


def test_count_features_background_not_counted(classes):
    elements = [{"type": "way", "id": 1, "tags": {"golf": "rough"}}]
    counts = count_features_from_elements(elements, classes)
    assert counts == {}


# ── OverpassClient (mocked) ───────────────────────────────────────────────────

def _mock_response(elements: list) -> dict:
    return {"version": 0.6, "elements": elements}


@patch("golf_mapper.osm.requests.post")
def test_overpass_client_success(mock_post, tmp_path):
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = _mock_response(
        [{"type": "way", "id": 1, "tags": {"golf": "fairway"}}]
    )
    mock_post.return_value.raise_for_status = lambda: None

    cfg = OverpassConfig()
    client = OverpassClient(cfg)
    result = client.query("[out:json];way(1);out;")
    assert len(result["elements"]) == 1


@patch("golf_mapper.osm.requests.post")
def test_overpass_client_uses_cache(mock_post, tmp_path):
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = _mock_response([])
    mock_post.return_value.raise_for_status = lambda: None

    cfg = OverpassConfig()
    client = OverpassClient(cfg, cache_dir=tmp_path / "cache")
    client.query("ql", cache_key="mykey")
    client.query("ql", cache_key="mykey")  # second call should hit cache
    assert mock_post.call_count == 1       # only one real HTTP call


@patch("golf_mapper.osm.requests.post")
def test_overpass_client_rotates_on_failure(mock_post, tmp_path):
    import requests as req

    call_count = [0]

    def side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] < 3:
            r = MagicMock()
            r.status_code = 500
            r.raise_for_status.side_effect = req.HTTPError("500")
            return r
        # third call succeeds
        r = MagicMock()
        r.status_code = 200
        r.json.return_value = _mock_response([])
        r.raise_for_status = lambda: None
        return r

    mock_post.side_effect = side_effect

    cfg = OverpassConfig(max_retries=5, backoff_base=0.001)
    client = OverpassClient(cfg)
    result = client.query("ql")
    assert result["elements"] == []
    assert call_count[0] == 3


@patch("golf_mapper.osm.requests.post")
def test_overpass_client_exhausted_raises(mock_post):
    import requests as req

    r = MagicMock()
    r.status_code = 500
    r.raise_for_status.side_effect = req.HTTPError("500")
    mock_post.return_value = r

    cfg = OverpassConfig(max_retries=2, backoff_base=0.001)
    client = OverpassClient(cfg)
    with pytest.raises(RuntimeError, match="Overpass failed"):
        client.query("ql")


# ── OSM XML writer ────────────────────────────────────────────────────────────

def _square_gdf(tags_list: list[dict]) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame with simple unit squares, one per tag dict."""
    polys = []
    for i, _ in enumerate(tags_list):
        off = i * 5.0
        polys.append(Polygon([(off, 0), (off + 1, 0), (off + 1, 1), (off, 1)]))
    return gpd.GeoDataFrame(
        {"osm_tags": tags_list, "geometry": polys},
        crs="EPSG:4326",
    )


def test_write_osm_xml_valid_xml(tmp_path):
    gdf = _square_gdf([{"golf": "green"}])
    out = tmp_path / "test.osm"
    write_osm_xml(gdf, out)
    assert out.exists()
    tree = ET.parse(out)
    assert tree.getroot().tag == "osm"


def test_write_osm_xml_root_attributes(tmp_path):
    gdf = _square_gdf([{"golf": "tee"}])
    out = tmp_path / "test.osm"
    write_osm_xml(gdf, out)
    root = ET.parse(out).getroot()
    assert root.get("version") == "0.6"
    assert "golf-mapper" in root.get("generator", "")


def test_write_osm_xml_correct_osm_tags(tmp_path):
    gdf = _square_gdf([{"golf": "green"}, {"golf": "bunker"}])
    out = tmp_path / "test.osm"
    write_osm_xml(gdf, out)
    root = ET.parse(out).getroot()
    tag_values = {
        tag.get("v")
        for way in root.findall("way")
        for tag in way.findall("tag")
        if tag.get("k") == "golf"
    }
    assert "green" in tag_values
    assert "bunker" in tag_values


def test_write_osm_xml_way_and_node_counts(tmp_path):
    # Each unit square has 4 unique exterior coords → 4 nodes, 1 way.
    gdf = _square_gdf([{"golf": "green"}, {"golf": "bunker"}])
    out = tmp_path / "test.osm"
    write_osm_xml(gdf, out)
    root = ET.parse(out).getroot()
    assert len(root.findall("way")) == 2
    assert len(root.findall("node")) == 8   # 4 coords × 2 polygons


def test_write_osm_xml_negative_ids(tmp_path):
    gdf = _square_gdf([{"golf": "fairway"}])
    out = tmp_path / "test.osm"
    write_osm_xml(gdf, out)
    root = ET.parse(out).getroot()
    for node in root.findall("node"):
        assert int(node.get("id")) < 0, "Node IDs must be negative (new elements)"
    for way in root.findall("way"):
        assert int(way.get("id")) < 0, "Way IDs must be negative (new elements)"


def test_write_osm_xml_ways_are_closed(tmp_path):
    gdf = _square_gdf([{"golf": "fairway"}])
    out = tmp_path / "test.osm"
    write_osm_xml(gdf, out)
    root = ET.parse(out).getroot()
    for way in root.findall("way"):
        refs = [nd.get("ref") for nd in way.findall("nd")]
        assert refs[0] == refs[-1], "Each way must be closed (first nd == last nd)"


def test_write_osm_xml_multipolygon_emits_two_ways(tmp_path):
    mp = MultiPolygon([
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(5, 5), (6, 5), (6, 6), (5, 6)]),
    ])
    gdf = gpd.GeoDataFrame(
        {"osm_tags": [{"natural": "water"}], "geometry": [mp]},
        crs="EPSG:4326",
    )
    out = tmp_path / "mp.osm"
    write_osm_xml(gdf, out)
    root = ET.parse(out).getroot()
    assert len(root.findall("way")) == 2   # one way per Polygon part


def test_write_osm_xml_rejects_non_4326(tmp_path):
    gdf = gpd.GeoDataFrame(
        {"osm_tags": [{"golf": "green"}], "geometry": [Polygon([(0, 0), (1, 0), (1, 1)])]},
        crs="EPSG:3857",
    )
    with pytest.raises(AssertionError):
        write_osm_xml(gdf, tmp_path / "bad.osm")


def test_write_osm_xml_attribution_comment(tmp_path):
    gdf = _square_gdf([{"golf": "green"}])
    out = tmp_path / "test.osm"
    write_osm_xml(gdf, out, attribution="Derived from Esri World Imagery (ODbL)")
    content = out.read_text(encoding="utf-8")
    assert "Esri" in content


def test_write_osm_xml_empty_gdf_writes_valid_file(tmp_path):
    gdf = gpd.GeoDataFrame({"osm_tags": [], "geometry": []}, crs="EPSG:4326")
    out = tmp_path / "empty.osm"
    write_osm_xml(gdf, out)
    root = ET.parse(out).getroot()
    assert root.tag == "osm"
    assert len(root.findall("way")) == 0
