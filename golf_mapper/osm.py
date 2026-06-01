"""Overpass API client, OSM tag utilities, query builders, and OSM XML writer."""
from __future__ import annotations

import logging
import random
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import requests
from shapely.geometry import MultiPolygon, Polygon

from .config import ClassDef, GolfMapperConfig, OverpassConfig
from .utils import DiskCache, get_logger

log = get_logger(__name__)

# Overpass converts relations to areas with this offset applied to the relation ID.
RELATION_AREA_OFFSET = 3_600_000_000


# ── ID / reference helpers ────────────────────────────────────────────────────

def parse_osm_ref(ref: str) -> tuple[str, int]:
    """Parse 'relation/5179090' → ('relation', 5179090).

    Supported types: node, way, relation.
    """
    parts = ref.strip().split("/")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid OSM ref {ref!r}. Expected '<type>/<id>', e.g. 'relation/5179090'"
        )
    osm_type, osm_id_str = parts
    if osm_type not in ("node", "way", "relation"):
        raise ValueError(
            f"Unknown OSM element type {osm_type!r}. Must be node, way, or relation."
        )
    return osm_type, int(osm_id_str)


def osm_element_to_area_id(osm_type: str, osm_id: int) -> int:
    """Compute the Overpass area filter ID for a closed way or relation.

    Relations: area_id = osm_id + RELATION_AREA_OFFSET (3,600,000,000)
    Ways:      area_id = osm_id  (no offset; Overpass uses the way ID directly)
    """
    if osm_type == "relation":
        return osm_id + RELATION_AREA_OFFSET
    elif osm_type == "way":
        return osm_id
    raise ValueError(
        f"Only 'way' and 'relation' can be converted to Overpass areas, got {osm_type!r}"
    )


# ── Tag matching ──────────────────────────────────────────────────────────────

def tags_match_class(tags: dict[str, str], class_def: ClassDef) -> bool:
    """Return True if *any* entry in class_def.osm_read_tags is a full subset of tags."""
    for tag_set in class_def.osm_read_tags:
        if all(tags.get(k) == v for k, v in tag_set.items()):
            return True
    return False


def classify_tags(
    tags: dict[str, str],
    classes: list[ClassDef],
    skip_background: bool = True,
) -> ClassDef | None:
    """Return the first matching ClassDef for a set of OSM element tags, or None.

    Classes are checked in id order. Background classes (is_background=True)
    are skipped by default so that rough is not returned for untagged areas.
    """
    for cls in sorted(classes, key=lambda c: c.id):
        if skip_background and cls.is_background:
            continue
        if tags_match_class(tags, cls):
            return cls
    return None


# ── Overpass query builders ───────────────────────────────────────────────────

def build_boundary_query(osm_type: str, osm_id: int) -> str:
    """Return QL that fetches the full boundary geometry of one course."""
    return (
        f"[out:json][timeout:120];\n"
        f"{osm_type}({osm_id});\n"
        f"out body;\n"
        f">;\n"
        f"out skel qt;"
    )


def build_discovery_query(
    bbox: list[float] | None = None,
    south: float = 0.0,
    west: float = 0.0,
    north: float = 0.0,
    east: float = 0.0,
) -> str:
    """Return QL that discovers all leisure=golf_course elements in a bounding box.

    bbox, if given, must be [min_lon, min_lat, max_lon, max_lat] (EPSG:4326).
    """
    if bbox is not None:
        west, south, east, north = bbox[0], bbox[1], bbox[2], bbox[3]
    return (
        "[out:json][timeout:120];\n"
        "(\n"
        f'  way["leisure"="golf_course"]({south},{west},{north},{east});\n'
        f'  relation["leisure"="golf_course"]({south},{west},{north},{east});\n'
        ");\n"
        "out body;\n"
        ">;\n"
        "out skel qt;"
    )


def build_single_bbox_query(bbox: list[float]) -> str:
    """Return QL that fetches ALL course boundaries AND ALL golf features in one request.

    Replaces the previous N+1 pattern (one boundary query + one feature-count query
    per course) with a single round-trip. Feature counts are then computed in Python
    via a geopandas spatial join rather than per-course Overpass area filters.

    bbox must be [min_lon, min_lat, max_lon, max_lat] (EPSG:4326).
    """
    west, south, east, north = bbox
    return (
        f"[out:json][timeout:180];\n"
        f"(\n"
        # Course boundaries
        f'  way["leisure"="golf_course"]({south},{west},{north},{east});\n'
        f'  relation["leisure"="golf_course"]({south},{west},{north},{east});\n'
        # Golf features — all golf=* tags, water, woodland
        f'  way["golf"~"."]({south},{west},{north},{east});\n'
        f'  way["natural"~"water|wood|tree_row"]({south},{west},{north},{east});\n'
        f'  way["landuse"="forest"]({south},{west},{north},{east});\n'
        f'  relation["golf"~"."]({south},{west},{north},{east});\n'
        f");\n"
        f"out body;\n"
        f">;\n"
        f"out skel qt;"
    )


def build_feature_count_query(area_id: int) -> str:
    """Return QL that fetches *tags only* of qualifying features in a course area.

    'out tags;' omits geometry — sufficient for counting and more efficient.
    """
    return (
        f"[out:json][timeout:120];\n"
        f"area({area_id})->.course;\n"
        "(\n"
        '  way["golf"~"."](area.course);\n'
        '  way["natural"~"water|wood|tree_row"](area.course);\n'
        '  way["landuse"="forest"](area.course);\n'
        '  relation["golf"~"."](area.course);\n'
        '  relation["natural"~"water|wood|tree_row"](area.course);\n'
        ");\n"
        "out tags;"
    )


def build_feature_geometry_query(area_id: int) -> str:
    """Return QL that fetches full geometry of qualifying features in a course area.

    Used in Step 4 (label generation) when full polygon coordinates are needed.
    """
    return (
        f"[out:json][timeout:120];\n"
        f"area({area_id})->.course;\n"
        "(\n"
        '  way["golf"~"."](area.course);\n'
        '  way["natural"~"water|wood|tree_row"](area.course);\n'
        '  way["landuse"="forest"](area.course);\n'
        '  relation["golf"~"."](area.course);\n'
        '  relation["natural"~"water|wood|tree_row"](area.course);\n'
        ");\n"
        "out body;\n"
        ">;\n"
        "out skel qt;"
    )


# ── Feature counting ──────────────────────────────────────────────────────────

def count_features_from_elements(
    elements: list[dict[str, Any]],
    classes: list[ClassDef],
) -> dict[str, int]:
    """Count qualifying ways/relations per class from a list of Overpass elements.

    Nodes are skipped (golf features are always areas/ways). Background class
    elements (rough) are not counted toward the eligibility total.
    """
    counts: dict[str, int] = {}
    for elem in elements:
        if elem.get("type") not in ("way", "relation"):
            continue
        tags = elem.get("tags", {})
        cls = classify_tags(tags, classes, skip_background=True)
        if cls is not None:
            counts[cls.name] = counts.get(cls.name, 0) + 1
    return counts


def compute_eligibility(feature_counts: dict[str, int], min_features: int) -> bool:
    """Return True if total qualifying feature count meets the training threshold."""
    return sum(feature_counts.values()) >= min_features


# ── Overpass HTTP client ──────────────────────────────────────────────────────

class OverpassClient:
    """Overpass API HTTP client with endpoint rotation, exponential backoff, and disk cache.

    On each failure the client rotates to the next configured mirror before retrying,
    up to cfg.max_retries total attempts. A 10-second HTTP margin is added above the
    Overpass [timeout:N] statement to allow the server time to respond.
    """

    def __init__(self, cfg: OverpassConfig, cache_dir: Path | None = None) -> None:
        self._cfg = cfg
        self._cache: DiskCache | None = (
            DiskCache(cache_dir, serializer="pickle") if cache_dir else None
        )

    def query(self, ql: str, cache_key: str | None = None) -> dict[str, Any]:
        """Execute an Overpass QL query and return the parsed JSON response dict.

        If cache_key is set and a cached result exists, return it without a network call.
        Otherwise query Overpass (with rotation + backoff) and cache the result.

        Raises RuntimeError if all attempts fail.
        """
        if cache_key and self._cache and self._cache.has(cache_key):
            log.debug("Overpass cache hit: %.60s", cache_key)
            return self._cache.get(cache_key)  # type: ignore[return-value]

        endpoints = self._cfg.endpoints
        n = len(endpoints)
        last_exc: Exception | None = None

        for attempt in range(self._cfg.max_retries):
            endpoint = endpoints[attempt % n]
            try:
                result = self._post(endpoint, ql)
                if cache_key and self._cache:
                    self._cache.set(cache_key, result)
                return result
            except Exception as exc:
                last_exc = exc
                is_last = attempt == self._cfg.max_retries - 1
                delay = self._cfg.backoff_base ** attempt * (1.0 + random.random() * 0.25)
                if is_last:
                    log.warning(
                        "Overpass attempt %d/%d on %s failed (%s). No more retries.",
                        attempt + 1, self._cfg.max_retries, endpoint, exc,
                    )
                else:
                    log.warning(
                        "Overpass attempt %d/%d on %s failed (%s). Retry in %.1fs…",
                        attempt + 1, self._cfg.max_retries, endpoint, exc, delay,
                    )
                    time.sleep(delay)

        raise RuntimeError(
            f"Overpass failed after {self._cfg.max_retries} attempts. "
            f"Last error: {last_exc}"
        ) from last_exc

    def _post(self, endpoint: str, ql: str) -> dict[str, Any]:
        http_timeout = self._cfg.timeout + 10  # server timeout + network margin
        resp = requests.post(
            endpoint,
            data={"data": ql},
            timeout=http_timeout,
        )
        if resp.status_code == 429:
            # Treat 429 as a retryable error so the backoff kicks in
            raise requests.HTTPError(f"429 Too Many Requests from {endpoint}", response=resp)
        resp.raise_for_status()
        data: dict = resp.json()
        if "elements" not in data:
            raise ValueError(
                f"Overpass response from {endpoint} missing 'elements' key: "
                f"{str(data)[:200]}"
            )
        log.info(
            "Overpass OK [%s] — %d elements", endpoint.split("/")[2], len(data["elements"])
        )
        return data


# ── OSM XML writer ────────────────────────────────────────────────────────────

def write_osm_xml(
    gdf: "gpd.GeoDataFrame",  # type: ignore[name-defined]  # noqa: F821
    output_path: Path | str,
    attribution: str = "",
    tag_column: str = "osm_tags",
) -> None:
    """Write a GeoDataFrame of polygon features to an OSM XML file readable by JOSM.

    All new elements use negative IDs — the JOSM convention for elements not yet
    submitted to the OSM API. Polygons are written as closed ways; MultiPolygons
    are split into one closed way per part.

    Args:
        gdf:          GeoDataFrame in EPSG:4326. Must have a column of OSM tag dicts.
        output_path:  Destination path for the .osm file.
        attribution:  Inserted as an XML comment (provenance note).
        tag_column:   Column holding {key: value} OSM tag dicts per row.

    Raises:
        AssertionError: if gdf.crs is not EPSG:4326.
    """
    assert gdf.crs is None or gdf.crs.to_epsg() == 4326, (
        f"write_osm_xml requires EPSG:4326 input, got {gdf.crs}"
    )

    root = ET.Element("osm", version="0.6", generator="golf-mapper")
    if attribution:
        root.append(ET.Comment(f" {attribution} "))

    node_elements: list[ET.Element] = []
    way_elements: list[ET.Element] = []

    node_id = -1      # negative, decrement for each new node
    way_id = -1000    # start ways at -1000 to separate from node ID space visually

    def _emit_polygon(poly: Polygon, tags: dict[str, str]) -> None:
        nonlocal node_id, way_id

        coords = list(poly.exterior.coords)
        # Exterior ring: first == last; create one node per unique coordinate.
        ring_node_ids: list[int] = []
        for lon, lat in coords[:-1]:
            node_elements.append(
                ET.Element(
                    "node",
                    id=str(node_id),
                    lat=f"{lat:.7f}",
                    lon=f"{lon:.7f}",
                    visible="true",
                )
            )
            ring_node_ids.append(node_id)
            node_id -= 1

        way_elem = ET.Element("way", id=str(way_id), visible="true")
        for nid in ring_node_ids:
            ET.SubElement(way_elem, "nd", ref=str(nid))
        ET.SubElement(way_elem, "nd", ref=str(ring_node_ids[0]))  # close the ring
        for k, v in (tags or {}).items():
            ET.SubElement(way_elem, "tag", k=k, v=str(v))
        way_elements.append(way_elem)
        way_id -= 1

    for _, row in gdf.iterrows():
        geom = row.geometry
        tags: dict[str, str] = row.get(tag_column) or {}
        if not isinstance(tags, dict):
            tags = {}
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, Polygon):
            _emit_polygon(geom, tags)
        elif isinstance(geom, MultiPolygon):
            for part in geom.geoms:
                _emit_polygon(part, tags)
        else:
            log.debug("Skipping non-polygon geometry: %s", geom.geom_type)

    for elem in node_elements:
        root.append(elem)
    for elem in way_elements:
        root.append(elem)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fh:
        tree.write(fh, xml_declaration=True, encoding="utf-8")

    n_ways = len(way_elements)
    n_nodes = len(node_elements)
    log.info("Wrote %s: %d way(s), %d node(s)", output_path.name, n_ways, n_nodes)
