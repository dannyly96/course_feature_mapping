"""Interactive preview map and human confirmation gate.

Renders detected polygons colour-coded by class on top of the course imagery,
prints a per-class feature count summary, then asks for explicit confirmation
before the export stage writes any output files.
"""
from __future__ import annotations

import logging
from typing import Any

import geopandas as gpd

from .config import GolfMapperConfig
from .utils import get_logger

log = get_logger(__name__)

# ── folium / leafmap import guard ─────────────────────────────────────────────

try:
    import folium as _folium
    _HAS_FOLIUM = True
except ImportError:
    _folium = None  # type: ignore[assignment]
    _HAS_FOLIUM = False
    log.warning("folium not installed — interactive maps unavailable. pip install folium")


# ── Preview map ───────────────────────────────────────────────────────────────

def render_preview_map(
    gdf: gpd.GeoDataFrame,
    boundary: Any,             # shapely geometry in EPSG:4326
    cfg: GolfMapperConfig,
    course_name: str = "",
) -> Any:                      # folium.Map | None
    """Build a folium choropleth map of detected polygons overlaid on OSM base tiles.

    Each class is drawn in its configured colour. The course boundary is shown
    as a blue outline.

    Returns the folium.Map object (displayable in Jupyter/Colab), or None if
    folium is not installed.
    """
    if not _HAS_FOLIUM:
        log.warning("folium not installed — cannot render preview map.")
        return None

    centroid = boundary.centroid
    m = _folium.Map(
        location=[centroid.y, centroid.x],
        zoom_start=15,
        tiles="CartoDB positron",
    )

    # Course boundary
    _folium.GeoJson(
        gpd.GeoDataFrame({"geometry": [boundary]}, crs="EPSG:4326").__geo_interface__,
        style_function=lambda _: {
            "color": "#2196F3", "weight": 3, "fillOpacity": 0.05,
        },
        tooltip=course_name or "Course boundary",
    ).add_to(m)

    # Class polygons
    class_colours = {c.name: c.color for c in cfg.classes}

    for cls_name, group in gdf.groupby("class_name"):
        colour = class_colours.get(str(cls_name), "#FFFFFF")
        _folium.GeoJson(
            group.__geo_interface__,
            style_function=lambda _, col=colour: {
                "color": col,
                "weight": 1.5,
                "fillColor": col,
                "fillOpacity": 0.45,
            },
            tooltip=str(cls_name),
            name=str(cls_name),
        ).add_to(m)

    _folium.LayerControl().add_to(m)
    return m


# ── Confirmation gate ─────────────────────────────────────────────────────────

def confirmation_gate(
    gdf: gpd.GeoDataFrame,
    boundary: Any,
    course_id: str,
    cfg: GolfMapperConfig,
    course_name: str = "",
) -> str:
    """Display a summary and ask for explicit confirmation before export.

    Prints:
      - Per-class feature counts
      - A prompt to render the map (in a notebook environment)

    Returns:
      'confirm' — user typed 'y'; proceed with export.
      'skip'    — user typed 's'; skip this course, continue pipeline.
      'abort'   — user typed 'a' or EOF; stop the entire pipeline.

    This gate is REQUIRED by the spec.  No output files are written unless
    the user explicitly confirms with 'y'.
    """
    _print_summary(gdf, course_id, course_name)

    m = render_preview_map(gdf, boundary, cfg, course_name)
    if m is not None:
        try:
            from IPython.display import display
            display(m)
        except ImportError:
            log.info("Not in a notebook environment — map not displayed.")
            log.info("To view, save the map: m.save('preview.html')")

    print("\n" + "=" * 60)
    print("EXPORT CONFIRMATION REQUIRED")
    print("  y = export this course (writes .geojson, .osm, provenance.json)")
    print("  s = skip this course (no files written)")
    print("  a = abort pipeline")
    print("=" * 60)

    try:
        answer = input("Export this course? [y/s/a] > ").strip().lower()
    except EOFError:
        # Non-interactive environment (e.g. CI) — default to skip
        log.warning("Non-interactive environment — defaulting to 'skip'.")
        answer = "s"

    if answer == "y":
        log.info("User confirmed export for %s.", course_id)
        return "confirm"
    elif answer == "a":
        log.warning("User chose abort — stopping pipeline.")
        return "abort"
    else:
        log.info("User skipped %s.", course_id)
        return "skip"


def _print_summary(gdf: gpd.GeoDataFrame, course_id: str, course_name: str) -> None:
    """Print a per-class feature count table to stdout."""
    print()
    print("=" * 60)
    print(f"  Course : {course_name or course_id}")
    print(f"  OSM ID : {course_id}")
    print(f"  Total  : {len(gdf)} predicted polygons")
    print("-" * 60)
    if gdf.empty:
        print("  (no features detected)")
    else:
        counts = gdf.groupby("class_name").size().sort_values(ascending=False)
        for cls, cnt in counts.items():
            print(f"  {cls:<20} {cnt:>4}")
    print("=" * 60)
