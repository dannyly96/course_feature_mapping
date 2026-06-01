# Golf Course Feature Mapper

Automatically extracts golf course features (fairways, greens, tees, bunkers, water
hazards, and woodland) from Esri World Imagery satellite imagery and produces
**OSM-ready vector polygons** for review and upload via JOSM.

The model learns from courses that are already well-mapped in OpenStreetMap (using
their existing feature polygons as free, pixel-aligned training labels) and applies
that knowledge to courses that are not yet mapped.

**No data is uploaded to OSM automatically.** All output is reviewed by a human in
JOSM before any upload. The export stage has a mandatory interactive confirmation gate.

---

## Licensing

> **Read this before using.**

| Resource | License / Terms |
|---|---|
| Esri World Imagery (source tiles) | Esri grants permission to **trace** World Imagery for the purpose of creating OSM vector data. Raw tiles/GeoTIFFs **must not** be redistributed, re-hosted, or committed to any repository. They are stored in a local gitignored cache only. |
| Derived vector polygons (output) | **ODbL** — OpenStreetMap's database license. Attribution required: *"© OpenStreetMap contributors"*. |
| OSM training labels | **ODbL** — fetched from OpenStreetMap via Overpass API. |

Every output file includes a `provenance.json` with the imagery source, license
statement, model version, and run timestamp.

---

## Colab Quickstart

```
notebooks/colab_quickstart.ipynb
```

1. Open the notebook in Google Colab (**Runtime → Change runtime type → GPU**).
2. Run **Cell 1** to verify GPU availability.
3. Run **Cell 2** to clone the repo and install dependencies.
4. Run **Cell 3** to write `versions.lock`.
5. Run **Cell 5** to load the configuration (targets Augusta National by default).
6. Run each stage cell in sequence. Stages cache results — re-running skips completed work.
7. At **Stage 6 (Export)**, review the interactive Folium preview map and type `y` to confirm before output files are written.

### Google Drive (persistent storage)

The default config uses `/content/golf_mapper` (ephemeral Colab disk). For persistence across sessions, mount Drive and update all `data.*` paths in `config.yaml`:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then set `data.cache_dir: "/content/drive/MyDrive/golf_mapper/cache"`, etc.

---

## Local Setup

```bash
git clone <this repo>
cd golf_course_feature_mapping
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip freeze > versions.lock   # record exact resolved versions
```

Requires a CUDA GPU for practical training speed. CPU fallback works but is very slow.

---

## Pipeline Stages

| Stage | Module | Description |
|---|---|---|
| 1 | `discovery.py` | Query Overpass for golf courses; count features; write course manifest |
| 2 | `imagery.py` | Download Esri World Imagery via samgeo; validate CRS + coverage |
| 3 | `labels.py` | OSM features → raster masks → tiled YOLO-seg dataset (split by course) |
| 4 | `train.py` | Train YOLO26-seg with optional proximity-weighted regional fine-tuning |
| 5 | `infer.py` | Tiled inference + SAM 2.1 boundary refinement + ExG canopy layer |
| 6 | `export.py` | Confirmation gate → `.geojson` + `.osm` + `provenance.json` |
| 7 | `evaluate.py` | Held-out per-class P/R/IoU evaluation + geometry QA report |

### Run individual stages

```bash
python -m golf_mapper.discovery --config config.yaml
python -m golf_mapper.imagery   --config config.yaml
# ... etc.
```

---

## Output Files (per course)

| File | Description |
|---|---|
| `<course>.geojson` | EPSG:4326 tagged polygons — primary review artifact |
| `<course>.osm` | OSM XML — open directly in JOSM for review and manual upload |
| `<course>_provenance.json` | Imagery source, license, model version, config hash, timestamp |
| `evaluation_metrics.csv` | Per-class precision / recall / IoU on held-out courses |
| `evaluation_metrics.md` | Human-readable markdown summary with macro-average |

### Uploading via JOSM

1. Open `<course>.osm` in JOSM (`File → Open`).
2. Download the existing OSM data for the area (`File → Update Data`).
3. Review every polygon — fix errors, add missing attributes (hole number, name, etc.).
4. Upload with your OSM account. **Never upload without human review.**

Suggested changeset comment:
> *Traced from Esri World Imagery with golf-mapper (ML-assisted); imagery © Esri*

---

## Configuration

All parameters are in `config.yaml`. Key values:

| Key | Default | Description |
|---|---|---|
| `aoi.osm_id` | `relation/5179090` | Target course (Augusta National) |
| `labels.min_features` | 50 | Minimum OSM features for a training course |
| `imagery.zoom` | 19 | Tile zoom level (18–20; 19 ≈ 25 cm/px at 33.5°N) |
| `labels.tile_size` | 1024 | Patch size in pixels |
| `labels.overlap` | 128 | Tile overlap in pixels |
| `model.variant` | `yolo26-seg` | `yolo26-seg` or `yolo11-seg` (fallback) |
| `training.weighting_mode` | `regional` | `global` or proximity-weighted `regional` |
| `training.sigma_km` | 150 | Gaussian decay σ for proximity weighting |
| `training.max_radius_km` | 500 | Hard cutoff for training course inclusion |
| `model.use_sam3` | `false` | Enable SAM 3 text prompting (requires HF access) |
| `export.simplify_tolerance_m` | 0.5 | Douglas-Peucker simplification tolerance |

---

## SAM 3 (optional upgrade)

SAM 3 enables text-prompted boundary refinement (e.g. "sand bunker", "pond") instead
of box-prompted SAM 2.1. It is **off by default** (`use_sam3: false`).

To enable:
1. Accept the model license at `https://huggingface.co/facebook/sam3`
2. Log in: `huggingface-cli login`
3. Set `model.use_sam3: true` in `config.yaml`

The pipeline falls back to SAM 2.1 automatically if access is denied or the weights
cannot be downloaded.

---

## OSM Class Mapping

| Class | OSM tags read (training labels) | OSM tags written (export) |
|---|---|---|
| rough *(background)* | `golf=rough` | `golf=rough` |
| fairway | `golf=fairway` | `golf=fairway` |
| green | `golf=green` | `golf=green` |
| tee | `golf=tee` | `golf=tee` |
| bunker | `golf=bunker` | `golf=bunker` |
| water_hazard | `natural=water`, `golf=water_hazard`, `golf=lateral_water_hazard` | `natural=water` |
| woods | `natural=wood`, `landuse=forest`, `natural=tree_row` | `natural=wood` |

Tree/canopy is emitted as `natural=wood` area polygons (OSM convention for woodland),
**not** `natural=tree` point features.

---

## Tests

```bash
pytest tests/ -v
```

187 tests covering: config loading, OSM tag matching, Overpass client (mocked),
OSM XML writer, course discovery (mocked), imagery cache key / coverage validation,
geometry rasterize↔vectorize round-trip, YOLO label format, ExG vegetation mask,
proximity weighting math, course-level train/val/test split, evaluation metrics
(IoU, precision, recall), geometry QA, and report output.

---

## Project Structure

```
golf_mapper/
  config.yaml            Main configuration
  requirements.txt       Dependency constraints
  versions.lock          Exact resolved versions (written after first install)
  golf_mapper/
    config.py            Typed config loader (pydantic v2)
    utils.py             Logging, disk cache, retry, seeds
    osm.py               Overpass client + tag matching + OSM XML writer
    discovery.py         Course discovery + eligibility filter → manifest
    imagery.py           Esri imagery download (samgeo) + validation
    geometry.py          Rasterize / vectorize / simplify / validate / ExG
    labels.py            OSM → YOLO-seg dataset builder (split by course)
    weighting.py         Gaussian proximity weighting for regional fine-tune
    train.py             YOLO training loop (global + regional modes)
    infer.py             Tiled inference + SAM 2.1 + cross-tile stitching
    viz.py               Folium preview map + confirmation gate
    export.py            Simplify → tag → schema check → export GeoJSON/OSM
    evaluate.py          Held-out P/R/IoU evaluation + geometry QA report
  notebooks/
    colab_quickstart.ipynb
  tests/
    test_config.py
    test_utils.py
    test_osm.py
    test_discovery.py
    test_imagery.py
    test_geometry.py
    test_labels.py
    test_weighting.py
    test_evaluate.py
```
