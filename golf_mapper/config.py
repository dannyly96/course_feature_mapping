"""Typed configuration loader for Golf Course Feature Mapper."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

log = logging.getLogger(__name__)


class AOIConfig(BaseModel):
    type: Literal["osm_id", "bbox", "admin_name"] = "osm_id"
    osm_id: str = "relation/5179090"
    name: str = "Augusta National Golf Club"
    bbox: list[float] | None = None  # [min_lon, min_lat, max_lon, max_lat]
    buffer_m: float = 50.0

    @model_validator(mode="after")
    def _check_bbox_when_needed(self) -> "AOIConfig":
        if self.type == "bbox" and self.bbox is None:
            raise ValueError("aoi.bbox is required when aoi.type='bbox'")
        if self.bbox is not None and len(self.bbox) != 4:
            raise ValueError("aoi.bbox must be [min_lon, min_lat, max_lon, max_lat]")
        return self


class DataConfig(BaseModel):
    cache_dir: Path = Path("/content/golf_mapper/cache")
    imagery_cache: Path = Path("/content/golf_mapper/cache/imagery")
    overpass_cache: Path = Path("/content/golf_mapper/cache/overpass")
    dataset_dir: Path = Path("/content/golf_mapper/dataset")
    checkpoints_dir: Path = Path("/content/golf_mapper/checkpoints")
    output_dir: Path = Path("/content/golf_mapper/output")

    def make_dirs(self) -> None:
        """Create all configured directories (idempotent)."""
        for field_name in DataConfig.model_fields:
            path: Path = getattr(self, field_name)
            path.mkdir(parents=True, exist_ok=True)
            log.debug("Ensured directory: %s", path)


class ImageryConfig(BaseModel):
    zoom: int = Field(default=19, ge=18, le=20)
    source: str = "Esri.WorldImagery"
    esri_xyz_url: str = (
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}"
    )


class LabelsConfig(BaseModel):
    tile_size: int = Field(default=1024, gt=0)
    overlap: int = Field(default=128, ge=0)
    min_features: int = Field(default=50, ge=1)
    empty_tile_max_ratio: float = Field(default=0.10, ge=0.0, le=1.0)


class ModelConfig(BaseModel):
    variant: Literal["yolo26-seg", "yolo11-seg"] = "yolo26-seg"
    sam_model_version: str = "sam2"
    sam_model_type: str = "hiera_large"
    use_sam3: bool = False


class TrainingConfig(BaseModel):
    weighting_mode: Literal["global", "regional"] = "regional"
    sigma_km: float = Field(default=150.0, gt=0)
    max_radius_km: float = Field(default=500.0, gt=0)
    epochs: int = Field(default=100, gt=0)
    batch_size: int = Field(default=16, gt=0)
    img_size: int = Field(default=1024, gt=0)
    seed: int = 42
    resume: bool = False
    val_split: float = Field(default=0.15, gt=0, lt=1)
    test_split: float = Field(default=0.10, gt=0, lt=1)


class InferenceConfig(BaseModel):
    conf_threshold: float = Field(default=0.25, gt=0, lt=1)
    iou_threshold: float = Field(default=0.45, gt=0, lt=1)
    min_area_m2: float = Field(default=10.0, gt=0)
    tile_overlap: int = Field(default=128, ge=0)


class ExportConfig(BaseModel):
    simplify_tolerance_m: float = Field(default=0.5, gt=0)
    output_crs: str = "EPSG:4326"
    attribution: str = (
        "Imagery source: Esri World Imagery. "
        "Derived vector data licensed under ODbL (OpenStreetMap contributors)."
    )


class OverpassConfig(BaseModel):
    endpoints: list[str] = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    ]
    timeout: int = Field(default=120, gt=0)
    max_retries: int = Field(default=5, ge=1)
    backoff_base: float = Field(default=2.0, gt=0)


class ClassDef(BaseModel):
    id: int
    name: str
    color: str = "#FFFFFF"
    osm_read_tags: list[dict[str, str]] = Field(default_factory=list)
    osm_write_tags: dict[str, str] = Field(default_factory=dict)
    is_background: bool = False


_DEFAULT_CLASSES: list[dict[str, Any]] = [
    {
        "id": 0, "name": "rough", "color": "#8BC34A",
        "osm_read_tags": [{"golf": "rough"}],
        "osm_write_tags": {"golf": "rough"},
        "is_background": True,
    },
    {
        "id": 1, "name": "fairway", "color": "#4CAF50",
        "osm_read_tags": [{"golf": "fairway"}],
        "osm_write_tags": {"golf": "fairway"},
    },
    {
        "id": 2, "name": "green", "color": "#1B5E20",
        "osm_read_tags": [{"golf": "green"}],
        "osm_write_tags": {"golf": "green"},
    },
    {
        "id": 3, "name": "tee", "color": "#FFF9C4",
        "osm_read_tags": [{"golf": "tee"}],
        "osm_write_tags": {"golf": "tee"},
    },
    {
        "id": 4, "name": "bunker", "color": "#FFF176",
        "osm_read_tags": [{"golf": "bunker"}],
        "osm_write_tags": {"golf": "bunker"},
    },
    {
        "id": 5, "name": "water_hazard", "color": "#2196F3",
        "osm_read_tags": [
            {"natural": "water"},
            {"golf": "water_hazard"},
            {"golf": "lateral_water_hazard"},
        ],
        "osm_write_tags": {"natural": "water"},
    },
    {
        "id": 6, "name": "woods", "color": "#33691E",
        "osm_read_tags": [
            {"natural": "wood"},
            {"landuse": "forest"},
            {"natural": "tree_row"},
        ],
        "osm_write_tags": {"natural": "wood"},
    },
]


class GolfMapperConfig(BaseModel):
    aoi: AOIConfig = Field(default_factory=AOIConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    imagery: ImageryConfig = Field(default_factory=ImageryConfig)
    labels: LabelsConfig = Field(default_factory=LabelsConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    overpass: OverpassConfig = Field(default_factory=OverpassConfig)
    classes: list[ClassDef] = Field(
        default_factory=lambda: [ClassDef(**c) for c in _DEFAULT_CLASSES]
    )

    @property
    def class_names(self) -> list[str]:
        return [c.name for c in sorted(self.classes, key=lambda c: c.id)]

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def class_by_name(self, name: str) -> ClassDef:
        for c in self.classes:
            if c.name == name:
                return c
        raise KeyError(f"Unknown class name: {name!r}")

    def class_by_id(self, class_id: int) -> ClassDef:
        for c in self.classes:
            if c.id == class_id:
                return c
        raise KeyError(f"Unknown class id: {class_id}")

    def config_hash(self) -> str:
        """16-char SHA-256 of the serialized config for provenance records."""
        raw = self.model_dump_json(indent=None)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_config(path: str | Path = "config.yaml") -> GolfMapperConfig:
    """Load and validate configuration from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as fh:
        raw = yaml.safe_load(fh)
    cfg = GolfMapperConfig(**(raw or {}))
    log.info("Loaded config from %s (hash=%s)", path, cfg.config_hash())
    return cfg
