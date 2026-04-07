from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class AppConfig(BaseModel):
    title: str = "localLatin Scholar Review"
    version: str = "0.1.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000


class PathsConfig(BaseModel):
    data_root: str = os.environ.get("LOCALLATIN_DATA_ROOT", ".")
    canon_unlabelled: str = "data/canon_unlabelled"
    canon_labelled: str = "data/canon_labelled"
    predictions_combined: str = "runs/active/resubmit/unlabelled/unlabelled_predictions.csv"
    predictions_dir: str = "runs/active/resubmit/unlabelled"
    ig_artifacts_dir: str = "runs/active/ig_examples/artifacts"
    ig_examples_csv: str = "runs/active/ig_examples/phase12f_examples.csv"
    feedback_db: str = "runs/active/resubmit/webapp/feedback.db"

    def resolve(self, relative: str) -> Path:
        return Path(self.data_root) / relative


class CorsConfig(BaseModel):
    allow_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]
    allow_methods: list[str] = ["*"]
    allow_headers: list[str] = ["*"]


class PaginationConfig(BaseModel):
    default_page_size: int = 50
    max_page_size: int = 200


class Settings(BaseModel):
    app: AppConfig = AppConfig()
    paths: PathsConfig = PathsConfig()
    cors: CorsConfig = CorsConfig()
    pagination: PaginationConfig = PaginationConfig()


def load_settings(config_path: str | Path | None = None) -> Settings:
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        return Settings(**data)
    return Settings()
