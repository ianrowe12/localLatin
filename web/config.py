from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, field_validator, model_validator

from web.variants import DEFAULT_VARIANT, VARIANTS


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
    # One predictions CSV per post-processing variant. The pre-variant frozen
    # file (unlabelled_predictions.csv) predates the 2026-04-07 Task B split
    # redesign and is deliberately NOT a fallback -- see issue #45.
    predictions_variant_pattern: str = (
        "runs/active/resubmit/unlabelled/unlabelled_predictions_{variant}.csv"
    )
    variants: list[str] = list(VARIANTS)
    default_variant: str = DEFAULT_VARIANT
    predictions_dir: str = "runs/active/resubmit/unlabelled"
    ig_artifacts_dir: str = "runs/active/ig_examples/artifacts"
    ig_examples_csv: str = "runs/active/ig_examples/phase12f_examples.csv"
    feedback_db: str = "runs/active/resubmit/webapp/feedback.db"

    @field_validator("variants")
    @classmethod
    def _known_variants(cls, value: list[str]) -> list[str]:
        unknown = [v for v in value if v not in VARIANTS]
        if unknown:
            raise ValueError(f"Unknown prediction variants: {unknown}")
        if not value:
            raise ValueError("At least one prediction variant must be configured")
        return value

    @model_validator(mode="after")
    def _default_variant_is_configured(self) -> "PathsConfig":
        if self.default_variant not in self.variants:
            raise ValueError(
                f"default_variant '{self.default_variant}' is not in variants {self.variants}"
            )
        return self

    def resolve(self, relative: str) -> Path:
        return Path(self.data_root) / relative

    def resolve_variant(self, variant: str) -> Path:
        """Path to the predictions CSV for one post-processing variant."""
        return self.resolve(self.predictions_variant_pattern.format(variant=variant))


class CorsConfig(BaseModel):
    allow_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]
    allow_methods: list[str] = ["*"]
    allow_headers: list[str] = ["*"]


class PaginationConfig(BaseModel):
    default_page_size: int = 50
    max_page_size: int = 200


class AuthConfig(BaseModel):
    session_cookie: str = "locallatin_session"
    session_days: int = 14
    secure_cookies: bool = False
    admin_registration_code: str | None = os.environ.get("LOCALLATIN_ADMIN_CODE")
    # Password change / reset throttling, per actor, in a rolling window.
    password_rate_limit_max_attempts: int = 10
    password_rate_limit_window_seconds: int = 900


class Settings(BaseModel):
    app: AppConfig = AppConfig()
    paths: PathsConfig = PathsConfig()
    cors: CorsConfig = CorsConfig()
    pagination: PaginationConfig = PaginationConfig()
    auth: AuthConfig = AuthConfig()


def load_settings(config_path: str | Path | None = None) -> Settings:
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        return Settings(**data)
    return Settings()
