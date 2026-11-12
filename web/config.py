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
    # Query-query cosine matrices, one per model, at the deployed sif_abtt
    # configuration. Reviewer-created directories are scored from these; a
    # deployment without them simply serves no reviewer-directory candidates.
    qq_matrix_pattern: str = (
        "runs/active/resubmit/unlabelled/qq_sim_{model_slug}.npz"
    )
    ig_artifacts_dir: str = "runs/active/ig_examples/artifacts"
    ig_examples_csv: str = "runs/active/ig_examples/phase12f_examples.csv"
    feedback_db: str = "runs/active/resubmit/webapp/feedback.db"
    # Where deploy/deploy.sh records the data release it installed
    # (`<tag> <sha256>`, under its default DATA_CACHE_DIR). Read once at
    # startup so exports can say which release the server is serving; absent
    # (local development, a host that never synced a release) means unknown.
    data_release_state: str = ".deploy-cache/installed.state"

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

    def resolve_qq_matrix(self, model_slug: str) -> Path:
        """Path to one model's query-query cosine matrix."""
        return self.resolve(self.qq_matrix_pattern.format(model_slug=model_slug))


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


class CorpusChangeConfig(BaseModel):
    """What a corpus change at startup may do (issue #278, see web/CLAUDE.md).

    A data release whose unlabelled listing differs from the one the feedback
    DB was last bound to is re-keyed by xml_id at startup. Startup refuses it
    when more than `max_missing_reviewed_share` of the reviewed witnesses the
    previous corpus served are gone, which is usually a broken release. To
    accept an intended removal, set `accept_missing_reviewed_sha256` to the
    new corpus digest the refusal prints: the override applies to that exact
    corpus only, so leaving it set cannot let a later broken release through.
    """

    max_missing_reviewed_share: float = 0.05
    accept_missing_reviewed_sha256: str = os.environ.get(
        "LOCALLATIN_ACCEPT_MISSING_REVIEWED_SHA256", ""
    )

    @field_validator("max_missing_reviewed_share")
    @classmethod
    def _share_is_a_fraction(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("max_missing_reviewed_share must be between 0 and 1")
        return value


class Settings(BaseModel):
    app: AppConfig = AppConfig()
    paths: PathsConfig = PathsConfig()
    cors: CorsConfig = CorsConfig()
    pagination: PaginationConfig = PaginationConfig()
    auth: AuthConfig = AuthConfig()
    corpus_change: CorpusChangeConfig = CorpusChangeConfig()


def load_settings(config_path: str | Path | None = None) -> Settings:
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        return Settings(**data)
    return Settings()
