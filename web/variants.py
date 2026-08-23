"""Canonical prediction-variant vocabulary.

Deliberately dependency-free so both the config layer (`web.config`) and the
API schema layer (`web.models`) can import it without one depending on the
other.
"""

from __future__ import annotations

from enum import StrEnum


class PredictionVariant(StrEnum):
    """Post-processing variant a set of predictions was computed with.

    Each variant has its own predictions CSV (see PathsConfig.resolve_variant).
    Anything outside this set is rejected by FastAPI/pydantic with a 422.
    """

    RAW = "raw"
    ABTT = "abtt"
    SIF = "sif"
    SIF_ABTT = "sif_abtt"


VARIANTS: tuple[str, ...] = tuple(variant.value for variant in PredictionVariant)

#: Fallback default. The variant actually served when a request omits the
#: parameter is `PathsConfig.default_variant`, resolved per request from the
#: store -- routes must never hardcode this constant.
DEFAULT_VARIANT: str = PredictionVariant.SIF_ABTT.value
