"""Shared model metadata for attribution example generation.

Run 2 fixed the Run 3 attribution layer contract on 2026-04-29:
use the train-selected operational retrieval layer, not the recovered-collapse
diagnostic layer. CLI layer overrides remain available for diagnostics, but
production Run 3 attribution should use these defaults.
"""
from __future__ import annotations

from copy import deepcopy


METHODS_AVAILABLE = (
    "ig",
    "bertscore",
    "ot",
    "attention_weighted",
    "dla",
    "attention_standalone",
    "retrieval_mark",
)

DEFAULT_MODELS = ["bowphs/LaTa", "bowphs/PhilTa", "google/mt5-base"]

# D=10 universal per the 2026-03-31 meeting decision: abtt_fixed (D=10)
# matches abtt_optimal across layers, so per-layer D tuning adds no benefit.
# Layers below are the Run 2 contract:
#   LaTa L7, PhilTa L1, mT5-base L1.
FEATURED_MODELS = {
    "bowphs/LaTa": {
        "model_short": "LaTa",
        "model_type": "t5",
        "layer": 7,
        "D": 10,
        "tau": 0.5025125628140703,
        "baseline_tau": 0.9949748743718592,
        "abtt_tau": 0.5025125628140703,
    },
    "bowphs/PhilTa": {
        "model_short": "PhilTa",
        "model_type": "t5",
        "layer": 1,
        "D": 10,
        "tau": 0.4170854271356784,
        "baseline_tau": 0.8944723618090452,
        "abtt_tau": 0.4170854271356784,
    },
    "google/mt5-base": {
        "model_short": "mT5-base",
        "model_type": "t5",
        "layer": 1,
        "D": 10,
        "tau": 0.4371859296482412,
        "baseline_tau": 0.9949748743718592,
        "abtt_tau": 0.4371859296482412,
    },
    "sentence-transformers/LaBSE": {
        "model_short": "LaBSE",
        "model_type": "bert",
        "layer": 12,
        "D": 10,
        "tau": 0.5829145728643216,
        "baseline_tau": 0.9195979899497488,
        "abtt_tau": 0.5829145728643216,
    },
    "Qwen/Qwen3-Embedding-0.6B": {
        "model_short": "Qwen3-0.6B",
        "model_type": "decoder",
        "layer": 23,
        "D": 10,
        "tau": 0.5226130653266332,
        "baseline_tau": 0.984924623115578,
        "abtt_tau": 0.5226130653266332,
    },
    # KaLM-mini layer 1 is the train-selected operational retrieval layer:
    # it maximises train_aucroc (0.984993) and overall_assignment_acc
    # (0.920746) in runs/active/resubmit/results/phase_resubmit_results.csv.
    # Thresholds are the abtt_optimal / baseline taus for that row.
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5": {
        "model_short": "KaLM-mini",
        "model_type": "decoder",
        "layer": 1,
        "D": 10,
        "tau": 0.4120603015075377,
        "baseline_tau": 0.9748743718592965,
        "abtt_tau": 0.4120603015075377,
    },
}


def model_slug(name: str) -> str:
    return name.replace("/", "_")


def methods_available_string() -> str:
    return ",".join(METHODS_AVAILABLE)


def parse_layer_overrides(values: list[str] | None) -> dict[str, int]:
    """Parse CLI overrides like ``google/mt5-base=4`` or ``google_mt5-base=4``."""
    overrides: dict[str, int] = {}
    for raw in values or []:
        if "=" not in raw:
            raise ValueError(f"Layer override must be MODEL=LAYER, got {raw!r}")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Layer override has empty model key: {raw!r}")
        try:
            layer = int(value)
        except ValueError as exc:
            raise ValueError(f"Layer override must use an integer layer: {raw!r}") from exc
        overrides[key] = layer
    return overrides


def model_config(model_name: str, layer_overrides: dict[str, int] | None = None) -> dict:
    if model_name not in FEATURED_MODELS:
        known = ", ".join(FEATURED_MODELS)
        raise KeyError(f"Unknown model {model_name!r}; known models: {known}")
    cfg = deepcopy(FEATURED_MODELS[model_name])
    overrides = layer_overrides or {}
    slug = model_slug(model_name)
    short = str(cfg["model_short"])
    for key in (model_name, slug, short):
        if key in overrides:
            cfg["layer"] = overrides[key]
            break
    return cfg


def slug_layer_map(layer_overrides: dict[str, int] | None = None) -> dict[str, int]:
    return {
        model_slug(name): int(model_config(name, layer_overrides)["layer"])
        for name in FEATURED_MODELS
    }
