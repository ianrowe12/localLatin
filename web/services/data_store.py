from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from web.config import Settings
from web.models import DEFAULT_VARIANT

logger = logging.getLogger(__name__)

# Slug mapping: filesystem-safe slug <-> HuggingFace model ID
_SLUG_TO_HF: dict[str, str] = {
    "bowphs_LaTa": "bowphs/LaTa",
    "bowphs_PhilTa": "bowphs/PhilTa",
    "sentence-transformers_LaBSE": "sentence-transformers/LaBSE",
    "google_mt5-base": "google/mt5-base",
    "KaLM-Embedding_KaLM-embedding-multilingual-mini-instruct-v2.5": (
        "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5"
    ),
    "Qwen_Qwen3-Embedding-0.6B": "Qwen/Qwen3-Embedding-0.6B",
}
_HF_TO_SLUG: dict[str, str] = {v: k for k, v in _SLUG_TO_HF.items()}

# Short display names
_DISPLAY_NAMES: dict[str, str] = {
    "bowphs_LaTa": "LaTa",
    "bowphs_PhilTa": "PhilTa",
    "sentence-transformers_LaBSE": "LaBSE",
    "google_mt5-base": "mT5-base",
    "KaLM-Embedding_KaLM-embedding-multilingual-mini-instruct-v2.5": "KaLM-mini",
    "Qwen_Qwen3-Embedding-0.6B": "Qwen3-0.6B",
}


def normalize_slug(slug_or_hf: str) -> str:
    """Accept either filesystem slug or HF ID, return filesystem slug."""
    if slug_or_hf in _SLUG_TO_HF:
        return slug_or_hf
    if slug_or_hf in _HF_TO_SLUG:
        return _HF_TO_SLUG[slug_or_hf]
    return slug_or_hf.replace("/", "_")


@dataclass
class ModelMeta:
    slug: str
    display_name: str
    hf_id: str
    layer: int | None
    pooling: str | None
    prediction_count: int
    variant: str = DEFAULT_VARIANT


@dataclass
class DataStore:
    # Query metadata: file_id -> (filename, file_path)
    file_ids: list[int] = field(default_factory=list)
    file_id_to_filename: dict[int, str] = field(default_factory=dict)
    file_id_to_path: dict[int, str] = field(default_factory=dict)

    # Cached text content
    unlabelled_texts: dict[int, str] = field(default_factory=dict)
    labelled_texts: dict[str, dict[str, str]] = field(default_factory=dict)
    labelled_dir_files: dict[str, list[str]] = field(default_factory=dict)

    # Predictions: (slug, variant) -> list of dicts (one per file_id).
    # Only the default variant is loaded at startup; the rest are read on first
    # use so a single uvicorn worker does not hold all four CSVs at once.
    predictions: dict[tuple[str, str], list[dict]] = field(default_factory=dict)

    # Model metadata, also keyed by (slug, variant)
    model_meta: dict[tuple[str, str], ModelMeta] = field(default_factory=dict)
    model_slugs: list[str] = field(default_factory=list)

    # Variant wiring
    variants: list[str] = field(default_factory=list)
    default_variant: str = DEFAULT_VARIANT
    variant_paths: dict[str, Path] = field(default_factory=dict)
    loaded_variants: set[str] = field(default_factory=set)
    _load_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    # IG examples
    ig_examples: pd.DataFrame | None = None
    ig_artifact_paths: dict[int, Path] = field(default_factory=dict)

    # --- Variant-aware prediction access ---

    def ensure_variant(self, variant: str) -> bool:
        """Load a variant's predictions if not loaded yet. False if unavailable."""
        if variant in self.loaded_variants:
            return True
        if variant not in self.variant_paths:
            return False
        with self._load_lock:
            # Another request may have loaded it while we waited for the lock.
            if variant not in self.loaded_variants:
                _load_variant(self, variant)
        return variant in self.loaded_variants

    def get_predictions(self, slug: str, variant: str) -> list[dict] | None:
        if not self.ensure_variant(variant):
            return None
        return self.predictions.get((slug, variant))

    def has_model(self, slug: str, variant: str) -> bool:
        return self.get_predictions(slug, variant) is not None


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _parse_predictions_row(row: pd.Series) -> dict:
    """Parse a single row from predictions CSV into structured dict."""
    preds = []
    for rank in range(1, 11):
        dir_col = f"rank{rank}_dir"
        score_col = f"rank{rank}_score"
        if dir_col in row and pd.notna(row[dir_col]):
            preds.append({
                "rank": rank,
                "dir_name": str(row[dir_col]),
                "score": float(row[score_col]),
            })
    return {
        "file_id": int(row["file_id"]),
        "filename": str(row["filename"]),
        "predictions": preds,
    }


def _load_variant(store: DataStore, variant: str) -> None:
    """Read one variant's predictions CSV into the store. Caller holds the lock."""
    path = store.variant_paths[variant]
    logger.info("Loading '%s' predictions from %s", variant, path)
    df = pd.read_csv(path)

    if "variant" in df.columns:
        declared = {str(value) for value in df["variant"].dropna().unique()}
        if declared and declared != {variant}:
            logger.warning(
                "Predictions file %s declares variant(s) %s but is wired as '%s'",
                path,
                sorted(declared),
                variant,
            )

    for hf_id, group in df.groupby("model"):
        slug = normalize_slug(str(hf_id))
        rows = []
        layer = None
        pooling = None
        for _, row in group.iterrows():
            rows.append(_parse_predictions_row(row))
            if layer is None and "layer" in row:
                layer = int(row["layer"])
            if pooling is None and "pooling" in row:
                pooling = str(row["pooling"])

        # Index by file_id for O(1) lookup
        store.predictions[(slug, variant)] = sorted(rows, key=lambda r: r["file_id"])
        store.model_meta[(slug, variant)] = ModelMeta(
            slug=slug,
            display_name=_DISPLAY_NAMES.get(slug, slug),
            hf_id=str(hf_id),
            layer=layer,
            pooling=pooling,
            prediction_count=len(rows),
            variant=variant,
        )

    store.loaded_variants.add(variant)


def build_store(settings: Settings) -> DataStore:
    """Build the in-memory DataStore at startup."""
    store = DataStore()
    paths = settings.paths
    root = Path(paths.data_root)

    # --- Discover per-variant predictions CSVs ---
    # There is no fallback to the pre-variant frozen unlabelled_predictions.csv:
    # it predates the current Task B split and would serve stale rankings (#45).
    store.default_variant = paths.default_variant
    for variant in paths.variants:
        variant_path = paths.resolve_variant(variant)
        if variant_path.exists():
            store.variant_paths[variant] = variant_path
        else:
            logger.warning(
                "Predictions CSV for variant '%s' not found at %s; not served",
                variant,
                variant_path,
            )
    store.variants = [v for v in paths.variants if v in store.variant_paths]

    if store.default_variant not in store.variant_paths:
        raise FileNotFoundError(
            "Default prediction variant "
            f"'{store.default_variant}' has no CSV at "
            f"{paths.resolve_variant(store.default_variant)}"
        )

    # Eagerly load the default variant only; the rest load on first request.
    _load_variant(store, store.default_variant)

    store.model_slugs = sorted(
        {slug for slug, _variant in store.model_meta if _variant == store.default_variant}
    )
    logger.info(
        "Loaded %d models for variant '%s': %s (variants available: %s)",
        len(store.model_slugs),
        store.default_variant,
        store.model_slugs,
        store.variants,
    )

    # Build file_id -> filename mapping from first model's default-variant rows
    first_slug = store.model_slugs[0]
    for row in store.predictions[(first_slug, store.default_variant)]:
        fid = row["file_id"]
        store.file_ids.append(fid)
        store.file_id_to_filename[fid] = row["filename"]
        store.file_id_to_path[fid] = f"data/canon_unlabelled/{row['filename']}"
    store.file_ids.sort()

    # --- Cache unlabelled text files ---
    unlabelled_dir = paths.resolve(paths.canon_unlabelled)
    logger.info("Caching unlabelled texts from %s", unlabelled_dir)
    for fid in store.file_ids:
        fname = store.file_id_to_filename[fid]
        fpath = unlabelled_dir / fname
        if fpath.exists():
            store.unlabelled_texts[fid] = _load_text(fpath)
        else:
            store.unlabelled_texts[fid] = ""
    logger.info("Cached %d unlabelled texts", len(store.unlabelled_texts))

    # --- Cache labelled text files ---
    labelled_dir = paths.resolve(paths.canon_labelled)
    logger.info("Caching labelled texts from %s", labelled_dir)
    if labelled_dir.exists():
        for dir_entry in sorted(labelled_dir.iterdir()):
            if dir_entry.is_dir():
                dir_name = dir_entry.name
                files: dict[str, str] = {}
                filenames: list[str] = []
                for txt_file in sorted(dir_entry.glob("*.txt")):
                    files[txt_file.name] = _load_text(txt_file)
                    filenames.append(txt_file.name)
                if files:
                    store.labelled_texts[dir_name] = files
                    store.labelled_dir_files[dir_name] = filenames
    logger.info(
        "Cached %d labelled directories, %d files",
        len(store.labelled_texts),
        sum(len(v) for v in store.labelled_texts.values()),
    )

    # --- Load IG example metadata ---
    ig_csv_path = paths.resolve(paths.ig_examples_csv)
    ig_artifacts_root = paths.resolve(paths.ig_artifacts_dir)
    if ig_csv_path.exists():
        logger.info("Loading IG examples from %s", ig_csv_path)
        store.ig_examples = pd.read_csv(ig_csv_path)
        # Each example_id has exactly one canonical owning model (per the CSV).
        # The IG regen pipeline writes NPZ files keyed by global example_id, but
        # historically leaked stray copies into wrong model dirs. Use the CSV as
        # the source of truth: only register an artifact when its parent dir slug
        # matches the example_id's CSV-declared owner.
        eid_to_owner_slug: dict[int, str] = {
            int(row["example_id"]): normalize_slug(str(row["model_name"]))
            for _, row in store.ig_examples.iterrows()
        }
        skipped_stray = 0
        for model_dir in ig_artifacts_root.iterdir():
            if not model_dir.is_dir():
                continue
            dir_slug = normalize_slug(model_dir.name)
            for npz_file in model_dir.glob("*.npz"):
                # Extract example_id from filename like "example001_pair_example.npz"
                name = npz_file.stem
                try:
                    eid = int(name.split("_")[0].replace("example", ""))
                except (ValueError, IndexError):
                    continue
                expected_slug = eid_to_owner_slug.get(eid)
                if expected_slug is None or expected_slug != dir_slug:
                    skipped_stray += 1
                    continue
                store.ig_artifact_paths[eid] = npz_file
        logger.info(
            "Indexed %d IG artifacts (skipped %d stray/orphan files)",
            len(store.ig_artifact_paths),
            skipped_stray,
        )
    else:
        logger.warning("IG examples CSV not found at %s", ig_csv_path)

    return store
