from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from web.config import Settings

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

    # Predictions: slug -> list of dicts (one per file_id)
    predictions: dict[str, list[dict]] = field(default_factory=dict)

    # Model metadata
    model_meta: dict[str, ModelMeta] = field(default_factory=dict)
    model_slugs: list[str] = field(default_factory=list)

    # IG examples
    ig_examples: pd.DataFrame | None = None
    ig_artifact_paths: dict[int, Path] = field(default_factory=dict)


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


def build_store(settings: Settings) -> DataStore:
    """Build the in-memory DataStore at startup."""
    store = DataStore()
    paths = settings.paths
    root = Path(paths.data_root)

    # --- Load combined predictions CSV ---
    combined_path = paths.resolve(paths.predictions_combined)
    logger.info("Loading predictions from %s", combined_path)
    df = pd.read_csv(combined_path)

    # Build per-model prediction indices
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
        store.predictions[slug] = sorted(rows, key=lambda r: r["file_id"])
        store.model_meta[slug] = ModelMeta(
            slug=slug,
            display_name=_DISPLAY_NAMES.get(slug, slug),
            hf_id=str(hf_id),
            layer=layer,
            pooling=pooling,
            prediction_count=len(rows),
        )

    store.model_slugs = sorted(store.model_meta.keys())
    logger.info("Loaded %d models: %s", len(store.model_slugs), store.model_slugs)

    # Build file_id -> filename mapping from first model's predictions
    first_slug = store.model_slugs[0]
    for row in store.predictions[first_slug]:
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
        # Build artifact path index
        for model_dir in ig_artifacts_root.iterdir():
            if model_dir.is_dir():
                for npz_file in model_dir.glob("*.npz"):
                    # Extract example_id from filename like "example001_pair_example.npz"
                    name = npz_file.stem
                    try:
                        eid = int(name.split("_")[0].replace("example", ""))
                        store.ig_artifact_paths[eid] = npz_file
                    except (ValueError, IndexError):
                        pass
        logger.info("Indexed %d IG artifacts", len(store.ig_artifact_paths))
    else:
        logger.warning("IG examples CSV not found at %s", ig_csv_path)

    return store
