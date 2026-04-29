"""Compute layerwise geometry diagnostics from cached resubmission embeddings.

The diagnostics are label-free: PCA/ABTT quantities are fit on train rows
only, and cosine concentration is computed without directory labels. Retrieval
outcomes are joined by ``build_layer_geometry_evidence.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from cli_utils import parse_layers
from canon_retrieval import l2_normalize
from sif_abtt import EmbeddingCleaner


MODEL_DISPLAY = {
    "bowphs/LaTa": "LaTa",
    "bowphs/PhilTa": "PhilTa",
    "google/mt5-base": "mT5-base",
    "sentence-transformers/LaBSE": "LaBSE",
    "Qwen/Qwen3-Embedding-0.6B": "Qwen3-0.6B",
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5": "KaLM-mini",
}

MAIN_MODELS = [
    "bowphs/LaTa",
    "bowphs/PhilTa",
    "google/mt5-base",
]

APPENDIX_MODELS = [
    "sentence-transformers/LaBSE",
    "Qwen/Qwen3-Embedding-0.6B",
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
]

SKIPPED_COLUMNS = [
    "model",
    "pooling",
    "layer",
    "reason",
    "embedding_path",
    "shape",
    "n_split_rows",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split_csv",
        default="runs/active/resubmit/data/phase_resubmit_split.csv",
        help="Resubmission train/test split CSV.",
    )
    parser.add_argument(
        "--runs_root",
        default="runs/active/resubmit_bases",
        help="Root containing phase9_bases/<model_slug>/ embedding dirs.",
    )
    parser.add_argument(
        "--out_dir",
        default="runs/active/resubmit/layer_diagnostics",
        help="Output directory for diagnostics CSV and manifest.",
    )
    parser.add_argument(
        "--models",
        default=",".join(MAIN_MODELS),
        help="Comma-separated model names, or 'main', 'appendix', or 'all'.",
    )
    parser.add_argument(
        "--poolings",
        default="mean",
        help="Comma-separated pooling views to inspect: mean,sif.",
    )
    parser.add_argument(
        "--layers",
        default="",
        help="Layer list/range such as '1,6,12' or '1-12'. Empty discovers all.",
    )
    parser.add_argument("--repr", default="hidden", help="Representation name.")
    parser.add_argument("--D", type=int, default=10, help="Fixed ABTT D.")
    parser.add_argument(
        "--max_models",
        type=int,
        default=0,
        help="Optional smoke limit on number of models after expansion.",
    )
    return parser.parse_args()


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "_")


def expand_models(value: str) -> List[str]:
    value = value.strip()
    if value == "main":
        return MAIN_MODELS
    if value == "appendix":
        return APPENDIX_MODELS
    if value == "all":
        return MAIN_MODELS + APPENDIX_MODELS
    return [m.strip() for m in value.split(",") if m.strip()]


def pooling_subdir(pooling: str) -> str:
    if pooling == "mean":
        return "hidden_mean_tokempty"
    if pooling == "sif":
        return "hidden_sif_tokempty"
    raise ValueError(f"Unsupported pooling: {pooling}")


def pooling_suffix(pooling: str) -> str:
    if pooling == "mean":
        return ""
    if pooling == "sif":
        return "_sif"
    raise ValueError(f"Unsupported pooling: {pooling}")


def embedding_file(
    runs_root: Path,
    model_name: str,
    repr_name: str,
    pooling: str,
    layer: int,
) -> Path:
    slug = model_slug(model_name)
    subdir = pooling_subdir(pooling)
    suffix = pooling_suffix(pooling)
    return (
        runs_root
        / "phase9_bases"
        / slug
        / subdir
        / f"{repr_name}_layer{layer}_embeddings{suffix}.npy"
    )


def config_file(runs_root: Path, model_name: str, pooling: str) -> Path:
    return runs_root / "phase9_bases" / model_slug(model_name) / pooling_subdir(pooling) / "config.json"


def discover_layers(
    runs_root: Path,
    model_name: str,
    repr_name: str,
    pooling: str,
) -> List[int]:
    base = runs_root / "phase9_bases" / model_slug(model_name) / pooling_subdir(pooling)
    suffix = pooling_suffix(pooling)
    pattern = f"{repr_name}_layer*_embeddings{suffix}.npy"
    layers: List[int] = []
    for path in sorted(base.glob(pattern)):
        stem = path.stem
        try:
            layer = int(stem.split("_layer", 1)[1].split("_", 1)[0])
        except (IndexError, ValueError):
            continue
        if layer >= 1:
            layers.append(layer)
    return sorted(set(layers))


def upper_triangle_values(sim: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(sim.shape[0], k=1)
    return sim[idx]


def cosine_stats(x: np.ndarray) -> Dict[str, float]:
    if x.shape[0] < 2:
        return {
            "anisotropy_mean_cosine": float("nan"),
            "cosine_mean": float("nan"),
            "cosine_std": float("nan"),
            "cosine_p05": float("nan"),
            "cosine_p50": float("nan"),
            "cosine_p95": float("nan"),
            "cosine_iqr": float("nan"),
        }
    x_norm = l2_normalize(x.astype(np.float64, copy=False))
    sims = upper_triangle_values(x_norm @ x_norm.T)
    p05, p25, p50, p75, p95 = np.percentile(sims, [5, 25, 50, 75, 95])
    return {
        "anisotropy_mean_cosine": float(np.mean(sims)),
        "cosine_mean": float(np.mean(sims)),
        "cosine_std": float(np.std(sims)),
        "cosine_p05": float(p05),
        "cosine_p50": float(p50),
        "cosine_p95": float(p95),
        "cosine_iqr": float(p75 - p25),
    }


def pca_stats(x: np.ndarray) -> Dict[str, float]:
    if x.shape[0] < 2:
        return {
            "pc1_variance_ratio": float("nan"),
            "pc10_cumulative_variance_ratio": float("nan"),
            "effective_rank_entropy": float("nan"),
        }
    centered = x.astype(np.float64, copy=False) - np.mean(x, axis=0, keepdims=True)
    _, svals, _ = np.linalg.svd(centered, full_matrices=False)
    eig = np.square(svals)
    total = float(eig.sum())
    if total <= 0.0:
        return {
            "pc1_variance_ratio": float("nan"),
            "pc10_cumulative_variance_ratio": float("nan"),
            "effective_rank_entropy": float("nan"),
        }
    probs = eig / total
    nonzero = probs[probs > 0.0]
    entropy = -float(np.sum(nonzero * np.log(nonzero)))
    return {
        "pc1_variance_ratio": float(probs[0]),
        "pc10_cumulative_variance_ratio": float(probs[: min(10, len(probs))].sum()),
        "effective_rank_entropy": float(np.exp(entropy)),
    }


def geometry_stats(x: np.ndarray) -> Dict[str, float]:
    out = {
        "n": int(x.shape[0]),
        "dim": int(x.shape[1]) if x.ndim == 2 else 0,
    }
    out.update(cosine_stats(x))
    out.update(pca_stats(x))
    return out


def add_delta_columns(raw: Dict[str, float], abtt: Dict[str, float]) -> None:
    delta_pairs = {
        "delta_anisotropy_mean_cosine": "anisotropy_mean_cosine",
        "delta_pc1_variance_ratio": "pc1_variance_ratio",
        "delta_pc10_cumulative_variance_ratio": "pc10_cumulative_variance_ratio",
        "delta_effective_rank_entropy": "effective_rank_entropy",
        "delta_cosine_std": "cosine_std",
        "delta_cosine_iqr": "cosine_iqr",
    }
    for dst, src in delta_pairs.items():
        raw_val = raw.get(src, float("nan"))
        abtt_val = abtt.get(src, float("nan"))
        delta = abtt_val - raw_val
        raw[dst] = float(delta)
        abtt[dst] = float(delta)


def read_token_filter(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return ""
    return str(payload.get("token_filter", ""))


def iter_requested_layers(
    runs_root: Path,
    model_name: str,
    repr_name: str,
    pooling: str,
    requested: Iterable[int],
) -> List[int]:
    requested_layers = list(requested)
    if requested_layers:
        return requested_layers
    return discover_layers(runs_root, model_name, repr_name, pooling)


def main() -> None:
    args = parse_args()
    split_csv = Path(args.split_csv)
    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_meta = pd.read_csv(split_csv)
    train_mask = split_meta["split"].to_numpy() == "train"
    test_mask = split_meta["split"].to_numpy() == "test"
    if not train_mask.any() or not test_mask.any():
        raise SystemExit("split_csv must contain non-empty train and test splits.")

    requested_layers = parse_layers(args.layers)
    models = expand_models(args.models)
    if args.max_models > 0:
        models = models[: args.max_models]
    poolings = [p.strip() for p in args.poolings.split(",") if p.strip()]

    rows: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []

    for model_name in models:
        if model_name not in MODEL_DISPLAY:
            raise SystemExit(f"Unknown model name: {model_name}")
        for pooling in poolings:
            token_filter = read_token_filter(config_file(runs_root, model_name, pooling))
            layers = iter_requested_layers(
                runs_root, model_name, args.repr, pooling, requested_layers
            )
            for layer in layers:
                emb_path = embedding_file(runs_root, model_name, args.repr, pooling, layer)
                if not emb_path.exists():
                    skipped.append(
                        {
                            "model": model_name,
                            "pooling": pooling,
                            "layer": layer,
                            "reason": "embedding_missing",
                            "embedding_path": str(emb_path),
                        }
                    )
                    continue

                print(f"[{model_name} {pooling} L{layer}] computing geometry", flush=True)
                emb_all = np.load(emb_path)
                if emb_all.ndim != 2 or emb_all.shape[0] != len(split_meta):
                    skipped.append(
                        {
                            "model": model_name,
                            "pooling": pooling,
                            "layer": layer,
                            "reason": "shape_mismatch",
                            "shape": str(tuple(emb_all.shape)),
                            "n_split_rows": len(split_meta),
                            "embedding_path": str(emb_path),
                        }
                    )
                    continue

                train_raw = emb_all[train_mask]
                test_raw = emb_all[test_mask]
                cleaner = EmbeddingCleaner(num_components=args.D, center=True)
                cleaner.fit(train_raw)
                split_arrays = {
                    "train": (train_raw, cleaner.transform(train_raw)),
                    "test": (test_raw, cleaner.transform(test_raw)),
                }

                for split_name, (raw_x, abtt_x) in split_arrays.items():
                    raw_stats = geometry_stats(raw_x)
                    abtt_stats = geometry_stats(abtt_x)
                    add_delta_columns(raw_stats, abtt_stats)

                    common = {
                        "model": model_name,
                        "model_display": MODEL_DISPLAY[model_name],
                        "repr": args.repr,
                        "pooling": pooling,
                        "layer": int(layer),
                        "split": split_name,
                        "D": int(args.D),
                        "embedding_path": str(emb_path),
                        "split_csv": str(split_csv),
                        "token_filter": token_filter,
                    }
                    rows.append({**common, "view": "raw", **raw_stats})
                    rows.append({**common, "view": f"abtt_d{args.D}", **abtt_stats})

    diag = pd.DataFrame(rows)
    diag_path = out_dir / "geometry_per_layer.csv"
    diag.to_csv(diag_path, index=False)

    skipped_df = pd.DataFrame(skipped, columns=SKIPPED_COLUMNS)
    skipped_path = out_dir / "skipped_inputs.csv"
    skipped_df.to_csv(skipped_path, index=False)

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/resubmit/run_layer_geometry_diagnostics.py",
        "split_csv": str(split_csv),
        "runs_root": str(runs_root),
        "out_dir": str(out_dir),
        "models": models,
        "poolings": poolings,
        "repr": args.repr,
        "layers": requested_layers or "discovered",
        "D": args.D,
        "n_rows": int(len(diag)),
        "n_skipped": int(len(skipped_df)),
        "outputs": {
            "geometry_per_layer": str(diag_path),
            "skipped_inputs": str(skipped_path),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(diag)} diagnostic rows to {diag_path}")
    if skipped:
        print(f"Skipped {len(skipped)} inputs; see {skipped_path}")


if __name__ == "__main__":
    main()
