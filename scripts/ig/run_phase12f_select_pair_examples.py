"""Select pair-level IG examples for four professor-review buckets.

This script builds a balanced set of pair examples from the Phase 9 test split,
using ABTT pair classification as the correctness basis. Each selected pair is
assigned to one of four buckets:

1. correct_similar: gold similar, ABTT predicts similar
2. correct_not_similar: gold not similar, ABTT predicts not similar
3. wrong_similar: gold similar, ABTT predicts not similar
4. wrong_not_similar: gold not similar, ABTT predicts similar

The output CSV is designed to plug directly into
scripts/run_phase12e_pair_explanations.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from canon_retrieval import l2_normalize, similarity_matrix
from sif_abtt import EmbeddingCleaner


FEATURED_MODELS = {
    "bowphs/LaTa": {
        "model_type": "t5",
        "layer": 4,
        "model_short": "LaTa",
    },
    "bowphs/PhilTa": {
        "model_type": "t5",
        "layer": 6,
        "model_short": "PhilTa",
    },
    "sentence-transformers/LaBSE": {
        "model_type": "bert",
        "layer": 12,
        "model_short": "LaBSE",
    },
    "Qwen/Qwen3-Embedding-0.6B": {
        "model_type": "decoder",
        "layer": 23,
        "model_short": "Qwen3-0.6B",
    },
}

BUCKET_ORDER = [
    "correct_similar",
    "correct_not_similar",
    "wrong_similar",
    "wrong_not_similar",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select Phase 12f pair-level IG examples.")
    parser.add_argument(
        "--results_csv",
        default="runs/phase11_filtered/phase11_results.csv",
        help="Phase 11 results CSV. Falls back to runs/phase11/phase11_results.csv when missing.",
    )
    parser.add_argument(
        "--split_csv",
        default="runs/phase9/phase9_split.csv",
        help="Phase 9 train/test split CSV.",
    )
    parser.add_argument(
        "--runs_root",
        default="runs",
        help="Repo runs root containing phase9_bases/ embeddings.",
    )
    parser.add_argument(
        "--out_csv",
        default="runs/active/ig_examples/phase12f_examples.csv",
        help="Output CSV for Phase 12f examples.",
    )
    parser.add_argument(
        "--examples_per_bucket_per_model",
        type=int,
        default=5,
        help="How many examples to keep for each bucket for each featured model.",
    )
    parser.add_argument(
        "--hidden_mean_subdir",
        default="hidden_mean_tokempty",
        help="Preferred hidden/mean embedding subdirectory.",
    )
    parser.add_argument(
        "--fallback_hidden_mean_subdir",
        default="hidden_mean",
        help="Fallback hidden/mean embedding subdirectory.",
    )
    parser.add_argument(
        "--bases_root",
        default=None,
        help="Embedding cache root holding <slug>/<subdir>/hidden_layer{N}_embeddings.npy. "
        "Defaults to <runs_root>/phase9_bases; pass runs/active/encoder_bases for the "
        "canonical cache.",
    )
    return parser.parse_args()


def resolve_results_csv(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path
    fallback = Path("runs/phase11/phase11_results.csv")
    if path == Path("runs/phase11_filtered/phase11_results.csv") and fallback.exists():
        return fallback
    raise FileNotFoundError(f"Results CSV not found: {path}")


def model_slug(name: str) -> str:
    return name.replace("/", "_")


def find_embedding_file(
    runs_root: Path,
    slug: str,
    layer: int,
    preferred_subdir: str,
    fallback_subdir: str,
    bases_root: Path | None = None,
) -> Path:
    """Locate the pooled hidden-state cache for ``slug`` at ``layer``.

    ``bases_root`` defaults to the legacy ``<runs_root>/phase9_bases`` layout.
    The canonical cache now lives at ``runs/active/encoder_bases``, so callers
    can point at it explicitly without rewriting the historical default.
    """
    roots = [bases_root] if bases_root is not None else [runs_root / "phase9_bases"]
    fname = f"hidden_layer{layer}_embeddings.npy"
    tried: list[Path] = []
    for root in roots:
        for subdir in (preferred_subdir, fallback_subdir):
            candidate = root / slug / subdir / fname
            tried.append(candidate)
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"Embedding file not found for {slug} layer {layer}: "
        + " or ".join(str(p) for p in tried)
    )


def load_config_row(
    results: pd.DataFrame,
    model_name: str,
    layer: int,
    method: str,
) -> pd.Series:
    sub = results[
        (results["model"] == model_name)
        & (results["repr"] == "hidden")
        & (results["pooling"] == "mean")
        & (results["layer"] == layer)
        & (results["method"] == method)
    ].copy()
    if sub.empty:
        raise ValueError(
            f"Missing Phase 11 config for model={model_name}, layer={layer}, method={method}"
        )
    sub = sub.sort_values(
        by=["train_aucroc", "aucroc", "dir_acc_at_1"],
        ascending=False,
    )
    return sub.iloc[0]


def bucket_name(gold_similar: np.ndarray, abtt_pred: np.ndarray) -> np.ndarray:
    out = np.empty(len(gold_similar), dtype=object)
    out[(gold_similar == 1) & (abtt_pred == 1)] = "correct_similar"
    out[(gold_similar == 0) & (abtt_pred == 0)] = "correct_not_similar"
    out[(gold_similar == 1) & (abtt_pred == 0)] = "wrong_similar"
    out[(gold_similar == 0) & (abtt_pred == 1)] = "wrong_not_similar"
    return out


def select_diverse_examples(
    bucket_df: pd.DataFrame,
    n_keep: int,
) -> pd.DataFrame:
    sorted_df = bucket_df.sort_values(
        by=[
            "baseline_disagrees",
            "abtt_margin_abs",
            "baseline_margin_abs",
            "query_path",
            "candidate_path",
        ],
        ascending=[False, False, False, True, True],
    )

    selected_idx: list[int] = []
    used_queries: set[str] = set()
    used_paths: set[str] = set()

    def acceptable(row: pd.Series, level: int) -> bool:
        if level == 0:
            return row["query_path"] not in used_paths and row["candidate_path"] not in used_paths
        if level == 1:
            return row["query_path"] not in used_queries
        return True

    for level in range(3):
        for idx, row in sorted_df.iterrows():
            if idx in selected_idx:
                continue
            if not acceptable(row, level):
                continue
            selected_idx.append(idx)
            used_queries.add(str(row["query_path"]))
            used_paths.add(str(row["query_path"]))
            used_paths.add(str(row["candidate_path"]))
            if len(selected_idx) >= n_keep:
                break
        if len(selected_idx) >= n_keep:
            break

    if len(selected_idx) < n_keep:
        raise ValueError(
            f"Only found {len(selected_idx)} examples for bucket "
            f"{bucket_df['bucket'].iloc[0]} in model {bucket_df['model_name'].iloc[0]}"
        )

    selected = sorted_df.loc[selected_idx].copy()
    selected["bucket_slot"] = np.arange(1, len(selected) + 1, dtype=np.int32)
    return selected


def build_pair_frame(
    model_name: str,
    cfg: dict[str, object],
    baseline_row: pd.Series,
    abtt_row: pd.Series,
    split_df: pd.DataFrame,
    runs_root: Path,
    preferred_subdir: str,
    fallback_subdir: str,
    bases_root: Path | None = None,
) -> pd.DataFrame:
    slug = model_slug(model_name)
    emb_path = find_embedding_file(
        runs_root,
        slug,
        int(cfg["layer"]),
        preferred_subdir,
        fallback_subdir,
        bases_root=bases_root,
    )
    emb_all = np.load(emb_path)

    train_mask = split_df["split"].to_numpy() == "train"
    test_mask = split_df["split"].to_numpy() == "test"
    test_meta = split_df.loc[test_mask].reset_index(drop=True)

    train_emb = emb_all[train_mask]
    test_emb = emb_all[test_mask]
    baseline_tau = float(baseline_row["tau"])
    abtt_tau = float(abtt_row["tau"])
    abtt_d = int(abtt_row["D"])

    cleaner = EmbeddingCleaner(num_components=abtt_d, center=True)
    cleaner.fit(train_emb)
    test_emb_abtt = cleaner.transform(test_emb)

    baseline_sim = similarity_matrix(l2_normalize(test_emb))
    abtt_sim = similarity_matrix(l2_normalize(test_emb_abtt))
    pair_i, pair_j = np.triu_indices(len(test_meta), k=1)

    query_meta = test_meta.iloc[pair_i].reset_index(drop=True)
    candidate_meta = test_meta.iloc[pair_j].reset_index(drop=True)
    gold_similar = (
        query_meta["folder_id"].to_numpy() == candidate_meta["folder_id"].to_numpy()
    ).astype(np.int8)

    baseline_scores = baseline_sim[pair_i, pair_j]
    abtt_scores = abtt_sim[pair_i, pair_j]
    baseline_pred = (baseline_scores >= baseline_tau).astype(np.int8)
    abtt_pred = (abtt_scores >= abtt_tau).astype(np.int8)

    df = pd.DataFrame(
        {
            "model_name": model_name,
            "model_short": str(cfg["model_short"]),
            "model_type": str(cfg["model_type"]),
            "layer": int(cfg["layer"]),
            "method": "abtt_optimal",
            "repr": "hidden",
            "pooling": "mean",
            "D": abtt_d,
            "tau": abtt_tau,
            "baseline_tau": baseline_tau,
            "abtt_tau": abtt_tau,
            "query_index": pair_i.astype(np.int32),
            "candidate_index": pair_j.astype(np.int32),
            "query_file_id": query_meta["file_id"].to_numpy(dtype=np.int32),
            "candidate_file_id": candidate_meta["file_id"].to_numpy(dtype=np.int32),
            "query_path": query_meta["path"].astype(str).to_numpy(),
            "candidate_path": candidate_meta["path"].astype(str).to_numpy(),
            "query_folder_id": query_meta["folder_id"].astype(str).to_numpy(),
            "candidate_folder_id": candidate_meta["folder_id"].astype(str).to_numpy(),
            "candidate_label": candidate_meta["folder_id"].astype(str).to_numpy(),
            "candidate_role": "pair_example",
            "gold_similar": gold_similar,
            "baseline_score": baseline_scores.astype(np.float32),
            "abtt_score": abtt_scores.astype(np.float32),
            "baseline_pred": baseline_pred,
            "abtt_pred": abtt_pred,
        }
    )
    df["bucket"] = bucket_name(df["gold_similar"].to_numpy(), df["abtt_pred"].to_numpy())
    df["baseline_disagrees"] = (df["baseline_pred"] != df["abtt_pred"]).astype(np.int8)
    df["baseline_margin"] = df["baseline_score"] - df["baseline_tau"]
    df["abtt_margin"] = df["abtt_score"] - df["abtt_tau"]
    df["baseline_margin_abs"] = df["baseline_margin"].abs()
    df["abtt_margin_abs"] = df["abtt_margin"].abs()
    return df


def main() -> None:
    args = parse_args()
    results_path = resolve_results_csv(args.results_csv)
    results = pd.read_csv(results_path)
    split_df = pd.read_csv(args.split_csv)
    runs_root = Path(args.runs_root)

    selected_frames: list[pd.DataFrame] = []
    for model_name, cfg in FEATURED_MODELS.items():
        layer = int(cfg["layer"])
        baseline_row = load_config_row(results, model_name, layer, "baseline")
        abtt_row = load_config_row(results, model_name, layer, "abtt_optimal")

        pair_df = build_pair_frame(
            model_name,
            cfg,
            baseline_row,
            abtt_row,
            split_df,
            runs_root,
            args.hidden_mean_subdir,
            args.fallback_hidden_mean_subdir,
            bases_root=Path(args.bases_root) if args.bases_root else None,
        )

        for bucket in BUCKET_ORDER:
            bucket_df = pair_df[pair_df["bucket"] == bucket].copy()
            if bucket_df.empty:
                raise SystemExit(f"No examples found for {model_name} / {bucket}")
            selected = select_diverse_examples(
                bucket_df,
                args.examples_per_bucket_per_model,
            )
            selected_frames.append(selected)

    out_df = pd.concat(selected_frames, ignore_index=True)
    out_df = out_df.sort_values(
        by=["bucket", "model_short", "bucket_slot", "query_path", "candidate_path"],
        key=lambda col: col.map({b: i for i, b in enumerate(BUCKET_ORDER)}) if col.name == "bucket" else col,
    ).reset_index(drop=True)
    out_df.insert(0, "example_id", np.arange(1, len(out_df) + 1, dtype=np.int32))

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    summary = (
        out_df.groupby(["bucket", "model_short"]).size().rename("n_examples").reset_index()
    )
    print(f"Saved {len(out_df)} Phase 12f examples to {out_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
