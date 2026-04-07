"""Generate exact-rank Task B outputs for the human-in-the-loop paper story.

This script reads Phase 11 results, selects the best hidden-state configuration
per model for a single method, reconstructs directory rankings on the test set,
and writes:

1. A per-query top-k CSV with the representative document behind each directory.
2. A model-level rank-distribution CSV with buckets rank1..rankK and other.
3. A paper-friendly table figure (PNG/PDF) summarizing the exact-rank shares.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from canon_retrieval import l2_normalize, similarity_matrix
from sif_abtt import EmbeddingCleaner


SHORT = {
    "bowphs/LaTa": "LaTa",
    "bowphs/PhilTa": "PhilTa",
    "sentence-transformers/LaBSE": "LaBSE",
    "Qwen/Qwen3-Embedding-0.6B": "Qwen3-0.6B",
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5": "KaLM-mini",
    "google/mt5-base": "mt5-base",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 11 Task B human-simulation report.")
    parser.add_argument("--results_csv", required=True, help="Phase 11 results CSV.")
    parser.add_argument("--split_csv", required=True, help="phase9_split.csv")
    parser.add_argument("--runs_root", required=True, help="Repo runs root.")
    parser.add_argument("--out_dir", required=True, help="Output directory.")
    parser.add_argument(
        "--method",
        default="abtt_optimal",
        help="Single method to report in the paper-facing Task B table.",
    )
    parser.add_argument(
        "--repr_name",
        default="hidden",
        help="Representation to report (paper default: hidden).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Top-k directory options to save per query.",
    )
    parser.add_argument(
        "--hidden_mean_subdir",
        default="hidden_mean_tokempty",
        help="Subdirectory containing hidden/mean paper-facing embeddings.",
    )
    parser.add_argument(
        "--hidden_sif_subdir",
        default="hidden_sif_tokempty",
        help="Subdirectory containing hidden/sif paper-facing embeddings.",
    )
    return parser.parse_args()


def model_slug(name: str) -> str:
    return name.replace("/", "_")


def find_embedding_file(
    runs_root: Path,
    slug: str,
    repr_name: str,
    pooling: str,
    layer: int,
    subdir_override: Optional[str] = None,
) -> Path:
    suffix = "" if pooling == "mean" else "_sif"
    fname = f"{repr_name}_layer{layer}_embeddings{suffix}.npy"
    subdir = subdir_override or f"{repr_name}_{pooling}"
    candidate = runs_root / "phase9_bases" / slug / subdir / fname
    if candidate.exists():
        return candidate
    fallback = runs_root / "phase9_bases" / slug / fname
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Embedding file not found for {slug} {repr_name} {pooling} L{layer}")


def apply_postprocess(
    train_emb: np.ndarray,
    test_emb: np.ndarray,
    method: str,
    d_value: int,
) -> tuple[np.ndarray, np.ndarray]:
    if method in {"baseline", "sif_only"}:
        return train_emb, test_emb
    if method in {"abtt_fixed", "abtt_optimal", "sif_abtt_fixed", "sif_abtt_optimal"}:
        cleaner = EmbeddingCleaner(num_components=d_value, center=True)
        cleaner.fit(train_emb)
        return cleaner.transform(train_emb), cleaner.transform(test_emb)
    if method == "whitening":
        pca = PCA(whiten=True)
        pca.fit(train_emb)
        return pca.transform(train_emb), pca.transform(test_emb)
    raise ValueError(f"Unsupported method: {method}")


def build_rankings(
    sim: np.ndarray,
    test_paths: np.ndarray,
    test_folder_ids: np.ndarray,
    test_has_partner: np.ndarray,
    tau: float,
    top_k: int,
) -> list[dict]:
    dir_members: Dict[str, List[int]] = {}
    for idx, folder_id in enumerate(test_folder_ids):
        dir_members.setdefault(str(folder_id), []).append(idx)

    rows: list[dict] = []
    for i in range(sim.shape[0]):
        my_dir = str(test_folder_ids[i])
        correct_label = my_dir if test_has_partner[i] else "__NEW__"

        ranked_entries: list[dict] = []
        for directory, members in dir_members.items():
            best_idx = None
            best_score = None
            for j in members:
                if j == i:
                    continue
                score = float(sim[i, j])
                if best_score is None or score > best_score:
                    best_score = score
                    best_idx = j
            if best_idx is None:
                continue
            ranked_entries.append(
                {
                    "label": directory,
                    "score": best_score,
                    "rep_idx": best_idx,
                    "rep_path": str(test_paths[best_idx]),
                    "rep_folder_id": str(test_folder_ids[best_idx]),
                }
            )

        ranked_entries.append(
            {
                "label": "__NEW__",
                "score": float(tau),
                "rep_idx": -1,
                "rep_path": "",
                "rep_folder_id": "__NEW__",
            }
        )
        ranked_entries.sort(key=lambda item: (-item["score"], item["label"]))

        rank_lookup = {entry["label"]: idx + 1 for idx, entry in enumerate(ranked_entries)}
        correct_rank = rank_lookup[correct_label]
        bucket = f"rank{correct_rank}" if correct_rank <= top_k else "other"

        row = {
            "query_index": i,
            "query_path": str(test_paths[i]),
            "query_folder_id": my_dir,
            "has_partner": bool(test_has_partner[i]),
            "correct_label": correct_label,
            "correct_rank": int(correct_rank),
            "correct_bucket": bucket,
            "tau": float(tau),
        }
        for rank in range(1, top_k + 1):
            if rank <= len(ranked_entries):
                entry = ranked_entries[rank - 1]
                row[f"rank{rank}_label"] = entry["label"]
                row[f"rank{rank}_score"] = float(entry["score"])
                row[f"rank{rank}_rep_path"] = entry["rep_path"]
                row[f"rank{rank}_rep_folder_id"] = entry["rep_folder_id"]
            else:
                row[f"rank{rank}_label"] = ""
                row[f"rank{rank}_score"] = np.nan
                row[f"rank{rank}_rep_path"] = ""
                row[f"rank{rank}_rep_folder_id"] = ""
        rows.append(row)
    return rows


def write_taskb_table(summary_df: pd.DataFrame, out_dir: Path) -> None:
    order = list(SHORT.values())
    cols = ["rank1", "rank2", "rank3", "rank4", "rank5", "other"]
    pivot = (
        summary_df.pivot(index="model_short", columns="bucket", values="pct")
        .reindex(index=order, columns=cols)
        .fillna(0.0)
    )
    display_df = pivot.apply(lambda col: col.map(lambda value: f"{value:.1f}"))
    display_df.insert(0, "Model", display_df.index)
    display_df = display_df.reset_index(drop=True)

    display_df.to_csv(out_dir / "taskb_rank_distribution_table.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.6, 2.8))
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values.tolist(),
        colLabels=list(display_df.columns),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.4)
    for j in range(len(display_df.columns)):
        table[0, j].set_facecolor("#1f4e79")
        table[0, j].set_text_props(color="white", fontweight="bold")

    rank1_values = pivot["rank1"].to_numpy() if "rank1" in pivot.columns else np.array([])
    if len(rank1_values) > 0:
        best_idx = int(np.argmax(rank1_values))
        table[best_idx + 1, 1].set_facecolor("#d4edda")
    fig.tight_layout()
    fig.savefig(out_dir / "taskb_rank_distribution_table.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "taskb_rank_distribution_table.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results = pd.read_csv(args.results_csv)
    split_df = pd.read_csv(args.split_csv)
    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_rows = []
    for model_name in sorted(results["model"].unique()):
        sub = results[
            (results["model"] == model_name)
            & (results["method"] == args.method)
            & (results["repr"] == args.repr_name)
        ]
        if sub.empty:
            continue
        best_idx = sub["train_dir_acc_at_1"].idxmax()
        selected_rows.append(sub.loc[best_idx].to_dict())

    if not selected_rows:
        raise SystemExit("No rows matched the requested Task B selection.")

    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(out_dir / "taskb_selected_configs.csv", index=False)

    test_mask = split_df["split"].values == "test"
    train_mask = split_df["split"].values == "train"
    test_paths = split_df.loc[test_mask, "path"].to_numpy()
    test_folder_ids = split_df.loc[test_mask, "folder_id"].astype(str).to_numpy()
    test_has_partner = split_df.loc[test_mask, "has_test_partner"].astype(bool).to_numpy()

    query_rows = []
    summary_rows = []
    for row in selected_rows:
        model_name = row["model"]
        slug = model_slug(model_name)
        pooling = str(row["pooling"])
        layer = int(row["layer"])
        method = str(row["method"])
        tau = float(row["tau"])
        d_value = int(row["D"])
        hidden_subdir = args.hidden_mean_subdir if pooling == "mean" else args.hidden_sif_subdir
        emb_path = find_embedding_file(
            runs_root,
            slug,
            args.repr_name,
            pooling,
            layer,
            subdir_override=hidden_subdir,
        )
        emb_all = np.load(emb_path)
        train_emb = emb_all[train_mask]
        test_emb = emb_all[test_mask]
        train_emb, test_emb = apply_postprocess(train_emb, test_emb, method, d_value)
        sim = similarity_matrix(l2_normalize(test_emb))

        rows = build_rankings(
            sim=sim,
            test_paths=test_paths,
            test_folder_ids=test_folder_ids,
            test_has_partner=test_has_partner,
            tau=tau,
            top_k=args.top_k,
        )
        for record in rows:
            record["model"] = model_name
            record["model_short"] = SHORT.get(model_name, model_name)
            record["method"] = method
            record["repr"] = args.repr_name
            record["pooling"] = pooling
            record["layer"] = layer
            record["D"] = d_value
        query_rows.extend(rows)

        buckets = [record["correct_bucket"] for record in rows]
        total = len(rows)
        for bucket in [f"rank{k}" for k in range(1, args.top_k + 1)] + ["other"]:
            count = sum(1 for value in buckets if value == bucket)
            summary_rows.append(
                {
                    "model": model_name,
                    "model_short": SHORT.get(model_name, model_name),
                    "method": method,
                    "repr": args.repr_name,
                    "pooling": pooling,
                    "layer": layer,
                    "D": d_value,
                    "bucket": bucket,
                    "count": count,
                    "pct": count / total * 100.0,
                    "n_test": total,
                }
            )

    query_df = pd.DataFrame(query_rows)
    summary_df = pd.DataFrame(summary_rows)
    query_df.to_csv(out_dir / "taskb_top5_predictions.csv", index=False)
    summary_df.to_csv(out_dir / "taskb_rank_distribution.csv", index=False)
    write_taskb_table(summary_df, out_dir)

    print(f"Saved selected configs to {out_dir / 'taskb_selected_configs.csv'}")
    print(f"Saved per-query predictions to {out_dir / 'taskb_top5_predictions.csv'}")
    print(f"Saved exact-rank summary to {out_dir / 'taskb_rank_distribution.csv'}")


if __name__ == "__main__":
    main()
