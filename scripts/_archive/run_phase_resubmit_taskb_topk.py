"""Generate cumulative top-K Task B outputs for the human-in-the-loop paper story.

This script reads Phase 11 results, selects the best hidden-state configuration
per model for a single method, reconstructs directory rankings on the test set,
and writes:

1. A per-query top-k CSV with the representative document behind each directory.
2. A model-level summary CSV with both exclusive and cumulative percentages.
3. A paper-friendly cumulative top-K table figure (PNG/PDF) and LaTeX snippet.
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
    parser = argparse.ArgumentParser(description="Phase 11 Task B cumulative top-K report.")
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
    rect_sim: np.ndarray,
    query_paths: np.ndarray,
    query_folder_ids: np.ndarray,
    query_has_reference: np.ndarray,
    ref_paths: np.ndarray,
    ref_folder_ids: np.ndarray,
    tau: float,
    top_k: int,
) -> list[dict]:
    """Rank reference directories for each query using rectangular similarity.

    rect_sim: (n_query, n_ref) cosine similarities.
    Each reference directory's score = max similarity among its member files.
    """
    # Build reference directory -> column indices mapping
    ref_dir_members: Dict[str, List[int]] = {}
    for j, folder_id in enumerate(ref_folder_ids):
        ref_dir_members.setdefault(str(folder_id), []).append(j)

    rows: list[dict] = []
    for i in range(rect_sim.shape[0]):
        my_dir = str(query_folder_ids[i])
        correct_label = my_dir if query_has_reference[i] else "__NEW__"

        ranked_entries: list[dict] = []
        for directory, members in ref_dir_members.items():
            best_idx = None
            best_score = None
            for j in members:
                score = float(rect_sim[i, j])
                if best_score is None or score > best_score:
                    best_score = score
                    best_idx = j
            ranked_entries.append(
                {
                    "label": directory,
                    "score": best_score,
                    "rep_idx": best_idx,
                    "rep_path": str(ref_paths[best_idx]),
                    "rep_folder_id": str(ref_folder_ids[best_idx]),
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
            "query_path": str(query_paths[i]),
            "query_folder_id": my_dir,
            "has_reference_dir": bool(query_has_reference[i]),
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


def write_topk_table(summary_df: pd.DataFrame, out_dir: Path, top_k: int) -> None:
    order = list(SHORT.values())
    exclusive_cols = [f"rank{k}" for k in range(1, top_k + 1)] + ["other"]

    # Pivot exclusive percentages
    pivot_excl = (
        summary_df.pivot(index="model_short", columns="bucket", values="exclusive_pct")
        .reindex(index=order, columns=exclusive_cols)
        .fillna(0.0)
    )

    # Compute cumulative top-K from exclusive rank columns
    rank_cols = [f"rank{k}" for k in range(1, top_k + 1)]
    cumulative = pivot_excl[rank_cols].cumsum(axis=1)
    cumulative.columns = [f"Top-{k}" for k in range(1, top_k + 1)]

    # --- CSV output ---
    csv_df = cumulative.copy()
    csv_df.insert(0, "Model", csv_df.index)
    csv_df = csv_df.reset_index(drop=True)
    csv_df.to_csv(out_dir / "taskb_topk_table.csv", index=False)

    # --- Display table ---
    display_df = cumulative.apply(lambda col: col.map(lambda v: f"{v:.1f}"))
    display_df.insert(0, "Model", display_df.index)
    display_df = display_df.reset_index(drop=True)

    # --- PNG/PDF figure ---
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
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

    # Highlight best Top-1
    top1_values = cumulative["Top-1"].to_numpy()
    if len(top1_values) > 0:
        best_idx = int(np.argmax(top1_values))
        table[best_idx + 1, 1].set_facecolor("#d4edda")

    fig.tight_layout()
    fig.savefig(out_dir / "taskb_topk_table.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "taskb_topk_table.pdf", bbox_inches="tight")
    plt.close(fig)

    # --- LaTeX snippet ---
    write_latex_table(cumulative, out_dir, top_k)


def write_latex_table(cumulative: pd.DataFrame, out_dir: Path, top_k: int) -> None:
    col_spec = "l" + "r" * top_k
    header_cells = " & ".join(f"\\textbf{{Top-{k}}}" for k in range(1, top_k + 1))
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        f"\\textbf{{Model}} & {header_cells} \\\\",
        "\\midrule",
    ]
    for model_name in cumulative.index:
        vals = " & ".join(f"{cumulative.loc[model_name, f'Top-{k}']:.1f}" for k in range(1, top_k + 1))
        lines.append(f"{model_name} & {vals} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    (out_dir / "taskb_topk_table.tex").write_text("\n".join(lines) + "\n")


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

    train_mask = split_df["split"].values == "train"

    # Detect query/reference paradigm (new) vs legacy
    if "taskb_role" not in split_df.columns:
        raise SystemExit(
            "Split CSV missing 'taskb_role' column. "
            "Re-run run_resubmit_data_prep.py to generate the updated split."
        )

    query_mask_full = split_df["taskb_role"].values == "query"
    ref_mask_full = split_df["taskb_role"].values == "reference"
    query_paths = split_df.loc[query_mask_full, "path"].to_numpy()
    query_folder_ids = split_df.loc[query_mask_full, "folder_id"].astype(str).to_numpy()
    query_has_reference = split_df.loc[query_mask_full, "has_reference_dir"].astype(bool).to_numpy()
    ref_paths = split_df.loc[ref_mask_full, "path"].to_numpy()
    ref_folder_ids = split_df.loc[ref_mask_full, "folder_id"].astype(str).to_numpy()

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
        # Concatenate query + ref for post-processing (fit on train, transform both)
        query_emb = emb_all[query_mask_full]
        ref_emb = emb_all[ref_mask_full]
        combined_test = np.concatenate([query_emb, ref_emb], axis=0)
        train_emb, combined_pp = apply_postprocess(train_emb, combined_test, method, d_value)
        n_q = len(query_emb)
        query_emb_pp = combined_pp[:n_q]
        ref_emb_pp = combined_pp[n_q:]
        rect_sim = l2_normalize(query_emb_pp) @ l2_normalize(ref_emb_pp).T

        rows = build_rankings(
            rect_sim=rect_sim,
            query_paths=query_paths,
            query_folder_ids=query_folder_ids,
            query_has_reference=query_has_reference,
            ref_paths=ref_paths,
            ref_folder_ids=ref_folder_ids,
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

        # Compute exclusive counts per bucket
        buckets = [record["correct_bucket"] for record in rows]
        total = len(rows)

        # Build exclusive percentages first, then cumulative
        exclusive_pcts = {}
        for bucket in [f"rank{k}" for k in range(1, args.top_k + 1)] + ["other"]:
            count = sum(1 for value in buckets if value == bucket)
            exclusive_pcts[bucket] = count / total * 100.0

        # Cumulative: top-K = sum of rank1..rankK
        cumulative_pcts = {}
        running = 0.0
        for k in range(1, args.top_k + 1):
            running += exclusive_pcts[f"rank{k}"]
            cumulative_pcts[f"rank{k}"] = running
        cumulative_pcts["other"] = 100.0  # by definition

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
                    "exclusive_pct": exclusive_pcts[bucket],
                    "cumulative_pct": cumulative_pcts[bucket],
                    "n_query": total,
                }
            )

    query_df = pd.DataFrame(query_rows)
    summary_df = pd.DataFrame(summary_rows)
    query_df.to_csv(out_dir / "taskb_top5_predictions.csv", index=False)
    summary_df.to_csv(out_dir / "taskb_rank_distribution.csv", index=False)
    write_topk_table(summary_df, out_dir, args.top_k)

    print(f"Saved selected configs to {out_dir / 'taskb_selected_configs.csv'}")
    print(f"Saved per-query predictions to {out_dir / 'taskb_top5_predictions.csv'}")
    print(f"Saved rank summary to {out_dir / 'taskb_rank_distribution.csv'}")
    print(f"Saved cumulative top-K table to {out_dir / 'taskb_topk_table.csv'}")
    print(f"Saved LaTeX snippet to {out_dir / 'taskb_topk_table.tex'}")


if __name__ == "__main__":
    main()
