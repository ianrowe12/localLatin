"""Render Phase 12f pair-level IG example figures."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer


SHORT = {
    "bowphs/LaTa": "LaTa",
    "bowphs/PhilTa": "PhilTa",
}

BUCKET_LABELS = {
    "correct_similar": "Correct prediction | Similar pair",
    "correct_not_similar": "Correct prediction | Not similar pair",
    "wrong_similar": "Wrong prediction | Similar pair",
    "wrong_not_similar": "Wrong prediction | Not similar pair",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Phase 12f pair IG artifacts.")
    parser.add_argument("--examples_csv", required=True, help="CSV from run_phase12f_select_pair_examples.py")
    parser.add_argument("--artifacts_dir", required=True, help="artifacts/ root from run_phase12e_pair_explanations.py")
    parser.add_argument("--fig_dir", required=True, help="Output figure directory")
    parser.add_argument("--max_tokens", type=int, default=18, help="Maximum query/document tokens to display")
    return parser.parse_args()


def model_slug(name: str) -> str:
    return name.replace("/", "_")


def load_artifact(artifacts_dir: Path, row: pd.Series) -> dict[str, np.ndarray]:
    path = artifacts_dir / model_slug(row["model_name"]) / f"example{int(row['example_id']):03d}_{row['candidate_role']}.npz"
    return dict(np.load(path))


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-12)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-12)
    return a_norm @ b_norm.T


def clean_tokens(hidden: np.ndarray, pcs: np.ndarray, mean_vec: np.ndarray) -> np.ndarray:
    centered = hidden - mean_vec
    proj = centered @ pcs.T @ pcs
    return centered - proj


def select_positions(ig_base: np.ndarray, ig_abtt: np.ndarray, seq_len: int, max_tokens: int) -> np.ndarray:
    importance = np.maximum(np.abs(ig_base[:seq_len]), np.abs(ig_abtt[:seq_len]))
    top_idx = np.argsort(importance)[-max_tokens:]
    return np.sort(top_idx)


def decode_tokens(tokenizer, input_ids: np.ndarray, positions: np.ndarray) -> list[str]:
    ids = input_ids[0, positions].tolist()
    return [tokenizer.decode([token_id]) for token_id in ids]


def build_pair_matrix(
    query_hidden: np.ndarray,
    candidate_hidden: np.ndarray,
    query_ig: np.ndarray,
    candidate_ig: np.ndarray,
) -> np.ndarray:
    cos = cosine_matrix(query_hidden, candidate_hidden)
    weight = np.sqrt(np.abs(query_ig)[:, None] * np.abs(candidate_ig)[None, :])
    sign = np.sign(query_ig)[:, None] * np.sign(candidate_ig)[None, :]
    return cos * weight * sign


def label_text(value: int) -> str:
    return "similar" if int(value) == 1 else "not similar"


def output_name(row: pd.Series) -> str:
    model_short = row.get("model_short", SHORT.get(row["model_name"], model_slug(row["model_name"])))
    return f"{row['bucket']}__{model_short}__{int(row['bucket_slot']):03d}.png"


def render_pair_heatmaps(
    row: pd.Series,
    data: dict[str, np.ndarray],
    tokenizer,
    out_path: Path,
    max_tokens: int,
) -> None:
    q_len = int(data["query_attention_mask"][0].sum())
    c_len = int(data["candidate_attention_mask"][0].sum())
    q_pos = select_positions(data["query_ig_baseline"], data["query_ig_abtt"], q_len, max_tokens)
    c_pos = select_positions(
        data["candidate_ig_baseline"],
        data["candidate_ig_abtt"],
        c_len,
        max_tokens,
    )

    q_tokens = decode_tokens(tokenizer, data["query_input_ids"], q_pos)
    c_tokens = decode_tokens(tokenizer, data["candidate_input_ids"], c_pos)

    q_hidden = data["query_hidden"][q_pos]
    c_hidden = data["candidate_hidden"][c_pos]
    q_hidden_abtt = clean_tokens(data["query_hidden"], data["pcs"], data["mean_vec"])[q_pos]
    c_hidden_abtt = clean_tokens(data["candidate_hidden"], data["pcs"], data["mean_vec"])[c_pos]

    q_ig_base = data["query_ig_baseline"][q_pos]
    q_ig_abtt = data["query_ig_abtt"][q_pos]
    c_ig_base = data["candidate_ig_baseline"][c_pos]
    c_ig_abtt = data["candidate_ig_abtt"][c_pos]

    baseline_matrix = build_pair_matrix(q_hidden, c_hidden, q_ig_base, c_ig_base)
    abtt_matrix = build_pair_matrix(q_hidden_abtt, c_hidden_abtt, q_ig_abtt, c_ig_abtt)
    vmax = np.percentile(np.abs(np.concatenate([baseline_matrix.ravel(), abtt_matrix.ravel()])), 95)
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, matrix, title in zip(
        axes,
        [baseline_matrix, abtt_matrix],
        ["Baseline", "ABTT"],
    ):
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(c_tokens)))
        ax.set_xticklabels(c_tokens, rotation=90, fontsize=7, fontfamily="monospace")
        ax.set_yticks(range(len(q_tokens)))
        ax.set_yticklabels(q_tokens, fontsize=7, fontfamily="monospace")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Document")
        ax.set_ylabel("Query")

    gold_text = label_text(int(row["gold_similar"]))
    baseline_text = label_text(int(row["baseline_pred"]))
    abtt_text = label_text(int(row["abtt_pred"]))
    model_short = row.get("model_short", SHORT.get(row["model_name"], row["model_name"]))
    bucket_label = BUCKET_LABELS.get(str(row["bucket"]), str(row["bucket"]).replace("_", " "))
    fig.suptitle(
        f"{model_short} | {bucket_label}\n"
        f"Gold: {gold_text} | ABTT: {abtt_text} | Baseline: {baseline_text}",
        fontweight="bold",
        fontsize=12,
    )
    fig.colorbar(im, ax=axes.tolist(), shrink=0.8)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    if out_path.suffix.lower() == ".png":
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    examples = pd.read_csv(args.examples_csv)
    artifacts_dir = Path(args.artifacts_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    tokenizers = {}
    manifest_rows = []
    for _, row in examples.iterrows():
        model_name = row["model_name"]
        if model_name not in tokenizers:
            tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
        data = load_artifact(artifacts_dir, row)
        out_path = fig_dir / output_name(row)
        render_pair_heatmaps(row, data, tokenizers[model_name], out_path, args.max_tokens)
        manifest_rows.append(
            {
                "example_id": int(row["example_id"]),
                "bucket": row["bucket"],
                "bucket_slot": int(row["bucket_slot"]),
                "model_name": model_name,
                "model_short": row.get("model_short", SHORT.get(model_name, model_name)),
                "gold_similar": int(row["gold_similar"]),
                "baseline_pred": int(row["baseline_pred"]),
                "abtt_pred": int(row["abtt_pred"]),
                "baseline_score": float(row["baseline_score"]),
                "abtt_score": float(row["abtt_score"]),
                "query_path": row["query_path"],
                "candidate_path": row["candidate_path"],
                "figure_path_png": str(out_path),
                "figure_path_pdf": str(out_path.with_suffix(".pdf")),
            }
        )

    manifest_df = pd.DataFrame(manifest_rows).sort_values(
        by=["bucket", "model_short", "bucket_slot"]
    )
    manifest_df.to_csv(fig_dir / "phase12f_manifest.csv", index=False)
    print(f"Saved {len(manifest_df)} Phase 12f figures to {fig_dir}")


if __name__ == "__main__":
    main()
