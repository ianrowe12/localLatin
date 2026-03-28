"""Compare IG-based pair matrices against token-alignment methods.

Produces side-by-side 4-panel comparison figures for selected example pairs:
  Panel A: Current approach (ABTT + IG pair matrix)
  Panel B: BERTScore-style greedy alignment on ABTT-cleaned token embeddings
  Panel C: Optimal Transport (EMD) with IG-derived mass on ABTT-cleaned tokens
  Panel D: Attention-weighted cross-similarity (Ditto-inspired)

All methods operate on pre-computed artifacts from Phase 12e (hidden states,
attention matrices, IG scores, PCs). No model re-run is needed.

Research basis (from Run 1 agents):
  - BERTScore greedy alignment: run1_geometry_alignment.md Area 2 (Priority 1)
    "The alignment matrix IS the explanation" -- no IG needed.
  - Optimal Transport (EMD): run1_nlp_interpretability.md Section 2.2
    Transport plan T[i,j] directly shows token-to-token mass flow.
  - Attention-weighted cross-sim: run1_geometry_alignment.md Area 1 (Ditto)
    Per-instance attention diagonal as pooling weights, no corpus-level stats.

Usage:
    python scripts/run_resubmit_ig_comparison.py \\
        --examples_csv runs/phase12e_release/phase12e_examples.csv \\
        --artifacts_dir runs/phase12e_release/artifacts \\
        --out_dir runs/phase_resubmit/ig_comparison \\
        --max_tokens 20
"""
from __future__ import annotations

import argparse
import string
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

try:
    import ot
except ImportError:
    raise ImportError("POT is required: pip install POT")


# ---------------------------------------------------------------------------
# Utility helpers (aligned with run_phase12e_visualize.py)
# ---------------------------------------------------------------------------

def model_slug(name: str) -> str:
    return name.replace("/", "_")


def load_artifact(artifacts_dir: Path, row: pd.Series) -> dict[str, np.ndarray]:
    path = (
        artifacts_dir
        / model_slug(row["model_name"])
        / f"example{int(row['example_id']):03d}_{row['candidate_role']}.npz"
    )
    return dict(np.load(path))


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity matrix between a (N,D) and b (M,D)."""
    a_n = a / np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-12)
    b_n = b / np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-12)
    return a_n @ b_n.T


def clean_tokens(hidden: np.ndarray, pcs: np.ndarray, mean_vec: np.ndarray) -> np.ndarray:
    """Per-token ABTT: center and project out top-D PCs."""
    centered = hidden - mean_vec
    proj = centered @ pcs.T @ pcs
    return centered - proj


def normalize_token(tok: str) -> str:
    return tok.strip().replace("\u2581", "").replace("\u0120", "").replace("</s>", "")


def is_empty_token(tok: str) -> bool:
    return normalize_token(tok) == ""


def is_content_token(tok: str) -> bool:
    text = normalize_token(tok)
    return bool(text) and not all(ch in string.punctuation for ch in text)


def select_positions(
    ig_base: np.ndarray, ig_abtt: np.ndarray, seq_len: int, max_tokens: int
) -> np.ndarray:
    importance = np.maximum(np.abs(ig_base[:seq_len]), np.abs(ig_abtt[:seq_len]))
    top_idx = np.argsort(importance)[-max_tokens:]
    return np.sort(top_idx)


def decode_tokens(tokenizer, input_ids: np.ndarray, positions: np.ndarray) -> list[str]:
    ids = input_ids[0, positions].tolist()
    return [tokenizer.decode([tid]) for tid in ids]


def pretty_role(role: str) -> str:
    if role == "correct_option":
        return "Same-source"
    if role == "predicted_directory":
        return "Predicted (wrong)"
    return role.replace("_", " ")


# ---------------------------------------------------------------------------
# Method A: Current IG + ABTT pair matrix (baseline from Phase 12e)
# ---------------------------------------------------------------------------

def build_ig_pair_matrix(
    q_hidden: np.ndarray,
    c_hidden: np.ndarray,
    q_ig: np.ndarray,
    c_ig: np.ndarray,
) -> np.ndarray:
    """cos(q_tok, c_tok) * sqrt(|IG_q| * |IG_c|) * sign(IG_q) * sign(IG_c)."""
    cos = cosine_matrix(q_hidden, c_hidden)
    weight = np.sqrt(np.abs(q_ig)[:, None] * np.abs(c_ig)[None, :])
    sign = np.sign(q_ig)[:, None] * np.sign(c_ig)[None, :]
    return cos * weight * sign


# ---------------------------------------------------------------------------
# Method B: BERTScore greedy alignment on ABTT-cleaned token embeddings
# ---------------------------------------------------------------------------

def build_bertscore_matrix(
    q_hidden_clean: np.ndarray,
    c_hidden_clean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """BERTScore-style greedy alignment.

    Returns:
        cos_matrix: dense cosine similarity (n_q, n_c)
        combined_alignment: union of recall + precision greedy matches
    """
    cos = cosine_matrix(q_hidden_clean, c_hidden_clean)
    nq, nc = cos.shape
    combined = np.zeros_like(cos)
    # Recall: for each query token, best matching candidate
    for i in range(nq):
        j = np.argmax(cos[i])
        combined[i, j] = max(combined[i, j], cos[i, j])
    # Precision: for each candidate token, best matching query
    for j in range(nc):
        i = np.argmax(cos[:, j])
        combined[i, j] = max(combined[i, j], cos[i, j])
    return cos, combined


# ---------------------------------------------------------------------------
# Method C: Optimal Transport (EMD) with IG mass on ABTT-cleaned tokens
# ---------------------------------------------------------------------------

def build_ot_transport_plan(
    q_hidden_clean: np.ndarray,
    c_hidden_clean: np.ndarray,
    q_ig: np.ndarray,
    c_ig: np.ndarray,
    use_uniform: bool = False,
) -> np.ndarray:
    """EMD transport plan. Cost = 1 - cosine. Mass from |IG| scores."""
    cos = cosine_matrix(q_hidden_clean, c_hidden_clean)
    C = np.clip(1.0 - cos, 0, None).astype(np.float64)

    n_q, n_c = C.shape
    if use_uniform:
        a = np.ones(n_q, dtype=np.float64) / n_q
        b = np.ones(n_c, dtype=np.float64) / n_c
    else:
        eps = 1e-8
        a = np.abs(q_ig).astype(np.float64) + eps
        b = np.abs(c_ig).astype(np.float64) + eps
        a /= a.sum()
        b /= b.sum()

    T = ot.emd(a, b, C)
    return T


# ---------------------------------------------------------------------------
# Method D: Attention-diagonal weighted cross-similarity (Ditto-inspired)
# ---------------------------------------------------------------------------

def build_attention_crosssim(
    q_hidden_clean: np.ndarray,
    c_hidden_clean: np.ndarray,
    q_attention: np.ndarray,
    c_attention: np.ndarray,
) -> np.ndarray:
    """Cross-similarity weighted by self-attention diagonal.

    Self-attention diagonal = how much each token attends to itself.
    This acts as per-instance token importance without corpus-level stats.
    """
    cos = cosine_matrix(q_hidden_clean, c_hidden_clean)
    q_diag = np.diag(q_attention).clip(min=0)
    c_diag = np.diag(c_attention).clip(min=0)
    q_w = q_diag / (q_diag.sum() + 1e-12)
    c_w = c_diag / (c_diag.sum() + 1e-12)
    weight = np.sqrt(q_w[:, None] * c_w[None, :])
    return cos * weight


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def sparsity_ratio(m: np.ndarray, threshold_frac: float = 0.05) -> float:
    """Fraction of entries above threshold_frac * max(|m|)."""
    if np.abs(m).max() == 0:
        return 0.0
    return float((np.abs(m) > threshold_frac * np.abs(m).max()).sum()) / m.size


def content_focus(m: np.ndarray, q_tok: list[str], c_tok: list[str]) -> float:
    """Fraction of total |weight| on content-token pairs."""
    q_c = np.array([is_content_token(t) for t in q_tok])
    c_c = np.array([is_content_token(t) for t in c_tok])
    mask = q_c[:, None] & c_c[None, :]
    total = np.abs(m).sum()
    if total == 0:
        return 0.0
    return float(np.abs(m[mask]).sum() / total)


def shared_token_score(m: np.ndarray, q_tok: list[str], c_tok: list[str]) -> float:
    """Of total |weight|, fraction on cells where query and candidate tokens match."""
    q_n = [normalize_token(t).lower() for t in q_tok]
    c_n = [normalize_token(t).lower() for t in c_tok]
    shared = 0.0
    total = np.abs(m).sum()
    if total == 0:
        return 0.0
    for i, qt in enumerate(q_n):
        if not qt:
            continue
        for j, ct in enumerate(c_n):
            if qt == ct:
                shared += np.abs(m[i, j])
    return shared / total


# ---------------------------------------------------------------------------
# Visualization: 4-panel comparison
# ---------------------------------------------------------------------------

def render_comparison(
    row: pd.Series,
    data: dict[str, np.ndarray],
    tokenizer,
    out_path: Path,
    max_tokens: int,
) -> dict:
    """Render 4-panel comparison figure and return summary metrics."""
    q_len = int(data["query_attention_mask"][0].sum())
    c_len = int(data["candidate_attention_mask"][0].sum())

    q_pos = select_positions(
        data["query_ig_baseline"], data["query_ig_abtt"], q_len, max_tokens
    )
    c_pos = select_positions(
        data["candidate_ig_baseline"], data["candidate_ig_abtt"], c_len, max_tokens
    )

    q_tokens = decode_tokens(tokenizer, data["query_input_ids"], q_pos)
    c_tokens = decode_tokens(tokenizer, data["candidate_input_ids"], c_pos)

    q_hidden = data["query_hidden"][q_pos]
    c_hidden = data["candidate_hidden"][c_pos]

    # Per-token ABTT cleaning
    q_hc = clean_tokens(data["query_hidden"], data["pcs"], data["mean_vec"])[q_pos]
    c_hc = clean_tokens(data["candidate_hidden"], data["pcs"], data["mean_vec"])[c_pos]

    q_ig = data["query_ig_abtt"][q_pos]
    c_ig = data["candidate_ig_abtt"][c_pos]

    q_attn = data["query_attention"][np.ix_(q_pos, q_pos)]
    c_attn = data["candidate_attention"][np.ix_(c_pos, c_pos)]

    # --- Build panels ---
    panel_a = build_ig_pair_matrix(q_hc, c_hc, q_ig, c_ig)
    _, panel_b = build_bertscore_matrix(q_hc, c_hc)
    panel_c = build_ot_transport_plan(q_hc, c_hc, q_ig, c_ig)
    panel_d = build_attention_crosssim(q_hc, c_hc, q_attn, c_attn)

    # --- Render figure ---
    fig = plt.figure(figsize=(22, 13))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

    panels = [
        (gs[0, 0], panel_a, "A: ABTT + IG pair matrix (current)", "RdBu_r", True),
        (gs[0, 1], panel_b, "B: BERTScore greedy alignment\n(per-token ABTT, no IG)", "YlOrRd", False),
        (gs[1, 0], panel_c, "C: Optimal Transport (EMD)\n(IG mass, per-token ABTT cost)", "YlOrRd", False),
        (gs[1, 1], panel_d, "D: Attention-weighted cross-sim\n(Ditto-inspired, per-token ABTT)", "YlOrRd", False),
    ]

    for subplot_spec, matrix, title, cmap, diverging in panels:
        ax = fig.add_subplot(subplot_spec)
        if diverging:
            vmax = max(np.percentile(np.abs(matrix), 95), 1e-6)
            im = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        else:
            pos_vals = matrix[matrix > 0]
            vmax = np.percentile(pos_vals, 95) if len(pos_vals) > 0 else 1e-6
            im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=max(vmax, 1e-6), aspect="auto")
        ax.set_xticks(range(len(c_tokens)))
        ax.set_xticklabels(c_tokens, rotation=90, fontsize=6, fontfamily="monospace")
        ax.set_yticks(range(len(q_tokens)))
        ax.set_yticklabels(q_tokens, fontsize=6, fontfamily="monospace")
        ax.set_xlabel("Candidate tokens", fontsize=8)
        ax.set_ylabel("Query tokens", fontsize=8)
        ax.set_title(title, fontweight="bold", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)

    fig.suptitle(
        f"{row['model_name']}  Layer {int(row['layer'])}  |  "
        f"Example {int(row['example_id'])}  |  "
        f"{pretty_role(row['candidate_role'])}  |  "
        f"Correct rank: {int(row['correct_rank'])}",
        fontweight="bold", fontsize=12,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if out_path.suffix.lower() == ".png":
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    # --- Compute metrics ---
    metrics = {
        "example_id": int(row["example_id"]),
        "model_name": row["model_name"],
        "candidate_role": row["candidate_role"],
        "correct_rank": int(row["correct_rank"]),
    }
    for label, mat in [
        ("ig_abtt", panel_a),
        ("bertscore", panel_b),
        ("ot_emd", panel_c),
        ("attn_crosssim", panel_d),
    ]:
        metrics[f"{label}_sparsity"] = sparsity_ratio(mat)
        metrics[f"{label}_content_focus"] = content_focus(mat, q_tokens, c_tokens)
        metrics[f"{label}_shared_token"] = shared_token_score(mat, q_tokens, c_tokens)

    return metrics


# ---------------------------------------------------------------------------
# Detail figures for individual methods
# ---------------------------------------------------------------------------

def render_bertscore_detail(
    row: pd.Series,
    data: dict[str, np.ndarray],
    tokenizer,
    out_path: Path,
    max_tokens: int,
) -> None:
    """BERTScore detail: dense cosine, recall alignment, combined."""
    q_len = int(data["query_attention_mask"][0].sum())
    c_len = int(data["candidate_attention_mask"][0].sum())
    q_pos = select_positions(data["query_ig_baseline"], data["query_ig_abtt"], q_len, max_tokens)
    c_pos = select_positions(data["candidate_ig_baseline"], data["candidate_ig_abtt"], c_len, max_tokens)
    q_tokens = decode_tokens(tokenizer, data["query_input_ids"], q_pos)
    c_tokens = decode_tokens(tokenizer, data["candidate_input_ids"], c_pos)
    q_hc = clean_tokens(data["query_hidden"], data["pcs"], data["mean_vec"])[q_pos]
    c_hc = clean_tokens(data["candidate_hidden"], data["pcs"], data["mean_vec"])[c_pos]

    cos_dense, combined = build_bertscore_matrix(q_hc, c_hc)
    # Also build recall-only for comparison
    nq, nc = cos_dense.shape
    recall_only = np.zeros_like(cos_dense)
    for i in range(nq):
        j = np.argmax(cos_dense[i])
        recall_only[i, j] = cos_dense[i, j]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    for ax, mat, title in [
        (axes[0], cos_dense, "Dense cosine (per-token ABTT)"),
        (axes[1], recall_only, "Recall alignment (query -> best candidate)"),
        (axes[2], combined, "Combined (recall + precision)"),
    ]:
        pos_v = mat[mat > 0]
        vmax = np.percentile(pos_v, 95) if len(pos_v) > 0 else 1.0
        im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=max(vmax, 1e-6), aspect="auto")
        ax.set_xticks(range(len(c_tokens)))
        ax.set_xticklabels(c_tokens, rotation=90, fontsize=6, fontfamily="monospace")
        ax.set_yticks(range(len(q_tokens)))
        ax.set_yticklabels(q_tokens, fontsize=6, fontfamily="monospace")
        ax.set_title(title, fontweight="bold", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.7)

    fig.suptitle(
        f"BERTScore Detail | {row['model_name']} L{int(row['layer'])} | "
        f"Ex {int(row['example_id'])} | {pretty_role(row['candidate_role'])}",
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_ot_detail(
    row: pd.Series,
    data: dict[str, np.ndarray],
    tokenizer,
    out_path: Path,
    max_tokens: int,
) -> None:
    """OT detail: IG-weighted vs uniform mass EMD."""
    q_len = int(data["query_attention_mask"][0].sum())
    c_len = int(data["candidate_attention_mask"][0].sum())
    q_pos = select_positions(data["query_ig_baseline"], data["query_ig_abtt"], q_len, max_tokens)
    c_pos = select_positions(data["candidate_ig_baseline"], data["candidate_ig_abtt"], c_len, max_tokens)
    q_tokens = decode_tokens(tokenizer, data["query_input_ids"], q_pos)
    c_tokens = decode_tokens(tokenizer, data["candidate_input_ids"], c_pos)
    q_hc = clean_tokens(data["query_hidden"], data["pcs"], data["mean_vec"])[q_pos]
    c_hc = clean_tokens(data["candidate_hidden"], data["pcs"], data["mean_vec"])[c_pos]
    q_ig = data["query_ig_abtt"][q_pos]
    c_ig = data["candidate_ig_abtt"][c_pos]

    T_ig = build_ot_transport_plan(q_hc, c_hc, q_ig, c_ig, use_uniform=False)
    T_unif = build_ot_transport_plan(q_hc, c_hc, q_ig, c_ig, use_uniform=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for ax, mat, title in [
        (axes[0], T_ig, "EMD: IG-weighted mass"),
        (axes[1], T_unif, "EMD: uniform mass"),
    ]:
        pos_v = mat[mat > 0]
        vmax = np.percentile(pos_v, 95) if len(pos_v) > 0 else 1e-6
        im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=max(vmax, 1e-6), aspect="auto")
        ax.set_xticks(range(len(c_tokens)))
        ax.set_xticklabels(c_tokens, rotation=90, fontsize=6, fontfamily="monospace")
        ax.set_yticks(range(len(q_tokens)))
        ax.set_yticklabels(q_tokens, fontsize=6, fontfamily="monospace")
        ax.set_title(title, fontweight="bold", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.7)

    fig.suptitle(
        f"OT Detail | {row['model_name']} L{int(row['layer'])} | "
        f"Ex {int(row['example_id'])} | {pretty_role(row['candidate_role'])}",
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare IG vs token-alignment methods.")
    p.add_argument("--examples_csv", required=True)
    p.add_argument("--artifacts_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_tokens", type=int, default=20)
    p.add_argument(
        "--max_examples", type=int, default=10,
        help="Max examples to process (default: 10).",
    )
    p.add_argument(
        "--only_correct", action="store_true",
        help="Only process correct_option (same-source) examples.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    examples = pd.read_csv(args.examples_csv)
    artifacts_dir = Path(args.artifacts_dir)
    out_dir = Path(args.out_dir)
    comp_dir = out_dir / "comparisons"
    detail_dir = out_dir / "details"
    comp_dir.mkdir(parents=True, exist_ok=True)
    detail_dir.mkdir(parents=True, exist_ok=True)

    if args.only_correct:
        examples = examples[examples["candidate_role"] == "correct_option"].copy()

    from transformers import AutoTokenizer
    tokenizers: dict[str, object] = {}
    all_metrics: list[dict] = []
    processed = 0

    for _, row in examples.iterrows():
        if processed >= args.max_examples:
            break

        slug = model_slug(row["model_name"])
        artifact_path = (
            artifacts_dir / slug
            / f"example{int(row['example_id']):03d}_{row['candidate_role']}.npz"
        )
        if not artifact_path.exists():
            print(f"[SKIP] Artifact missing: {artifact_path}")
            continue

        model_name = row["model_name"]
        if model_name not in tokenizers:
            tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)

        data = load_artifact(artifacts_dir, row)
        tag = f"example{int(row['example_id']):03d}_{row['candidate_role']}"

        # 4-panel comparison
        comp_path = comp_dir / f"{slug}_{tag}_comparison.png"
        metrics = render_comparison(row, data, tokenizers[model_name], comp_path, args.max_tokens)
        all_metrics.append(metrics)

        # Detail figures
        bs_path = detail_dir / f"{slug}_{tag}_bertscore.png"
        render_bertscore_detail(row, data, tokenizers[model_name], bs_path, args.max_tokens)

        ot_path = detail_dir / f"{slug}_{tag}_ot.png"
        render_ot_detail(row, data, tokenizers[model_name], ot_path, args.max_tokens)

        print(f"[OK] {slug} example {int(row['example_id'])} ({row['candidate_role']})")
        processed += 1

    # Save metrics
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = out_dir / "comparison_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\nSaved {len(all_metrics)} comparison sets to {comp_dir}")
        print(f"Detail figures in {detail_dir}")
        print(f"Metrics: {metrics_path}")

        # Summary
        print("\n=== Metrics Summary (mean across examples) ===")
        methods = ["ig_abtt", "bertscore", "ot_emd", "attn_crosssim"]
        for suffix in ["content_focus", "shared_token", "sparsity"]:
            cols = [f"{m}_{suffix}" for m in methods]
            present = [c for c in cols if c in metrics_df.columns]
            if present:
                print(f"\n{suffix}:")
                for c in present:
                    label = c.replace(f"_{suffix}", "")
                    print(f"  {label:20s}: {metrics_df[c].mean():.4f}")
    else:
        print("No examples were processed.")


if __name__ == "__main__":
    main()
