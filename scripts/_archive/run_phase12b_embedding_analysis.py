"""Phase 12b: Embedding-level metrics that connect ABTT to AUCROC improvement.

Three metrics that do NOT require Captum / GPU — pure NumPy on pre-extracted
embeddings:
  1) Relative Projection Ratio (RPR)  — fraction of embedding energy in the
     top-D PC subspace that ABTT removes.
  2) Cosine Gap Shift (CGS) — how much ABTT improves same-vs-different pair
     cosine similarity discrimination.
  3) Isotropy Gain (IG_iso) — ratio of isotropy after/before ABTT.

All metrics are computed per model/layer, then correlated with Phase 11
delta-AUCROC to validate alignment.

Usage:
  python scripts/run_phase12b_embedding_analysis.py \
    --emb_base runs/phase9_bases \
    --pc_dir  runs/phase12/pcs \
    --split_csv runs/phase9/phase9_split.csv \
    --phase11_csv runs/phase11/phase11_results.csv \
    --out_dir runs/phase12b
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Model configs — same conventions as run_phase12_aggregate.py
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "LaTa": {
        "slug": "bowphs_LaTa",
        "hf": "bowphs/LaTa",
        "layers": list(range(1, 13)),
    },
    "PhilTa": {
        "slug": "bowphs_PhilTa",
        "hf": "bowphs/PhilTa",
        "layers": list(range(1, 13)),
    },
    "LaBSE": {
        "slug": "sentence-transformers_LaBSE",
        "hf": "sentence-transformers/LaBSE",
        "layers": list(range(1, 13)),
    },
    "Qwen3-0.6B": {
        "slug": "Qwen_Qwen3-Embedding-0.6B",
        "hf": "Qwen/Qwen3-Embedding-0.6B",
        "layers": list(range(1, 29)),
    },
    "KaLM-mini": {
        "slug": "KaLM-Embedding_KaLM-embedding-multilingual-mini-instruct-v2.5",
        "hf": "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
        "layers": list(range(1, 25)),
    },
}

MARKERS = {"LaTa": "o", "PhilTa": "s", "LaBSE": "^", "Qwen3-0.6B": "D", "KaLM-mini": "v"}
COLORS = {"LaTa": "#1f77b4", "PhilTa": "#ff7f0e", "LaBSE": "#2ca02c",
           "Qwen3-0.6B": "#d62728", "KaLM-mini": "#9467bd"}


# ── helpers ────────────────────────────────────────────────────────────────
def _load_embeddings(emb_base: Path, slug: str, layer: int) -> np.ndarray | None:
    """Load hidden-mean embedding .npy for a model/layer (shape: n_files × dim)."""
    p = emb_base / slug / "hidden_mean" / f"hidden_layer{layer}_embeddings.npy"
    if not p.exists():
        return None
    return np.load(p)


def _load_pcs(pc_dir: Path, slug: str, layer: int):
    """Return (pcs, mean_vec) from saved .npz, or (None, None)."""
    p = pc_dir / slug / f"layer{layer}_pcs.npz"
    if not p.exists():
        return None, None
    d = np.load(p)
    return d["pcs"], d["mean_vec"]


def _cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    """Row-normalised cosine similarity matrix."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    Xn = X / norms
    return Xn @ Xn.T


def _apply_abtt(X: np.ndarray, pcs: np.ndarray, mean_vec: np.ndarray) -> np.ndarray:
    """Center X, project out top-D PCs."""
    Xc = X - mean_vec
    proj = Xc @ pcs.T @ pcs          # (n, dim)
    return Xc - proj


def _isotropy(X_centered: np.ndarray) -> float:
    """Isotropy = min(singular_val) / max(singular_val) of centred embedding matrix."""
    _, S, _ = np.linalg.svd(X_centered, full_matrices=False)
    return float(S[-1] / S[0]) if S[0] > 0 else 0.0


# ── metric functions ──────────────────────────────────────────────────────
def compute_rpr(X: np.ndarray, pcs: np.ndarray, mean_vec: np.ndarray) -> float:
    """Mean Relative Projection Ratio across documents."""
    Xc = X - mean_vec
    norms_sq = np.sum(Xc ** 2, axis=1)           # (n,)
    proj = Xc @ pcs.T                             # (n, D)
    proj_norms_sq = np.sum(proj ** 2, axis=1)     # (n,)
    # avoid division by zero for near-zero embeddings
    ratio = np.where(norms_sq > 1e-12, proj_norms_sq / norms_sq, 0.0)
    return float(np.mean(ratio))


def compute_cgs(X: np.ndarray, pcs: np.ndarray, mean_vec: np.ndarray,
                folder_ids: np.ndarray) -> float:
    """Cosine Gap Shift: gap_abtt - gap_baseline."""
    def _gap(sim_mat, fids):
        n = len(fids)
        same_sum, same_cnt = 0.0, 0
        diff_sum, diff_cnt = 0.0, 0
        # Use upper triangle only
        for i in range(n):
            for j in range(i + 1, n):
                if fids[i] == fids[j]:
                    same_sum += sim_mat[i, j]
                    same_cnt += 1
                else:
                    diff_sum += sim_mat[i, j]
                    diff_cnt += 1
        s_mean = same_sum / max(same_cnt, 1)
        d_mean = diff_sum / max(diff_cnt, 1)
        return s_mean - d_mean

    sim_base = _cosine_sim_matrix(X)
    X_abtt = _apply_abtt(X, pcs, mean_vec)
    sim_abtt = _cosine_sim_matrix(X_abtt)

    gap_base = _gap(sim_base, folder_ids)
    gap_abtt = _gap(sim_abtt, folder_ids)
    return float(gap_abtt - gap_base)


def compute_cgs_vectorized(X: np.ndarray, pcs: np.ndarray, mean_vec: np.ndarray,
                           folder_ids: np.ndarray) -> float:
    """Vectorised Cosine Gap Shift (fast for large n)."""
    sim_base = _cosine_sim_matrix(X)
    X_abtt = _apply_abtt(X, pcs, mean_vec)
    sim_abtt = _cosine_sim_matrix(X_abtt)

    n = len(folder_ids)
    # Build same-folder mask (upper triangle only)
    fid_arr = np.array(folder_ids)
    same_mask = np.equal.outer(fid_arr, fid_arr)
    triu = np.triu(np.ones((n, n), dtype=bool), k=1)
    same_mask = same_mask & triu
    diff_mask = (~np.equal.outer(fid_arr, fid_arr)) & triu

    def _gap(sim):
        s = sim[same_mask].mean() if same_mask.any() else 0.0
        d = sim[diff_mask].mean() if diff_mask.any() else 0.0
        return s - d

    return float(_gap(sim_abtt) - _gap(sim_base))


def compute_isotropy_gain(X: np.ndarray, pcs: np.ndarray,
                          mean_vec: np.ndarray) -> tuple[float, float, float]:
    """Return (isotropy_baseline, isotropy_abtt, gain=abtt/base)."""
    Xc = X - X.mean(axis=0)
    iso_base = _isotropy(Xc)

    X_abtt = _apply_abtt(X, pcs, mean_vec)
    Xc_abtt = X_abtt - X_abtt.mean(axis=0)
    iso_abtt = _isotropy(Xc_abtt)

    gain = iso_abtt / iso_base if iso_base > 1e-15 else float("inf")
    return iso_base, iso_abtt, gain


def compute_pc1_var_explained(X: np.ndarray, pcs: np.ndarray,
                              mean_vec: np.ndarray) -> float:
    """Fraction of total variance explained by PC1."""
    Xc = X - mean_vec
    total_var = np.var(Xc, axis=0).sum()
    if total_var < 1e-15:
        return 0.0
    proj = Xc @ pcs[0]
    return float(np.var(proj) / total_var)


# ── main pipeline ─────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Phase 12b embedding-level metrics.")
    p.add_argument("--emb_base", required=True, help="Base dir for phase9_bases embeddings.")
    p.add_argument("--pc_dir", required=True, help="PC directory (runs/phase12/pcs).")
    p.add_argument("--split_csv", required=True, help="phase9_split.csv for folder labels.")
    p.add_argument("--phase11_csv", required=True, help="phase11_results.csv for AUCROC data.")
    p.add_argument("--out_dir", required=True, help="Output directory for metrics + figures.")
    return p.parse_args()


def main():
    args = parse_args()
    emb_base = Path(args.emb_base)
    pc_dir = Path(args.pc_dir)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── load metadata ─────────────────────────────────────────────────
    split_df = pd.read_csv(args.split_csv)
    folder_ids_all = split_df["folder_id"].values      # indexed by file_id
    test_mask = (split_df["split"] == "test").values

    # Phase 11 AUCROC lookup: {(hf_model, method, layer): aucroc}
    p11 = pd.read_csv(args.phase11_csv)
    p11_base = p11[(p11["method"] == "baseline") & (p11["repr"] == "hidden")]
    p11_abtt = p11[(p11["method"] == "abtt_optimal") & (p11["repr"] == "hidden")]

    def _get_aucroc(hf_model: str, layer: int, method_df: pd.DataFrame) -> float | None:
        row = method_df[(method_df["model"] == hf_model) & (method_df["layer"] == layer)]
        if row.empty:
            return None
        return float(row.iloc[0]["aucroc"])

    # ── compute metrics per model / layer ─────────────────────────────
    rows = []
    for mname, cfg in MODEL_CONFIGS.items():
        slug = cfg["slug"]
        hf = cfg["hf"]
        print(f"\n{'='*60}\n  Model: {mname}\n{'='*60}")
        for layer in cfg["layers"]:
            X = _load_embeddings(emb_base, slug, layer)
            pcs, mean_vec = _load_pcs(pc_dir, slug, layer)
            if X is None or pcs is None:
                continue

            # Use TEST split only for pairwise metrics (leak-free)
            X_test = X[test_mask]
            fids_test = folder_ids_all[test_mask]

            # Metrics
            rpr = compute_rpr(X_test, pcs, mean_vec)
            cgs = compute_cgs_vectorized(X_test, pcs, mean_vec, fids_test)
            iso_base, iso_abtt, ig_iso = compute_isotropy_gain(X_test, pcs, mean_vec)
            pc1_var = compute_pc1_var_explained(X_test, pcs, mean_vec)

            # Phase 11 AUCROC
            auc_base = _get_aucroc(hf, layer, p11_base)
            auc_abtt = _get_aucroc(hf, layer, p11_abtt)
            d_auc = (auc_abtt - auc_base) if (auc_base is not None and auc_abtt is not None) else None

            row = {
                "model": mname,
                "layer": layer,
                "rpr": rpr,
                "cgs": cgs,
                "isotropy_baseline": iso_base,
                "isotropy_abtt": iso_abtt,
                "isotropy_gain": ig_iso,
                "pc1_var_explained": pc1_var,
                "aucroc_baseline": auc_base,
                "aucroc_abtt": auc_abtt,
                "delta_aucroc": d_auc,
            }
            rows.append(row)
            print(f"  L{layer:2d}  RPR={rpr:.4f}  CGS={cgs:+.4f}  "
                  f"IG_iso={ig_iso:.2f}  PC1%={pc1_var*100:.1f}  "
                  f"ΔAUC={d_auc:+.4f}" if d_auc is not None else
                  f"  L{layer:2d}  RPR={rpr:.4f}  CGS={cgs:+.4f}  "
                  f"IG_iso={ig_iso:.2f}  PC1%={pc1_var*100:.1f}  ΔAUC=N/A")

    df = pd.DataFrame(rows)
    csv_path = out_dir / "metrics_per_layer.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved metrics to {csv_path}")

    # ── global correlation summary ────────────────────────────────────
    valid = df.dropna(subset=["delta_aucroc"])
    for metric in ["rpr", "cgs", "pc1_var_explained"]:
        r, p = spearmanr(valid[metric], valid["delta_aucroc"])
        print(f"  Spearman({metric}, ΔAUCROC) = {r:+.4f}  (p={p:.2e})")

    # ── Figure 1: RPR vs ΔAUCROC scatter ──────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for mname in MODEL_CONFIGS:
        sub = valid[valid["model"] == mname]
        if sub.empty:
            continue
        ax.scatter(sub["rpr"], sub["delta_aucroc"],
                   marker=MARKERS[mname], color=COLORS[mname],
                   s=50, label=mname, alpha=0.8, edgecolors="k", linewidths=0.5)
    r, p = spearmanr(valid["rpr"], valid["delta_aucroc"])
    ax.set_xlabel("Relative Projection Ratio (RPR)", fontsize=12)
    ax.set_ylabel("ΔAUCROC (ABTT_optimal − baseline)", fontsize=12)
    ax.set_title(f"RPR vs ΔAUCROC  (Spearman ρ = {r:+.3f}, p = {p:.1e})", fontsize=13)
    ax.legend(fontsize=9)
    ax.axhline(0, color="gray", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_rpr_vs_delta_aucroc.pdf", dpi=150)
    fig.savefig(fig_dir / "fig_rpr_vs_delta_aucroc.png", dpi=150)
    plt.close(fig)
    print("  Saved RPR vs ΔAUCROC scatter")

    # ── Figure 2: CGS vs ΔAUCROC scatter ──────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for mname in MODEL_CONFIGS:
        sub = valid[valid["model"] == mname]
        if sub.empty:
            continue
        ax.scatter(sub["cgs"], sub["delta_aucroc"],
                   marker=MARKERS[mname], color=COLORS[mname],
                   s=50, label=mname, alpha=0.8, edgecolors="k", linewidths=0.5)
    r, p = spearmanr(valid["cgs"], valid["delta_aucroc"])
    ax.set_xlabel("Cosine Gap Shift (CGS)", fontsize=12)
    ax.set_ylabel("ΔAUCROC (ABTT_optimal − baseline)", fontsize=12)
    ax.set_title(f"CGS vs ΔAUCROC  (Spearman ρ = {r:+.3f}, p = {p:.1e})", fontsize=13)
    ax.legend(fontsize=9)
    ax.axhline(0, color="gray", ls="--", alpha=0.4)
    ax.axvline(0, color="gray", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_cgs_vs_delta_aucroc.pdf", dpi=150)
    fig.savefig(fig_dir / "fig_cgs_vs_delta_aucroc.png", dpi=150)
    plt.close(fig)
    print("  Saved CGS vs ΔAUCROC scatter")

    # ── Figure 3: Isotropy Gain by layer (line per model) ─────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for mname in MODEL_CONFIGS:
        sub = df[df["model"] == mname].sort_values("layer")
        if sub.empty:
            continue
        ax.plot(sub["layer"], sub["isotropy_gain"],
                marker=MARKERS[mname], color=COLORS[mname],
                markersize=4, label=mname, alpha=0.8)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Isotropy Gain (after / before ABTT)", fontsize=12)
    ax.set_title("Isotropy Gain by Layer", fontsize=13)
    ax.axhline(1, color="gray", ls="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_isotropy_gain_by_layer.pdf", dpi=150)
    fig.savefig(fig_dir / "fig_isotropy_gain_by_layer.png", dpi=150)
    plt.close(fig)
    print("  Saved isotropy gain plot")

    # ── Figure 4: RPR by layer (line per model) with ΔAUCROC on twin ax
    fig, axes = plt.subplots(len(MODEL_CONFIGS), 1, figsize=(10, 3.5 * len(MODEL_CONFIGS)),
                             sharex=False)
    for ax, mname in zip(axes, MODEL_CONFIGS):
        sub = df[df["model"] == mname].dropna(subset=["delta_aucroc"]).sort_values("layer")
        if sub.empty:
            continue
        ax.plot(sub["layer"], sub["rpr"], "b-o", markersize=4, label="RPR")
        ax.set_ylabel("RPR", color="b")
        ax.set_title(mname, fontsize=11)
        ax.tick_params(axis="y", labelcolor="b")

        ax2 = ax.twinx()
        ax2.plot(sub["layer"], sub["delta_aucroc"], "r--s", markersize=4, label="ΔAUCROC", alpha=0.7)
        ax2.set_ylabel("ΔAUCROC", color="r")
        ax2.tick_params(axis="y", labelcolor="r")
        ax.set_xlabel("Layer")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(fig_dir / "fig_rpr_delta_aucroc_by_layer.pdf", dpi=150)
    fig.savefig(fig_dir / "fig_rpr_delta_aucroc_by_layer.png", dpi=150)
    plt.close(fig)
    print("  Saved RPR + ΔAUCROC layerwise panel")

    # ── Diagnosis: CGS sign check ─────────────────────────────────────
    n_positive_cgs = (valid["cgs"] > 0).sum()
    n_total = len(valid)
    n_positive_dauc = (valid["delta_aucroc"] > 0).sum()
    print(f"\n  CGS > 0: {n_positive_cgs}/{n_total}  "
          f"(ΔAUCROC > 0: {n_positive_dauc}/{n_total})")

    print("\nPhase 12b embedding analysis complete.")


if __name__ == "__main__":
    main()
