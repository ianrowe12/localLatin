"""Compute MarK-adapted attribution-quality metrics on existing IG pair NPZs.

Reads every per-pair NPZ under
``runs/active/ig_examples/artifacts/<model_slug>/`` and computes four metrics
(sufficiency, comprehensiveness, compactness, LOO rank correlation) for every
attribution method present in the NPZ, separately for {baseline, abtt} ABTT
variants. Methods are auto-discovered from ``pair_matrix_*_baseline`` key
prefixes, so ``retrieval_mark``/MaRC sidecars appear in the table whenever
they have been merged into the artifact NPZs.

In addition to the stored attribution methods, the driver synthesizes two
diagnostic baselines per pair × variant:

  - ``random``  : i.i.d. uniform-random per-token scores (averaged over 5 seeds)
  - ``inverse`` : negated stored ``ig`` per-token scores (selects bottom-k)

These ground the table: a real attribution method should beat ``random`` and
the gap to ``inverse`` is the cleanest single-number summary of "does the
ranking matter".

Outputs:

  runs/active/ig_examples/attribution_metrics/<slug>/<example_tag>.json   per pair
  runs/active/ig_examples/attribution_metrics/summary.csv                 per (model, method, variant)
  overleaf_drafts/tables/attribution_metrics.tex                          paper-ready table

Usage:
  python scripts/ig/run_attribution_metrics.py \\
      --examples_csv runs/active/ig_examples/phase12f_examples.csv \\
      --artifacts_root runs/active/ig_examples/artifacts \\
      --out_root runs/active/ig_examples/attribution_metrics \\
      --tex_out overleaf_drafts/tables/attribution_metrics.tex
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from attribution_metrics import (  # noqa: E402
    METHOD_SCORE_REDUCER,
    METRIC_REGISTRY,
    REDUCER_FALLBACK,
    PairContext,
    scores_from_pair_matrix,
)


# ---------------------------------------------------------------------------
# Model loading (mirrors scripts/_archive/run_phase12e_pair_explanations.py:60)
# ---------------------------------------------------------------------------
def load_model(model_name: str, model_type: str, half_precision: bool,
               trust_remote_code: bool, device: str):
    kwargs = {"trust_remote_code": trust_remote_code}
    if half_precision:
        kwargs["torch_dtype"] = torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model_type == "t5":
        full_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)
        model = full_model.get_encoder()
        resolved_type = "t5"
    else:
        model = AutoModel.from_pretrained(model_name, **kwargs)
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            resolved_type = "decoder_wrapped"
        elif hasattr(model, "layers"):
            resolved_type = "decoder"
        elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            resolved_type = "bert"
        else:
            resolved_type = model_type
    model.to(device).eval()
    model.requires_grad_(False)
    return model, tokenizer, resolved_type


def model_slug(name: str) -> str:
    return name.replace("/", "_")


# ---------------------------------------------------------------------------
# Embedding + ABTT helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def forward_pooled(model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                   layer: int) -> torch.Tensor:
    """Mean-pool the layer-L hidden states using ``attention_mask`` weights.

    Returns shape (hidden,). Casts to float32 before pooling for numerical
    stability when the model runs in fp16.
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True, return_dict=True)
    hidden = out.hidden_states[layer].float()  # (1, seq, hidden)
    mask = attention_mask.float()              # (1, seq)
    pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    return pooled.squeeze(0)


def abtt_clean(vec: torch.Tensor, pcs: torch.Tensor, mean_vec: torch.Tensor) -> torch.Tensor:
    """Subtract train-fit mean and project out top-D principal components.

    Mirrors ``scripts/resubmit/run_resubmit_ig_comparison.py:78``.
    """
    centered = vec - mean_vec
    proj = centered @ pcs.T @ pcs
    return centered - proj


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    na = a.norm().clamp(min=1e-12)
    nb = b.norm().clamp(min=1e-12)
    return float((a @ b) / (na * nb))


# ---------------------------------------------------------------------------
# NPZ method discovery
# ---------------------------------------------------------------------------
def infer_methods(keys: List[str]) -> List[str]:
    """Return methods present as ``pair_matrix_<method>_baseline`` AND
    ``pair_matrix_<method>_abtt``. Order follows first-seen."""
    base = {}
    abtt = {}
    for k in keys:
        if k.startswith("pair_matrix_") and k.endswith("_baseline"):
            base[k[len("pair_matrix_"):-len("_baseline")]] = True
        elif k.startswith("pair_matrix_") and k.endswith("_abtt"):
            abtt[k[len("pair_matrix_"):-len("_abtt")]] = True
    methods = [m for m in base if m in abtt]
    return methods


# ---------------------------------------------------------------------------
# Per-pair driver
# ---------------------------------------------------------------------------
def process_pair(npz_path: Path, model, tokenizer, device: str,
                 fractions: Tuple[float, ...], compactness_threshold: float,
                 method_filter: Optional[List[str]],
                 random_seeds: int = 5,
                 compute_aopc: bool = False) -> List[dict]:
    """Process one NPZ. Returns a list of result rows (one per method × variant)."""
    data = np.load(npz_path)
    layer = int(data["layer"].item())
    n_q = int(data["query_attention_mask"].sum())
    n_c = int(data["candidate_attention_mask"].sum())

    # Trim to actual length (NPZs were saved with padding=False so lengths match,
    # but trim defensively in case future artifacts add padding).
    q_ids_full = torch.from_numpy(data["query_input_ids"][:, :n_q].astype(np.int64)).to(device)
    q_mask_full = torch.from_numpy(data["query_attention_mask"][:, :n_q].astype(np.int64)).to(device)
    c_ids = torch.from_numpy(data["candidate_input_ids"][:, :n_c].astype(np.int64)).to(device)
    c_mask = torch.from_numpy(data["candidate_attention_mask"][:, :n_c].astype(np.int64)).to(device)

    pcs = torch.from_numpy(data["pcs"].astype(np.float32)).to(device)
    mean_vec = torch.from_numpy(data["mean_vec"].astype(np.float32)).to(device)

    methods_present = infer_methods(list(data.files))
    if method_filter is not None:
        methods_present = [m for m in methods_present if m in method_filter]

    pad_id = int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else 0
    rows: List[dict] = []

    for variant in ("baseline", "abtt"):
        # 1. Candidate embedding (computed once per variant; query is masked many times)
        c_pooled = forward_pooled(model, c_ids, c_mask, layer)
        if variant == "abtt":
            c_pooled = abtt_clean(c_pooled, pcs, mean_vec)

        # Closure: take a length-n_q binary mask, return cosine under variant.
        def make_eval():
            cap = c_pooled  # capture by reference
            def eval_masked_cos(mask01: np.ndarray) -> float:
                m_bool = mask01.astype(bool)
                # Replace masked positions with PAD; zero attention for them.
                ids = q_ids_full.clone()
                attn = q_mask_full.clone()
                if (~m_bool).any():
                    drop_idx = torch.from_numpy(np.where(~m_bool)[0]).to(device)
                    ids[0, drop_idx] = pad_id
                    attn[0, drop_idx] = 0
                # If everything is masked, attention sum is 0 -> avoid NaN by bailing.
                if attn.sum() == 0:
                    return 0.0
                q_pooled = forward_pooled(model, ids, attn, layer)
                if variant == "abtt":
                    q_pooled = abtt_clean(q_pooled, pcs, mean_vec)
                return cos(q_pooled, cap)
            return eval_masked_cos
        eval_masked_cos = make_eval()

        full_cos = eval_masked_cos(np.ones(n_q, dtype=np.int64))

        # Single-token leave-one-out cache: shared across all methods for this variant.
        single_ablation_cos = np.empty(n_q, dtype=np.float64)
        for i in range(n_q):
            mask = np.ones(n_q, dtype=np.int64)
            mask[i] = 0
            single_ablation_cos[i] = eval_masked_cos(mask)

        ctx = PairContext(
            n_q=n_q, variant=variant, full_cos=full_cos,
            single_ablation_cos=single_ablation_cos,
            eval_masked_cos=eval_masked_cos,
            metadata={"layer": layer, "n_c": n_c},
        )

        # Real methods.
        for method in methods_present:
            pm = data[f"pair_matrix_{method}_{variant}"]
            stored = data.get(f"query_ig_{variant}") if method == "ig" else None
            if stored is not None:
                stored = stored[:n_q]
            reducer = METHOD_SCORE_REDUCER.get(method, REDUCER_FALLBACK)
            scores = scores_from_pair_matrix(pm[:n_q, :n_c], stored, reducer)
            rows.append(_eval_one(ctx, method, scores, fractions, compactness_threshold, compute_aopc))

        # Pseudo-baseline: random scores. Emit one row per seed so seed variance
        # propagates into the across-pair std (per statistician audit).
        rng = np.random.default_rng(0)
        for seed in range(random_seeds):
            r = rng.uniform(-1.0, 1.0, size=n_q)
            rows.append(_eval_one(ctx, "random", r, fractions, compactness_threshold, compute_aopc))

        # Pseudo-baseline: inverse. We want a method whose ranking is the
        # *reverse* of IG (low |IG| ranked highest) AND whose |score| is
        # anti-correlated with |IG| (so loo_rho is meaningful, not trivially
        # equal to IG's via the np.abs in loo_correlation). Using
        # ``1 / (eps + |IG|)`` satisfies both.
        ig_key = f"query_ig_{variant}"
        if ig_key in data.files:
            ig_abs = np.abs(data[ig_key][:n_q].astype(np.float64))
            inv_scores = 1.0 / (1e-9 + ig_abs)
            rows.append(_eval_one(ctx, "inverse", inv_scores, fractions, compactness_threshold, compute_aopc))

    return rows


def _eval_one(ctx: PairContext, method: str, scores: np.ndarray,
              fractions: Tuple[float, ...], compactness_threshold: float,
              compute_aopc: bool = False) -> dict:
    """Run all registered metrics on one (ctx, scores) pair. Returns a flat row."""
    row: dict = {"method": method, "variant": ctx.variant,
                 "n_q": ctx.n_q, "full_cos": ctx.full_cos}
    for name, fn in METRIC_REGISTRY.items():
        if name == "sufficiency":
            row.update(fn(ctx, scores, fractions=fractions, compute_aopc=compute_aopc))
        elif name == "comprehensiveness":
            row.update(fn(ctx, scores, fractions=fractions, compute_aopc=compute_aopc))
        elif name == "compactness":
            row.update(fn(ctx, scores, threshold=compactness_threshold))
        else:
            row.update(fn(ctx, scores))
    return row


# ---------------------------------------------------------------------------
# Aggregation + LaTeX
# ---------------------------------------------------------------------------
HEADLINE_METRIC_KEYS = (
    "suff@0.25_ratio",
    "comp@0.25_ratio",
    "compactness@0.80",
    "loo_rho",
)
HEADLINE_LABELS = {
    "suff@0.25_ratio": r"Suff@25\%~$\uparrow$",
    "comp@0.25_ratio": r"Comp@25\%~$\uparrow$",
    "compactness@0.80": r"Cmpct@0.8~$\downarrow$",
    "loo_rho": r"$\rho_{\text{LOO}}$~$\uparrow$",
}
MODEL_SHORT = {
    "bowphs/LaTa": "LaTa",
    "bowphs/PhilTa": "PhilTa",
    "google/mt5-base": "mT5-base",
    "sentence-transformers/LaBSE": "LaBSE",
    "Qwen/Qwen3-Embedding-0.6B": "Qwen3-0.6B",
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5": "KaLM-mini",
    "Qwen/Qwen3-Embedding-8B": "Qwen3-8B",
}

METHOD_DISPLAY_ORDER = (
    "ig", "bertscore", "ot",
    "attention_weighted", "attention_standalone", "dla",
    "retrieval_mark",
    "random", "inverse",
)
METHOD_LABELS = {
    "ig": "IG",
    "bertscore": "BERTScore",
    "ot": "OT",
    "attention_weighted": "Att-weighted",
    "attention_standalone": "Att-standalone",
    "dla": "DLA",
    "retrieval_mark": "MaRC",
    "random": "\\textit{random}",
    "inverse": "\\textit{inverse}",
}


def aggregate(per_pair_rows: List[dict]) -> pd.DataFrame:
    """Aggregate per-pair rows into per (model, method, variant) means + std + n.

    Drops NaNs per metric column before averaging.
    """
    df = pd.DataFrame(per_pair_rows)
    grouped = df.groupby(["model", "method", "variant"])
    out_rows = []
    metric_cols = [c for c in df.columns if c not in
                   ("model", "method", "variant", "example_tag", "n_q", "full_cos", "layer")]
    for (model, method, variant), g in grouped:
        row = {"model": model, "method": method, "variant": variant, "n": len(g),
               "n_q_mean": float(g["n_q"].mean()),
               "full_cos_mean": float(g["full_cos"].mean())}
        for c in metric_cols:
            if c not in g.columns:
                continue
            vals = g[c].astype(float).dropna()
            n_used = len(vals)
            row[f"{c}_mean"] = float(vals.mean()) if n_used else float("nan")
            row[f"{c}_std"] = float(vals.std(ddof=1)) if n_used > 1 else float("nan")
            row[f"{c}_se"] = float(vals.std(ddof=1) / np.sqrt(n_used)) if n_used > 1 else float("nan")
            row[f"{c}_n"] = int(n_used)
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def render_latex(summary: pd.DataFrame, out_path: Path) -> None:
    """Render the methods × {baseline | abtt} table grouped by model.

    Uses booktabs + multicolumn. Bolds the best method per (metric, variant)
    column within each model block. Direction-aware: higher-is-better for
    sufficiency, comprehensiveness, loo_rho; lower-is-better for compactness.
    """
    HIGHER_BETTER = {
        "suff@0.25_ratio": True,
        "comp@0.25_ratio": True,
        "compactness@0.80": False,
        "loo_rho": True,
    }

    def fmt(v: float) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "--"
        return f"{v:.3f}"

    lines: List[str] = []
    lines.append("% Auto-generated by scripts/ig/run_attribution_metrics.py")
    lines.append("% Edit the script, not this file.")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")

    # Column spec: method | (baseline abtt) for each of 4 metrics = 1 + 8 = 9 cols.
    # Right-aligned numeric cells for cleaner decimal alignment without siunitx.
    colspec = "l" + "rr" * len(HEADLINE_METRIC_KEYS)
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\toprule")

    # Header row 1: metric group names spanning 2 cols each.
    h1 = ["Method"]
    for k in HEADLINE_METRIC_KEYS:
        h1.append(rf"\multicolumn{{2}}{{c}}{{{HEADLINE_LABELS[k]}}}")
    lines.append(" & ".join(h1) + r" \\")

    # Header row 2: baseline / abtt under each.
    cmid_parts = []
    for i, _ in enumerate(HEADLINE_METRIC_KEYS):
        # Method col is col 1; metric i has cols 2+2i and 3+2i.
        a = 2 + 2 * i
        b = 3 + 2 * i
        cmid_parts.append(rf"\cmidrule(lr){{{a}-{b}}}")
    lines.append(" ".join(cmid_parts))
    h2 = [""]
    for _ in HEADLINE_METRIC_KEYS:
        h2.append("base")
        h2.append("abtt")
    lines.append(" & ".join(h2) + r" \\")
    lines.append(r"\midrule")

    # One block per model, ordered to match phase12f.
    seen_models = list(dict.fromkeys(summary["model"].tolist()))
    for model in seen_models:
        short = MODEL_SHORT.get(model, model)
        lines.append(rf"\multicolumn{{{1 + 2 * len(HEADLINE_METRIC_KEYS)}}}{{l}}{{\textit{{{short}}}}} \\")

        sub = summary[summary["model"] == model]
        # Determine bolding: best per (metric, variant) over real methods only.
        real_methods = [m for m in METHOD_DISPLAY_ORDER
                        if m in sub["method"].values and m not in ("random", "inverse")]
        order_key = {m: i for i, m in enumerate(METHOD_DISPLAY_ORDER)}
        best_idx: Dict[Tuple[str, str], str] = {}
        for k in HEADLINE_METRIC_KEYS:
            for v in ("baseline", "abtt"):
                col = f"{k}_mean"
                if col not in sub.columns:
                    continue
                cand = sub[(sub["variant"] == v) & sub["method"].isin(real_methods)][["method", col]]
                cand = cand.dropna()
                if cand.empty:
                    continue
                # Sort deterministically by display order so ties resolve consistently.
                cand = cand.assign(_ord=cand["method"].map(order_key)).sort_values("_ord")
                if HIGHER_BETTER[k]:
                    best = cand.loc[cand[col].idxmax(), "method"]
                else:
                    best = cand.loc[cand[col].idxmin(), "method"]
                best_idx[(k, v)] = best

        # Methods rows.
        for method in METHOD_DISPLAY_ORDER:
            if method not in sub["method"].values:
                continue
            row_label = METHOD_LABELS.get(method, method)
            row_cells = [row_label]
            for k in HEADLINE_METRIC_KEYS:
                col_mean = f"{k}_mean"
                for v in ("baseline", "abtt"):
                    cell = sub[(sub["method"] == method) & (sub["variant"] == v)]
                    if cell.empty or col_mean not in cell.columns:
                        row_cells.append("--")
                        continue
                    val = float(cell.iloc[0][col_mean])
                    s = fmt(val)
                    if best_idx.get((k, v)) == method:
                        s = rf"\textbf{{{s}}}"
                    row_cells.append(s)
            lines.append(" & ".join(row_cells) + r" \\")
        lines.append(r"\midrule")

    # Replace last midrule with bottomrule.
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")

    real = summary[~summary["method"].isin(["random", "inverse"])].copy()
    if real.empty:
        counts = summary.groupby("model")["n"].max()
    else:
        counts = real.groupby("model")["n"].max()
    model_bits = [
        f"{MODEL_SHORT.get(model, model)} n={int(n)}" for model, n in counts.items()
    ]
    model_phrase = "; ".join(model_bits)
    decoder_caveat = ""
    if any("Qwen" in model or "KaLM" in model for model in counts.index):
        decoder_caveat = (
            r"For decoder-only models, masking replaces tokens with PAD under "
            r"the causal mask, which does not fully remove their influence on "
            r"downstream hidden states; those rows should be interpreted with "
            r"this caveat. "
        )

    caption = (
        rf"Retrieval-adapted attribution-quality metrics on cached "
        rf"query--candidate attribution examples ({model_phrase}). "
        r"For each model $\times$ \{baseline, ABTT\} $\times$ method we report: "
        r"\textbf{Suff@25\%}: $S_v(q_{\text{top-25\%}}, c) / S_v(q_{\text{full}}, c)$ "
        r"(higher is better); restricted to pairs with $S_v(q_{\text{full}}, c) \ge 0.05$ "
        r"to avoid small-denominator instability. "
        r"\textbf{Comp@25\%}: relative drop when the top 25\% of tokens are masked "
        r"(higher is better). "
        r"\textbf{Compact@0.8}: smallest token fraction that recovers "
        r"$\ge 0.8 \cdot S_v(q_{\text{full}}, c)$ (lower is better). "
        r"\textbf{$\rho_{\text{LOO}}$}: Spearman correlation between $|a|$ and "
        r"single-token leave-one-out $\Delta\cos$ "
        r"(higher is better; tokens with $|\Delta| < 10^{-6}$ excluded as noise). "
        r"\textit{random} and \textit{inverse} rows serve as lower- and upper-bound "
        r"sanity checks; a useful method should beat \textit{random} on every metric, "
        r"and \textit{inverse} (using $1 / (\varepsilon + |\textsc{IG}|)$ as the per-token "
        r"score, so its ranking is the reverse of \textsc{IG}'s) should be the worst. "
        r"Bolded values mark the best real method per (metric, variant) within each model. "
        r"Cross-variant deltas conflate two effects (ABTT changes both the attribution "
        r"scores and the decision function $\cos(\text{embed}_{\text{ABTT}}(q), \text{embed}_{\text{ABTT}}(c))$ "
        r"being explained) and should be read as descriptive shifts on this "
        r"example set rather than as a measure of fidelity loss caused by ABTT alone. "
        + decoder_caveat +
        r"\textsc{IG} and \textsc{OT} produce numerically identical rows because the "
        r"\textsc{OT} pair-matrix uses $|\textsc{IG}|$ as transport mass and our "
        r"row-sum-positive reduction recovers the same per-token magnitudes; "
        r"any methodological gap between them shows up only in the pair-matrix "
        r"heatmaps, not in the per-token aggregate. "
        r"Per-pair rows and the full sweep over fractions $\{0.10, 0.25, 0.50\}$ are in "
        r"the companion summary CSV."
    )
    lines.append(rf"\caption{{{caption}}}")
    lines.append(r"\label{tab:attribution_metrics}")
    lines.append(r"\end{table*}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--examples_csv",
                   default=str(REPO_ROOT / "runs/active/ig_examples/phase12f_examples.csv"))
    p.add_argument("--artifacts_root",
                   default=str(REPO_ROOT / "runs/active/ig_examples/artifacts"))
    p.add_argument("--out_root",
                   default=str(REPO_ROOT / "runs/active/ig_examples/attribution_metrics"))
    p.add_argument("--tex_out",
                   default=str(REPO_ROOT / "overleaf_drafts/tables/attribution_metrics.tex"))
    p.add_argument("--models", nargs="*", default=None,
                   help="Restrict to these model_name strings (default: auto-detect from CSV).")
    p.add_argument("--methods", nargs="*", default=None,
                   help="Restrict to these method names (default: auto-detect from NPZ).")
    p.add_argument("--sufficiency_fractions", default="0.10,0.25,0.50",
                   help="Comma-separated fractions for suff/comp sweeps.")
    p.add_argument("--compactness_threshold", type=float, default=0.8)
    p.add_argument("--max_pairs_per_model", type=int, default=None)
    p.add_argument("--half_precision", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--device", default=None,
                   help="Override device (default: cuda if available else cpu).")
    p.add_argument("--dry_run", action="store_true",
                   help="List planned work and exit; do no forward passes.")
    p.add_argument("--dry_run_require_existing", action="store_true",
                   help="With --dry_run, fail if planned artifact NPZs are missing.")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip per-pair JSONs that already exist on disk.")
    p.add_argument("--render_only", action="store_true",
                   help="Skip metric computation; rebuild summary + TEX from existing per-pair JSONs.")
    p.add_argument("--compute_aopc", action="store_true",
                   help="Also compute AOPC over k=1..n for sufficiency and comprehensiveness "
                        "(adds ~2*n_q forward passes per method per variant; off by default).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fractions = tuple(float(x) for x in args.sufficiency_fractions.split(","))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    examples = pd.read_csv(args.examples_csv)
    if args.models:
        examples = examples[examples["model_name"].isin(args.models)]
    examples = examples.copy()
    examples["slug"] = examples["model_name"].apply(model_slug)
    examples = examples.sort_values(["model_name", "example_id"]).reset_index(drop=True)

    artifacts_root = Path(args.artifacts_root)
    out_root = Path(args.out_root)
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        failures: list[str] = []
        for slug, sub in examples.groupby("slug"):
            planned = sub
            if args.max_pairs_per_model is not None:
                planned = planned.head(args.max_pairs_per_model)
            print(f"{slug}: {len(planned)} examples")
            for ex in planned.to_dict(orient="records"):
                ex_id = int(ex["example_id"])
                role = str(ex.get("candidate_role", "pair_example"))
                npz_path = artifacts_root / slug / f"example{ex_id:03d}_{role}.npz"
                status = "exists" if npz_path.exists() else "missing"
                print(f"  [DRY] example{ex_id:03d}_{role}.npz {status}: {npz_path}")
                if args.dry_run_require_existing and not npz_path.exists():
                    failures.append(f"missing artifact: {npz_path}")
        if failures:
            raise SystemExit("Dry run failed:\n" + "\n".join(failures))
        return

    all_rows: List[dict] = []

    if not args.render_only:
        # Group by model so we load the model exactly once.
        for (model_name, model_type), sub in examples.groupby(["model_name", "model_type"]):
            slug = model_slug(model_name)
            slug_dir = out_root / slug
            slug_dir.mkdir(parents=True, exist_ok=True)
            ex_rows = sub.to_dict(orient="records")
            if args.max_pairs_per_model is not None:
                ex_rows = ex_rows[: args.max_pairs_per_model]
            print(f"\n=== {model_name} ({model_type}) - {len(ex_rows)} pairs ===",
                  flush=True)
            t_load = time.time()
            model, tokenizer, resolved_type = load_model(
                model_name, model_type, args.half_precision, args.trust_remote_code, device,
            )
            print(f"  loaded model in {time.time() - t_load:.1f}s (resolved={resolved_type})",
                  flush=True)
            for ex in ex_rows:
                ex_id = int(ex["example_id"])
                role = str(ex.get("candidate_role", "pair_example"))
                example_tag = f"example{ex_id:03d}_{role}"
                npz_path = artifacts_root / slug / f"{example_tag}.npz"
                if not npz_path.exists():
                    print(f"  [skip] missing NPZ: {npz_path}")
                    continue
                json_path = slug_dir / f"{example_tag}.json"
                if args.skip_existing and json_path.exists():
                    print(f"  [skip-existing] {example_tag}")
                    with open(json_path) as f:
                        cached = json.load(f)
                    all_rows.extend(cached)
                    continue
                t_pair = time.time()
                rows = process_pair(
                    npz_path, model, tokenizer, device,
                    fractions=fractions,
                    compactness_threshold=args.compactness_threshold,
                    method_filter=args.methods,
                    compute_aopc=args.compute_aopc,
                )
                # Annotate with model/example identifiers.
                for r in rows:
                    r["model"] = model_name
                    r["example_tag"] = example_tag
                    r["layer"] = int(ex["layer"])
                json_path.write_text(json.dumps(rows, indent=2))
                all_rows.extend(rows)
                print(f"  {example_tag}: {len(rows)} rows in {time.time() - t_pair:.1f}s",
                      flush=True)
            del model, tokenizer
            if device == "cuda":
                torch.cuda.empty_cache()
    else:
        # Load existing per-pair JSONs.
        for json_path in sorted(out_root.glob("*/*.json")):
            cached = json.loads(json_path.read_text())
            all_rows.extend(cached)
        print(f"Loaded {len(all_rows)} cached rows from {out_root}")

    if not all_rows:
        print("No rows produced; bailing.")
        return

    summary = aggregate(all_rows)
    summary_csv = out_root / "summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"Wrote {summary_csv} ({len(summary)} rows)")

    render_latex(summary, Path(args.tex_out))


if __name__ == "__main__":
    main()
