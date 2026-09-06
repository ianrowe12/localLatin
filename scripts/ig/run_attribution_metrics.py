"""Compute MarK-adapted attribution-quality metrics on existing IG pair NPZs.

Reads every per-pair NPZ under
``runs/active/ig_examples/artifacts/<model_slug>/`` and computes the registered
attribution-quality metrics for every attribution method present in the NPZ,
separately for {baseline, abtt} ABTT variants. Methods are auto-discovered from
``pair_matrix_*_baseline`` key prefixes, so ``retrieval_mark``/MaRC sidecars
appear in the table whenever they have been merged into the artifact NPZs.

Two backends compute the masked cosine that every metric is built on:

  ``--backend model``  (default, historical) re-runs the encoder with PAD in the
      masked positions. Input-level erasure; needs the model and a GPU to be
      practical, so it only affords the four threshold metrics.
  ``--backend hidden`` recomputes the masked cosine from the ``query_hidden`` /
      ``candidate_hidden`` states already cached in the NPZ. Representation-level
      erasure: exact for the full-query cosine (it reproduces the stored
      ``cos_orig_*``), CPU-only, and cheap enough for the whole k=1..n curve,
      which is what AOPC, deletion/insertion AUC and the randomization check
      need. Numbers from the two backends are NOT interchangeable.

In addition to the stored attribution methods, the driver synthesizes two
diagnostic baselines per pair × variant:

  - ``random``  : i.i.d. uniform-random per-token scores (averaged over 5 seeds)
  - ``inverse`` : negated stored ``ig`` per-token scores (selects bottom-k)

These ground the table: a real attribution method should beat ``random`` and
the gap to ``inverse`` is the cleanest single-number summary of "does the
ranking matter".

Outputs:

  runs/active/ig_examples/attribution_metrics/<slug>/<example_tag>.json   per pair
  runs/active/ig_examples/attribution_metrics/summary.csv                 wide per (model, method, variant)
  runs/active/ig_examples/attribution_metrics/summary_sweep_long.csv      long sweep summary
  overleaf_drafts/tables/attribution_metrics.tex                          headline table

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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from attribution_metrics import (  # noqa: E402
    DEFAULT_COMPACTNESS_THRESHOLDS,
    DEFAULT_RANDOM_ORDER_DRAWS,
    DEFAULT_SHUFFLE_DRAWS,
    FULL_COS_FLOOR,
    LOO_NOISE_FLOOR,
    METHOD_SCORE_REDUCER,
    METRIC_REGISTRY,
    RANDOMIZATION_CHECK_KEYS,
    REDUCER_FALLBACK,
    PairContext,
    scores_from_pair_matrix,
)

# torch and transformers are imported lazily. The ``hidden`` backend recomputes
# masked cosines from the pooled hidden states already stored in each artifact
# NPZ, so it needs neither, and importing them anyway would make a CPU-only run
# depend on a GPU-sized dependency stack.
torch = None  # type: ignore[assignment]


def _require_torch():
    """Import torch and transformers on demand (``--backend model`` only)."""
    global torch
    import torch as _torch
    from transformers import (  # noqa: F401
        AutoModel,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
    )

    torch = _torch
    return _torch, AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Model loading (mirrors scripts/_archive/run_phase12e_pair_explanations.py:60)
# ---------------------------------------------------------------------------
def load_model(model_name: str, model_type: str, half_precision: bool,
               trust_remote_code: bool, device: str):
    torch, AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer = _require_torch()
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
def forward_pooled(model, input_ids: "torch.Tensor", attention_mask: "torch.Tensor",
                   layer: int) -> "torch.Tensor":
    """Mean-pool the layer-L hidden states using ``attention_mask`` weights.

    Returns shape (hidden,). Casts to float32 before pooling for numerical
    stability when the model runs in fp16. ``no_grad`` is applied inside the
    body rather than as a decorator so this module still imports without torch.
    """
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True, return_dict=True)
        hidden = out.hidden_states[layer].float()  # (1, seq, hidden)
        mask = attention_mask.float()              # (1, seq)
        pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return pooled.squeeze(0)


def forward_pooled_batch(model, input_ids: "torch.Tensor", attention_mask: "torch.Tensor",
                         layer: int) -> "torch.Tensor":
    """Batched twin of :func:`forward_pooled`. Returns shape (batch, hidden).

    Rows of a batch do not interact inside the encoder, so stacking B masks of
    the same query is numerically the same computation as B separate forwards,
    up to the reduction order inside the matmuls. It is what makes the whole
    k=1..n masking curve affordable under ``--backend model``: one forward per
    ``MODEL_MASK_BATCH`` masks instead of one per mask.
    """
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True, return_dict=True)
        hidden = out.hidden_states[layer].float()      # (batch, seq, hidden)
        mask = attention_mask.float()                  # (batch, seq)
        return (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)


# Masks evaluated per encoder forward under ``--backend model``. Bounded so a
# long query does not blow the activation buffer (output_hidden_states keeps
# every layer).
MODEL_MASK_BATCH: int = 16


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
# Hidden-state backend (CPU, no model)
# ---------------------------------------------------------------------------
# Every artifact NPZ already stores ``query_hidden`` and ``candidate_hidden``:
# the layer-L hidden states that the pooled embedding is built from. Because
# the pooling is a plain mean over kept positions, masking a subset of query
# tokens is a linear operation on those cached rows, so the masked cosine can be
# recomputed exactly without the model. Two consequences:
#
#   * a full k=1..n masking curve costs one cumulative sum instead of n forward
#     passes, which is what makes AOPC, deletion/insertion AUC and the
#     randomization check affordable at all;
#   * the erasure operator is *representation-level* (drop the token from the
#     mean) rather than *input-level* (re-run the encoder with PAD in its
#     place), so it does not re-contextualise the surviving tokens.
#
# The second point is a real semantic difference from the ``model`` backend and
# the numbers are not interchangeable; see the decision memo. Full-query
# cosines are identical to the stored ``cos_orig_*`` values under both backends,
# which is what pins the two to the same decision scalar.
class HiddenPairEvaluator:
    """Masked-cosine evaluator over cached hidden states for one (pair, variant).

    Exposes exactly the three things ``PairContext`` needs: the full cosine, the
    leave-one-out cosine vector, and a masked-cosine closure -- plus the
    vectorised ``prefix_curves`` fast path.
    """

    def __init__(self, query_hidden: np.ndarray, candidate_hidden: np.ndarray,
                 pcs: np.ndarray, mean_vec: np.ndarray, variant: str) -> None:
        self.q = np.asarray(query_hidden, dtype=np.float64)
        self.n = self.q.shape[0]
        self.variant = variant
        self._pcs = np.asarray(pcs, dtype=np.float64)
        self._mean = np.asarray(mean_vec, dtype=np.float64)
        c_vec = self._clean(np.asarray(candidate_hidden, dtype=np.float64).mean(axis=0))
        self._c = c_vec
        self._c_norm = float(np.linalg.norm(c_vec))
        self._q_total = self.q.sum(axis=0)
        self._curve_cache: Dict[bytes, Tuple[np.ndarray, np.ndarray]] = {}

    def _clean(self, vecs: np.ndarray) -> np.ndarray:
        if self.variant != "abtt":
            return vecs
        centered = vecs - self._mean
        return centered - (centered @ self._pcs.T) @ self._pcs

    def _cos_rows(self, pooled: np.ndarray) -> np.ndarray:
        """Cosine of each row of ``pooled`` (m, d) against the candidate."""
        cleaned = self._clean(pooled)
        norms = np.linalg.norm(cleaned, axis=1)
        denom = norms * self._c_norm
        out = np.zeros(cleaned.shape[0], dtype=np.float64)
        ok = denom > 1e-12
        out[ok] = (cleaned[ok] @ self._c) / denom[ok]
        return out

    @property
    def full_cos(self) -> float:
        return float(self._cos_rows((self._q_total / self.n)[None, :])[0])

    def single_ablation_cos(self) -> np.ndarray:
        if self.n < 2:
            return np.zeros(self.n, dtype=np.float64)
        pooled = (self._q_total[None, :] - self.q) / (self.n - 1)
        return self._cos_rows(pooled)

    def eval_masked_cos(self, mask01: np.ndarray) -> float:
        m = np.asarray(mask01, dtype=np.float64)
        kept = m.sum()
        if kept <= 0:
            return 0.0
        return float(self._cos_rows(((self.q * m[:, None]).sum(axis=0) / kept)[None, :])[0])

    def prefix_curves(self, order: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        key = np.asarray(order, dtype=np.int64).tobytes()
        hit = self._curve_cache.get(key)
        if hit is not None:
            return hit
        n = self.n
        prefix = np.cumsum(self.q[order], axis=0)            # prefix[k-1] = sum of top-k
        keep_pooled = np.zeros((n + 1, self.q.shape[1]), dtype=np.float64)
        keep_pooled[1:] = prefix / np.arange(1, n + 1, dtype=np.float64)[:, None]
        drop_pooled = np.zeros((n + 1, self.q.shape[1]), dtype=np.float64)
        drop_pooled[0] = self._q_total / n
        if n > 1:
            sizes = np.arange(n - 1, 0, -1, dtype=np.float64)  # n-1 ... 1
            drop_pooled[1:n] = (self._q_total[None, :] - prefix[: n - 1]) / sizes[:, None]
        keep = self._cos_rows(keep_pooled)
        drop = self._cos_rows(drop_pooled)
        # Endpoints are the empty query; ``_clean`` would otherwise turn the
        # zero row into ``-mean_vec`` and report a spurious cosine for it.
        keep[0] = 0.0
        drop[n] = 0.0
        result = (keep, drop)
        self._curve_cache[key] = result
        return result

    def context(self, metadata: Optional[Dict[str, object]] = None) -> PairContext:
        return PairContext(
            n_q=self.n,
            variant=self.variant,
            full_cos=self.full_cos,
            single_ablation_cos=self.single_ablation_cos(),
            eval_masked_cos=self.eval_masked_cos,
            metadata=metadata or {},
            prefix_curves=self.prefix_curves,
        )


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
# The four metrics that predate this driver's curve-based additions. They are
# the only ones the ``model`` backend can afford, because everything else needs
# the whole k=1..n masking curve rather than a few thresholds.
LEGACY_METRICS: Tuple[str, ...] = (
    "sufficiency", "comprehensiveness", "compactness", "loo_correlation",
)
CURVE_METRICS: Tuple[str, ...] = (
    "aopc", "deletion_auc", "insertion_auc", "randomization_check",
)


@dataclass(frozen=True)
class MetricOptions:
    """Everything the metric dispatcher needs beyond ``(ctx, scores)``."""

    fractions: Tuple[float, ...]
    compactness_thresholds: Tuple[float, ...]
    compute_aopc: bool = False
    metric_names: Optional[Tuple[str, ...]] = None
    random_order_draws: int = DEFAULT_RANDOM_ORDER_DRAWS
    shuffle_draws: int = DEFAULT_SHUFFLE_DRAWS
    random_seeds: int = 5
    # The ``random`` and ``inverse`` diagnostic rows cost six extra scored
    # methods per (pair, variant). Under ``--backend model`` that is the
    # difference between a bounded operator spot check and an unaffordable one,
    # so it can be turned off. Off is never the default: the full tables need
    # those controls.
    skip_pseudo_baselines: bool = False

    def selected(self) -> Tuple[str, ...]:
        if self.metric_names is None:
            return tuple(METRIC_REGISTRY)
        return tuple(n for n in METRIC_REGISTRY if n in self.metric_names)


def _pseudo_baseline_rows(data, ctx: PairContext, variant: str, n_q: int,
                          opts: MetricOptions) -> List[dict]:
    """The ``random`` and ``inverse`` diagnostic rows for one (pair, variant)."""
    rows: List[dict] = []
    if opts.skip_pseudo_baselines:
        return rows
    # Random scores. Emit one row per seed so seed variance propagates into the
    # across-pair std (per statistician audit).
    rng = np.random.default_rng(0)
    for _ in range(opts.random_seeds):
        r = rng.uniform(-1.0, 1.0, size=n_q)
        rows.append(_eval_one(ctx, "random", r, opts))

    # Inverse. We want a method whose ranking is the *reverse* of IG (low |IG|
    # ranked highest) AND whose |score| is anti-correlated with |IG| (so loo_rho
    # is meaningful, not trivially equal to IG's via the np.abs in
    # loo_correlation). Using ``1 / (eps + |IG|)`` satisfies both.
    ig_key = f"query_ig_{variant}"
    if ig_key in data.files:
        ig_abs = np.abs(data[ig_key][:n_q].astype(np.float64))
        inv_scores = 1.0 / (1e-9 + ig_abs)
        rows.append(_eval_one(ctx, "inverse", inv_scores, opts))
    return rows


def _method_rows(data, ctx: PairContext, variant: str, n_q: int, n_c: int,
                 methods_present: Sequence[str], opts: MetricOptions) -> List[dict]:
    """One result row per stored attribution method for this (pair, variant)."""
    rows: List[dict] = []
    for method in methods_present:
        pm = data[f"pair_matrix_{method}_{variant}"]
        stored = data[f"query_ig_{variant}"] if (
            method == "ig" and f"query_ig_{variant}" in data.files
        ) else None
        if stored is not None:
            stored = stored[:n_q]
        reducer = METHOD_SCORE_REDUCER.get(method, REDUCER_FALLBACK)
        scores = scores_from_pair_matrix(pm[:n_q, :n_c], stored, reducer)
        rows.append(_eval_one(ctx, method, scores, opts))
    return rows


def process_pair_hidden(npz_path: Path, method_filter: Optional[List[str]],
                        opts: MetricOptions) -> List[dict]:
    """CPU-only twin of :func:`process_pair` backed by cached hidden states.

    Recomputes every masked cosine from ``query_hidden`` / ``candidate_hidden``
    instead of re-running the encoder, which makes the curve-based metrics
    affordable and removes the model dependency entirely. The erasure operator
    differs from the model backend's (see :class:`HiddenPairEvaluator`), so the
    two backends' numbers must not be mixed inside one table.
    """
    data = np.load(npz_path)
    layer = int(data["layer"].item())
    n_q = int(data["query_attention_mask"].sum())
    n_c = int(data["candidate_attention_mask"].sum())

    q_hidden = data["query_hidden"][:n_q]
    c_hidden = data["candidate_hidden"][:n_c]
    pcs = data["pcs"]
    mean_vec = data["mean_vec"]

    methods_present = infer_methods(list(data.files))
    if method_filter is not None:
        methods_present = [m for m in methods_present if m in method_filter]

    rows: List[dict] = []
    for variant in ("baseline", "abtt"):
        evaluator = HiddenPairEvaluator(q_hidden, c_hidden, pcs, mean_vec, variant)
        ctx = evaluator.context(metadata={"layer": layer, "n_c": n_c})
        variant_rows = _method_rows(data, ctx, variant, n_q, n_c, methods_present, opts)
        variant_rows.extend(_pseudo_baseline_rows(data, ctx, variant, n_q, opts))
        # The consistency check that ties this backend to the artifact: the
        # *unmasked* cosine must reproduce the value stored at build time, when
        # the model actually ran. Carried per row so the aggregate can report it.
        stored_key = f"cos_orig_{variant}"
        if stored_key in data.files:
            drift = abs(ctx.full_cos - float(data[stored_key]))
            for row in variant_rows:
                row["full_cos_drift"] = drift
        rows.extend(variant_rows)
    return rows


def process_pair(npz_path: Path, model, tokenizer, device: str,
                 method_filter: Optional[List[str]],
                 opts: MetricOptions) -> List[dict]:
    """Process one NPZ with the model backend. One row per method × variant."""
    fractions = opts.fractions
    compactness_thresholds = opts.compactness_thresholds
    compute_aopc = opts.compute_aopc
    random_seeds = opts.random_seeds
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

        def make_batch_eval():
            cap = c_pooled
            cap_norm = cap.norm().clamp(min=1e-12)

            def eval_masked_cos_many(masks: np.ndarray) -> np.ndarray:
                """Cosines for a stack of length-n_q masks, batched over forwards."""
                masks = np.asarray(masks, dtype=bool)
                out = np.zeros(masks.shape[0], dtype=np.float64)
                live = np.where(masks.any(axis=1))[0]  # all-masked rows stay 0.0
                for start in range(0, len(live), MODEL_MASK_BATCH):
                    idx = live[start:start + MODEL_MASK_BATCH]
                    keep = torch.from_numpy(masks[idx].astype(np.int64)).to(device)
                    ids = q_ids_full.repeat(len(idx), 1)
                    ids = torch.where(keep.bool(), ids, torch.full_like(ids, pad_id))
                    attn = q_mask_full.repeat(len(idx), 1) * keep
                    pooled = forward_pooled_batch(model, ids, attn, layer)
                    if variant == "abtt":
                        pooled = abtt_clean(pooled, pcs, mean_vec)
                    num = pooled @ cap
                    den = pooled.norm(dim=1).clamp(min=1e-12) * cap_norm
                    out[idx] = (num / den).double().cpu().numpy()
                return out
            return eval_masked_cos_many
        eval_masked_cos_many = make_batch_eval()

        def make_prefix_curves():
            def prefix_curves(order: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
                """Input-level twin of ``HiddenPairEvaluator.prefix_curves``.

                Builds the 2(n-1) interior masks of the k-sweep and evaluates
                them in batches. Endpoints are pinned by ``PairContext.curves``.
                """
                order = np.asarray(order, dtype=np.int64)
                keep_masks = np.zeros((n_q + 1, n_q), dtype=bool)
                for k in range(1, n_q + 1):
                    keep_masks[k, order[:k]] = True
                stacked = np.concatenate([keep_masks[1:n_q], ~keep_masks[1:n_q]], axis=0)
                vals = eval_masked_cos_many(stacked)
                keep = np.zeros(n_q + 1, dtype=np.float64)
                drop = np.zeros(n_q + 1, dtype=np.float64)
                keep[1:n_q] = vals[: n_q - 1]
                drop[1:n_q] = vals[n_q - 1:]
                return keep, drop
            return prefix_curves
        prefix_curves = make_prefix_curves()

        full_cos = eval_masked_cos(np.ones(n_q, dtype=np.int64))

        # Single-token leave-one-out cache: shared across all methods for this
        # variant. Deliberately still one forward per mask: the published
        # summary.csv was produced this way, and batching would change the
        # matmul reduction order and so the last bits of every legacy number.
        # It is n forwards against the curves' 2n(n-1), so nothing is gained.
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
            prefix_curves=prefix_curves,
        )

        rows.extend(_method_rows(data, ctx, variant, n_q, n_c, methods_present, opts))
        rows.extend(_pseudo_baseline_rows(data, ctx, variant, n_q, opts))

    return rows


def _eval_one(ctx: PairContext, method: str, scores: np.ndarray,
              opts: MetricOptions) -> dict:
    """Run the selected registered metrics on one (ctx, scores). Flat row out."""
    row: dict = {"method": method, "variant": ctx.variant,
                 "n_q": ctx.n_q, "full_cos": ctx.full_cos}
    selected = opts.selected()
    for name, fn in METRIC_REGISTRY.items():
        if name not in selected:
            continue
        if name in ("sufficiency", "comprehensiveness"):
            row.update(fn(ctx, scores, fractions=opts.fractions,
                          compute_aopc=opts.compute_aopc))
        elif name == "compactness":
            row.update(fn(ctx, scores, thresholds=opts.compactness_thresholds))
        elif name in ("deletion_auc", "insertion_auc"):
            row.update(fn(ctx, scores, random_draws=opts.random_order_draws))
        elif name == "randomization_check":
            row.update(fn(ctx, scores, shuffle_draws=opts.shuffle_draws,
                          random_draws=opts.random_order_draws))
        else:
            row.update(fn(ctx, scores))
    return row


# ---------------------------------------------------------------------------
# Aggregation + LaTeX
# ---------------------------------------------------------------------------
def _pct_label(frac: float) -> str:
    return str(int(round(frac * 100)))


def _latex_float(value: float) -> str:
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    if abs(exponent) < 3:
        return f"{value:g}"
    coeff = value / (10 ** exponent)
    if np.isclose(coeff, 1.0):
        return rf"10^{{{exponent}}}"
    return rf"{coeff:g}\times 10^{{{exponent}}}"


def _metric_key(kind: str, value: float) -> str:
    if kind == "suff":
        return f"suff@{value:.2f}_ratio"
    if kind == "comp":
        return f"comp@{value:.2f}_ratio"
    if kind == "compactness":
        return f"compactness@{value:.2f}"
    raise ValueError(f"unknown metric kind: {kind!r}")


def headline_metric_keys(headline_fraction: float,
                         headline_compactness_threshold: float) -> Tuple[str, ...]:
    """Global headline columns. The same values apply to every model/method."""
    return (
        "loo_rho",
        _metric_key("suff", headline_fraction),
        _metric_key("comp", headline_fraction),
        _metric_key("compactness", headline_compactness_threshold),
    )


def metric_label(metric_key: str) -> str:
    if metric_key == "loo_rho":
        return r"$\rho_{\text{LOO}}$~$\uparrow$"
    if metric_key.startswith("suff@") and metric_key.endswith("_ratio"):
        frac = float(metric_key.split("@", 1)[1].split("_", 1)[0])
        return rf"Suff@{_pct_label(frac)}\%~$\uparrow$"
    if metric_key.startswith("comp@") and metric_key.endswith("_ratio"):
        frac = float(metric_key.split("@", 1)[1].split("_", 1)[0])
        return rf"Comp@{_pct_label(frac)}\%~$\uparrow$"
    if metric_key.startswith("compactness@"):
        threshold = float(metric_key.split("@", 1)[1])
        return rf"MinFrac@{threshold:.2f}~$\downarrow$"
    return metric_key


def higher_is_better(metric_key: str) -> bool:
    return not metric_key.startswith("compactness@")


def required_result_keys(fractions: Sequence[float],
                         compactness_thresholds: Sequence[float],
                         metric_names: Optional[Sequence[str]] = None,
                         compute_aopc: bool = False) -> set[str]:
    """Keys a cached per-pair JSON must carry to be reusable under --skip_existing.

    ``metric_names`` defaults to the legacy four so an old cache written before
    the curve metrics existed still validates when only those are requested.
    """
    selected = set(metric_names) if metric_names is not None else set(LEGACY_METRICS)
    keys: set[str] = set()
    if "loo_correlation" in selected:
        keys.update({"loo_rho", "loo_p", "loo_n_used", "loo_n_total"})
    if "sufficiency" in selected:
        for frac in fractions:
            keys.update({f"suff@{frac:.2f}_raw", f"suff@{frac:.2f}_ratio"})
        if compute_aopc:
            keys.update({"suff_aopc_raw", "suff_aopc_ratio"})
    if "comprehensiveness" in selected:
        for frac in fractions:
            keys.update({f"comp@{frac:.2f}_drop", f"comp@{frac:.2f}_ratio"})
        if compute_aopc:
            keys.update({"comp_aopc_drop", "comp_aopc_ratio"})
    if "compactness" in selected:
        for threshold in compactness_thresholds:
            keys.add(f"compactness@{threshold:.2f}")
    if "kendall_tau_loo" in selected:
        keys.update({"loo_tau", "loo_tau_p", "loo_tau_n_used"})
    if "aopc" in selected:
        keys.update({"aopc_suff_raw", "aopc_suff_ratio",
                     "aopc_comp_raw", "aopc_comp_ratio"})
    if "deletion_auc" in selected:
        keys.update({"del_auc", "del_auc_random", "del_auc_gap"})
    if "insertion_auc" in selected:
        keys.update({"ins_auc", "ins_auc_random", "ins_auc_gap"})
    if "randomization_check" in selected:
        for base in RANDOMIZATION_CHECK_KEYS:
            keys.update({f"rand_{base}", f"rand_{base}_gap"})
    return keys
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


def aggregate_sweep_long(summary: pd.DataFrame,
                         fractions: Sequence[float],
                         compactness_thresholds: Sequence[float]) -> pd.DataFrame:
    """Return a tidy sweep summary from the wide aggregate table."""
    rows: List[dict] = []
    specs: List[Tuple[str, str, float | None]] = [
        ("rho_LOO", "loo_rho", None),
    ]
    for frac in fractions:
        specs.append(("sufficiency", f"suff@{frac:.2f}_ratio", float(frac)))
    for frac in fractions:
        specs.append(("comprehensiveness", f"comp@{frac:.2f}_ratio", float(frac)))
    for threshold in compactness_thresholds:
        specs.append(("min_recovery_fraction", f"compactness@{threshold:.2f}", float(threshold)))
    # Threshold-free additions. They carry threshold=None because there is no
    # grid to sweep; a consumer filtering on `metric` still gets them.
    specs.extend([
        ("kendall_tau_loo", "loo_tau", None),
        ("aopc_sufficiency", "aopc_suff_ratio", None),
        ("aopc_comprehensiveness", "aopc_comp_ratio", None),
        ("deletion_auc", "del_auc", None),
        ("deletion_auc_random", "del_auc_random", None),
        ("deletion_auc_gap", "del_auc_gap", None),
        ("insertion_auc", "ins_auc", None),
        ("insertion_auc_random", "ins_auc_random", None),
        ("insertion_auc_gap", "ins_auc_gap", None),
        ("randomization_rho_gap", "rand_loo_rho_gap", None),
        ("randomization_tau_gap", "rand_loo_tau_gap", None),
        ("randomization_aopc_suff_gap", "rand_aopc_suff_ratio_gap", None),
        ("randomization_aopc_comp_gap", "rand_aopc_comp_ratio_gap", None),
        ("randomization_ins_auc_gap_gap", "rand_ins_auc_gap_gap", None),
        ("randomization_del_auc_gap_gap", "rand_del_auc_gap_gap", None),
    ])

    for _, src in summary.iterrows():
        base = {
            "model": src["model"],
            "method": src["method"],
            "variant": src["variant"],
            "n": int(src["n"]),
            "n_q_mean": float(src["n_q_mean"]),
            "full_cos_mean": float(src["full_cos_mean"]),
        }
        for metric, key, threshold in specs:
            mean_col = f"{key}_mean"
            if mean_col not in summary.columns:
                continue
            rows.append({
                **base,
                "metric": metric,
                "metric_key": key,
                "threshold": threshold,
                "mean": float(src[mean_col]) if pd.notna(src[mean_col]) else float("nan"),
                "std": float(src[f"{key}_std"]) if f"{key}_std" in summary.columns and pd.notna(src[f"{key}_std"]) else float("nan"),
                "se": float(src[f"{key}_se"]) if f"{key}_se" in summary.columns and pd.notna(src[f"{key}_se"]) else float("nan"),
                "metric_n": int(src[f"{key}_n"]) if f"{key}_n" in summary.columns and pd.notna(src[f"{key}_n"]) else 0,
            })
    return pd.DataFrame(rows)


def render_sweep_latex(summary: pd.DataFrame, out_path: Path,
                       fractions: Sequence[float],
                       compactness_thresholds: Sequence[float]) -> None:
    """Render an appendix-oriented table exposing every threshold column."""
    metric_keys: List[str] = ["loo_rho"]
    metric_keys.extend(_metric_key("suff", frac) for frac in fractions)
    metric_keys.extend(_metric_key("comp", frac) for frac in fractions)
    metric_keys.extend(_metric_key("compactness", threshold) for threshold in compactness_thresholds)

    def fmt(v: float) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "--"
        return f"{v:.3f}"

    lines: List[str] = []
    lines.append("% Auto-generated by scripts/ig/run_attribution_metrics.py")
    lines.append("% Edit the script, not this file.")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{2pt}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    colspec = "l" + "rr" * len(metric_keys)
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\toprule")

    h1 = ["Method"]
    for key in metric_keys:
        h1.append(rf"\multicolumn{{2}}{{c}}{{{metric_label(key)}}}")
    lines.append(" & ".join(h1) + r" \\")
    cmid_parts = []
    for i, _ in enumerate(metric_keys):
        a = 2 + 2 * i
        b = 3 + 2 * i
        cmid_parts.append(rf"\cmidrule(lr){{{a}-{b}}}")
    lines.append(" ".join(cmid_parts))
    h2 = [""]
    for _ in metric_keys:
        h2.extend(["base", "abtt"])
    lines.append(" & ".join(h2) + r" \\")
    lines.append(r"\midrule")

    seen_models = list(dict.fromkeys(summary["model"].tolist()))
    for model in seen_models:
        short = MODEL_SHORT.get(model, model)
        lines.append(rf"\multicolumn{{{1 + 2 * len(metric_keys)}}}{{l}}{{\textit{{{short}}}}} \\")
        sub = summary[summary["model"] == model]
        for method in METHOD_DISPLAY_ORDER:
            if method not in sub["method"].values:
                continue
            row_cells = [METHOD_LABELS.get(method, method)]
            for key in metric_keys:
                col_mean = f"{key}_mean"
                for variant in ("baseline", "abtt"):
                    cell = sub[(sub["method"] == method) & (sub["variant"] == variant)]
                    if cell.empty or col_mean not in cell.columns:
                        row_cells.append("--")
                    else:
                        row_cells.append(fmt(float(cell.iloc[0][col_mean])))
            lines.append(" & ".join(row_cells) + r" \\")
        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    fraction_bits = ", ".join(f"{_pct_label(frac)}\\%" for frac in fractions)
    threshold_bits = ", ".join(f"{threshold:.2f}" for threshold in compactness_thresholds)
    lines.append(
        r"\caption{Full attribution metric threshold sweep. "
        r"$\rho_{\text{LOO}}$ is threshold-free and is reported once; "
        rf"Sufficiency and Comprehensiveness are shown at $\{{{fraction_bits}}}$; "
        rf"MinFrac is shown at $\tau \in \{{{threshold_bits}}}$ "
        r"(stored internally as compactness for compatibility). Higher is better for "
        r"$\rho_{\text{LOO}}$, Sufficiency, and Comprehensiveness; lower is better for MinFrac. "
        rf"Ratio metrics use $|S_v(q_{{\text{{full}}}}, c)| \ge {_latex_float(FULL_COS_FLOOR)}$ and "
        rf"$\rho_{{\text{{LOO}}}}$ excludes tokens with $|\Delta| < {_latex_float(LOO_NOISE_FLOOR)}$.}}"
    )
    lines.append(r"\label{tab:attribution_metrics_sweep}")
    lines.append(r"\end{table*}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


def render_latex(summary: pd.DataFrame, out_path: Path,
                 headline_fraction: float = 0.25,
                 headline_compactness_threshold: float = 0.80) -> None:
    """Render the methods × {baseline | abtt} table grouped by model.

    Uses booktabs + multicolumn. Bolds the best method per (metric, variant)
    column within each model block. Direction-aware: higher-is-better for
    sufficiency, comprehensiveness, loo_rho; lower-is-better for MinFrac.
    """
    headline_keys = headline_metric_keys(headline_fraction, headline_compactness_threshold)

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
    colspec = "l" + "rr" * len(headline_keys)
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\toprule")

    # Header row 1: metric group names spanning 2 cols each.
    h1 = ["Method"]
    for k in headline_keys:
        h1.append(rf"\multicolumn{{2}}{{c}}{{{metric_label(k)}}}")
    lines.append(" & ".join(h1) + r" \\")

    # Header row 2: baseline / abtt under each.
    cmid_parts = []
    for i, _ in enumerate(headline_keys):
        # Method col is col 1; metric i has cols 2+2i and 3+2i.
        a = 2 + 2 * i
        b = 3 + 2 * i
        cmid_parts.append(rf"\cmidrule(lr){{{a}-{b}}}")
    lines.append(" ".join(cmid_parts))
    h2 = [""]
    for _ in headline_keys:
        h2.append("base")
        h2.append("abtt")
    lines.append(" & ".join(h2) + r" \\")
    lines.append(r"\midrule")

    # One block per model, ordered to match phase12f.
    seen_models = list(dict.fromkeys(summary["model"].tolist()))
    for model in seen_models:
        short = MODEL_SHORT.get(model, model)
        lines.append(rf"\multicolumn{{{1 + 2 * len(headline_keys)}}}{{l}}{{\textit{{{short}}}}} \\")

        sub = summary[summary["model"] == model]
        # Determine bolding: best per (metric, variant) over real methods only.
        real_methods = [m for m in METHOD_DISPLAY_ORDER
                        if m in sub["method"].values and m not in ("random", "inverse")]
        order_key = {m: i for i, m in enumerate(METHOD_DISPLAY_ORDER)}
        best_idx: Dict[Tuple[str, str], str] = {}
        for k in headline_keys:
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
                if higher_is_better(k):
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
            for k in headline_keys:
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
        r"For each model $\times$ \{baseline, ABTT\} $\times$ method we report one "
        r"global headline threshold choice: "
        rf"\textbf{{Suff@{_pct_label(headline_fraction)}\%}}: "
        rf"$S_v(q_{{\text{{top-{_pct_label(headline_fraction)}\%}}}}, c) / "
        r"S_v(q_{\text{full}}, c)$ "
        rf"(higher is better); restricted to pairs with $|S_v(q_{{\text{{full}}}}, c)| \ge {_latex_float(FULL_COS_FLOOR)}$ "
        r"to avoid small-denominator instability. "
        rf"\textbf{{Comp@{_pct_label(headline_fraction)}\%}}: relative drop when the top "
        rf"{_pct_label(headline_fraction)}\% of tokens are masked "
        r"(higher is better). "
        rf"\textbf{{MinFrac@{headline_compactness_threshold:.2f}}}: smallest token fraction that recovers "
        rf"$\ge {headline_compactness_threshold:.2f} \cdot S_v(q_{{\text{{full}}}}, c)$ "
        r"(lower is better; internally stored as compactness for compatibility). "
        r"\textbf{$\rho_{\text{LOO}}$}: Spearman correlation between $|a|$ and "
        r"single-token leave-one-out $\Delta\cos$ "
        rf"(higher is better; tokens with $|\Delta| < {_latex_float(LOO_NOISE_FLOOR)}$ excluded as noise). "
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
        r"Per-pair rows, wide summary columns, and the long sweep summary expose "
        r"the full threshold grid."
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
    p.add_argument("--compactness_thresholds",
                   default=",".join(f"{t:.2f}" for t in DEFAULT_COMPACTNESS_THRESHOLDS),
                   help="Comma-separated thresholds for MinFrac/compactness sweeps.")
    p.add_argument("--compactness_threshold", type=float, default=None,
                   help="Backward-compatible single-threshold alias. When provided, "
                        "it overrides --compactness_thresholds.")
    p.add_argument("--headline_fraction", type=float, default=0.25,
                   help="Global Suff/Comp fraction used in the headline TEX table.")
    p.add_argument("--headline_compactness_threshold", type=float, default=0.80,
                   help="Global MinFrac threshold used in the headline TEX table.")
    p.add_argument("--sweep_tex_out", default=None,
                   help="Optional appendix-style TEX table exposing the full sweep.")
    p.add_argument("--max_pairs_per_model", type=int, default=None)
    p.add_argument("--half_precision", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--device", default=None,
                   help="Override device (default: cuda if available else cpu).")
    p.add_argument("--dry_run", action="store_true",
                   help="List planned work and exit; do no forward passes.")
    p.add_argument("--dry_run_require_existing", action="store_true",
                   help="With --dry_run, fail if planned artifact NPZs are missing.")
    p.add_argument("--require_artifacts", action="store_true",
                   help="Fail during metric computation if any planned artifact NPZ is missing.")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip per-pair JSONs that already exist on disk.")
    p.add_argument("--render_only", action="store_true",
                   help="Skip metric computation; rebuild summary + TEX from existing per-pair JSONs.")
    p.add_argument("--compute_aopc", "--aopc", dest="compute_aopc", action="store_true",
                   help="Also emit the suff_aopc_*/comp_aopc_* columns from the "
                        "sufficiency and comprehensiveness metrics. Cheap under "
                        "--backend hidden; ~2*n_q extra forward passes per method "
                        "per variant under --backend model, hence off by default. "
                        "The registered `aopc` metric reports the same numbers.")
    p.add_argument("--backend", choices=("model", "hidden"), default="model",
                   help="'model' re-runs the encoder with PAD in the masked positions "
                        "(the original, GPU-shaped path). 'hidden' recomputes masked "
                        "cosines from the hidden states cached in each NPZ: CPU-only, "
                        "no model download, and cheap enough for the full k=1..n "
                        "curve metrics. The two use different erasure operators and "
                        "their outputs must not be pooled into one table.")
    p.add_argument("--metrics", default=None,
                   help="Comma-separated registry metric names, or 'legacy' "
                        f"({', '.join(LEGACY_METRICS)}) or 'all'. Default: 'legacy' "
                        "for --backend model, 'all' for --backend hidden.")
    p.add_argument("--out_subdir", default="",
                   help="Optional subdirectory of --out_root for the per-pair JSONs "
                        "and summaries, so a second parameterisation does not "
                        "overwrite an existing per-pair cache.")
    p.add_argument("--summary_name", default="summary.csv",
                   help="Filename for the wide per (model, method, variant) summary.")
    p.add_argument("--sweep_summary_name", default="summary_sweep_long.csv",
                   help="Filename for the long/tidy sweep summary.")
    p.add_argument("--summary_out", default=None,
                   help="Explicit path for the wide summary, overriding "
                        "--out_root/--out_subdir/--summary_name. Use it to keep the "
                        "per-pair JSON cache in its own subdirectory while the "
                        "summary lands next to an existing one.")
    p.add_argument("--sweep_summary_out", default=None,
                   help="Explicit path for the long sweep summary; see --summary_out.")
    p.add_argument("--random_order_draws", type=int, default=DEFAULT_RANDOM_ORDER_DRAWS,
                   help="Random orderings averaged for the deletion/insertion AUC reference.")
    p.add_argument("--skip_pseudo_baselines", action="store_true",
                   help="Skip the synthetic 'random' and 'inverse' rows. They "
                        "cost six extra scored methods per (pair, variant), "
                        "which the model backend cannot afford for a curve "
                        "metric. Never use this for a table the memo quotes: "
                        "criterion 4 is read off the 'random' rows.")
    p.add_argument("--shuffle_draws", type=int, default=DEFAULT_SHUFFLE_DRAWS,
                   help="Attribution shuffles averaged for the randomization check.")
    return p.parse_args()


def resolve_metric_names(raw: Optional[str], backend: str) -> Tuple[str, ...]:
    """Turn --metrics into a concrete tuple of registry names."""
    if raw is None:
        raw = "legacy" if backend == "model" else "all"
    raw = raw.strip()
    if raw == "all":
        names: Tuple[str, ...] = tuple(METRIC_REGISTRY)
    elif raw == "legacy":
        names = LEGACY_METRICS
    else:
        names = tuple(x.strip() for x in raw.split(",") if x.strip())
    unknown = [n for n in names if n not in METRIC_REGISTRY]
    if unknown:
        raise SystemExit(
            f"unknown metric(s) {unknown}; registered: {sorted(METRIC_REGISTRY)}"
        )
    if backend == "model":
        pricey = [n for n in names if n in CURVE_METRICS]
        if pricey:
            print(
                f"WARNING: {pricey} sweep the whole k=1..n curve, which costs O(n) "
                f"masked encoder evaluations per ordering ({MODEL_MASK_BATCH} masks "
                "per forward) under --backend model. --backend hidden is the intended "
                "path for a full table; the model backend is for bounded "
                "erasure-operator spot checks, and its numbers must not be mixed with "
                "hidden-backend numbers in one table.",
                flush=True,
            )
    return names


def parse_float_tuple(raw: str, *, name: str) -> Tuple[float, ...]:
    vals = tuple(float(x.strip()) for x in raw.split(",") if x.strip())
    if not vals:
        raise ValueError(f"{name} must contain at least one value")
    return vals


def main() -> None:
    args = parse_args()
    fractions = parse_float_tuple(args.sufficiency_fractions, name="--sufficiency_fractions")
    if args.compactness_threshold is not None:
        compactness_thresholds = (float(args.compactness_threshold),)
    else:
        compactness_thresholds = parse_float_tuple(args.compactness_thresholds, name="--compactness_thresholds")
    metric_names = resolve_metric_names(args.metrics, args.backend)
    opts = MetricOptions(
        fractions=fractions,
        compactness_thresholds=compactness_thresholds,
        compute_aopc=args.compute_aopc,
        metric_names=metric_names,
        random_order_draws=args.random_order_draws,
        shuffle_draws=args.shuffle_draws,
        skip_pseudo_baselines=args.skip_pseudo_baselines,
    )
    required_keys = required_result_keys(
        fractions, compactness_thresholds, metric_names, args.compute_aopc,
    )
    needs_model = args.backend == "model" and not args.dry_run and not args.render_only
    if needs_model:
        torch_mod, _, _, _ = _require_torch()
        device = args.device or ("cuda" if torch_mod.cuda.is_available() else "cpu")
    else:
        device = args.device or "cpu"
    print(f"backend={args.backend} device={device} metrics={list(metric_names)}", flush=True)

    examples = pd.read_csv(args.examples_csv)
    if args.models:
        examples = examples[examples["model_name"].isin(args.models)]
    examples = examples.copy()
    examples["slug"] = examples["model_name"].apply(model_slug)
    examples = examples.sort_values(["model_name", "example_id"]).reset_index(drop=True)

    artifacts_root = Path(args.artifacts_root)
    out_root = Path(args.out_root)
    if args.out_subdir:
        out_root = out_root / args.out_subdir
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
                if (args.dry_run_require_existing or args.require_artifacts) and not npz_path.exists():
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
            model = tokenizer = None
            if args.backend == "model":
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
                    msg = f"missing NPZ: {npz_path}"
                    if args.require_artifacts:
                        raise FileNotFoundError(msg)
                    print(f"  [skip] {msg}")
                    continue
                json_path = slug_dir / f"{example_tag}.json"
                if args.skip_existing and json_path.exists():
                    with open(json_path) as f:
                        cached = json.load(f)
                    if cached and all(required_keys <= set(row) for row in cached):
                        print(f"  [skip-existing] {example_tag}")
                        all_rows.extend(cached)
                        continue
                    print(f"  [recompute] {example_tag}: cached JSON lacks requested sweep keys")
                t_pair = time.time()
                if args.backend == "hidden":
                    rows = process_pair_hidden(npz_path, args.methods, opts)
                else:
                    rows = process_pair(
                        npz_path, model, tokenizer, device,
                        method_filter=args.methods, opts=opts,
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
            if args.backend == "model" and device == "cuda":
                torch.cuda.empty_cache()
    else:
        # Load existing per-pair JSONs.
        for json_path in sorted(out_root.glob("*/*.json")):
            cached = json.loads(json_path.read_text())
            if args.require_artifacts and cached and not all(required_keys <= set(row) for row in cached):
                raise ValueError(f"cached JSON lacks requested sweep keys: {json_path}")
            all_rows.extend(cached)
        print(f"Loaded {len(all_rows)} cached rows from {out_root}")

    if not all_rows:
        print("No rows produced; bailing.")
        return

    summary = aggregate(all_rows)
    summary_csv = Path(args.summary_out) if args.summary_out else out_root / args.summary_name
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    print(f"Wrote {summary_csv} ({len(summary)} rows)")

    summary_long = aggregate_sweep_long(summary, fractions, compactness_thresholds)
    summary_long_csv = (
        Path(args.sweep_summary_out) if args.sweep_summary_out
        else out_root / args.sweep_summary_name
    )
    summary_long_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_long.to_csv(summary_long_csv, index=False)
    print(f"Wrote {summary_long_csv} ({len(summary_long)} rows)")

    if not args.tex_out:
        print("--tex_out empty: skipping LaTeX rendering.")
        return

    render_latex(
        summary,
        Path(args.tex_out),
        headline_fraction=args.headline_fraction,
        headline_compactness_threshold=args.headline_compactness_threshold,
    )
    if args.sweep_tex_out:
        render_sweep_latex(summary, Path(args.sweep_tex_out), fractions, compactness_thresholds)


if __name__ == "__main__":
    main()
