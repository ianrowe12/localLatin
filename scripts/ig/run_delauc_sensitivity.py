"""Sensitivity of the chance-corrected deletion AUC gap to its own knobs.

Issue #195 (G9), part of epic #109. The second main-table attribution column,
``DelAUC gap = mean(random-order deletion AUC) - attribution-order deletion
AUC``, is a constructed statistic with at least five free choices behind it.
Prof. Siddique's question at the 2026-09-14 meeting was whether the very
uneven base-versus-ABTT gaps in Table 4 (0.842 vs 0.180 in one cell, 0.042 vs
0.464 in another) are a property of the representations or of those choices.

This script answers that by recomputing the metric over the *same* stored
hidden states as the published run, one knob at a time plus a small
interaction block, and reporting the paired ABTT-minus-baseline difference per
cell under every setting. It is a sensitivity analysis: the predeclared
setting (:data:`PREDECLARED`) is fixed by
``docs/research/attribution_metrics_decision.md`` and is never chosen here on
the basis of its result.

Everything runs on the ``hidden`` erasure operator over the NPZs in
``runs/active/ig_examples_200pos_v1/artifacts/``: CPU only, no model, no GPU.

Outputs (under ``--out_dir``):

  ``per_pair.csv``      one row per (config, model, view, pair): the two
                        variants' gap values and their difference
  ``cells.csv``         one row per (config, model, view): paired mean, SE,
                        SE ratio, verdict, both variants' cell means
  ``configs.csv``       the knob table, one row per configuration
  ``verification.json`` predeclared-config agreement with the published
                        per-pair cache

Usage:

    python scripts/ig/run_delauc_sensitivity.py \\
        --run_dir runs/active/ig_examples_200pos_v1 \\
        --out_dir runs/active/ig_examples_200pos_v1/attribution_metrics/sensitivity \\
        --verify_cache runs/active/ig_examples_200pos_v1/attribution_metrics/v2_hidden
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ig"))

from attribution_metrics import (  # noqa: E402
    FULL_COS_FLOOR,
    METHOD_SCORE_REDUCER,
    RANDOM_ORDER_SEED,
    REDUCER_FALLBACK,
    DEFAULT_RANDOM_ORDER_DRAWS,
    rank_order,
    scores_from_pair_matrix,
)
from token_filtering import build_token_keep_lookup  # noqa: E402
from run_attribution_metrics import keep_positions, model_slug  # noqa: E402

# The two views the main table reports. ``ig`` is Integrated Gradients, and
# ``retrieval_mark`` is the retrieval-adapted MaRC mask.
VIEWS: Tuple[str, ...] = ("ig", "retrieval_mark")
VARIANTS: Tuple[str, ...] = ("baseline", "abtt")
# The paper's caption rule: a cell whose paired difference is inside two
# standard errors of zero is a tie, not a win for either side.
TIE_SE: float = 2.0


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DelAucConfig:
    """One setting of every knob the deletion-gap metric has.

    ``knob`` names the single axis this configuration moves away from
    :data:`PREDECLARED` ("-" for the predeclared setting itself, "mixed" for
    the interaction block). ``pooling_matches_generator`` is False when the
    token filter differs from the one the artifacts were generated with, in
    which case the pooled vector is not the vector the ABTT components were fit
    on and the arm is a diagnostic rather than a candidate setting.
    """

    name: str
    knob: str
    schedule: str = "every_token"
    erasure: str = "drop"
    draws: int = DEFAULT_RANDOM_ORDER_DRAWS
    seed: int = RANDOM_ORDER_SEED
    token_filter: str = "tokenizer_empty"
    side: str = "query"

    @property
    def pooling_matches_generator(self) -> bool:
        return self.token_filter == "tokenizer_empty"


# The predeclared main-table setting. Every field here is the value the
# published run used, and none of them is revisited by this script.
PREDECLARED = DelAucConfig(name="predeclared", knob="-")


def build_configs() -> List[DelAucConfig]:
    """The sweep: one-factor-at-a-time around :data:`PREDECLARED`, then a small
    interaction block over the two knobs that change the operator itself.

    Twenty configurations. The budget matters because every configuration is
    600 pairs x 2 variants x 2 views, and because a sweep large enough to
    contain a flattering cell by chance is not a sensitivity analysis.
    """
    cfgs: List[DelAucConfig] = [PREDECLARED]
    # Knob 1: deletion step schedule. The published curve evaluates every k
    # from 0 to n; a coarser grid is what most deletion-AUC papers use.
    for frac in ("0.05", "0.10", "0.20"):
        cfgs.append(DelAucConfig(name=f"sched_frac{frac}", knob="schedule",
                                 schedule=f"frac_{frac}"))
    # Knob 2: what a deleted token is replaced with.
    cfgs.append(DelAucConfig(name="erase_zero", knob="erasure", erasure="zero"))
    cfgs.append(DelAucConfig(name="erase_centroid", knob="erasure", erasure="centroid"))
    # Knob 3: Monte-Carlo size of the chance correction.
    for draws in (1, 20, 50):
        cfgs.append(DelAucConfig(name=f"draws{draws}", knob="draws", draws=draws))
    # Knob 4: the seed behind those draws, at the published draw count.
    for seed in (20260101, 7):
        cfgs.append(DelAucConfig(name=f"seed{seed}", knob="seed", seed=seed))
    # Knob 5: the token filter. Both alternatives pool a different vector than
    # the generator pooled and the components were fit on, so they are
    # diagnostics; see the memo.
    for tf in ("all", "no_empty"):
        cfgs.append(DelAucConfig(name=f"filter_{tf}", knob="token_filter", token_filter=tf))
    # Knob 6: which side of the pair is erased.
    cfgs.append(DelAucConfig(name="side_both", knob="side", side="both"))
    # Interaction block: schedule x erasure x side, the three knobs that change
    # the operator rather than the reference.
    cfgs.append(DelAucConfig(name="sched0.10_zero", knob="mixed",
                             schedule="frac_0.10", erasure="zero"))
    cfgs.append(DelAucConfig(name="sched0.10_centroid", knob="mixed",
                             schedule="frac_0.10", erasure="centroid"))
    cfgs.append(DelAucConfig(name="sched0.10_both", knob="mixed",
                             schedule="frac_0.10", side="both"))
    cfgs.append(DelAucConfig(name="zero_both", knob="mixed",
                             erasure="zero", side="both"))
    cfgs.append(DelAucConfig(name="centroid_both", knob="mixed",
                             erasure="centroid", side="both"))
    cfgs.append(DelAucConfig(name="sched0.10_zero_both", knob="mixed",
                             schedule="frac_0.10", erasure="zero", side="both"))
    return cfgs


# ---------------------------------------------------------------------------
# The metric, with every knob exposed
# ---------------------------------------------------------------------------
def deletion_grid(n: int, schedule: str) -> np.ndarray:
    """Token counts k at which the deletion curve is evaluated.

    ``every_token`` is the published schedule: k = 0, 1, ..., n. ``frac_<f>``
    evaluates at the fractions 0, f, 2f, ..., 1 of the query, rounded to whole
    tokens and de-duplicated, so a short query silently falls back to a finer
    grid than requested rather than to a degenerate two-point curve.
    """
    if schedule == "every_token":
        return np.arange(n + 1, dtype=np.int64)
    if not schedule.startswith("frac_"):
        raise ValueError(f"unknown schedule: {schedule!r}")
    step = float(schedule[len("frac_"):])
    if not 0.0 < step <= 1.0:
        raise ValueError(f"schedule fraction must be in (0, 1]: {step}")
    fracs = np.arange(0.0, 1.0 + 1e-9, step)
    if fracs[-1] < 1.0:
        fracs = np.append(fracs, 1.0)
    ks = np.unique(np.rint(fracs * n).astype(np.int64))
    if ks[0] != 0:
        ks = np.insert(ks, 0, 0)
    if ks[-1] != n:
        ks = np.append(ks, n)
    return ks


def pooled_after_deletion(hidden: np.ndarray, order: np.ndarray, ks: np.ndarray,
                          erasure: str, replacement: np.ndarray) -> np.ndarray:
    """Pooled vectors after deleting the first ``k`` tokens of ``order``.

    Three erasure operators, all at the representation level:

    ``drop``      the deleted tokens leave the mean entirely, so the
                  denominator shrinks to ``n - k``. This is the published
                  operator, and the k = n point has no vector at all: it is
                  returned as the zero vector, matching the empty-query
                  convention ``PairContext.curves`` pins.
    ``zero``      the deleted tokens are replaced by the zero vector and stay
                  in the mean, so the denominator stays ``n``. The curve is the
                  ``drop`` curve scaled by ``(n - k) / n``, which is a pure
                  rescaling of each pooled vector and therefore leaves the
                  cosine unchanged under the baseline variant but *not* under
                  ABTT, where the mean subtraction is not scale-equivariant.
    ``centroid``  the deleted tokens are replaced by a fixed vector (the
                  corpus mean the ABTT cleaner was fit with), denominator
                  ``n``. The null point at k = n is the replacement vector
                  itself rather than an empty query.
    """
    n = hidden.shape[0]
    total = hidden.sum(axis=0)
    prefix = np.cumsum(hidden[order], axis=0)
    out = np.zeros((len(ks), hidden.shape[1]), dtype=np.float64)
    for i, k in enumerate(ks):
        k = int(k)
        remaining = total if k == 0 else total - prefix[k - 1]
        if erasure == "drop":
            out[i] = remaining / (n - k) if k < n else 0.0
        elif erasure == "zero":
            out[i] = remaining / n
        elif erasure == "centroid":
            out[i] = (remaining + k * replacement) / n
        else:
            raise ValueError(f"unknown erasure: {erasure!r}")
    return out


def curve_auc(values: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal AUC of ``values`` over ``x``, which need not be uniform."""
    if len(values) < 2:
        return float("nan")
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(values, x))


class PairEvaluator:
    """Deletion curves for one (pair, variant) under one erasure operator.

    Mirrors ``run_attribution_metrics.HiddenPairEvaluator`` in how it cleans and
    how it pools, and generalises it in the two directions this sweep needs: an
    arbitrary set of deletion points, and deletion applied to the candidate
    side as well as the query.
    """

    def __init__(self, q_hidden: np.ndarray, c_hidden: np.ndarray,
                 pcs: np.ndarray, mean_vec: np.ndarray, variant: str,
                 erasure: str, side: str) -> None:
        self.q = np.asarray(q_hidden, dtype=np.float64)
        self.c = np.asarray(c_hidden, dtype=np.float64)
        self.n_q = self.q.shape[0]
        self.n_c = self.c.shape[0]
        self.variant = variant
        self.erasure = erasure
        self.side = side
        self._pcs = np.asarray(pcs, dtype=np.float64)
        self._mean = np.asarray(mean_vec, dtype=np.float64)
        self._c_full = self._clean(self.c.mean(axis=0))
        self._c_full_norm = float(np.linalg.norm(self._c_full))

    def _clean(self, vecs: np.ndarray) -> np.ndarray:
        if self.variant != "abtt":
            return vecs
        centered = vecs - self._mean
        return centered - (centered @ self._pcs.T) @ self._pcs

    def _cos_rows(self, q_rows: np.ndarray, c_rows: Optional[np.ndarray]) -> np.ndarray:
        """Cosine per row, with the empty-input convention applied first.

        A row whose *raw* pooled vector is the zero vector is an empty query
        (or candidate), and its cosine is 0 by convention. The check has to be
        made before cleaning, because under ABTT ``_clean`` maps the zero
        vector to minus the corpus mean and would report a spurious cosine for
        an input that has no tokens left. ``PairContext.curves`` pins the same
        endpoint for the same reason.
        """
        cq = self._clean(q_rows)
        nq = np.linalg.norm(cq, axis=1)
        empty = np.linalg.norm(q_rows, axis=1) <= 0.0
        if c_rows is None:
            num = cq @ self._c_full
            denom = nq * self._c_full_norm
        else:
            cc = self._clean(c_rows)
            num = np.einsum("ij,ij->i", cq, cc)
            denom = nq * np.linalg.norm(cc, axis=1)
            empty = empty | (np.linalg.norm(c_rows, axis=1) <= 0.0)
        out = np.zeros(len(num), dtype=np.float64)
        ok = (denom > 1e-12) & ~empty
        out[ok] = num[ok] / denom[ok]
        return out

    @property
    def full_cos(self) -> float:
        q_full = self.q.mean(axis=0)[None, :]
        return float(self._cos_rows(q_full, None)[0])

    def drop_curve(self, q_order: np.ndarray, c_order: np.ndarray,
                   ks: np.ndarray) -> np.ndarray:
        """Cosine after deleting ``ks`` query tokens (and the matching fraction
        of candidate tokens when ``side == 'both'``)."""
        q_rows = pooled_after_deletion(self.q, q_order, ks, self.erasure, self._mean)
        if self.side == "query":
            return self._cos_rows(q_rows, None)
        if self.side != "both":
            raise ValueError(f"unknown side: {self.side!r}")
        # Match by fraction, not by count: the two sequences have different
        # lengths, and deleting the same *number* of tokens from a short
        # candidate and a long query is not the same intervention.
        ks_c = np.rint(ks / max(self.n_q, 1) * self.n_c).astype(np.int64)
        ks_c = np.clip(ks_c, 0, self.n_c)
        c_rows = pooled_after_deletion(self.c, c_order, ks_c, self.erasure, self._mean)
        return self._cos_rows(q_rows, c_rows)


def del_auc_gap(evaluator: PairEvaluator, q_scores: np.ndarray,
                c_scores: Optional[np.ndarray], cfg: DelAucConfig) -> Dict[str, float]:
    """``(random-order AUC) - (attribution-order AUC)``, higher is better.

    Undefined, as in production, when the full-query cosine is under
    :data:`FULL_COS_FLOOR`: the curve is normalised by it, so a near-zero
    denominator turns the ratio into noise.
    """
    full = evaluator.full_cos
    if abs(full) < FULL_COS_FLOOR:
        return {"del_auc": float("nan"), "del_auc_random": float("nan"),
                "del_auc_gap": float("nan"), "full_cos": full}
    ks = deletion_grid(evaluator.n_q, cfg.schedule)
    x = ks.astype(np.float64) / max(evaluator.n_q, 1)
    q_order = rank_order(q_scores)
    c_order = (rank_order(c_scores) if c_scores is not None
               else np.arange(evaluator.n_c, dtype=np.int64))
    attr = curve_auc(evaluator.drop_curve(q_order, c_order, ks) / full, x)
    rng = np.random.default_rng(cfg.seed)
    rand_vals = []
    for _ in range(cfg.draws):
        rq = rng.permutation(evaluator.n_q)
        # Only consume the stream for the candidate side when that side is
        # actually erased, so the query-only arms draw exactly the orderings
        # ``attribution_metrics._random_orders`` draws at the same seed.
        rc = (rng.permutation(evaluator.n_c) if evaluator.side == "both"
              else np.arange(evaluator.n_c, dtype=np.int64))
        rand_vals.append(curve_auc(evaluator.drop_curve(rq, rc, ks) / full, x))
    rand_mean = float(np.mean(rand_vals)) if rand_vals else float("nan")
    return {"del_auc": attr, "del_auc_random": rand_mean,
            "del_auc_gap": rand_mean - attr, "full_cos": full}


# ---------------------------------------------------------------------------
# Driving one NPZ through every configuration
# ---------------------------------------------------------------------------
def candidate_scores(data, view: str, variant: str, q_idx: np.ndarray,
                     c_idx: np.ndarray) -> np.ndarray:
    """Per-candidate-token scores, the column-wise twin of the query reducer.

    Only the ``side == 'both'`` arm needs these. IG stores a per-token vector
    for the candidate exactly as it does for the query; every other view is
    reduced from the pair matrix along the query axis.
    """
    pm = np.asarray(data[f"pair_matrix_{view}_{variant}"], dtype=np.float64)[np.ix_(q_idx, c_idx)]
    stored_key = f"candidate_ig_{variant}"
    if view == "ig" and stored_key in data.files:
        return np.asarray(data[stored_key], dtype=np.float64)[c_idx]
    reducer = METHOD_SCORE_REDUCER.get(view, REDUCER_FALLBACK)
    if reducer == "row_max":
        return pm.max(axis=0)
    if reducer == "row_sum_positive":
        return np.where(pm > 0, pm, 0.0).sum(axis=0)
    return pm.sum(axis=0)


def query_scores(data, view: str, variant: str, q_idx: np.ndarray,
                 c_idx: np.ndarray) -> np.ndarray:
    pm = np.asarray(data[f"pair_matrix_{view}_{variant}"], dtype=np.float64)
    stored = None
    if view == "ig" and f"query_ig_{variant}" in data.files:
        stored = np.asarray(data[f"query_ig_{variant}"], dtype=np.float64)[q_idx]
    reducer = METHOD_SCORE_REDUCER.get(view, REDUCER_FALLBACK)
    return scores_from_pair_matrix(pm[np.ix_(q_idx, c_idx)], stored, reducer)


def process_npz(npz_path: Path, configs: Sequence[DelAucConfig],
                keep_lookups: Dict[str, Optional[np.ndarray]]) -> List[dict]:
    """One row per (config, view, variant) for this pair."""
    data = np.load(npz_path)
    n_q = int(data["query_attention_mask"].sum())
    n_c = int(data["candidate_attention_mask"].sum())
    pcs = data["pcs"]
    mean_vec = data["mean_vec"]
    rows: List[dict] = []
    # Configs sharing a token filter share their pooled token set, and the
    # filter is the only knob that changes which rows are pooled at all.
    by_filter: Dict[str, List[DelAucConfig]] = {}
    for cfg in configs:
        by_filter.setdefault(cfg.token_filter, []).append(cfg)
    for token_filter, cfgs in by_filter.items():
        lookup = keep_lookups.get(token_filter)
        q_idx = keep_positions(data["query_input_ids"], n_q, lookup)
        c_idx = keep_positions(data["candidate_input_ids"], n_c, lookup)
        if len(q_idx) == 0 or len(c_idx) == 0:
            continue
        q_hidden = data["query_hidden"][q_idx]
        c_hidden = data["candidate_hidden"][c_idx]
        for variant in VARIANTS:
            scores = {v: query_scores(data, v, variant, q_idx, c_idx) for v in VIEWS}
            c_scores = {v: candidate_scores(data, v, variant, q_idx, c_idx) for v in VIEWS}
            for cfg in cfgs:
                ev = PairEvaluator(q_hidden, c_hidden, pcs, mean_vec, variant,
                                   cfg.erasure, cfg.side)
                for view in VIEWS:
                    res = del_auc_gap(
                        ev, scores[view],
                        c_scores[view] if cfg.side == "both" else None, cfg)
                    rows.append({
                        "config": cfg.name, "view": view, "variant": variant,
                        "n_q": len(q_idx), "n_c": len(c_idx), **res,
                    })
    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def verdict(mean: float, se: float, tie_se: float = TIE_SE) -> str:
    """ABTT win / tie / baseline win under the paper's 2-SE caption rule."""
    if not np.isfinite(mean) or not np.isfinite(se) or se <= 0:
        return "tie"
    ratio = mean / se
    if ratio >= tie_se:
        return "win"
    if ratio <= -tie_se:
        return "loss"
    return "tie"


def paired_cells(per_pair: pd.DataFrame, tie_se: float = TIE_SE) -> pd.DataFrame:
    """Paired ABTT-minus-baseline statistics per (config, model, view).

    Same statistic as ``build_main_attribution_artifacts.paired_cell_stats``:
    the difference is taken within a pair, over the pairs where both variants
    are defined, and the standard error is the SE of that difference.
    """
    wide = per_pair.pivot_table(
        index=["config", "model", "view", "example_tag"],
        columns="variant", values="del_auc_gap", aggfunc="first")
    for variant in VARIANTS:
        if variant not in wide.columns:
            wide[variant] = np.nan
    wide = wide.reset_index()
    wide = wide[np.isfinite(wide["baseline"]) & np.isfinite(wide["abtt"])]
    wide["diff"] = wide["abtt"] - wide["baseline"]
    out = []
    for (config, model, view), grp in wide.groupby(["config", "model", "view"]):
        diffs = grp["diff"].to_numpy(dtype=float)
        n = len(diffs)
        se = float(diffs.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        mean = float(diffs.mean()) if n else float("nan")
        out.append({
            "config": config, "model": model, "view": view,
            "n_pairs": n,
            "gap_baseline": float(grp["baseline"].mean()),
            "gap_abtt": float(grp["abtt"].mean()),
            "paired_mean": mean, "paired_se": se,
            "se_ratio": (mean / se) if (np.isfinite(se) and se > 0) else float("nan"),
            "verdict": verdict(mean, se, tie_se),
        })
    return pd.DataFrame(out)


def summarise_configs(cells: pd.DataFrame, configs: Sequence[DelAucConfig]) -> pd.DataFrame:
    """Win/tie/loss over the six cells, plus the spread of the cell gaps."""
    meta = {c.name: c for c in configs}
    rows = []
    for config, grp in cells.groupby("config"):
        cfg = meta[config]
        counts = grp["verdict"].value_counts()
        rows.append({
            "config": config,
            "knob": cfg.knob,
            "schedule": cfg.schedule,
            "erasure": cfg.erasure,
            "draws": cfg.draws,
            "seed": cfg.seed,
            "token_filter": cfg.token_filter,
            "side": cfg.side,
            "pooling_matches_generator": cfg.pooling_matches_generator,
            "n_cells": len(grp),
            "wins": int(counts.get("win", 0)),
            "ties": int(counts.get("tie", 0)),
            "losses": int(counts.get("loss", 0)),
            "sign_wins": int((grp["paired_mean"] > 0).sum()),
            "gap_base_min": float(grp["gap_baseline"].min()),
            "gap_base_max": float(grp["gap_baseline"].max()),
            "gap_abtt_min": float(grp["gap_abtt"].min()),
            "gap_abtt_max": float(grp["gap_abtt"].max()),
            "gap_base_spread": float(grp["gap_baseline"].max() - grp["gap_baseline"].min()),
            "mean_abs_paired": float(grp["paired_mean"].abs().mean()),
            "min_pairs": int(grp["n_pairs"].min()),
        })
    order = {c.name: i for i, c in enumerate(configs)}
    df = pd.DataFrame(rows)
    return df.sort_values("config", key=lambda s: s.map(order)).reset_index(drop=True)


def verify_against_cache(per_pair: pd.DataFrame, cache_root: Path,
                         tolerance: float = 1e-9) -> dict:
    """Does the predeclared configuration reproduce the published per-pair cache?

    The point of the check is that the sweep's own reimplementation of the
    metric is not a second opinion but the same computation: if the predeclared
    row does not land on the published numbers, nothing downstream of it means
    anything.

    A pair that is undefined under both (below ``FULL_COS_FLOOR``) agrees. A
    pair undefined under exactly one of them is a **disagreement about the
    floor**, counted in ``floor_mismatches`` rather than folded into the
    numeric difference: NaN minus a number is NaN, and any max that skips NaNs
    would report perfect agreement for a run that had silently changed which
    pairs it scores at all.
    """
    published: Dict[Tuple[str, str, str], float] = {}
    for path in sorted(cache_root.rglob("*.json")):
        try:
            rows = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        for row in rows:
            if row.get("method") in VIEWS and "del_auc_gap" in row:
                published[(path.stem, str(row["method"]), str(row["variant"]))] = row["del_auc_gap"]
    ours = per_pair[per_pair["config"] == PREDECLARED.name]
    diffs: List[float] = []
    missing = 0
    both_undefined = 0
    floor_mismatches: List[str] = []
    for row in ours.itertuples():
        key = (row.example_tag, row.view, row.variant)
        if key not in published:
            missing += 1
            continue
        # The driver writes NaN through ``json.dumps``, which emits the
        # non-standard ``NaN`` literal that ``json.loads`` reads back as a
        # float; a stricter writer would emit ``null``. Both mean "undefined".
        a = float(row.del_auc_gap) if row.del_auc_gap is not None else float("nan")
        b = float(published[key]) if published[key] is not None else float("nan")
        a_ok, b_ok = np.isfinite(a), np.isfinite(b)
        if not a_ok and not b_ok:
            both_undefined += 1
            continue
        if a_ok != b_ok:
            floor_mismatches.append("/".join(key))
            continue
        diffs.append(abs(a - b))
    arr = np.asarray(diffs, dtype=float)
    max_abs = float(arr.max()) if len(arr) else float("nan")
    return {
        "compared": int(len(arr)),
        "missing_from_cache": int(missing),
        "both_undefined": int(both_undefined),
        "floor_mismatches": len(floor_mismatches),
        "floor_mismatch_examples": floor_mismatches[:10],
        "max_abs_diff": max_abs,
        "mean_abs_diff": float(arr.mean()) if len(arr) else float("nan"),
        "tolerance": tolerance,
        "agrees": bool(len(arr) and missing == 0 and not floor_mismatches
                       and max_abs <= tolerance),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", default="runs/active/ig_examples_200pos_v1")
    p.add_argument("--examples_csv", default=None,
                   help="defaults to <run_dir>/positive200_examples.csv")
    p.add_argument("--artifacts_root", default=None,
                   help="defaults to <run_dir>/artifacts")
    p.add_argument("--out_dir", default=None,
                   help="defaults to <run_dir>/attribution_metrics/sensitivity")
    p.add_argument("--verify_cache", default=None,
                   help="per-pair JSON cache of the published run (v2_hidden)")
    p.add_argument("--max_pairs_per_model", type=int, default=None)
    p.add_argument("--configs", default=None,
                   help="comma-separated subset of configuration names")
    p.add_argument("--trust_remote_code", action="store_true")
    return p.parse_args(argv)


def resolve_lookups(model_name: str, filters: Iterable[str],
                    trust_remote_code: bool) -> Dict[str, Optional[np.ndarray]]:
    lookups: Dict[str, Optional[np.ndarray]] = {}
    tokenizer = None
    for tf in sorted(set(filters)):
        if tf == "all":
            lookups[tf] = None
            continue
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=trust_remote_code)
        lookups[tf] = build_token_keep_lookup(tokenizer, tf)
    return lookups


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    examples_csv = Path(args.examples_csv or run_dir / "positive200_examples.csv")
    artifacts_root = Path(args.artifacts_root or run_dir / "artifacts")
    out_dir = Path(args.out_dir or run_dir / "attribution_metrics" / "sensitivity")
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = build_configs()
    if args.configs:
        wanted = {n.strip() for n in args.configs.split(",")}
        configs = [c for c in configs if c.name in wanted]
    print(f"{len(configs)} configurations: {[c.name for c in configs]}", flush=True)

    examples = pd.read_csv(examples_csv)
    examples["slug"] = examples["model_name"].apply(model_slug)
    examples = examples.sort_values(["model_name", "example_id"]).reset_index(drop=True)

    rows: List[dict] = []
    for model_name, sub in examples.groupby("model_name"):
        slug = model_slug(model_name)
        ex_rows = sub.to_dict(orient="records")
        if args.max_pairs_per_model is not None:
            ex_rows = ex_rows[: args.max_pairs_per_model]
        lookups = resolve_lookups(model_name, [c.token_filter for c in configs],
                                  args.trust_remote_code)
        print(f"\n=== {model_name}: {len(ex_rows)} pairs ===", flush=True)
        t0 = time.time()
        for i, ex in enumerate(ex_rows):
            ex_id = int(ex["example_id"])
            role = str(ex.get("candidate_role", "pair_example"))
            tag = f"example{ex_id:03d}_{role}"
            npz_path = artifacts_root / slug / f"{tag}.npz"
            if not npz_path.exists():
                raise FileNotFoundError(f"missing NPZ: {npz_path}")
            for row in process_npz(npz_path, configs, lookups):
                row["model"] = model_name
                row["example_tag"] = tag
                row["layer"] = int(ex["layer"])
                rows.append(row)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(ex_rows)} pairs, {time.time() - t0:.1f}s",
                      flush=True)
        print(f"  done in {time.time() - t0:.1f}s", flush=True)

    per_pair = pd.DataFrame(rows)
    per_pair.to_csv(out_dir / "per_pair.csv", index=False)
    print(f"\nWrote {out_dir / 'per_pair.csv'} ({len(per_pair)} rows)")

    cells = paired_cells(per_pair)
    cells.to_csv(out_dir / "cells.csv", index=False)
    print(f"Wrote {out_dir / 'cells.csv'} ({len(cells)} rows)")

    summary = summarise_configs(cells, configs)
    summary.to_csv(out_dir / "configs.csv", index=False)
    print(f"Wrote {out_dir / 'configs.csv'} ({len(summary)} rows)")
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(summary[["config", "knob", "wins", "ties", "losses",
                       "gap_base_min", "gap_base_max", "gap_abtt_min",
                       "gap_abtt_max", "min_pairs"]].to_string(index=False))

    if args.verify_cache:
        report = verify_against_cache(per_pair, Path(args.verify_cache))
        (out_dir / "verification.json").write_text(json.dumps(report, indent=2))
        print(f"\nPredeclared vs published cache: {report}")
        if not report["agrees"]:
            # The sweep's outputs are already on disk, which is deliberate:
            # a disagreement is something to diff, not something to lose. But
            # it must not pass silently into a table.
            raise RuntimeError(
                "the predeclared configuration does not reproduce the published "
                f"per-pair cache at {args.verify_cache}; every other row of the "
                f"sweep is measured against it. Report: {report}")


if __name__ == "__main__":
    main()
