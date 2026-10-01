#!/usr/bin/env python3
"""Reframe experiment E2 (issue #252): token audit of the dominant directions.

Question: which tokens carry the directions that dominate the mean-pooled passage vectors
at collapsed T5 layers? E1 showed that no small set of coordinates does; ABTT with three
components restores every collapsed layer. E2 re-runs the encoders, decomposes each
passage's score on a direction into the contributions of its tokens, and tests three
accounts: special tokens, frequent tokens, passage length.

  audit   GPU forward pass per model (tokenization, model class, batch size and token
          filter of the extraction CLIs; one forward serves all layers), two passes:
            pass 1  the five pooling arms and the token projections on the directions;
            pass 2  pooling with carrier token types dropped (needs pass 1's ranking).
          Directions, fit on the cached TRAIN vectors (``hidden_mean_tokempty``): the
          train mean mu, PC1 to PC3 of the centered train vectors (EmbeddingCleaner's,
          the ones ABTT removes; sign fixed so that the largest loading is positive), the
          top-3 coordinates by train variance (E1's ranking; secondary block) and 20
          seeded random unit directions, orthogonal to the top-10 PCs and to each other.
          For a direction w, token t contributes c_t = (h_t - mu) . w and the passage
          score s_p = w . (pooled_p - mu) is the mean of c_t over the kept tokens.
          Readouts per model-layer:
            1. pooling arms   mean (the cache), mean_nospecial, sif_keepspecial, sif (the
                              CLI's SIF), mean_nofreq100: the E1 metric block and the
                              top-PC share / effective rank of the train vectors;
            2. token audit    token-mix explained variance (per-token-type means of c_t
                              fit on train tokens), shares of the score variance by token
                              group (special / frequent / other; share_g = Cov(s_g, s) /
                              Var(s)), carrier token types, Spearman of s with log length
                              and with the frequent-token mass; the same for the subspaces
                              span(PC2, PC3) and span(PC1, PC2, PC3) with traces of
                              covariances, which do not depend on the basis chosen inside
                              the subspace, and for random 2-D and 3-D subspaces;
            3. token ablation pooling with the top m carrier types dropped, m in
                              {1,3,10,30,100}, two rankings, against random types matched
                              in train frequency (5 draws);
            4. length check   |delta log n| of same- and different-directory pairs.
          Writes small CSVs only; token states and embeddings are never written.
  check   the gates on existing CSVs (no caches, no torch).
  render  the paper table and a facts file for the prose (no caches, no torch).

Pooling arms. All five use the CLIs' expression sum_t w_t h_t / max(sum_t w_t, 1) with
w_t = attention * lookup[token id]; only the lookup differs:
  mean             the ``tokenizer_empty`` keep lookup (what the cache holds);
  mean_nospecial   the same with ``tokenizer.all_special_ids`` set to 0;
  sif_keepspecial  sif_weights_from_ids without special_ids: a / (a + p) for a token type
                   with a train probability, 1 otherwise. Special tokens are never counted
                   by token_probabilities, so they keep weight 1. This is full SIF with
                   one factor changed (the zero weight on special tokens);
  sif              sif_weights_from_ids as the CLI calls it (the published ``sif_only``);
  mean_nofreq100   mean with the 100 most frequent train token types set to 0 (special
                   tokens kept).
A passage left with no token under an arm falls back to its ``mean`` vector and is counted
(``n_fallback``); the ``sif`` arm never falls back, because it must equal the CLI.

Decision rules (frozen before any E2 number was read; the constants below):
  R1  an arm rescues a collapsed layer if (AUROC_arm - AUROC_mean) / (AUROC_sif -
      AUROC_mean) >= 0.80, evaluated only where AUROC_sif - AUROC_mean >= 0.05;
  R2  the token mix carries the direction if test EV for PC1 >= 0.5;
  R3  token ablation restores a collapsed layer if test AUROC >= 0.90 for some m <= 100;
  R4  the length account holds if |Spearman(s_PC1, log n)| >= 0.5 on test and the mean
      |delta log n| is larger for same-directory than for different-directory pairs.
Collapsed = published baseline test AUROC below 0.70.

Gates (``audit --check`` or ``check``; exit status 3 on failure, after the CSVs are
written; skipped under --limit):
  1a. the mean arm's AUROC = the baseline cells (repr hidden, pooling mean) of
      runs/active/resubmit/results/phase_resubmit_results.csv within 1e-6, every layer;
  1b. the mean arm's vectors = the cached ``hidden_mean_tokempty`` vectors (largest
      relative L2 difference over passages within 1e-3; the absolute one is reported);
  2a. the sif arm's AUROC = the ``sif_only`` cells (pooling sif) within 1e-6;
  2b. the sif arm's vectors against ``hidden_sif_tokempty`` where that cache exists
      (reported, not gated);
  3.  the score identity: mean of c_t = w . (float64 mean-pooled vector - mu);
  4.  the group shares sum to 1.

Outputs (small CSVs, force-added; a rerun of some models replaces only their rows):
  runs/active/reframe/e2/e2_pooling_arms.csv      one row per (model, layer, arm)
  runs/active/reframe/e2/e2_direction_audit.csv   (model, layer, direction, split)
  runs/active/reframe/e2/e2_carriers.csv          top 30 token types per PC
  runs/active/reframe/e2/e2_token_ablation.csv    (model, layer, ranking, kind, draw, m)
  runs/active/reframe/e2/e2_length.csv            (model, split)
  runs/active/reframe/e2/e2_gate_check.csv
  runs/active/reframe/e2/facts_e2.md              (render)
  overleaf_drafts/tables/e2_token_audit.tex       (tab:e2_token_audit)

Run from the repo root, on a GPU node (slurm/reframe/reframe_e2.sbatch):
  python scripts/paper/reframe/e2_token_audit.py audit --check --workers 8 \
      --bases_root <checkout>/runs/active/resubmit_bases --models LaTa,PhilTa
  python scripts/paper/reframe/e2_token_audit.py render
Python 3.10, numpy / pandas / scipy / scikit-learn; torch and transformers 4.x for audit.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "resubmit"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import abtt_subspace_whiten as asw  # noqa: E402
import e1_coordinate_ablation as e1  # noqa: E402
from embedding_alignment import AlignmentResolver, find_manifest, load_row_order  # noqa: E402
from sif_abtt import EmbeddingCleaner, sif_weights_from_ids  # noqa: E402

# torch, transformers and the extraction CLIs are imported inside the functions that run
# the encoders: CI has no torch, and check / render must work without it.

SPLIT_CSV = asw.SPLIT_CSV
RES_CSV = asw.RES_CSV
BASES_ROOT = asw.BASES_ROOT
E1_COORD_CSV = e1.OUT_DIR / e1.COORD_NAME
OUT_DIR = asw.OUT_ROOT / "e2"
TAB_DIR = asw.TAB_DIR
ARMS_NAME = "e2_pooling_arms.csv"
AUDIT_NAME = "e2_direction_audit.csv"
CARRIER_NAME = "e2_carriers.csv"
ABL_NAME = "e2_token_ablation.csv"
LENGTH_NAME = "e2_length.csv"
GATE_NAME = "e2_gate_check.csv"
FACTS_NAME = "facts_e2.md"
TABLE_NAME = "e2_token_audit.tex"

MODELS = asw.MODELS
ALL_MODEL_IDS = tuple(m[0] for m in MODELS)
MODEL_INDEX = {m: i for i, m in enumerate(ALL_MODEL_IDS)}
DISP = {m[0]: m[1] for m in MODELS}
ORDER = [m[1] for m in MODELS]
T5 = [m[1] for m in MODELS if m[2]]
NON_T5 = [m[1] for m in MODELS if not m[2]]

# How each cache was extracted (slurm/resubmit/resubmit_extract_*.sbatch): which CLI, and
# whether it was given --trust_remote_code. No model was loaded in half precision.
EXTRACT = {
    "bowphs/LaTa": ("hidden", False),
    "bowphs/PhilTa": ("hidden", False),
    "google/mt5-base": ("hidden", False),
    "sentence-transformers/LaBSE": ("encoder", False),
    "Qwen/Qwen3-Embedding-0.6B": ("encoder", True),
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5": ("encoder", True),
}
MAX_LENGTH = 512
BATCH_SIZE = 8
TOKEN_FILTER = "tokenizer_empty"
SIF_A = 1e-3
MEAN_SUBDIR = asw.SUBDIR
SIF_SUBDIR = "hidden_sif_tokempty"

ARMS = ("mean", "mean_nospecial", "sif_keepspecial", "sif", "mean_nofreq100")
NO_FALLBACK = ("mean", "sif")  # the two arms that must equal the CLI's output
R1_ARMS = ("mean_nospecial", "sif_keepspecial", "mean_nofreq100")
ARM_LABEL = {"mean": "mean", "mean_nospecial": "mean, no special", "sif": "SIF",
             "sif_keepspecial": "SIF, special kept", "mean_nofreq100": "mean, no top-100"}
GROUPS = ("special", "frequent", "other")
N_FREQUENT = 100  # the `frequent` group and the mean_nofreq100 arm
N_PC = 3
N_COORD = 3
N_RANDOM = 20
ORTHO_PCS = 10  # random directions are orthogonal to this many top PCs
RANDOM_SEED = 233
PC_NAMES = tuple(f"pc{i + 1}" for i in range(N_PC))
COORD_NAMES = tuple(f"coord{i + 1}" for i in range(N_COORD))
RAND_NAMES = tuple(f"rand{i:02d}" for i in range(N_RANDOM))
DIRECTIONS = PC_NAMES + COORD_NAMES + RAND_NAMES
# Subspaces read jointly (no decision rule attached): (name, kind, member directions).
# PC2 and PC3 are individually arbitrary up to a rotation where their variances are
# close; a trace over span(PC2, PC3) is not. Controls: 10 disjoint pairs and 6 disjoint
# triples of the random directions.
SUBSPACES = (
    [("pcs2_3", "pc_subspace", ("pc2", "pc3")), ("pcs1_3", "pc_subspace", PC_NAMES)]
    + [(f"randpair{i:02d}", "random_subspace", RAND_NAMES[2 * i:2 * i + 2])
       for i in range(N_RANDOM // 2)]
    + [(f"randtriple{i:02d}", "random_subspace", RAND_NAMES[3 * i:3 * i + 3])
       for i in range(N_RANDOM // 3)])
TOP_CARRIERS = 30  # token types written per PC
ABL_MS = (1, 3, 10, 30, 100)
RANKINGS = ("pc1", "pc123")
RANK_LABEL = {"pc1": "PC1 share", "pc123": "mean of the PC1, PC2, PC3 shares"}
CONTROL_DRAWS = 5
MATCH_WINDOW = 10  # a matched random type is drawn among this many nearest in train count

# Decision rules, frozen before any E2 number was read. Do not tune.
COLLAPSE_AUROC = asw.COLLAPSE_AUROC  # collapsed = published baseline test AUROC below 0.70
R1_RESCUE_FRAC = 0.80  # share of the SIF gain an arm must recover to "rescue" a layer
R1_MIN_SIF_GAIN = 0.05  # R1 is evaluated only where SIF gains at least this much
R2_EV = 0.5  # test token-mix EV for PC1
R3_AUROC = 0.90  # test AUROC a token ablation must reach
R3_MAX_M = 100  # with at most this many token types dropped
R4_RHO = 0.5  # |Spearman(s_PC1, log n)| on test

GATE_TOL_AUROC = 1e-6
GATE_TOL_VEC_REL = 1e-3  # gate 1b: relative L2 difference of a re-derived vector to the cache
GATE_TOL_IDENTITY = 1e-9  # gate 3: relative to the passage vector's norm (float64 algebra)
GATE_TOL_SHARE_SUM = 1e-8  # gate 4
GATE_EXIT = e1.GATE_EXIT  # exit status of a failed gate: the CSVs are written


# --------------------------------------------------------------------------- #
# Pure functions (unit-tested on synthetic arrays)
# --------------------------------------------------------------------------- #

def fix_sign(w: np.ndarray) -> np.ndarray:
    """Orient a direction so that its largest-magnitude loading is positive.

    A principal component is defined up to sign; this makes the sign of a token's mean
    contribution comparable across layers and runs. Ties keep the lower index.
    """
    return w if w[int(np.argmax(np.abs(w)))] >= 0 else -w


def random_directions(d: int, basis: np.ndarray, n: int = N_RANDOM,
                      seed: Sequence[int] = (RANDOM_SEED,)) -> np.ndarray:
    """``n`` seeded random unit directions [n, d], orthogonal to the rows of ``basis``
    and to each other (QR of a Gaussian matrix after projecting the basis out)."""
    g = np.random.default_rng(list(seed)).standard_normal((n, d))
    if len(basis):
        q, _ = np.linalg.qr(np.asarray(basis, dtype=np.float64).T)
        g = g - (g @ q) @ q.T
    q, _ = np.linalg.qr(g.T)
    return np.stack([fix_sign(v) for v in q.T])


def fit_directions(train: np.ndarray, seed: Sequence[int] = (RANDOM_SEED,)) -> Dict:
    """mu and the audited directions of one model-layer, from the cached TRAIN vectors.

    The PCs are EmbeddingCleaner's (float32 SVD of the centered train vectors), so PC1 to
    PC3 span exactly what ABTT D=3 removes. The coordinates are E1's variance ranking.
    Returns {"mu": [d], "W": [K, d] float64, "names", "kinds", "coords"} with the rows of
    W in the order of DIRECTIONS.
    """
    d = train.shape[1]
    cleaner = EmbeddingCleaner(num_components=ORTHO_PCS, center=True).fit(train)
    pcs = np.asarray(cleaner.pcs, dtype=np.float64)
    if len(pcs) < N_PC:
        raise ValueError(f"need at least {N_PC} principal components, got {len(pcs)}")
    coords = [int(i) for i in e1.rank_coords(train, "variance")[:N_COORD]]
    unit = np.zeros((N_COORD, d))
    unit[np.arange(N_COORD), coords] = 1.0
    w = np.vstack([np.stack([fix_sign(p) for p in pcs[:N_PC]]), unit,
                   random_directions(d, pcs, N_RANDOM, seed)])
    return {"mu": np.asarray(cleaner.mean_vec, dtype=np.float64), "W": w,
            "names": list(DIRECTIONS),
            "kinds": ["pc"] * N_PC + ["coord"] * N_COORD + ["random"] * N_RANDOM,
            "coords": [-1] * N_PC + coords + [-1] * N_RANDOM}


def frequent_ids(token_probs: Dict[int, float], n: int = N_FREQUENT) -> np.ndarray:
    """The ``n`` token ids of largest train probability (ties: lower id first).

    token_probabilities never counts special tokens or tokens the filter drops, so these
    are the most frequent non-special train token types.
    """
    items = sorted(token_probs.items(), key=lambda kv: (-float(kv[1]), int(kv[0])))
    return np.array([int(k) for k, _ in items[:n]], dtype=np.int64)


def arm_weight_lookups(vocab_size: int, token_probs: Dict[int, float],
                       special_ids: Sequence[int], keep_lookup: np.ndarray,
                       sif_a: float = SIF_A) -> Dict[str, np.ndarray]:
    """Per-token-id pooling weight of every arm, float32 [vocab_size].

    The two SIF lookups are sif_weights_from_ids (the CLI's function) evaluated once on
    every token id, so lookup[ids] equals the CLI's per-batch weights. Each lookup already
    contains the keep lookup, as the CLI's product keep_mask * weights does.
    """
    keep = np.asarray(keep_lookup, dtype=np.float32)
    ids = np.arange(vocab_size, dtype=np.int64)[None, :]
    special = np.array(sorted(int(i) for i in special_ids if 0 <= int(i) < vocab_size),
                       dtype=np.int64)
    nospecial = keep.copy()
    nospecial[special] = 0.0
    nofreq = keep.copy()
    nofreq[frequent_ids(token_probs)] = 0.0
    return {
        "mean": keep.copy(),
        "mean_nospecial": nospecial,
        "sif_keepspecial": sif_weights_from_ids(ids, token_probs, a=sif_a, special_ids=None,
                                                token_keep_lookup=keep)[0],
        "sif": sif_weights_from_ids(ids, token_probs, a=sif_a,
                                    special_ids=[int(i) for i in special_ids],
                                    token_keep_lookup=keep)[0],
        "mean_nofreq100": nofreq,
    }


def pool_weighted(hidden, wm):
    """The extraction CLIs' pooling expression for per-token weights ``wm`` [B, T].

    With wm = keep_mask it is their mean pooling, with wm = keep_mask * SIF weights their
    SIF pooling, operation for operation (the weight sum is clamped to at least 1).
    """
    denom = wm.sum(dim=1, keepdim=True).clamp(min=1.0)
    return (hidden * wm.unsqueeze(-1)).sum(dim=1) / denom


@dataclass
class TokenTable:
    """The kept tokens of every passage (the ones mean pooling averages), flat."""

    pid: np.ndarray       # passage index of each token
    tid: np.ndarray       # token id
    first: np.ndarray     # True at the first kept token of its passage
    group: np.ndarray     # index into GROUPS
    n: np.ndarray         # kept tokens per passage [N]
    types: np.ndarray     # sorted distinct token ids
    type_idx: np.ndarray  # index into ``types`` of each token


def build_token_table(pid: np.ndarray, tid: np.ndarray, first: np.ndarray, n_passages: int,
                      special_ids: Iterable[int], frequent: Iterable[int]) -> TokenTable:
    pid = np.asarray(pid, dtype=np.int64)
    tid = np.asarray(tid, dtype=np.int64)
    group = np.full(len(tid), GROUPS.index("other"), dtype=np.int64)
    group[np.isin(tid, np.fromiter(frequent, dtype=np.int64))] = GROUPS.index("frequent")
    group[np.isin(tid, np.fromiter(special_ids, dtype=np.int64))] = GROUPS.index("special")
    types, type_idx = np.unique(tid, return_inverse=True)
    return TokenTable(pid=pid, tid=tid, first=np.asarray(first, dtype=bool), group=group,
                      n=np.bincount(pid, minlength=n_passages).astype(np.int64),
                      types=types, type_idx=type_idx.reshape(-1))


def _sum_by(index: np.ndarray, values: np.ndarray, size: int) -> np.ndarray:
    """Column-wise np.bincount: sums of ``values`` [n, j] by ``index`` -> [size, j]."""
    return np.stack([np.bincount(index, weights=values[:, k], minlength=size)
                     for k in range(values.shape[1])], axis=1)


def passage_scores(c: np.ndarray, tok: TokenTable) -> np.ndarray:
    """s_p = mean over the kept tokens of passage p of c_t; [N, j]. 0 where n_p = 0."""
    c = np.asarray(c, dtype=np.float64).reshape(len(tok.pid), -1)
    return _sum_by(tok.pid, c, len(tok.n)) / np.maximum(tok.n, 1)[:, None]


def type_means(c: np.ndarray, tok: TokenTable, train_rows: np.ndarray
               ) -> Tuple[np.ndarray, np.ndarray]:
    """(train count, mean of c_t over the TRAIN tokens) of every token type.

    A type with no train token gets the mean over all train tokens.
    """
    c = np.asarray(c, dtype=np.float64).reshape(len(tok.pid), -1)
    tr = train_rows[tok.pid]
    count = np.bincount(tok.type_idx[tr], minlength=len(tok.types))
    total = _sum_by(tok.type_idx[tr], c[tr], len(tok.types))
    overall = c[tr].mean(axis=0) if tr.any() else np.zeros(c.shape[1])
    mean = np.where(count[:, None] > 0, total / np.maximum(count, 1)[:, None], overall[None, :])
    return count, mean


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(spearmanr(a, b)[0])


def audit_scores(c: np.ndarray, tok: TokenTable, train_rows: np.ndarray,
                 test_rows: np.ndarray, equal_weight: bool = False) -> Dict:
    """Token-mix explained variance and additive token shares of a score.

    ``c`` holds the token contributions, [n_tok] for one direction or [n_tok, j] for a
    subspace with an orthonormal basis of j directions. With s_p the passage score (a
    j-vector), s_hat_p the mean over the passage's tokens of their type's train mean
    contribution, and Cov over the passages of a split:
      ev      = 1 - trace(Cov(s - s_hat)) / trace(Cov(s));
      token share a_t = (c_t / n_p) . (s_p - mean s) / (N trace(Cov(s))), so that the sum
      of a_t over any token set G is trace(Cov(s_G, s)) / trace(Cov(s)), and over all
      tokens 1.
    Both are unchanged by a rotation of the basis inside the subspace. ``equal_weight``
    first whitens the score with the TRAIN covariance of s (S^-1/2, symmetric): in the PC
    basis that divides each component by its train SD, and it is rotation invariant too.

    Returns {"s": [N, j] (not whitened), "type_count", "type_mean" (not whitened),
    "train" / "test": {"n_passages", "var_s", "ev", "r2" (j = 1 only), "token_share":
    [n_tok], "valid": [N] bool}}.
    """
    c = np.asarray(c, dtype=np.float64).reshape(len(tok.pid), -1)
    j = c.shape[1]
    s = passage_scores(c, tok)
    count, mean = type_means(c, tok, train_rows)
    s_hat = _sum_by(tok.pid, mean[tok.type_idx], len(tok.n)) / np.maximum(tok.n, 1)[:, None]
    out = {"s": s, "type_count": count, "type_mean": mean}
    cw, sw, hw = c, s, s_hat
    if equal_weight:
        tr = train_rows & (tok.n > 0)
        cov = np.atleast_2d(np.cov(s[tr].T, bias=True))
        vals, vecs = np.linalg.eigh(cov)
        with np.errstate(divide="ignore", invalid="ignore"):
            root = (vecs / np.sqrt(vals)) @ vecs.T
        cw, sw, hw = c @ root, s @ root, s_hat @ root
    inv_n = 1.0 / np.maximum(tok.n, 1)[tok.pid]
    for split, rows in (("train", train_rows), ("test", test_rows)):
        valid = rows & (tok.n > 0)
        nv = int(valid.sum())
        dev = sw[valid] - sw[valid].mean(axis=0) if nv else np.zeros((0, j))
        total = float((dev ** 2).sum() / nv) if nv else float("nan")
        res = (sw - hw)[valid]
        with np.errstate(divide="ignore", invalid="ignore"):
            ev = (1.0 - float(((res - res.mean(axis=0)) ** 2).sum() / nv) / total
                  if nv and total > 0 else float("nan"))
            scaled = np.zeros_like(sw)
            if nv and total > 0:
                scaled[valid] = dev / (nv * total)
            share = (cw * scaled[tok.pid]).sum(axis=1) * inv_n
            if not (nv and total > 0):
                share = np.full(len(tok.pid), np.nan)
        r2 = float("nan")
        if j == 1 and nv > 2 and np.ptp(sw[valid, 0]) > 0 and np.ptp(hw[valid, 0]) > 0:
            r2 = float(np.corrcoef(sw[valid, 0], hw[valid, 0])[0, 1] ** 2)
        out[split] = {"n_passages": nv, "var_s": total, "ev": ev, "r2": r2,
                      "token_share": share, "valid": valid}
    return out


def share_columns(token_share: np.ndarray, tok: TokenTable, suffix: str = "") -> Dict[str, float]:
    """Group shares, their sum minus 1, and the first-token share of one split."""
    g = np.bincount(tok.group, weights=token_share, minlength=len(GROUPS))
    out = {f"share_{name}{suffix}": float(g[i]) for i, name in enumerate(GROUPS)}
    out[f"share_sum_err{suffix}"] = float(g.sum() - 1.0)
    out[f"share_first{suffix}"] = float(token_share[tok.first].sum())
    return out


def rank_types(score: np.ndarray, train_count: np.ndarray) -> np.ndarray:
    """Indices of the token types seen in train, largest score first (ties: lower id)."""
    seen = np.flatnonzero(train_count > 0)
    return seen[np.argsort(-score[seen], kind="stable")]


def matched_random(train_count: np.ndarray, carriers: Sequence[int],
                   excluded: Iterable[int], rng: np.random.Generator,
                   window: int = MATCH_WINDOW) -> np.ndarray:
    """One draw of frequency-matched random token types, one per carrier, in carrier order.

    For each carrier in turn, the candidates are the ``window`` eligible types nearest to
    it in train count (absolute difference; ties: lower index) that this draw has not used
    yet, and one is drawn uniformly. Eligible = seen in train and not in ``excluded``. The
    first m entries are therefore the control for the top m carriers. Shorter than
    ``carriers`` only when the eligible types run out.
    """
    count = np.asarray(train_count, dtype=np.int64)
    free = count > 0
    free[np.fromiter(excluded, dtype=np.int64)] = False
    index = np.arange(len(count), dtype=np.int64)
    picks: List[int] = []
    for carrier in carriers:
        cand = index[free]
        if not len(cand):
            break
        # distance first, index second, in one integer key
        key = np.abs(count[cand] - count[int(carrier)]) * len(count) + cand
        k = min(window, len(cand))
        part = np.argpartition(key, k - 1)[:k] if k < len(cand) else np.arange(len(cand))
        near = cand[part[np.argsort(key[part])]]  # keys are distinct, so the order is fixed
        pick = int(near[int(rng.integers(len(near)))])
        picks.append(pick)
        free[pick] = False
    return np.array(picks, dtype=np.int64)


def ablation_arms(type_share: np.ndarray, train_count: np.ndarray,
                  seed: Sequence[int] = (RANDOM_SEED,)) -> List[Dict]:
    """The token-ablation arms of one model-layer, as sets of token-type indices.

    ``type_share`` [N_PC, n_types] holds each type's train share of the PC1, PC2 and PC3
    score variance. Two rankings (RANKINGS); per ranking the top m carriers for m in
    ABL_MS, then CONTROL_DRAWS matched random draws, each read at the same m. A control
    type is in the top max(ABL_MS) of neither ranking.
    """
    score = {"pc1": type_share[0], "pc123": type_share[:N_PC].mean(axis=0)}
    top = {r: rank_types(score[r], train_count)[:max(ABL_MS)] for r in RANKINGS}
    excluded = sorted({int(i) for r in RANKINGS for i in top[r]})
    arms: List[Dict] = []
    for ri, ranking in enumerate(RANKINGS):
        for m in ABL_MS:
            arms.append({"ranking": ranking, "kind": "carrier", "draw": -1, "m": m,
                         "types": top[ranking][:m]})
        for draw in range(CONTROL_DRAWS):
            rng = np.random.default_rng([*seed, ri, draw])
            picks = matched_random(train_count, top[ranking], excluded, rng)
            for m in ABL_MS:
                arms.append({"ranking": ranking, "kind": "control", "draw": draw, "m": m,
                             "types": picks[:m]})
    return arms


def length_stats(n: np.ndarray, labels_ut: np.ndarray) -> Dict[str, float]:
    """|delta log n| over same- and different-directory pairs (the pairs Task A scores).

    ``labels_ut`` is upper_triangle_labels of the passages' directories. Also the AUROC
    of -|delta log n| as a pair score (asw.absdiff_auroc).
    """
    logn = np.log(np.maximum(np.asarray(n, dtype=np.float64), 1.0))
    iu = np.triu_indices(len(logn), k=1)
    d = np.abs(logn[iu[0]] - logn[iu[1]])
    same = np.asarray(labels_ut).astype(bool)
    nan = float("nan")
    return {"n_same_pairs": int(same.sum()), "n_diff_pairs": int((~same).sum()),
            "mean_dlog_same": float(d[same].mean()) if same.any() else nan,
            "mean_dlog_diff": float(d[~same].mean()) if (~same).any() else nan,
            "median_dlog_same": float(np.median(d[same])) if same.any() else nan,
            "median_dlog_diff": float(np.median(d[~same])) if (~same).any() else nan,
            "auroc_neg_dlog": float(asw.absdiff_auroc(logn, labels_ut))}


def visible(text: str) -> str:
    """A token string with backslashes, line breaks and other non-printing characters
    escaped, so that it is one unambiguous CSV / markdown cell."""
    out = []
    for ch in str(text):
        if ch == "\\":
            out.append("\\\\")
        elif ch == " " or ch.isprintable():
            out.append(ch)
        else:
            out.append(ch.encode("unicode_escape").decode("ascii"))
    return "".join(out)


# --------------------------------------------------------------------------- #
# Decision rules (frozen)
# --------------------------------------------------------------------------- #

def r1_rescue(auc_arm: float, auc_mean: float, auc_sif: float) -> Tuple[str, float, float]:
    """R1 for one arm at one layer: (status, share of the SIF gain recovered, raw change).

    status is "rescue", "no_rescue", "no_sif_gain" (SIF gains less than R1_MIN_SIF_GAIN,
    so there is nothing to recover and the share is NaN) or "undefined" (a NaN input).
    """
    change = auc_arm - auc_mean
    gain = auc_sif - auc_mean
    if not (np.isfinite(change) and np.isfinite(gain)):
        return "undefined", float("nan"), float(change)
    if gain < R1_MIN_SIF_GAIN:
        return "no_sif_gain", float("nan"), float(change)
    frac = change / gain
    return ("rescue" if frac >= R1_RESCUE_FRAC else "no_rescue"), float(frac), float(change)


def r2_token_mix(ev_test: float) -> bool:
    """R2: the token mix carries the direction."""
    return bool(np.isfinite(ev_test) and ev_test >= R2_EV)


def r3_restoring_m(auc_by_m: Dict[int, float]) -> Optional[int]:
    """R3: the smallest m <= R3_MAX_M whose ablation reaches R3_AUROC, or None."""
    for m in sorted(auc_by_m):
        v = auc_by_m[m]
        if m <= R3_MAX_M and v is not None and np.isfinite(v) and v >= R3_AUROC:
            return int(m)
    return None


def r4_length(rho_test: float, mean_dlog_same: float, mean_dlog_diff: float) -> bool:
    """R4: the length account holds."""
    return bool(np.isfinite(rho_test) and abs(rho_test) >= R4_RHO
                and np.isfinite(mean_dlog_same) and np.isfinite(mean_dlog_diff)
                and mean_dlog_same > mean_dlog_diff)


# --------------------------------------------------------------------------- #
# CSV plumbing: a rerun of some models replaces only their rows
# --------------------------------------------------------------------------- #

STR_COLS = ("model", "arm", "direction", "kind", "split", "ranking", "types", "token",
            "decoded", "group", "gate", "cells_over_tolerance", "note")


def read_csv(path: Path) -> pd.DataFrame:
    """Read one of the E2 CSVs. Text columns stay text whatever they hold (a token may be
    the string "nan" or "NA" or be empty); only numeric columns parse "nan" as missing."""
    cols = list(pd.read_csv(path, nrows=0).columns)
    text = [c for c in cols if c in STR_COLS]
    return pd.read_csv(path, keep_default_na=False, dtype={c: str for c in text},
                       na_values={c: ["", "nan", "NaN"] for c in cols if c not in text})


def sort_models(df: pd.DataFrame) -> pd.DataFrame:
    """Panel order of the models; the row order inside a model is kept."""
    known = {m: i for i, m in enumerate(ALL_MODEL_IDS)}
    extra = {m: len(known) + i for i, m in enumerate(sorted(set(df["model"]) - set(known)))}
    key = df["model"].map({**known, **extra})
    return (df.assign(_m=key, _i=np.arange(len(df))).sort_values(["_m", "_i"], kind="stable")
            .drop(columns=["_m", "_i"]).reset_index(drop=True))


def merge_write(path: Path, new: pd.DataFrame) -> pd.DataFrame:
    """Write ``new`` to ``path``, keeping the rows of the models that ``new`` does not hold.

    The work is split across jobs by model, so a rerun replaces exactly its models' rows.
    Written through a temporary file and renamed, so a killed job leaves the old CSV.
    """
    df = new
    if path.exists():
        old = read_csv(path)
        old = old[~old["model"].isin(set(new["model"]))]
        if len(old):
            df = pd.concat([old, new], ignore_index=True)[list(new.columns)]
    df = sort_models(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, float_format="%.10g")
    os.replace(tmp, path)
    return df
