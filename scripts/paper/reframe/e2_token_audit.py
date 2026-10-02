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
                              {1,3,10,30,100}, two rankings, against two random controls
                              (5 draws each): the count-nearest control (as many types,
                              nearest in train count) and the mass-matched control (other
                              types holding at least as many train tokens);
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
  2a. the sif arm's AUROC = the ``sif_only`` cells (pooling sif) within 1e-6 (fails on
      the real run: see "Changed after the results were read" below);
  2b. the sif arm's vectors against ``hidden_sif_tokempty`` where that cache exists
      (reported, not gated);
  3.  the score identity: mean of c_t = w . (float64 mean-pooled vector - mu);
  4.  the group shares sum to 1.

Changed after the results were read (render only; the audit numbers, the rules and the
gates are untouched). Gate 2a failed on the real run: the sif arm, which equals the local
SIF re-extraction exactly, misses the published ``sif_only`` cells by up to 1.5e-2, and no
token-probability source we could reconstruct reproduces them. The pre-registered
consequence of a gate 2 failure was to block the pooling-control conclusions. It was
replaced by reporting under both references: the facts file gives R1 against the job's
own sif arm and, in "R1 under the published SIF reference", against the published cells,
and says loudly if a verdict differs between the two. ``check`` still exits 3. The table
caption no longer calls the SIF column the published SIF, and its cells print a math
minus sign and no negative zero.

Added after the first full run, on review, because the count-nearest control was not
mass-matched: the mass-matched control (``control_mass`` rows of the ablation CSV). The
carriers are the most frequent token types, so the types nearest to them in train count
hold far fewer tokens (LaTa, pc123 ranking, m = 3: 7,022 train tokens against 828, the
median over layers of the draw mean). The new control draws token types with probability
proportional to train count, from the train types outside the dropped carriers, until
they hold at least as many train tokens. R3 and the carrier arms are unchanged. The
table's Drop cell is chosen on train AUROC (it was the highest test AUROC) and its Rand.
cell is the mass-matched control.

Outputs (small CSVs, force-added; a rerun of some models replaces only their rows):
  runs/active/reframe/e2/e2_pooling_arms.csv      one row per (model, layer, arm)
  runs/active/reframe/e2/e2_direction_audit.csv   (model, layer, direction, split)
  runs/active/reframe/e2/e2_carriers.csv          top 30 token types per PC
  runs/active/reframe/e2/e2_token_ablation.csv    (model, layer, ranking, kind, draw, m);
                                                  kind = carrier, control (count-nearest)
                                                  or control_mass (mass-matched)
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
MATCH_WINDOW = 10  # a count-nearest control type is drawn among this many nearest in train count
MASS_STREAM = 1_000_003  # seed entry that keeps the mass-matched control's draws separate
TYPES_LISTED = 100  # a mass-matched control row lists its token ids only up to this many
# Kinds of token-ablation rows: the carriers, and two random controls.
CONTROL_KINDS = ("control", "control_mass")
CONTROL_LABEL = {"control": "count-nearest control", "control_mass": "mass-matched control"}

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
    if d - len(basis) < n:
        raise ValueError(f"{n} orthonormal directions do not fit in {d} dimensions next to "
                         f"{len(basis)} excluded ones")
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


def group_masses(tok: TokenTable) -> np.ndarray:
    """Share of each passage's kept tokens in every group, [N, len(GROUPS)]."""
    counts = np.stack([np.bincount(tok.pid[tok.group == i], minlength=len(tok.n))
                       for i in range(len(GROUPS))], axis=1)
    return counts / np.maximum(tok.n, 1)[:, None]


def type_table(tok: TokenTable, train_rows: np.ndarray) -> Dict[str, np.ndarray]:
    """Per token type: its group, its train token count (special tokens included) and
    the number of train passages that contain it."""
    tr = train_rows[tok.pid]
    group = np.zeros(len(tok.types), dtype=np.int64)
    group[tok.type_idx] = tok.group
    pairs = np.unique(tok.type_idx[tr] * len(tok.n) + tok.pid[tr])
    return {"group": group,
            "train_count": np.bincount(tok.type_idx[tr], minlength=len(tok.types)),
            "train_passages": np.bincount(pairs // len(tok.n), minlength=len(tok.types))}


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
    """One draw of the count-nearest control: one random token type per carrier, in carrier
    order, of about the same train count.

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


def mass_matched_random(train_count: np.ndarray, carriers: Sequence[int],
                        rng: np.random.Generator) -> np.ndarray:
    """One draw of the mass-matched control for one set of dropped carriers.

    Token types are drawn without replacement from the train types that are not among
    ``carriers`` (special tokens are eligible like any other type), each draw with
    probability proportional to train count, and added until their cumulative train count
    first meets or exceeds the carriers'. The control therefore drops at least as many
    train tokens as the carriers do, and more by less than the count of the last type
    added. Returns the type indices in draw order; all eligible types when together they
    hold fewer train tokens than the carriers.
    """
    count = np.asarray(train_count, dtype=np.int64)
    carriers = np.asarray(carriers, dtype=np.int64)
    eligible = count > 0
    eligible[carriers] = False
    index = np.flatnonzero(eligible)
    target = int(count[carriers].sum())
    if not len(index) or target <= 0:
        return np.array([], dtype=np.int64)
    # Successive draws proportional to count are the ascending order of Exp(1) / count.
    order = index[np.argsort(rng.exponential(size=len(index)) / count[index], kind="stable")]
    reached = int(np.searchsorted(np.cumsum(count[order]), target))  # first cumsum >= target
    return order[:reached + 1]


def ablation_arms(type_share: np.ndarray, train_count: np.ndarray,
                  seed: Sequence[int] = (RANDOM_SEED,)) -> List[Dict]:
    """The token-ablation arms of one model-layer, as sets of token-type indices.

    ``type_share`` [N_PC, n_types] holds each type's train share of the PC1, PC2 and PC3
    score variance. Two rankings (RANKINGS); per ranking:
      carrier       the top m types, m in ABL_MS;
      control       the count-nearest control: CONTROL_DRAWS draws of matched_random, each
                    read at the same m; a control type is in the top max(ABL_MS) of
                    neither ranking;
      control_mass  the mass-matched control: for every m, CONTROL_DRAWS draws of
                    mass_matched_random against the top m carriers of this ranking, seeded
                    per (ranking, m, draw).
    Every arm carries ``carrier_count``, the train tokens of the carriers it is matched to.
    """
    score = {"pc1": type_share[0], "pc123": type_share[:N_PC].mean(axis=0)}
    top = {r: rank_types(score[r], train_count)[:max(ABL_MS)] for r in RANKINGS}
    excluded = sorted({int(i) for r in RANKINGS for i in top[r]})
    arms: List[Dict] = []

    def add(ranking: str, kind: str, draw: int, m: int, types: np.ndarray) -> None:
        arms.append({"ranking": ranking, "kind": kind, "draw": draw, "m": m, "types": types,
                     "carrier_count": int(train_count[top[ranking][:m]].sum())})

    for ri, ranking in enumerate(RANKINGS):
        for m in ABL_MS:
            add(ranking, "carrier", -1, m, top[ranking][:m])
        for draw in range(CONTROL_DRAWS):
            rng = np.random.default_rng([*seed, ri, draw])
            picks = matched_random(train_count, top[ranking], excluded, rng)
            for m in ABL_MS:
                add(ranking, "control", draw, m, picks[:m])
        for m in ABL_MS:
            for draw in range(CONTROL_DRAWS):
                rng = np.random.default_rng([*seed, MASS_STREAM, ri, m, draw])
                add(ranking, "control_mass", draw, m,
                    mass_matched_random(train_count, top[ranking][:m], rng))
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


# --------------------------------------------------------------------------- #
# audit: split, caches and encoders
# --------------------------------------------------------------------------- #

@dataclass
class Context:
    """The passages of one run, in split-CSV order (the first --limit of each split)."""

    sp: pd.DataFrame            # the selected split rows, index reset
    sel: np.ndarray             # their row numbers in the full split CSV
    tr: np.ndarray              # train mask over ``sp``
    te: np.ndarray              # test mask over ``sp``
    resolver: AlignmentResolver  # aligns a cache to the FULL split by filename
    split_path: Path            # the CSV the metric workers load (asw._init)
    limit: int


def limit_rows(split: pd.DataFrame, limit: int) -> np.ndarray:
    """Row numbers of the first ``limit`` train and first ``limit`` test passages (all
    rows when ``limit`` is 0), in split order."""
    if not limit:
        return np.arange(len(split))
    parts = [np.flatnonzero((split["split"] == name).to_numpy())[:limit]
             for name in ("train", "test")]
    return np.sort(np.concatenate(parts))


def build_context(split_csv: Path, out_dir: Path, limit: int) -> Context:
    full = pd.read_csv(split_csv)
    sel = limit_rows(full, limit)
    sp = full.iloc[sel].reset_index(drop=True)
    split_path = Path(split_csv)
    if limit:
        out_dir.mkdir(parents=True, exist_ok=True)
        split_path = out_dir / f"split_limit{limit}.csv"
        sp.to_csv(split_path, index=False)
    return Context(sp=sp, sel=sel, tr=(sp["split"] == "train").to_numpy(),
                   te=(sp["split"] == "test").to_numpy(), resolver=AlignmentResolver(full),
                   split_path=split_path, limit=int(limit))


def cache_dir(bases_root: Path, model_id: str, subdir: str) -> Path:
    return Path(bases_root) / "phase9_bases" / asw.slug(model_id) / subdir


def load_cached(ctx: Context, bases_root: Path, model_id: str, subdir: str, layer: int
                ) -> Optional[np.ndarray]:
    """The cached vectors of the run's passages, aligned by filename; None if absent."""
    suffix = "_sif" if subdir == SIF_SUBDIR else ""
    path = cache_dir(bases_root, model_id, subdir) / f"hidden_layer{layer}_embeddings{suffix}.npy"
    if not path.exists():
        return None
    return ctx.resolver.load(path)[ctx.sel]


def forward_order(run_dir: Path, sp: pd.DataFrame) -> np.ndarray:
    """Rows of ``sp`` in the order the cache was extracted, so that the batches (and their
    padding) are the extraction's. Falls back to split order without a manifest."""
    manifest = find_manifest(run_dir)
    if manifest is None:
        print(f"WARNING: no row-order manifest in {run_dir}; batching in split order",
              flush=True)
        return np.arange(len(sp))
    pos = {str(f): i for i, f in enumerate(sp["filename"])}
    order = [pos[name] for name in load_row_order(manifest) if name in pos]
    if len(order) != len(sp):
        raise SystemExit(f"ERROR: {manifest} lists {len(order)} of the {len(sp)} passages")
    return np.array(order, dtype=np.int64)


@dataclass
class Encoder:
    """What the audit needs from a model; open_encoder builds it from HuggingFace, the
    tests from toy tensors.

    ``batches()`` yields (rows, input_ids [B, T], attention_mask [B, T], hidden_states)
    with ``rows`` the passage indices of the batch and hidden_states[layer] a [B, T, d]
    tensor, the same batches on every call.
    """

    batches: Callable[[], Iterator]
    keep_lookup: np.ndarray
    special_ids: List[int]
    token_probs: Dict[int, float]
    describe: Callable[[int], Tuple[str, str]]  # token id -> (vocabulary piece, decoded)
    reference_pool: Optional[Callable] = None   # the CLI's pooling function (self-check)
    close: Optional[Callable[[], None]] = None


def open_encoder(model_id: str, ctx: Context, bases_root: Path) -> Encoder:
    """Load a model and tokenizer exactly as its extraction CLI does (EXTRACT), with the
    CLI's keep lookup, special ids and train-only SIF token probabilities."""
    import torch
    from canon_retrieval import load_texts
    from sif_abtt import token_probabilities
    from token_filtering import build_token_keep_lookup

    cli_name, trust = EXTRACT[model_id]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cli_name == "hidden":
        import extract_hidden_cli as cli
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        net = model.get_encoder() if hasattr(model, "get_encoder") else model.encoder
        reference_pool = cli.pool_hidden
    else:
        import extract_encoder_cli as cli
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust)
        net = AutoModel.from_pretrained(model_id, trust_remote_code=trust)
        reference_pool = cli.pool_embeddings
    net.to(device)
    net.eval()
    keep_lookup = build_token_keep_lookup(tokenizer, TOKEN_FILTER)
    paths = ctx.sp["path"].tolist()
    train_texts = load_texts(ctx.sp.loc[ctx.tr, "path"].tolist())
    token_probs = token_probabilities(tokenizer, train_texts, batch_size=128,
                                      max_length=MAX_LENGTH, token_keep_lookup=keep_lookup)
    order = forward_order(cache_dir(bases_root, model_id, MEAN_SUBDIR), ctx.sp)

    def batches():
        for start in range(0, len(order), BATCH_SIZE):
            rows = order[start:start + BATCH_SIZE]
            enc = tokenizer(load_texts([paths[i] for i in rows]), truncation=True,
                            max_length=MAX_LENGTH, padding=True, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            with torch.no_grad():
                out = net(input_ids=input_ids, attention_mask=attention_mask,
                          output_hidden_states=True, return_dict=True)
            yield rows, input_ids, attention_mask, out.hidden_states

    def describe(token_id: int) -> Tuple[str, str]:
        return (str(tokenizer.convert_ids_to_tokens(int(token_id))),
                str(tokenizer.decode([int(token_id)])))

    def close() -> None:
        net.to("cpu")
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"  loaded {DISP.get(model_id, model_id)} on {device} ({cli_name} CLI"
          f"{', trust_remote_code' if trust else ''}); vocabulary {len(keep_lookup)}, "
          f"{len(token_probs)} train token types, {len(set(tokenizer.all_special_ids))} "
          f"special ids", flush=True)
    return Encoder(batches=batches, keep_lookup=keep_lookup,
                   special_ids=sorted(set(int(i) for i in tokenizer.all_special_ids)),
                   token_probs=token_probs, describe=describe,
                   reference_pool=reference_pool, close=close)


# --------------------------------------------------------------------------- #
# audit: the two forward passes
# --------------------------------------------------------------------------- #

def pass_one(enc: Encoder, layers: Sequence[int], dirs: Dict[int, Dict],
             lookups: Dict[str, np.ndarray], frequent: np.ndarray, n_passages: int) -> Dict:
    """Pooled vectors of the five arms and the token projections on every direction.

    Returns pooled [arm, layer, N, d] float32 (no fallback applied), the weight total and
    the special / frequent weight of every passage under each arm [arm, N], the flat kept
    tokens (pid, tid, first), their contributions c_t per layer ([n_tok, K] float64), the
    projections of the float64 mean-pooled vectors [layer, N, K] (score identity), the
    attention length of every passage, and the largest difference to the CLI's own
    pooling function on the first batch.
    """
    import torch
    from token_filtering import torch_token_keep_mask

    n_arm, n_layer, k = len(ARMS), len(layers), len(DIRECTIONS)
    pooled = None
    total = np.zeros((n_arm, n_passages))
    special_w = np.zeros((n_arm, n_passages))
    frequent_w = np.zeros((n_arm, n_passages))
    n_att = np.zeros(n_passages, dtype=np.int64)
    s_pool = np.zeros((n_layer, n_passages, k))
    pid: List[np.ndarray] = []
    tid: List[np.ndarray] = []
    first: List[np.ndarray] = []
    contrib: List[List[np.ndarray]] = [[] for _ in layers]
    selfcheck = {"mean": float("nan"), "sif": float("nan")}
    for rows, input_ids, attention_mask, hidden_states in enc.batches():
        rows = np.asarray(rows, dtype=np.int64)
        dev = input_ids.device
        first_batch = pooled is None
        if first_batch:
            d = hidden_states[layers[0]].shape[-1]
            pooled = np.zeros((n_arm, n_layer, n_passages, d), dtype=np.float32)
            w_arm = [torch.as_tensor(lookups[a], device=dev, dtype=torch.float32) for a in ARMS]
            vocab = len(enc.keep_lookup)
            is_special = torch.zeros(vocab, device=dev)
            inside = [i for i in enc.special_ids if 0 <= i < vocab]
            is_special[torch.as_tensor(inside, device=dev, dtype=torch.long)] = 1.0
            is_frequent = torch.zeros(vocab, device=dev)
            is_frequent[torch.as_tensor(frequent, device=dev, dtype=torch.long)] = 1.0
            mu = [torch.as_tensor(dirs[x]["mu"], device=dev, dtype=torch.float64) for x in layers]
            w_dir = [torch.as_tensor(dirs[x]["W"], device=dev, dtype=torch.float64)
                     for x in layers]
        keep = torch_token_keep_mask(input_ids, attention_mask, enc.keep_lookup)
        att = attention_mask.float()
        wm = [att * w[input_ids] for w in w_arm]
        sp_tok, fr_tok = is_special[input_ids], is_frequent[input_ids]
        for a, w in enumerate(wm):
            total[a, rows] = w.sum(dim=1).double().cpu().numpy()
            special_w[a, rows] = (w * sp_tok).sum(dim=1).double().cpu().numpy()
            frequent_w[a, rows] = (w * fr_tok).sum(dim=1).double().cpu().numpy()
        n_att[rows] = attention_mask.sum(dim=1).cpu().numpy()
        kept = keep > 0
        b_idx, t_idx = torch.nonzero(kept, as_tuple=True)  # row-major, as kept-mask indexing
        first_t = torch.argmax(kept.int(), dim=1)
        pid.append(rows[b_idx.cpu().numpy()])
        tid.append(input_ids[kept].cpu().numpy())
        first.append((t_idx == first_t[b_idx]).cpu().numpy())
        keep64 = keep.double()
        count = keep64.sum(dim=1, keepdim=True).clamp(min=1.0)
        vecs, scores = [], []
        for li, layer in enumerate(layers):
            h = hidden_states[layer]
            vecs.append(torch.stack([pool_weighted(h, w) for w in wm]))  # [arm, B, d]
            h64 = h.double()
            proj = (h64 - mu[li]) @ w_dir[li].T                           # [B, T, K]
            contrib[li].append(proj[kept].cpu().numpy())
            mean64 = (h64 * keep64.unsqueeze(-1)).sum(dim=1) / count
            scores.append((mean64 - mu[li]) @ w_dir[li].T)                # [B, K]
            if first_batch and enc.reference_pool is not None:
                # the two reference arms against the CLI's own function
                ref = enc.reference_pool(h, attention_mask, "mean", input_ids=input_ids,
                                         token_keep_lookup=enc.keep_lookup)
                dm = float((vecs[-1][ARMS.index("mean")] - ref).abs().max())
                ref = enc.reference_pool(h, attention_mask, "sif", input_ids=input_ids,
                                         token_probs=enc.token_probs, sif_a=SIF_A,
                                         special_ids=set(enc.special_ids),
                                         token_keep_lookup=enc.keep_lookup)
                ds = float((vecs[-1][ARMS.index("sif")] - ref).abs().max())
                selfcheck = {"mean": float(np.nanmax([selfcheck["mean"], dm])),
                             "sif": float(np.nanmax([selfcheck["sif"], ds]))}
        pooled[:, :, rows] = torch.stack(vecs, dim=1).float().cpu().numpy()
        s_pool[:, rows] = torch.stack(scores).cpu().numpy()
    if pooled is None:
        raise SystemExit("ERROR: the encoder yielded no batch")
    return {"pooled": pooled, "total": total, "special_w": special_w,
            "frequent_w": frequent_w, "n_att": n_att, "s_pool": s_pool,
            "pid": np.concatenate(pid), "tid": np.concatenate(tid),
            "first": np.concatenate(first), "contrib": contrib, "selfcheck": selfcheck}


def pass_two(enc: Encoder, layers: Sequence[int], drop: np.ndarray, mean_ref: np.ndarray
             ) -> Dict:
    """Mean pooling with token types dropped: ``drop`` is bool [layer, arm, vocab].

    A passage left with no token gets its mean-pooled vector. Returns vectors
    [layer, arm, N, d] float32, kept-token counts [layer, arm, N], and the largest
    difference of this pass's mean pooling to pass 1's (``mean_ref`` [layer, N, d]).
    """
    import torch
    from token_filtering import torch_token_keep_mask

    n_layer, n_arm, _ = drop.shape
    n_passages, d = mean_ref.shape[1], mean_ref.shape[2]
    out = np.zeros((n_layer, n_arm, n_passages, d), dtype=np.float32)
    kept_n = np.zeros((n_layer, n_arm, n_passages), dtype=np.float32)
    drop_t = None
    pass_diff = 0.0
    for rows, input_ids, attention_mask, hidden_states in enc.batches():
        rows = np.asarray(rows, dtype=np.int64)
        if drop_t is None:
            drop_t = torch.as_tensor(drop, device=input_ids.device)
        keep = torch_token_keep_mask(input_ids, attention_mask, enc.keep_lookup)
        vecs, counts = [], []
        for li, layer in enumerate(layers):
            h = hidden_states[layer]
            mean_vec = pool_weighted(h, keep)                               # [B, d]
            ref = torch.as_tensor(mean_ref[li, rows], device=h.device)
            pass_diff = max(pass_diff, float((mean_vec.float() - ref).abs().max()))
            k2 = keep.unsqueeze(0) * (~drop_t[li][:, input_ids]).to(keep.dtype)  # [arm, B, T]
            n2 = k2.sum(dim=2)                                              # [arm, B]
            v = torch.einsum("abt,btd->abd", k2.to(h.dtype), h) / n2.clamp(min=1.0).unsqueeze(-1)
            v = torch.where((n2 == 0).unsqueeze(-1), mean_vec.unsqueeze(0), v)
            vecs.append(v)
            counts.append(n2)
        out[:, :, rows] = torch.stack(vecs).float().cpu().numpy()
        kept_n[:, :, rows] = torch.stack(counts).float().cpu().numpy()
    return {"pooled": out, "kept_n": kept_n, "pass_diff": pass_diff}


# --------------------------------------------------------------------------- #
# audit: readouts of one model
# --------------------------------------------------------------------------- #

def _metric_task(task) -> Tuple[Tuple, Dict[str, float]]:
    """One metric cell in a worker: the E1 metric block and train geometry ("full"), or
    Task A test and train AUROC alone ("auroc"; the same number as the block's)."""
    key, mode, train, test = task
    if mode == "full":
        m = asw._metrics(train, test)
        return key, {**{c: m[c] for c in asw.KEEP if c in m}, **e1._geometry(train)}
    return key, {"aucroc": asw.pair_auroc(test, asw._CTX["lab_te"]),
                 "train_aucroc": asw.pair_auroc(train, asw._CTX["lab_tr"])}


def _vec_diff(ours: np.ndarray, ref: np.ndarray) -> Tuple[float, float]:
    """(largest absolute difference, largest relative L2 difference over passages)."""
    a, b = ours.astype(np.float64), ref.astype(np.float64)
    norm = np.linalg.norm(b, axis=1)
    rel = np.linalg.norm(a - b, axis=1)[norm > 0] / norm[norm > 0]
    return float(np.abs(a - b).max()), float(rel.max()) if len(rel) else float("nan")


AUDIT_NAN = ("r2_tokenmix", "ev_tokenmix_eqw", *(f"share_{g}_eqw" for g in GROUPS),
             "share_sum_err_eqw", "share_first_eqw", "rho_logn", "rho_freqmass",
             "identity_max_abs_diff", "identity_max_rel_diff", "cache_score_max_abs_diff")


def layer_audit(key: Dict, contrib: np.ndarray, tok: TokenTable, dirs: Dict,
                train_rows: np.ndarray, test_rows: np.ndarray, masses: np.ndarray,
                s_pool: np.ndarray, cached: np.ndarray, describe: Callable,
                type_info: Dict[str, np.ndarray], e1_agrees: int = -1
                ) -> Tuple[List[Dict], List[Dict], np.ndarray]:
    """Readout 2 of one model-layer: (audit rows, carrier rows, type shares [N_PC, types]).

    ``contrib`` [n_tok, K] holds c_t on the K directions, ``masses`` [N, group] the token
    mass fractions of every passage, ``s_pool`` [N, K] the projections of the float64
    mean-pooled vectors and ``cached`` [N, d] the cached vectors (float32).
    """
    rows = {"train": train_rows, "test": test_rows}
    logn = np.log(np.maximum(tok.n, 1))
    freq_mass = masses[:, GROUPS.index("frequent")]
    cached64 = cached.astype(np.float64)
    norms = np.linalg.norm(cached64, axis=1)
    s_cache = (cached64 - dirs["mu"]) @ dirs["W"].T
    seen_train = np.bincount(tok.type_idx[train_rows[tok.pid]], minlength=len(tok.types)) > 0
    base: Dict[str, Dict] = {}
    for split, r in rows.items():
        valid = r & (tok.n > 0)
        tok_in = valid[tok.pid]
        base[split] = {
            "total_var": float(cached64[valid].var(axis=0).sum()) if valid.any() else np.nan,
            **{f"mass_{g}": float(masses[valid, i].mean()) if valid.any() else np.nan
               for i, g in enumerate(GROUPS)},
            "unseen_token_frac": (float((~seen_train[tok.type_idx[tok_in]]).mean())
                                  if tok_in.any() else np.nan)}

    def row(name: str, kind: str, dim: int, coord: int, split: str, r: Dict) -> Dict:
        x = r[split]
        b = base[split]
        with np.errstate(divide="ignore", invalid="ignore"):
            var_share = x["var_s"] / b["total_var"] if b["total_var"] else np.nan
        return {**key, "direction": name, "kind": kind, "dim": dim, "coord": coord,
                "split": split, "n_passages": x["n_passages"], "var_s": x["var_s"],
                "var_share": var_share, "ev_tokenmix": x["ev"],
                **share_columns(x["token_share"], tok), **{c: np.nan for c in AUDIT_NAN},
                **{k: v for k, v in b.items() if k != "total_var"},
                "e1_rank_agrees": e1_agrees if kind == "coord" else -1}

    audit: List[Dict] = []
    carriers: List[Dict] = []
    type_share = np.zeros((N_PC, len(tok.types)))
    col = {name: i for i, name in enumerate(dirs["names"])}
    for k, name in enumerate(dirs["names"]):
        c = contrib[:, k]
        r = audit_scores(c, tok, train_rows, test_rows)
        s = r["s"][:, 0]
        for split in rows:
            valid = r[split]["valid"]
            out = row(name, dirs["kinds"][k], 1, dirs["coords"][k], split, r)
            out["r2_tokenmix"] = r[split]["r2"]
            out["rho_logn"] = _spearman(s[valid], logn[valid])
            out["rho_freqmass"] = _spearman(s[valid], freq_mass[valid])
            if valid.any():
                diff = np.abs(s - s_pool[:, k])[valid]
                out["identity_max_abs_diff"] = float(diff.max())
                out["identity_max_rel_diff"] = float(
                    (diff / np.maximum(norms[valid], 1e-300)).max())
                out["cache_score_max_abs_diff"] = float(np.abs(s - s_cache[:, k])[valid].max())
            audit.append(out)
        if k < N_PC:
            share = np.bincount(tok.type_idx, weights=np.nan_to_num(r["train"]["token_share"]),
                                minlength=len(tok.types))
            type_share[k] = share
            tr_tok = train_rows[tok.pid]
            count = r["type_count"]
            mean_abs = (np.bincount(tok.type_idx[tr_tok], weights=np.abs(c[tr_tok]),
                                    minlength=len(tok.types)) / np.maximum(count, 1))
            for rank, t in enumerate(rank_types(share, count)[:TOP_CARRIERS], start=1):
                piece, decoded = describe(int(tok.types[t]))
                carriers.append({**key, "direction": name, "rank": rank,
                                 "token_id": int(tok.types[t]), "token": visible(piece),
                                 "decoded": visible(decoded),
                                 "group": GROUPS[int(type_info["group"][t])],
                                 "train_count": int(count[t]),
                                 "train_passages": int(type_info["train_passages"][t]),
                                 "mean_c": float(r["type_mean"][t, 0]),
                                 "mean_abs_c": float(mean_abs[t]), "share": float(share[t])})
    for name, kind, members in SUBSPACES:
        c = contrib[:, [col[m] for m in members]]
        r = audit_scores(c, tok, train_rows, test_rows)
        q = audit_scores(c, tok, train_rows, test_rows, equal_weight=True)
        for split in rows:
            out = row(name, kind, len(members), -1, split, r)
            out["ev_tokenmix_eqw"] = q[split]["ev"]
            out.update(share_columns(q[split]["token_share"], tok, suffix="_eqw"))
            audit.append(out)
    return audit, carriers, type_share


def audit_model(model_id: str, enc: Encoder, ctx: Context, bases_root: Path, pool=None,
                control_metrics: str = "auroc", e1_coords: Optional[pd.DataFrame] = None
                ) -> Dict[str, pd.DataFrame]:
    """All E2 rows of one model: {csv name: frame}."""
    t0 = time.time()
    name = DISP.get(model_id, model_id)
    layers = asw.discover(Path(bases_root), model_id)
    if not layers:
        raise SystemExit(f"ERROR: no cached vectors for {name} under "
                         f"{cache_dir(bases_root, model_id, MEAN_SUBDIR)}")
    n_passages = len(ctx.sp)
    tr, te = ctx.tr, ctx.te
    seed = (RANDOM_SEED, MODEL_INDEX.get(model_id, len(MODEL_INDEX)))
    cached = {x: load_cached(ctx, bases_root, model_id, MEAN_SUBDIR, x) for x in layers}
    dirs = {x: fit_directions(cached[x][tr], seed=(*seed, x)) for x in layers}
    frequent = frequent_ids(enc.token_probs)
    lookups = arm_weight_lookups(len(enc.keep_lookup), enc.token_probs, enc.special_ids,
                                 enc.keep_lookup)
    print(f"  {name}: {len(layers)} layers, directions fit ({time.time() - t0:.0f}s)", flush=True)

    # ---- pass 1: pooling arms and token projections
    p1 = pass_one(enc, layers, dirs, lookups, frequent, n_passages)
    tok = build_token_table(p1["pid"], p1["tid"], p1["first"], n_passages, enc.special_ids,
                            frequent)
    pooled = p1["pooled"]
    print(f"  {name}: pass 1 done, {len(tok.pid)} kept tokens, {len(tok.types)} token types; "
          f"first batch against the CLI pooling function: mean {p1['selfcheck']['mean']:.2e}, "
          f"sif {p1['selfcheck']['sif']:.2e} ({time.time() - t0:.0f}s)", flush=True)
    if not np.array_equal(tok.n, np.rint(p1["total"][ARMS.index("mean")]).astype(np.int64)):
        raise SystemExit(f"ERROR: {name}: token table and mean-arm weights disagree on n_p")
    # the token table's train frequencies must be the SIF probabilities (same tokenization)
    tr_tok = tr[tok.pid] & (tok.group != GROUPS.index("special"))
    counts = np.bincount(tok.type_idx[tr_tok], minlength=len(tok.types))
    probs = np.array([enc.token_probs.get(int(t), 0.0) for t in tok.types])
    prob_diff = float(np.abs(counts / max(counts.sum(), 1) - probs).max())
    print(f"  {name}: train token frequencies against token_probabilities: max |diff| "
          f"{prob_diff:.2e}", flush=True)

    # vectors against the caches, before any fallback
    vec = {}
    for arm, subdir in (("mean", MEAN_SUBDIR), ("sif", SIF_SUBDIR)):
        for li, x in enumerate(layers):
            ref = cached[x] if arm == "mean" else load_cached(ctx, bases_root, model_id, subdir, x)
            vec[(arm, li)] = ((np.nan, np.nan) if ref is None
                              else _vec_diff(pooled[ARMS.index(arm), li], ref))
    empty = p1["total"] == 0                                     # [arm, N]
    for a, arm in enumerate(ARMS):
        if arm not in NO_FALLBACK and empty[a].any():
            pooled[a][:, empty[a]] = pooled[ARMS.index("mean")][:, empty[a]]

    # ---- readout 2 and 4 (CPU), and the carrier rankings pass 2 needs
    masses = group_masses(tok)
    type_info = type_table(tok, tr)
    train_count = type_info["train_count"]
    key0 = {"model": model_id}
    audit_rows: List[Dict] = []
    carrier_rows: List[Dict] = []
    arms_by_layer: List[List[Dict]] = []
    seen_tokens: Dict[int, Tuple[str, str]] = {}

    def describe(token_id: int) -> Tuple[str, str]:  # a carrier recurs across layers
        if token_id not in seen_tokens:
            seen_tokens[token_id] = enc.describe(token_id)
        return seen_tokens[token_id]

    for li, x in enumerate(layers):
        contrib = np.concatenate(p1["contrib"][li])
        p1["contrib"][li] = None
        agrees = -1
        if e1_coords is not None:
            ref = e1_coords[(e1_coords["model"] == model_id) & (e1_coords["layer"] == x)
                            & (e1_coords["ranking"] == "variance")].sort_values("rank")
            if len(ref) >= N_COORD:
                agrees = int(list(ref["coord"].astype(int))[:N_COORD]
                             == dirs[x]["coords"][N_PC:N_PC + N_COORD])
        a, c, share = layer_audit({**key0, "layer": int(x)}, contrib, tok, dirs[x], tr, te,
                                  masses, p1["s_pool"][li], cached[x], describe,
                                  type_info, agrees)
        audit_rows += a
        carrier_rows += c
        arms_by_layer.append(ablation_arms(share, train_count, seed=(*seed, x)))
        del contrib
    print(f"  {name}: token audit done ({time.time() - t0:.0f}s)", flush=True)

    lab = {"train": asw._CTX["lab_tr"], "test": asw._CTX["lab_te"]}
    length_rows = []
    for split, r in (("train", tr), ("test", te)):
        n = tok.n[r]
        length_rows.append({**key0, "split": split, "n_passages": int(r.sum()),
                            "n_zero_token": int((n == 0).sum()),
                            "n_truncated": int((p1["n_att"][r] >= MAX_LENGTH).sum()),
                            "mean_n": float(n.mean()), "median_n": float(np.median(n)),
                            **length_stats(n, lab[split])})

    # ---- pass 2: token ablation
    n_abl = len(arms_by_layer[0])
    drop = np.zeros((len(layers), n_abl, len(enc.keep_lookup)), dtype=bool)
    for li, arms in enumerate(arms_by_layer):
        for j, arm in enumerate(arms):
            drop[li, j, tok.types[arm["types"]]] = True
    p2 = pass_two(enc, layers, drop, pooled[ARMS.index("mean")])
    del drop
    print(f"  {name}: pass 2 done, mean pooling differs from pass 1 by at most "
          f"{p2['pass_diff']:.2e} ({time.time() - t0:.0f}s)", flush=True)

    # ---- metrics
    def tasks():
        for li in range(len(layers)):
            for a in range(len(ARMS)):
                yield ("arm", li, a), "full", pooled[a, li][tr], pooled[a, li][te]
            for j, arm in enumerate(arms_by_layer[li]):
                mode = "full" if arm["kind"] == "carrier" or control_metrics == "full" else "auroc"
                yield (("abl", li, j), mode, p2["pooled"][li, j][tr], p2["pooled"][li, j][te])

    it = pool.imap_unordered(_metric_task, tasks(), chunksize=1) if pool else map(
        _metric_task, tasks())
    metrics: Dict[Tuple, Dict[str, float]] = {}
    for k, m in it:
        metrics[k] = m
        if k[0] == "arm" and k[2] == 0:
            print(f"    {name} L{layers[k[1]]}: mean arm AUROC {m['aucroc']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    block = list(asw.KEEP) + ["pc1_share_train", "pc10_share_train", "eff_rank_train"]
    total = p1["total"]
    arm_rows, abl_rows = [], []
    for li, x in enumerate(layers):
        for a, arm in enumerate(ARMS):
            m = metrics[("arm", li, a)]
            has = total[a] > 0
            d_abs, d_rel = vec.get((arm, li), (np.nan, np.nan))
            arm_rows.append({
                **key0, "layer": int(x), "arm": arm, **{c: m.get(c, np.nan) for c in block},
                "n_no_token": int(empty[a].sum()),
                "n_fallback": 0 if arm in NO_FALLBACK else int(empty[a].sum()),
                "special_mass": float((p1["special_w"][a][has] / total[a][has]).mean()),
                "frequent_mass": float((p1["frequent_w"][a][has] / total[a][has]).mean()),
                "vec_max_abs_diff": d_abs, "vec_max_rel_diff": d_rel,
                "cli_pool_max_abs_diff": p1["selfcheck"].get(arm, np.nan),
                "n_train": int(tr.sum()), "n_test": int(te.sum()), "limit": ctx.limit})
        has = tok.n > 0
        for j, arm in enumerate(arms_by_layer[li]):
            m = metrics[("abl", li, j)]
            kept = p2["kept_n"][li, j]
            dropped = 1.0 - kept[has] / tok.n[has]
            # a mass-matched draw can hold thousands of types: its ids are listed only
            # when few (the draw is reproducible from its seed)
            listed = arm["kind"] != "control_mass" or len(arm["types"]) <= TYPES_LISTED
            abl_rows.append({
                **key0, "layer": int(x), "ranking": arm["ranking"], "kind": arm["kind"],
                "draw": arm["draw"], "m": arm["m"], "n_types": int(len(arm["types"])),
                "types": (";".join(str(int(t)) for t in tok.types[arm["types"]])
                          if listed else ""),
                **{c: m.get(c, np.nan) for c in block},
                "n_fallback": int(((kept == 0) & has).sum()),
                "dropped_mass_train": float(dropped[tr[has]].mean()),
                "dropped_mass_test": float(dropped[te[has]].mean()),
                "train_count_dropped": int(train_count[arm["types"]].sum()),
                "carrier_train_count": int(arm["carrier_count"])})
    print(f"  {name}: done in {time.time() - t0:.0f}s", flush=True)
    return {ARMS_NAME: pd.DataFrame(arm_rows), AUDIT_NAME: pd.DataFrame(audit_rows),
            CARRIER_NAME: pd.DataFrame(carrier_rows), ABL_NAME: pd.DataFrame(abl_rows),
            LENGTH_NAME: pd.DataFrame(length_rows)}


def cmd_audit(args) -> int:
    model_ids = asw.pick_models(args.models, MODELS)
    if not model_ids:
        raise SystemExit(f"--models {args.models!r} matches none of {ORDER}")
    for mid in model_ids:
        if not asw.discover(args.bases_root, mid):
            raise SystemExit(f"ERROR: no cached vectors for {DISP[mid]} under "
                             f"{cache_dir(args.bases_root, mid, MEAN_SUBDIR)}")
    t0 = time.time()
    ctx = build_context(args.split_csv, args.out_dir, args.limit)
    print(f"passages: {int(ctx.tr.sum())} train, {int(ctx.te.sum())} test"
          + (f" (--limit {args.limit}: smoke run, gates not enforced)" if args.limit else "")
          + f"; output {args.out_dir}", flush=True)
    pool = None
    if args.workers > 1:
        # forked here, before torch is imported and before any GPU context exists
        from multiprocessing import Pool

        pool = Pool(args.workers, initializer=asw._init, initargs=(str(ctx.split_path),))
    asw._init(str(ctx.split_path))
    e1_coords = None
    if not args.limit and Path(args.e1_coords_csv).exists():
        e1_coords = pd.read_csv(args.e1_coords_csv)
    try:
        for mid in model_ids:
            print(f"=== {DISP[mid]} ===", flush=True)
            enc = open_encoder(mid, ctx, args.bases_root)
            try:
                frames = audit_model(mid, enc, ctx, args.bases_root, pool,
                                     args.control_metrics, e1_coords)
            finally:
                if enc.close is not None:
                    enc.close()
                del enc
            for name, frame in frames.items():  # after every model: a killed job keeps them
                merge_write(args.out_dir / name, frame)
            print(f"wrote the rows of {DISP[mid]} to {args.out_dir} ({time.time() - t0:.0f}s)",
                  flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    if not args.check and not args.limit:
        return 0
    return run_gates(args, complete=True, write=True)


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #

def is_limit_run(arms: pd.DataFrame) -> bool:
    return bool("limit" in arms.columns and (arms["limit"].fillna(0) > 0).any())


def gate_table(arms: pd.DataFrame, audit: pd.DataFrame, res: Optional[pd.DataFrame],
               complete: bool = True, tol_auroc: float = GATE_TOL_AUROC,
               expected: Optional[Sequence[str]] = ALL_MODEL_IDS,
               auroc_gates: bool = True) -> pd.DataFrame:
    """One row per (gate, model): cells compared, the largest difference and the cells
    over tolerance by name. See the module docstring for the gates.

    Every number is a cell of the two CSVs or of the published results. A NaN on either
    side of a comparison counts as over tolerance, a missing published cell fails its
    gate, and so does an expected model with no rows (gate 0). ``complete`` also requires
    every published layer of a computed model. Gate 2b is reported and never fails
    (``gated`` = 0). ``auroc_gates=False`` leaves out 1a and 2a (a --limit run scores a
    subset of the passages, so the published cells do not apply).
    """
    ours = arms.set_index(["model", "layer", "arm"])
    pub: Dict[str, Dict] = {"mean": {}, "sif": {}}
    if res is not None:
        for arm, pooling, method in (("mean", "mean", "baseline"), ("sif", "sif", "sif_only")):
            r = res[(res["repr"] == "hidden") & (res["pooling"] == pooling)
                    & (res["method"] == method)]
            pub[arm] = {(m, int(x)): float(v) for m, x, v in zip(r["model"], r["layer"],
                                                                 r["aucroc"])}
    present = set(arms["model"])
    wanted = set(expected) if expected is not None else set()
    known = list(ALL_MODEL_IDS) + sorted((present | wanted) - set(ALL_MODEL_IDS))
    nan = float("nan")
    rows = []

    def add(gate: str, mid: str, gated: bool, tol: float, cells, absent: int = 0,
            note: str = "") -> None:
        # cells: (label, difference or None when the reference is missing)
        diffs = [(lab, float(d)) for lab, d in cells if d is not None]
        n_missing = sum(1 for _, d in cells if d is None)
        # np.max keeps a NaN visible, and "not d <= tol" counts a NaN as over tolerance
        mx = float(np.max([d for _, d in diffs])) if diffs else nan
        over = [(lab, d) for lab, d in diffs if not d <= tol]
        ok = (bool(diffs) or not gated) and n_missing == 0 and not absent and not over
        shown = "; ".join(f"{lab} {d:.2e}" for lab, d in over[:12])
        if len(over) > 12:
            shown += f"; and {len(over) - 12} more"
        rows.append({"gate": gate, "model": mid, "gated": int(gated), "n_cells": len(diffs),
                     "n_missing_reference": n_missing, "n_published_layers_absent": absent,
                     "max_diff": mx, "tolerance": tol, "n_over_tolerance": len(over),
                     "cells_over_tolerance": shown, "note": note,
                     "ok": bool(ok) if gated else True})

    for mid in [m for m in known if m in present or m in wanted]:
        if mid not in present:
            rows.append({"gate": "0 expected model has rows in the pooling CSV", "model": mid,
                         "gated": 1, "n_cells": 0, "n_missing_reference": 0,
                         "n_published_layers_absent": 0, "max_diff": nan, "tolerance": nan,
                         "n_over_tolerance": 0, "cells_over_tolerance": "", "note": "",
                         "ok": False})
            continue
        mine = sorted(int(x) for x in arms.loc[arms["model"] == mid, "layer"].unique())

        def cell(x: int, arm: str, column: str) -> float:
            return float(ours.loc[(mid, x, arm), column]) if (mid, x, arm) in ours.index else nan

        for tag, arm, cache, what in (("1", "mean", MEAN_SUBDIR, "baseline"),
                                      ("2", "sif", SIF_SUBDIR, "sif_only")):
            if auroc_gates:
                published = sorted(x for m, x in pub[arm] if m == mid)
                absent = len([x for x in published if x not in mine]) if complete else 0
                add(f"{tag}a {arm} arm AUROC vs published {what}", mid, True, tol_auroc,
                    [(f"L{x}", None if (mid, x) not in pub[arm]
                      else abs(cell(x, arm, "aucroc") - pub[arm][(mid, x)])) for x in mine],
                    absent=absent)
            rel = [(f"L{x}", cell(x, arm, "vec_max_rel_diff")) for x in mine]
            mx_abs = [cell(x, arm, "vec_max_abs_diff") for x in mine]
            if arm == "mean":
                add(f"1b mean arm vectors vs cached {cache} (relative L2)", mid, True,
                    GATE_TOL_VEC_REL, rel, note=f"max abs diff {np.max(mx_abs):.3e}")
            else:
                have = [(lab, d) for lab, d in rel if np.isfinite(d)]
                finite = [v for v in mx_abs if np.isfinite(v)]
                add(f"2b sif arm vectors vs cached {cache} (relative L2; reported, not gated)",
                    mid, False, GATE_TOL_VEC_REL, have,
                    note=(f"max abs diff {np.max(finite):.3e}" if finite
                          else f"no {cache} cache"))
        aud = audit[audit["model"] == mid]
        single = aud[aud["dim"] == 1]
        add("3 score identity (relative to the passage vector norm)", mid, True,
            GATE_TOL_IDENTITY,
            [(f"L{int(x)} {d} {s}", v) for x, d, s, v in zip(
                single["layer"], single["direction"], single["split"],
                single["identity_max_rel_diff"])],
            note=(f"max abs diff {single['identity_max_abs_diff'].max():.3e}; to the cached "
                  f"vectors' score {single['cache_score_max_abs_diff'].max():.3e}"
                  if len(single) else ""))
        joint = aud[aud["dim"] > 1]
        add("4 group shares sum to 1", mid, True, GATE_TOL_SHARE_SUM,
            [(f"L{int(x)} {d} {s}", abs(v)) for x, d, s, v in zip(
                aud["layer"], aud["direction"], aud["split"], aud["share_sum_err"])]
            + [(f"L{int(x)} {d} {s} equal-weight", abs(v)) for x, d, s, v in zip(
                joint["layer"], joint["direction"], joint["split"],
                joint["share_sum_err_eqw"])])
    return pd.DataFrame(rows)


def write_gates(gates: pd.DataFrame, path: Path) -> None:
    gates.to_csv(path, index=False, float_format="%.6g")


def gate_line(g) -> str:
    """One gate row in words, shared by the log and the facts file."""
    name = DISP.get(g.model, g.model)
    if g.gate.startswith("0"):
        return f"gate {g.gate}, {name}: no rows: FAIL"
    if not g.gated:
        text = (f"gate {g.gate}, {name}: {g.n_cells} cells"
                + (f", max diff {g.max_diff:.2e} (reference {g.tolerance:.0e}; "
                   f"{g.n_over_tolerance} over)" if g.n_cells else ""))
    else:
        text = (f"gate {g.gate}, {name}: {g.n_cells} cells, max diff {g.max_diff:.2e} "
                f"(tolerance {g.tolerance:.0e}): {'PASS' if g.ok else 'FAIL'}")
    if isinstance(g.note, str) and g.note:
        text += f"; {g.note}"
    if g.n_missing_reference or g.n_published_layers_absent:
        text += (f"; missing reference cells {g.n_missing_reference}, published layers absent "
                 f"{g.n_published_layers_absent}")
    if g.n_over_tolerance:
        text += f"; cells over tolerance: {g.cells_over_tolerance}"
    return text


def gates_passed(gates: pd.DataFrame) -> bool:
    return bool(len(gates) and gates.loc[gates["gated"] == 1, "ok"].all())


def gates_for(arms: pd.DataFrame, audit: pd.DataFrame, args, complete: bool = True
              ) -> pd.DataFrame:
    res = pd.read_csv(args.results_csv) if Path(args.results_csv).exists() else None
    limit = is_limit_run(arms)
    return gate_table(arms, audit, res, complete=complete and not limit,
                      tol_auroc=args.tol_auroc, expected=e1.expected_models(args),
                      auroc_gates=not limit)


def run_gates(args, complete: bool, write: bool) -> int:
    """Gate the CSVs as written (10 significant digits), so that audit --check, check and
    render report the same numbers."""
    arms = read_csv(args.out_dir / ARMS_NAME)
    gates = gates_for(arms, read_csv(args.out_dir / AUDIT_NAME), args, complete=complete)
    if write:
        write_gates(gates, args.out_dir / GATE_NAME)
    for g in gates.itertuples():
        print(gate_line(g))
    if is_limit_run(arms):
        print("limit run: the AUROC gates are skipped and the others are not enforced")
        return 0
    if not gates_passed(gates):
        print("GATES FAILED")
        return GATE_EXIT
    print("gates passed")
    return 0


def cmd_check(args) -> int:
    return run_gates(args, complete=True, write=not args.no_write)


# --------------------------------------------------------------------------- #
# render: one row per model-layer
# --------------------------------------------------------------------------- #

JOINT = ("pcs2_3", "pcs1_3")
# Token-ablation cells read at one fixed (ranking, m) for every layer, so that nothing is
# picked on test: two sizes of the pc123 ranking, and the largest drop of the pc1 ranking
# as the comparator that removes more token mass.
FIXED_CELLS = (("pc123", 3, ""), ("pc123", 30, ""),
               ("pc1", 100, ", the comparator that drops more token mass under the PC1-only "
                            "ranking"))
JOINT_LABEL = {"pcs2_3": "span(PC2, PC3)", "pcs1_3": "span(PC1, PC2, PC3)"}
READ_DIRS = PC_NAMES + COORD_NAMES + JOINT
f3 = e1.f3


def f2(x) -> str:
    return "--" if x is None or not np.isfinite(x) else f"{x:.2f}"


def published_cells(res: Optional[pd.DataFrame]) -> Dict[str, Dict[Tuple[str, int], float]]:
    """Published Task A test AUROC per (model, layer): baseline and sif_only."""
    out: Dict[str, Dict[Tuple[str, int], float]] = {"baseline": {}, "sif_only": {}}
    if res is None:
        return out
    for method, pooling in (("baseline", "mean"), ("sif_only", "sif")):
        r = res[(res["repr"] == "hidden") & (res["pooling"] == pooling)
                & (res["method"] == method)]
        out[method] = {(m, int(x)): float(v) for m, x, v in zip(r["model"], r["layer"],
                                                                r["aucroc"])}
    return out


def summarize(arms: pd.DataFrame, audit: pd.DataFrame, abl: pd.DataFrame,
              length: pd.DataFrame, res: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One row per model-layer with every cell the table and the facts file read, plus
    the frozen rules applied to them. Test split unless a column starts with ``tr_``.

    ``base`` is the published baseline AUROC (the model's own mean arm when the published
    CSV is not given); collapsed = a T5 layer with base below COLLAPSE_AUROC.
    """
    pub = published_cells(res)
    nan = float("nan")
    aud = {(r["model"], int(r["layer"]), r["direction"], r["split"]): r
           for r in audit.to_dict("records")}
    rnd = audit[audit["kind"].isin(["random", "random_subspace"])]
    rnd = {k: g for k, g in rnd.groupby(["model", "layer", "split", "dim"], sort=False)}
    ablg = {k: g for k, g in abl.groupby(["model", "layer"], sort=False)}
    ln = {(r["model"], r["split"]): r for r in length.to_dict("records")}
    rows = []
    for (mid, layer), g in arms.groupby(["model", "layer"], sort=False):
        layer = int(layer)
        g = g.set_index("arm")
        r: Dict = {"model": mid, "m": DISP.get(mid, mid), "layer": layer}
        for arm in ARMS:
            x = g.loc[arm] if arm in g.index else {}
            r[f"auc_{arm}"] = float(x.get("aucroc", nan))
            r[f"trauc_{arm}"] = float(x.get("train_aucroc", nan))
            r[f"pc1_{arm}"] = float(x.get("pc1_share_train", nan))
            r[f"erank_{arm}"] = float(x.get("eff_rank_train", nan))
            r[f"fb_{arm}"] = int(x.get("n_fallback", 0))
            r[f"empty_{arm}"] = int(x.get("n_no_token", 0))
            r[f"spmass_{arm}"] = float(x.get("special_mass", nan))
        r["base_published"] = int((mid, layer) in pub["baseline"])
        r["base"] = pub["baseline"].get((mid, layer), r["auc_mean"])
        r["sif_gain"] = r["auc_sif"] - r["auc_mean"]
        for arm in R1_ARMS:
            r[f"r1_{arm}"], r[f"r1frac_{arm}"], r[f"r1chg_{arm}"] = r1_rescue(
                r[f"auc_{arm}"], r["auc_mean"], r["auc_sif"])
        # R1 again with the published sif_only cell as AUROC_sif (the second reference)
        r["pub_sif"] = pub["sif_only"].get((mid, layer), nan)
        r["sif_minus_pub"] = r["auc_sif"] - r["pub_sif"]
        r["sif_gain_pub"] = r["pub_sif"] - r["auc_mean"]
        for arm in R1_ARMS:
            r[f"r1p_{arm}"], r[f"r1pfrac_{arm}"], _ = r1_rescue(
                r[f"auc_{arm}"], r["auc_mean"], r["pub_sif"])
        for d in READ_DIRS:
            for split, pre in (("test", ""), ("train", "tr_")):
                x = aud.get((mid, layer, d, split), {})
                r[f"{pre}ev_{d}"] = float(x.get("ev_tokenmix", nan))
                r[f"{pre}r2_{d}"] = float(x.get("r2_tokenmix", nan))
                r[f"{pre}eveq_{d}"] = float(x.get("ev_tokenmix_eqw", nan))
                for grp in GROUPS:
                    r[f"{pre}sh_{grp}_{d}"] = float(x.get(f"share_{grp}", nan))
                    r[f"{pre}sheq_{grp}_{d}"] = float(x.get(f"share_{grp}_eqw", nan))
                r[f"{pre}first_{d}"] = float(x.get("share_first", nan))
                r[f"{pre}rho_{d}"] = float(x.get("rho_logn", nan))
                r[f"{pre}rhof_{d}"] = float(x.get("rho_freqmass", nan))
                r[f"{pre}var_{d}"] = float(x.get("var_s", nan))
                r[f"{pre}varshare_{d}"] = float(x.get("var_share", nan))
        x = aud.get((mid, layer, "pc1", "test"), {})
        for grp in GROUPS:
            r[f"mass_{grp}"] = float(x.get(f"mass_{grp}", nan))
        r["unseen_token_frac"] = float(x.get("unseen_token_frac", nan))
        r["e1_rank_agrees"] = int(aud.get((mid, layer, "coord1", "train"), {}).get(
            "e1_rank_agrees", -1))
        with np.errstate(divide="ignore", invalid="ignore"):
            r["var_ratio_pc2_pc3"] = float(np.float64(r["tr_var_pc2"]) / r["tr_var_pc3"])
        for split, pre in (("test", ""), ("train", "tr_")):
            for dim, tag in ((1, "rand"), (2, "rand2"), (3, "rand3")):
                t = rnd.get((mid, layer, split, dim))
                ev = t["ev_tokenmix"] if t is not None else pd.Series(dtype=float)
                r[f"{pre}{tag}_ev_mean"] = float(ev.mean()) if len(ev) else nan
                r[f"{pre}{tag}_ev_min"] = float(ev.min()) if len(ev) else nan
                r[f"{pre}{tag}_ev_max"] = float(ev.max()) if len(ev) else nan
                r[f"{pre}{tag}_eveq_mean"] = (float(t["ev_tokenmix_eqw"].mean())
                                              if t is not None and dim > 1 else nan)
                for grp in GROUPS:
                    r[f"{pre}{tag}_sh_{grp}"] = (float(t[f"share_{grp}"].mean())
                                                 if t is not None else nan)
                if dim == 1:
                    r[f"{pre}rand_absrho_mean"] = (float(t["rho_logn"].abs().mean())
                                                   if t is not None else nan)
        shares = [r[f"sh_{grp}_pc1"] for grp in GROUPS]
        r["top_group"] = GROUPS[int(np.argmax(shares))] if np.all(np.isfinite(shares)) else ""
        # token ablation
        # Per (ranking, m): the carriers' cells (abl_, abltr_ train AUROC, dm_ share of
        # test tokens dropped, tc_ train tokens dropped) and, for the count-nearest control
        # (ctl_, cdm_, ctc_) and the mass-matched control (mctl_, mdm_, mtc_, mnt_ number
        # of types), the mean over draws. sel_ = the cell chosen on TRAIN AUROC of the
        # carrier arm; best_ = the highest TEST AUROC (a pick on test, as R3 allows).
        t = ablg.get((mid, layer))
        best = (nan, "", -1)
        chosen = (nan, "", -1)
        restores = {kind: False for kind in CONTROL_KINDS}
        for ranking in RANKINGS:
            by_m: Dict[int, float] = {}
            for m in ABL_MS:
                tag = f"{ranking}_m{m}"
                sel = (t[(t["ranking"] == ranking) & (t["m"] == m)] if t is not None
                       else pd.DataFrame({"kind": []}))
                c = sel[sel["kind"] == "carrier"]

                def cell(frame: pd.DataFrame, column: str, how: str = "mean") -> float:
                    if not len(frame) or column not in frame.columns:
                        return nan
                    return float(getattr(frame[column].astype(float), how)())

                auc = cell(c, "aucroc")
                by_m[m] = auc
                r[f"abl_{tag}"] = auc
                r[f"abltr_{tag}"] = cell(c, "train_aucroc")
                r[f"dm_{tag}"] = cell(c, "dropped_mass_test")
                r[f"tc_{tag}"] = cell(c, "train_count_dropped")
                for kind, pre, dm, tc in (("control", "ctl", "cdm", "ctc"),
                                          ("control_mass", "mctl", "mdm", "mtc")):
                    k = sel[sel["kind"] == kind]
                    for how in ("mean", "min", "max"):
                        r[f"{pre}_{tag}_{how}"] = cell(k, "aucroc", how)
                    r[f"{dm}_{tag}"] = cell(k, "dropped_mass_test")
                    r[f"{tc}_{tag}"] = cell(k, "train_count_dropped")
                    if m <= R3_MAX_M and r[f"{pre}_{tag}_mean"] >= R3_AUROC:
                        restores[kind] = True
                r[f"mnt_{tag}"] = cell(sel[sel["kind"] == "control_mass"], "n_types")
                if m <= R3_MAX_M and np.isfinite(auc) and not auc <= best[0]:
                    best = (auc, ranking, m)
                if (m <= R3_MAX_M and np.isfinite(r[f"abltr_{tag}"])
                        and not r[f"abltr_{tag}"] <= chosen[0]):
                    chosen = (r[f"abltr_{tag}"], ranking, m)
            m_ok = r3_restoring_m(by_m)
            r[f"r3m_{ranking}"] = -1 if m_ok is None else m_ok
        for name, (_, ranking, m) in (("best", best), ("sel", chosen)):
            tag = f"{ranking}_m{m}"
            r[f"{name}_abl"] = r.get(f"abl_{tag}", nan)
            r[f"{name}_abl_ranking"], r[f"{name}_abl_m"] = ranking, m
            r[f"{name}_ctl"] = r.get(f"ctl_{tag}_mean", nan)
            r[f"{name}_mctl"] = r.get(f"mctl_{tag}_mean", nan)
        r["sel_abl_train"] = chosen[0]
        r["r3"] = bool(any(r[f"r3m_{ranking}"] > 0 for ranking in RANKINGS))
        r["ctl_r3"] = restores["control"]
        r["mctl_r3"] = restores["control_mass"]
        r["r2"] = r2_token_mix(r["ev_pc1"])
        x = ln.get((mid, "test"), {})
        r["len_same"] = float(x.get("mean_dlog_same", nan))
        r["len_diff"] = float(x.get("mean_dlog_diff", nan))
        r["r4"] = r4_length(r["rho_pc1"], r["len_same"], r["len_diff"])
        rows.append(r)
    w = pd.DataFrame(rows)
    w["is_t5"] = w["m"].isin(T5)
    w["collapsed"] = w["is_t5"] & (w["base"] < COLLAPSE_AUROC)
    return w


def worst_layer(w: pd.DataFrame, name: str) -> Optional[pd.Series]:
    """The model's row at its lowest baseline test AUROC (first layer on ties)."""
    s = w[w["m"] == name].sort_values("layer")
    return None if s.empty else s.loc[s["base"].idxmin()]


def r1_reference_comparison(coll: pd.DataFrame) -> Dict:
    """R1 at the collapsed layers under the two SIF references, cell by cell.

    ``coll`` holds the collapsed rows of summarize(). One cell per (layer, arm): the R1
    status and recovered share with AUROC_sif = the job's own sif arm (``own``) and with
    AUROC_sif = the published sif_only cell (``pub``; "undefined" where that cell is
    missing). Returns {"cells": frame, "n": all cells, "n_compared": cells with a published
    reference, "differ": the compared cells whose status differs, "n_both": cells with a
    recovered share under both references, "max_dfrac": the largest |share_pub -
    share_own| over them (NaN if none), "max_cell": that cell's label}.
    """
    cells = pd.DataFrame(
        [{"m": x["m"], "layer": int(x["layer"]), "arm": arm, "own": x[f"r1_{arm}"],
          "pub": x[f"r1p_{arm}"], "frac_own": x[f"r1frac_{arm}"],
          "frac_pub": x[f"r1pfrac_{arm}"]} for _, x in coll.iterrows() for arm in R1_ARMS],
        columns=["m", "layer", "arm", "own", "pub", "frac_own", "frac_pub"])
    compared = cells[cells["pub"] != "undefined"]
    both = compared[np.isfinite(compared["frac_own"].astype(float))
                    & np.isfinite(compared["frac_pub"].astype(float))]
    dfrac = (both["frac_pub"] - both["frac_own"]).abs().astype(float)
    top = both.loc[dfrac.idxmax()] if len(both) else None
    return {"cells": cells, "n": len(cells), "n_compared": len(compared),
            "differ": compared[compared["own"] != compared["pub"]], "n_both": len(both),
            "max_dfrac": float(dfrac.max()) if len(both) else float("nan"),
            "max_cell": "" if top is None else f"{top['m']} L{int(top['layer'])} `{top['arm']}`"}


# --------------------------------------------------------------------------- #
# render: table
# --------------------------------------------------------------------------- #

def tex_num(x, digits: int) -> str:
    """A table cell: fixed decimals, the minus sign in math mode, and no negative zero
    (a value that rounds to 0 prints without a sign). ``--`` for a missing value."""
    if x is None or not np.isfinite(x):
        return "--"
    text = f"{x:.{digits}f}"
    if float(text) == 0.0:
        return text.lstrip("-")
    return "$-$" + text[1:] if text.startswith("-") else text


def sif_reference_gap(gates: Optional[pd.DataFrame]) -> float:
    """Largest |sif arm AUROC - published sif_only| over all models and layers, read from
    the gate 2a rows; NaN when that gate was not evaluated."""
    if gates is None or gates.empty:
        return float("nan")
    rows = gates[gates["gate"].str.startswith("2a")]
    return float(np.max(rows["max_diff"])) if len(rows) else float("nan")


def caption(gates: Optional[pd.DataFrame] = None) -> str:
    """Caption of tab:e2_token_audit. What it says about the SIF cells reported earlier
    in the paper (the published ``sif_only`` cells) is read from gate 2a: the largest
    difference over all models and layers, to three decimals."""
    gap = sif_reference_gap(gates)
    earlier = r"the SIF cells of Section~\ref{sec:geometry}"
    if not np.isfinite(gap):
        published = ""
    elif gap <= GATE_TOL_AUROC:
        published = f"; it reproduces {earlier}"
    else:
        published = (f"; its AUROC differs from {earlier} by up to {gap:.3f} over all models "
                     "and layers")
    return (
        r"\caption{Token audit at each model's worst baseline layer (L). Pooling: Task~A test "
        r"AUROC of mean pooling (Mean); mean pooling without special tokens (No spec.); SIF "
        r"frequency weights with special tokens kept (SIF+spec.); SIF pooling, which also "
        r"drops special tokens, recomputed in this experiment's forward pass with "
        r"training-only token frequencies (SIF" + published + r"); and mean pooling without "
        r"the 100 most frequent training tokens (No freq.). Token ablation: test AUROC of "
        r"mean pooling after dropping the $m$ token types that contribute most to the top "
        r"principal components (Drop), with $m$ (1 to 100) and the ranking (one of two) "
        r"chosen by training AUROC, and the mean test AUROC of five random sets of other "
        r"token types, sampled in proportion to training count, that hold at least as many "
        r"training tokens as the dropped types (Rand.). Token-mix EV: the share of the variance "
        r"of the first principal component's score across test passages that is explained by "
        r"which token types a passage contains, with one mean per token type fit on training "
        r"tokens (PC1), next to the mean of the same quantity over 20 random directions "
        r"orthogonal to the top ten principal components (Rand.). Share by token group: the "
        r"covariance share of that variance contributed by special tokens, by the 100 most "
        r"frequent tokens and by all other tokens; the three sum to one, but a single share "
        r"can fall below zero or exceed one. $|\rho|$: absolute Spearman correlation between "
        r"the PC1 score and the log token count of test passages (the sign of a principal "
        r"component is a convention). Components, token frequencies and token rankings are "
        r"fit on training passages only.}")


def write_table(w: pd.DataFrame, path: Path, gates: Optional[pd.DataFrame] = None
                ) -> List[str]:
    """Write tab:e2_token_audit. Returns the display names of omitted models."""
    rows = [(name, worst_layer(w, name)) for name in ORDER]
    omitted = [name for name, x in rows if x is None]
    lines = [asw.HEADER, r"\begin{table*}[t]", r"\centering", r"\footnotesize",
             # 2.6pt, not the 3pt of the E1 table: at 3pt a row with negative cells overflowed
             # \textwidth by 5.5pt; at 2.6pt three negative shares in one row still fit.
             r"\setlength{\tabcolsep}{2.6pt}", r"\begin{tabular}{@{}lcccccccccccccc@{}}",
             r"\toprule",
             r" & & \multicolumn{5}{c}{AUROC by pooling} & \multicolumn{2}{c}{Token ablation} "
             r"& \multicolumn{2}{c}{Token-mix EV} & \multicolumn{3}{c}{Share by token group} "
             r"& \\",
             r"\cmidrule(lr){3-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}\cmidrule(lr){12-14}",
             r"Model & L & Mean & No spec. & SIF+spec. & SIF & No freq. & Drop & Rand. & PC1 "
             r"& Rand. & Spec. & Freq. & Other & $|\rho|$ \\",
             r"\midrule"]
    for name, x in rows:
        if x is None:
            continue
        cells = ([tex_num(x[f"auc_{arm}"], 3) for arm in ARMS]
                 + [tex_num(x[c], 3) for c in ("sel_abl", "sel_mctl")]
                 + [tex_num(x[c], 2) for c in ("ev_pc1", "rand_ev_mean")]
                 + [tex_num(x[f"sh_{grp}_pc1"], 2) for grp in GROUPS]
                 + [tex_num(abs(x["rho_pc1"]), 2)])
        lines.append(f"{name} & {int(x['layer'])} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", caption(gates), r"\label{tab:e2_token_audit}",
              r"\end{table*}"]
    path.write_text("\n".join(lines) + "\n")
    return omitted


# --------------------------------------------------------------------------- #
# render: facts
# --------------------------------------------------------------------------- #

def _rng(s: pd.Series, fmt: str = ".3f") -> str:
    """median (min to max) of the finite values."""
    s = pd.Series(s, dtype=float).dropna()
    if s.empty:
        return "n/a"
    return f"{format(s.median(), fmt)} ({format(s.min(), fmt)} to {format(s.max(), fmt)})"


def _layers(t: pd.DataFrame) -> str:
    return ", ".join(str(int(v)) for v in sorted(t["layer"])) or "none"


def _labels(t: pd.DataFrame) -> str:
    return ", ".join(f"{m} {int(x)}" for m, x in zip(t["m"], t["layer"])) or "none"


def _md(head: Sequence[str], body: Sequence[Sequence[str]]) -> List[str]:
    return (["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
            + ["| " + " | ".join(r) + " |" for r in body])


def carrier_lines(carriers: pd.DataFrame, mid: str, layer: int, top: int = 10) -> List[str]:
    """The top carrier token types of PC1 to PC3 at one model-layer."""
    out: List[str] = []
    c = carriers[(carriers["model"] == mid) & (carriers["layer"] == layer)]
    for d in PC_NAMES:
        t = c[c["direction"] == d].sort_values("rank").head(top)
        if t.empty:
            out.append(f"  - {d.upper()}: no rows")
            continue
        out.append(f"  - {d.upper()}, top {len(t)} by train share (token `piece` \"decoded\" "
                   "[group], train count, passages, mean c, mean |c|, share):")
        for x in t.itertuples():
            out.append(f"    - {int(x.rank)}. `{x.token}` \"{x.decoded}\" [{x.group}], "
                       f"{int(x.train_count)}, {int(x.train_passages)}, {x.mean_c:+.4g}, "
                       f"{x.mean_abs_c:.4g}, {x.share:+.3f}")
        full = c[c["direction"] == d]
        out.append(f"    - sum of the {len(full)} listed shares {full['share'].sum():+.3f}; by "
                   "group: " + ", ".join(
                       f"{grp} {full.loc[full['group'] == grp, 'share'].sum():+.3f} "
                       f"({int((full['group'] == grp).sum())} types)" for grp in GROUPS))
    return out


def sif_deviation(gates: pd.DataFrame, cmp: Dict) -> str:
    """The statement that replaces "blocked" when gate 2a fails: what failed and by how
    much, what the sif arm is, and how R1 is reported. Every number is a gate cell or a
    count of R1 cells; it is loud when the two references disagree on a verdict."""
    g2a = gates[gates["gate"].str.startswith("2a")]
    g2b = gates[gates["gate"].str.startswith("2b")]
    failed = g2a[~g2a["ok"].astype(bool)]
    text = ("**Deviation (gate 2a).** Gate 2a failed: the `sif` arm does not reproduce the "
            "published `sif_only` cells. Largest AUROC difference per model: "
            + ", ".join(f"{DISP.get(g.model, g.model)} "
                        + (f"{g.max_diff:.2e}" if g.n_cells else "no published cell")
                        for g in failed.itertuples())
            + ". The `sif` arm is the repo CLI's SIF pooling on the tracked split with "
            "train-only token probabilities")
    cached = g2b[g2b["n_cells"] > 0]
    absent = [DISP.get(m, m) for m in g2b.loc[g2b["n_cells"] == 0, "model"]]
    if len(cached):
        gap = float(np.max(cached["max_diff"]))
        text += ((f", and it equals the local re-extraction (`{SIF_SUBDIR}`) exactly at the "
                  f"{len(cached)} models that have one" if gap == 0 else
                  f", and it differs from the local re-extraction (`{SIF_SUBDIR}`) by at most "
                  f"{gap:.2e} in relative L2 at the {len(cached)} models that have one")
                 + " (gate 2b" + (f"; none for {', '.join(absent)}" if absent else "") + ")")
    text += (". R1 is therefore reported against the job's own `sif` arm and, under \"R1 under "
             "the published SIF reference\" in section 3, against the published cells. Changed "
             "after the results were read: the pre-registered consequence of a gate 2 failure "
             "(block the pooling-control conclusions) was replaced by reporting under both "
             "references.")
    if cmp["n_compared"] < cmp["n"]:
        text += (f" The published cell is missing at {cmp['n'] - cmp['n_compared']} of the "
                 f"{cmp['n']} (collapsed layer, arm) cells, which are not compared.")
    if len(cmp["differ"]):
        text += (f" **THE R1 VERDICTS DIFFER BETWEEN THE TWO REFERENCES AT {len(cmp['differ'])} "
                 f"OF {cmp['n_compared']} (COLLAPSED LAYER, ARM) CELLS. QUOTE NO R1 COUNT "
                 "WITHOUT NAMING ITS REFERENCE.**")
    elif cmp["n_compared"]:
        text += (f" The R1 verdicts are the same under both references at all "
                 f"{cmp['n_compared']} (collapsed layer, arm) cells.")
    return text


def facts(w: pd.DataFrame, arms: pd.DataFrame, carriers: pd.DataFrame, length: pd.DataFrame,
          gates: Optional[pd.DataFrame], path: Path, omitted: Sequence[str] = ()) -> None:
    L: List[str] = []
    a = L.append
    present = [m for m in ORDER if (w["m"] == m).any()]
    coll = w[w["collapsed"]]
    n_coll = len(coll)
    limit = int(arms["limit"].max()) if "limit" in arms.columns and len(arms) else 0
    verdicts: List[Tuple[str, str]] = []
    cmp = r1_reference_comparison(coll)
    gate2_failed = bool(gates is not None and len(gates) and (
        ~gates.loc[(gates["gated"] == 1) & gates["gate"].str.startswith("2"), "ok"]
        .astype(bool)).any())

    def count(mask) -> str:
        return f"{int(np.sum(mask))}/{len(mask)}"

    a("# E2 token audit: facts (generated)")
    a("")
    a("Generated by `scripts/paper/reframe/e2_token_audit.py render` from "
      f"`{ARMS_NAME}`, `{AUDIT_NAME}`, `{CARRIER_NAME}`, `{ABL_NAME}` and `{LENGTH_NAME}` in "
      "this directory and the published baseline cells. Every number below is a cell of one "
      "of those CSVs, or a count, difference, ratio, median, minimum or maximum of such "
      "cells. AUROC is Task A test AUROC unless marked train; token-audit numbers are on "
      "test passages unless marked train.")
    a("")
    a("Changed after the results were read: one consequence, in `render` only. A failed gate "
      "2 was to block the pooling-control conclusions. This file instead reports R1 under "
      "both SIF references: against the job's own `sif` arm, and against the published "
      "`sif_only` cells (section 3). "
      + ("Gate 2a failed in this run (section 0). " if gate2_failed else
         "Gate 2a did not fail in this run. " if gates is not None and len(gates) else "")
      + "The gate, its tolerance, the decision rules and every audit number are unchanged. "
      "The table caption was corrected with it: it no longer calls the SIF column the "
      "published SIF.")
    a("")
    a("Added after the first full run, on review, because the count-nearest control was not "
      "mass-matched: a second random control for the token ablation, the mass-matched "
      "control (`control_mass`, section 6). R3 and the carrier arms are unchanged. With it "
      "the table changed: its Drop column is the cell chosen on train AUROC and no longer "
      "the highest test AUROC, its Rand. column beside it is the mass-matched control, and "
      "its last column prints |rho|.")
    if limit:
        a("")
        a(f"**SMOKE RUN (--limit {limit}): the first {limit} train and {limit} test passages "
          "only. Nothing below is a result; the AUROC gates are skipped.**")
    a("")
    a("## Frozen decision rules")
    a("Fixed after E1 and before any E2 number was read (constants at the top of the script):")
    a(f"- collapsed = published baseline test AUROC < {COLLAPSE_AUROC:.2f} at a T5 layer.")
    a(f"- R1 rescue: an arm rescues a collapsed layer if (AUROC_arm - AUROC_mean) / "
      f"(AUROC_sif - AUROC_mean) >= {R1_RESCUE_FRAC:.2f}, evaluated only where AUROC_sif - "
      f"AUROC_mean >= {R1_MIN_SIF_GAIN:.2f}; elsewhere \"no SIF gain to recover\" and the "
      "raw change.")
    a(f"- R2: the token mix carries the direction at a layer if test EV for PC1 >= {R2_EV}.")
    a(f"- R3: token ablation restores a collapsed layer if test AUROC >= {R3_AUROC:.2f} for "
      f"some m <= {R3_MAX_M} (either ranking).")
    a(f"- R4: the length account holds if |Spearman(s_PC1, log n)| >= {R4_RHO} on test and "
      "mean |delta log n| is larger for same-directory than for different-directory pairs.")
    a("- Expectations recorded before the run: in LaTa and PhilTa frequent tokens carry the "
      "direction (the `frequent` group has the largest share; `sif_keepspecial` rescues, "
      "`mean_nospecial` does not); in mT5-base no arm rescues and tokens SIF keeps (`other`) "
      "carry it.")
    a("- The subspace readouts (span(PC2, PC3), span(PC1, PC2, PC3)) were added to the "
      "design before the run and carry no rule.")
    a("")
    a("## Definitions")
    a(f"- Forward pass: each model loaded and tokenized as its extraction CLI does "
      f"(max_length {MAX_LENGTH}, batch size {BATCH_SIZE}, `{TOKEN_FILTER}` keep lookup, "
      "batches in the cache's row order); one forward with all hidden states. Passages and "
      "the train/test split come from the split CSV; cached vectors are aligned by filename.")
    a("- Kept tokens: the tokens mean pooling averages (attention mask times the keep "
      "lookup; special tokens such as `</s>` are kept). n_p = number of kept tokens.")
    a(f"- Directions, fit on the cached TRAIN vectors: mu = train mean; PC1 to PC3 = "
      "EmbeddingCleaner's components (the ones ABTT removes), sign fixed so that the "
      f"largest loading is positive; coord1 to coord3 = the {N_COORD} coordinates of largest "
      f"train variance (E1's ranking); {N_RANDOM} random unit directions (seed "
      f"{RANDOM_SEED}, per model-layer), orthogonal to the top {ORTHO_PCS} PCs and to each "
      "other.")
    a("- c_t = (h_t - mu) . w for token t; the passage score s_p = w . (pooled_p - mu) is "
      "the mean of c_t over the passage's kept tokens (gate 3).")
    a("- Token-mix EV: per-token-type means of c_t fit on TRAIN tokens (a type unseen in "
      "train gets the mean over all train tokens); s_hat_p = mean of its tokens' type means; "
      "EV = 1 - Var(s - s_hat) / Var(s) over the passages of a split. The train value is in "
      "sample. r2 = squared Pearson correlation of s and s_hat.")
    a(f"- Token groups: `special` = id in tokenizer.all_special_ids; `frequent` = the "
      f"{N_FREQUENT} token types of largest train probability (token_probabilities, which "
      "never counts special tokens); `other` = the rest. share_g = Cov(s_g, s) / Var(s) "
      "with s_g the part of s contributed by group g; the three shares sum to 1 (gate 4) "
      "and a share can be negative or exceed 1. `first` = the same share for the first "
      "kept token of each passage (overlaps the groups). Mass = mean share of a passage's "
      "kept tokens in the group.")
    a("- Subspaces: for span(PC2, PC3) and span(PC1, PC2, PC3) the score is a vector and "
      "variances become traces of covariance matrices: EV = 1 - tr Cov(s - s_hat) / "
      "tr Cov(s), share_g = tr Cov(s_g, s) / tr Cov(s). These do not change under a "
      "rotation of the basis inside the subspace, so they are defined where PC2 and PC3 "
      "are not individually identifiable (train variance ratio near 1). `variance-weighted` "
      "is that definition; `equal-weight` first whitens the score with its train covariance "
      "(in the PC basis: divides each component by its train SD), so PC1 does not swamp "
      "PC2 and PC3. Random controls: 10 disjoint pairs and 6 disjoint triples of the random "
      "directions.")
    a(f"- Carriers: per token type, share = Cov(s_type, s) / Var(s) on train; the top "
      f"{TOP_CARRIERS} per PC are in `{CARRIER_NAME}`. mean c = mean contribution of the "
      "type's train tokens (sign relative to the fixed sign of the PC), mean |c| its mean "
      "magnitude, train count includes special tokens.")
    a("- Pooling arms (all use the CLIs' expression sum w_t h_t / max(sum w_t, 1)): `mean` "
      "= the cache; `mean_nospecial` = special tokens dropped; `sif_keepspecial` = SIF "
      "weights a / (a + p) on token types with a train probability and weight 1 on special "
      "tokens (they have no train probability; this is full SIF with only the zero weight "
      "on special tokens removed, so a special token weighs more than a frequent token "
      "there: see the special-token weight mass per arm in section 3); `sif` = the CLI's "
      f"SIF; `mean_nofreq100` = the {N_FREQUENT} frequent token types dropped, special "
      "tokens kept. A passage left with no token falls back to its `mean` vector (counted; "
      "never for `sif`).")
    a(f"- Token ablation: token types ranked on train by PC1 share (`pc1`) and by the mean "
      f"of the PC1, PC2, PC3 shares (`pc123`); mean pooling with the top m types dropped, m "
      f"in {list(ABL_MS)}. Two random controls, {CONTROL_DRAWS} draws each, reported as the "
      "mean over draws. Count-nearest control (`control`): for each carrier in rank order, "
      f"one type drawn among the {MATCH_WINDOW} nearest in train count that are in the top "
      f"{max(ABL_MS)} of neither ranking, without replacement; the control for m is the "
      "first m of a draw. It matches the number of types, not the token mass: where the "
      "carriers are the most frequent types, their nearest non-carriers hold fewer tokens. "
      "Mass-matched control (`control_mass`): for each (ranking, m), token types drawn "
      "without replacement from the train types that are not among the m dropped carriers "
      "(special tokens are eligible), with probability proportional to train count, until "
      "their cumulative train count first meets or exceeds the carriers'. Carrier arms have "
      "the full metric block; controls have Task A AUROC (unless the run used "
      "--control_metrics full). The cell chosen on train AUROC is the (ranking, m) with the "
      "highest TRAIN AUROC of the carrier arm (on ties the first in the order pc1, pc123 "
      "and ascending m); R3 itself reads the highest test AUROC over the cells.")
    a("- Length: n_p under each model's tokenizer (truncated at 512); |delta log n| over "
      "the same- and different-directory pairs Task A scores.")
    a("- Worst layer = first argmin of the published baseline test AUROC.")
    a("")

    a("## 0. Coverage and gates")
    a("- model-layers: " + ", ".join(f"{m} {int((w['m'] == m).sum())}" for m in present)
      + f" (total {len(w)}); passages: train {int(arms['n_train'].max())}, test "
      f"{int(arms['n_test'].max())}")
    a(f"- collapsed layers (baseline AUROC < {COLLAPSE_AUROC:.2f}): {n_coll}: " + (", ".join(
        f"{m} {int((coll['m'] == m).sum())} ({_layers(coll[coll['m'] == m])})"
        for m in T5 if (coll["m"] == m).any()) or "none"))
    if not w["base_published"].all():
        a(f"- published baseline cell missing at {int((w['base_published'] == 0).sum())} "
          "model-layers: the mean arm's own AUROC defines collapsed and worst there")
    if omitted:
        a("- ABSENT from the CSVs, so omitted from the table and from every count below: "
          + ", ".join(omitted))
    if gates is None or gates.empty:
        a("- gates: not evaluated")
    else:
        for g in gates.itertuples():
            a("- " + gate_line(g))
        bad = gates[(gates["gated"] == 1) & ~gates["ok"].astype(bool)]
        if limit:
            a("- limit run: the AUROC gates (1a, 2a) are skipped and the others not enforced.")
        elif len(bad):
            a("- A gate FAILED. Read the cells it names before quoting numbers that depend "
              "on them.")
    if gate2_failed:
        a("- " + sif_deviation(gates, cmp))
    checked = w[w["e1_rank_agrees"] >= 0]
    a(f"- top-{N_COORD} variance coordinates against `e1_top_coordinates.csv`: "
      + (f"the same at {count(checked['e1_rank_agrees'] == 1)} model-layers" if len(checked)
         else "not compared (E1 CSV absent, or a limit run)")
      + (f"; DIFFERENT at {_labels(checked[checked['e1_rank_agrees'] == 0])}"
         if (checked["e1_rank_agrees"] == 0).any() else ""))
    sc = arms[arms["arm"].isin(["mean", "sif"])].groupby("arm")["cli_pool_max_abs_diff"].max()
    a("- pooling expression against the extraction CLI's own function on the first batch of "
      "each model, max |diff| over models and layers: "
      + ", ".join(f"{arm} {sc[arm]:.2e}" for arm in ("mean", "sif") if arm in sc.index))
    fb = arms.groupby("arm")[["n_no_token", "n_fallback"]].max()
    a("- passages with no token under an arm (max over model-layers), and how many fell back "
      "to the mean vector: " + ", ".join(
          f"{arm} {int(fb.loc[arm, 'n_no_token'])} / {int(fb.loc[arm, 'n_fallback'])}"
          for arm in ARMS if arm in fb.index))
    a("")

    a("## 1. Table rows: each model at its worst baseline layer")
    for name in present:
        x = worst_layer(w, name)
        a(f"- {name} L{int(x['layer'])} (published baseline {f3(x['base'])}):")
        a("  - AUROC by pooling arm: " + "; ".join(
            f"{arm} {f3(x[f'auc_{arm}'])}" for arm in ARMS)
          + "; R1: " + "; ".join(
              f"{arm} {x[f'r1_{arm}']}"
              + (f" ({100 * x[f'r1frac_{arm}']:.0f}% of the SIF gain)"
                 if np.isfinite(x[f"r1frac_{arm}"]) else f" (change {x[f'r1chg_{arm}']:+.3f})")
              for arm in R1_ARMS))
        a("  - top-PC share of the train vectors by arm: " + "; ".join(
            f"{arm} {f3(x[f'pc1_{arm}'])}" for arm in ARMS))
        a(f"  - token ablation at the cell chosen on train AUROC (ranking "
          f"{x['sel_abl_ranking']}, m={int(x['sel_abl_m'])}): test AUROC {f3(x['sel_abl'])}; "
          f"count-nearest control {f3(x['sel_ctl'])}; mass-matched control "
          f"{f3(x['sel_mctl'])}. Highest test AUROC over all cells (a pick on test): "
          f"{f3(x['best_abl'])} (ranking {x['best_abl_ranking']}, m={int(x['best_abl_m'])}; "
          f"count-nearest control {f3(x['best_ctl'])}, mass-matched control "
          f"{f3(x['best_mctl'])})")
        a(f"  - token-mix EV: PC1 {f3(x['ev_pc1'])} (train {f3(x['tr_ev_pc1'])}), PC2 "
          f"{f3(x['ev_pc2'])}, PC3 {f3(x['ev_pc3'])}; random directions mean "
          f"{f3(x['rand_ev_mean'])} ({f3(x['rand_ev_min'])} to {f3(x['rand_ev_max'])})")
        a("  - PC1 share by group: " + ", ".join(
            f"{grp} {x[f'sh_{grp}_pc1']:+.3f}" for grp in GROUPS)
          + f"; first token {x['first_pc1']:+.3f}; token mass: " + ", ".join(
              f"{grp} {f3(x[f'mass_{grp}'])}" for grp in GROUPS))
        a(f"  - Spearman of the PC1 score with log n {x['rho_pc1']:+.3f}, with the "
          f"frequent-token mass {x['rhof_pc1']:+.3f}")
    a("")

    # ---------------------------------------------------------------- R1
    a("## 2. Collapsed layers at a glance")
    if n_coll:
        a(f"- AUROC, median (min to max) over the {n_coll} collapsed layers: " + "; ".join(
            f"{arm} {_rng(coll[f'auc_{arm}'])}" for arm in ARMS))
        a(f"- top-PC share of the train vectors: " + "; ".join(
            f"{arm} {_rng(coll[f'pc1_{arm}'])}" for arm in ARMS))
    a("")
    a("## 3. R1: pooling control")
    if gate2_failed:
        a(sif_deviation(gates, cmp))
    a("The lines of this section up to the last subsection use the job's own `sif` arm as "
      "AUROC_sif.")
    for label, t in [("all collapsed layers", coll)] + [
            (m, coll[coll["m"] == m]) for m in T5 if (coll["m"] == m).any()]:
        if t.empty:
            continue
        gain_ok = t["sif_gain"] >= R1_MIN_SIF_GAIN
        a(f"### {label} (n = {len(t)})")
        a(f"- SIF gain (AUROC_sif - AUROC_mean): {_rng(t['sif_gain'], '+.3f')}; at least "
          f"{R1_MIN_SIF_GAIN:.2f} at {count(gain_ok)} layers"
          + (f" (below it at {_labels(t[~gain_ok])}: R1 reports \"no SIF gain to recover\" "
             "there)" if (~gain_ok).any() else ""))
        for arm in R1_ARMS:
            st = t[f"r1_{arm}"]
            a(f"- `{arm}`: rescues {count(st == 'rescue')}; does not rescue "
              f"{count(st == 'no_rescue')}; no SIF gain to recover {count(st == 'no_sif_gain')}"
              f". AUROC {_rng(t[f'auc_{arm}'])}; change against mean "
              f"{_rng(t[f'r1chg_{arm}'], '+.3f')}"
              f"; share of the SIF gain where evaluated {_rng(t[f'r1frac_{arm}'], '.2f')}")
        a(f"- special-token share of the pooling weight, median: " + "; ".join(
            f"{arm} {t[f'spmass_{arm}'].median():.3f}" for arm in ARMS))
    a("### R1 under the published SIF reference")
    a("- AUROC_sif is the published `sif_only` cell (repr hidden, pooling sif) of "
      "`phase_resubmit_results.csv`; AUROC_mean and AUROC_arm stay this job's arms, and the "
      "rule is unchanged.")
    if not cmp["n_compared"]:
        a("- no published `sif_only` cell at a collapsed layer: nothing to compare.")
    else:
        if cmp["n_compared"] < cmp["n"]:
            a(f"- published cell missing at {cmp['n'] - cmp['n_compared']} of the {cmp['n']} "
              "(collapsed layer, arm) cells: those are left out below.")
        a("- `sif` arm minus published `sif_only`, median (min to max):")
        for m in present:
            t_all = w[(w["m"] == m) & w["pub_sif"].notna()]
            t_coll = t_all[t_all["collapsed"]]
            a(f"  - {m}: " + (f"collapsed layers (n = {len(t_coll)}) "
                              f"{_rng(t_coll['sif_minus_pub'], '+.2e')}; " if len(t_coll) else "")
              + f"all layers (n = {len(t_all)}) {_rng(t_all['sif_minus_pub'], '+.2e')}")
        have = coll[coll["pub_sif"].notna()]
        a(f"- SIF gain with the published cell (published `sif_only` - AUROC_mean): "
          f"{_rng(have['sif_gain_pub'], '+.3f')}; at least {R1_MIN_SIF_GAIN:.2f} at "
          f"{count(have['sif_gain_pub'] >= R1_MIN_SIF_GAIN)} collapsed layers (job's `sif` "
          f"arm: {count(have['sif_gain'] >= R1_MIN_SIF_GAIN)})")
        a("- verdicts as rescues / does not rescue / no SIF gain to recover, job's `sif` arm "
          "then published `sif_only`:")
        for label, t in [("all collapsed layers", have)] + [
                (m, have[have["m"] == m]) for m in T5 if (have["m"] == m).any()]:
            def triple(col: pd.Series) -> str:
                return " / ".join(str(int((col == k).sum()))
                                  for k in ("rescue", "no_rescue", "no_sif_gain"))

            a(f"  - {label} (n = {len(t)}): " + "; ".join(
                f"`{arm}` {triple(t[f'r1_{arm}'])}, then {triple(t[f'r1p_{arm}'])}"
                for arm in R1_ARMS))
        differ = cmp["differ"]
        a(f"- verdicts that differ between the two references: {len(differ)} of "
          f"{cmp['n_compared']} (collapsed layer, arm) cells"
          + (": " + "; ".join(f"{x.m} L{int(x.layer)} `{x.arm}` {x.own} -> {x.pub}"
                              for x in differ.itertuples()) if len(differ) else ""))
        a("- largest absolute change of the recovered share of the SIF gain: "
          + (f"{cmp['max_dfrac']:.3f} ({cmp['max_cell']}), over the {cmp['n_both']} cells "
             "where the share is evaluated under both references" if cmp["n_both"]
             else "not defined (no cell is evaluated under both references)"))
        v = f"{cmp['n_compared'] - len(differ)}/{cmp['n_compared']}"
        verdicts.append(("R1 verdict is the same with the published sif_only cell as AUROC_sif "
                         "((collapsed layer, arm) cells)", v))
    a("")

    # ---------------------------------------------------------------- R2
    a("## 4. R2: does the token mix carry the direction?")
    if n_coll:
        v = f"{int(coll['r2'].sum())}/{n_coll}"
        a(f"- **R2 holds at {v} collapsed layers** (test EV for PC1 >= {R2_EV}); per model: "
          + ", ".join(f"{m} {count(coll[coll['m'] == m]['r2'])}" for m in T5
                      if (coll["m"] == m).any()))
        verdicts.append((f"R2 token mix carries PC1 (test EV >= {R2_EV}) at collapsed layers", v))
    for label, t in [("all collapsed layers", coll)] + [
            (m, coll[coll["m"] == m]) for m in T5 if (coll["m"] == m).any()]:
        if t.empty:
            continue
        a(f"### {label} (n = {len(t)}): median (min to max)")
        for d in PC_NAMES:
            a(f"- {d.upper()}: test EV {_rng(t[f'ev_{d}'])}, r2 {_rng(t[f'r2_{d}'])}; train EV "
              f"{_rng(t[f'tr_ev_{d}'])}; share of the train variance "
              f"{_rng(t[f'tr_varshare_{d}'])}")
        a(f"- random directions, mean over {N_RANDOM}: test EV {_rng(t['rand_ev_mean'])}; "
          f"lowest {_rng(t['rand_ev_min'])}, highest {_rng(t['rand_ev_max'])}; train EV "
          f"{_rng(t['tr_rand_ev_mean'])}")
        for d in JOINT:
            tag = "rand2" if d == "pcs2_3" else "rand3"
            a(f"- {JOINT_LABEL[d]}: test EV variance-weighted {_rng(t[f'ev_{d}'])}, "
              f"equal-weight {_rng(t[f'eveq_{d}'])}; train {_rng(t[f'tr_ev_{d}'])} and "
              f"{_rng(t[f'tr_eveq_{d}'])}; random {2 if tag == 'rand2' else 3}-D subspaces, "
              f"mean: variance-weighted {_rng(t[f'{tag}_ev_mean'])}, equal-weight "
              f"{_rng(t[f'{tag}_eveq_mean'])}")
        a(f"- train variance ratio var(PC2) / var(PC3): {_rng(t['var_ratio_pc2_pc3'], '.2f')}"
          f"; below 1.5 at {count(t['var_ratio_pc2_pc3'] < 1.5)} layers (there PC2 and PC3 "
          "are close to interchangeable and only the subspace readouts are stable)")
        a("- secondary, top-variance coordinates: test EV " + "; ".join(
            f"{d} {_rng(t[f'ev_{d}'])}" for d in COORD_NAMES))
    a("")

    # ---------------------------------------------------------------- shares
    a("## 5. Which token group carries the score variance?")
    for label, t in [("all collapsed layers", coll)] + [
            (m, coll[coll["m"] == m]) for m in T5 if (coll["m"] == m).any()]:
        if t.empty:
            continue
        a(f"### {label} (n = {len(t)}): median (min to max)")
        for d in PC_NAMES:
            a(f"- {d.upper()} share, test: " + "; ".join(
                f"{grp} {_rng(t[f'sh_{grp}_{d}'], '+.3f')}" for grp in GROUPS)
              + f"; first token {_rng(t[f'first_{d}'], '+.3f')}")
        a("- PC1 share, train: " + "; ".join(
            f"{grp} {_rng(t[f'tr_sh_{grp}_pc1'], '+.3f')}" for grp in GROUPS))
        a("- largest PC1 group (test): " + ", ".join(
            f"{grp} at {count(t['top_group'] == grp)}" for grp in GROUPS))
        for d in JOINT:
            a(f"- {JOINT_LABEL[d]} share, test, variance-weighted: " + "; ".join(
                f"{grp} {_rng(t[f'sh_{grp}_{d}'], '+.3f')}" for grp in GROUPS)
              + "; equal-weight: " + "; ".join(
                  f"{grp} {_rng(t[f'sheq_{grp}_{d}'], '+.3f')}" for grp in GROUPS))
        a("- random directions, mean share: " + "; ".join(
            f"{grp} {_rng(t[f'rand_sh_{grp}'], '+.3f')}" for grp in GROUPS))
        a("- token mass: " + "; ".join(f"{grp} {_rng(t[f'mass_{grp}'])}" for grp in GROUPS)
          + f"; test tokens of a type unseen in train {_rng(t['unseen_token_frac'])}")
        a(f"- Spearman of the PC1 score with the frequent-token mass: "
          f"{_rng(t['rhof_pc1'], '+.3f')}")
    a("")

    # ---------------------------------------------------------------- R3
    a("## 6. R3: token ablation")
    a(f"Two random controls, each the mean of {CONTROL_DRAWS} draws per (ranking, m) cell. "
      "The count-nearest control drops as many token types as the carriers, drawn among the "
      "types nearest in train count. The mass-matched control drops random other types that "
      "hold at least as many train tokens as the carriers; it was added after the first full "
      "run, on review, because the count-nearest control was not mass-matched. R3 and the "
      "carrier arms are unchanged.")
    groups6 = [("all collapsed layers", coll)] + [
        (m, coll[coll["m"] == m]) for m in T5 if (coll["m"] == m).any()]
    if n_coll:
        v = f"{int(coll['r3'].sum())}/{n_coll}"
        a(f"- **R3 holds at {v} collapsed layers** (test AUROC >= {R3_AUROC:.2f} for some "
          f"m <= {R3_MAX_M}, either ranking); per model: "
          + ", ".join(f"{m} {count(coll[coll['m'] == m]['r3'])}" for m in T5
                      if (coll["m"] == m).any())
          + f"; the count-nearest control reaches {R3_AUROC:.2f} at some cell at "
          f"{count(coll['ctl_r3'])}, the mass-matched control at {count(coll['mctl_r3'])}")
        verdicts.append((f"R3 token ablation restores collapsed layers (AUROC >= "
                         f"{R3_AUROC:.2f}, m <= {R3_MAX_M})", v))
        for kind, col in (("control", "ctl_r3"), ("control_mass", "mctl_r3")):
            verdicts.append((f"the {CONTROL_LABEL[kind]} of the token ablation reaches AUROC "
                             f">= {R3_AUROC:.2f} at some cell (collapsed layers)",
                             count(coll[col])))

    def med(t: pd.DataFrame, column: str, fmt: str = ".3f") -> str:
        v = t[column].dropna()
        return format(v.median(), fmt) if len(v) else "n/a"

    def spread(v: pd.Series) -> str:
        v = v.dropna()
        if v.empty:
            return "-- / -- / --"
        return f"{f3(v.min())} / {f3(v.median())} / {f3(v.max())}"

    for label, t in groups6:
        if t.empty:
            continue
        a(f"### {label} (n = {len(t)})")
        a(f"- cell chosen on train AUROC of the carrier arm: test AUROC {_rng(t['sel_abl'])}; "
          f"count-nearest control {_rng(t['sel_ctl'])}; mass-matched control "
          f"{_rng(t['sel_mctl'])}; mean arm {_rng(t['auc_mean'])}")
        a(f"- highest test AUROC over rankings and m (a pick on test): {_rng(t['best_abl'])}; "
          f"count-nearest control at that cell {_rng(t['best_ctl'])}; mass-matched control "
          f"{_rng(t['best_mctl'])}")
        for ranking in RANKINGS:
            ok = t[f"r3m_{ranking}"] > 0
            a(f"- ranking by {RANK_LABEL[ranking]}: restored at {count(ok)}"
              + (f" (smallest m: {_rng(t.loc[ok, f'r3m_{ranking}'], '.0f')})" if ok.any() else "")
              + ". Medians by m, in the order carriers / count-nearest control / mass-matched "
              "control:")
            a("  - test AUROC: " + "; ".join(
                f"m={m} {med(t, f'abl_{ranking}_m{m}')} / {med(t, f'ctl_{ranking}_m{m}_mean')} / "
                f"{med(t, f'mctl_{ranking}_m{m}_mean')}" for m in ABL_MS))
            a("  - train tokens dropped: " + "; ".join(
                f"m={m} {med(t, f'tc_{ranking}_m{m}', '.0f')} / "
                f"{med(t, f'ctc_{ranking}_m{m}', '.0f')} / {med(t, f'mtc_{ranking}_m{m}', '.0f')}"
                for m in ABL_MS))
            a("  - share of test tokens dropped: " + "; ".join(
                f"m={m} {med(t, f'dm_{ranking}_m{m}')} / {med(t, f'cdm_{ranking}_m{m}')} / "
                f"{med(t, f'mdm_{ranking}_m{m}')}" for m in ABL_MS))
            a("  - token types in a mass-matched draw: " + "; ".join(
                f"m={m} {med(t, f'mnt_{ranking}_m{m}', '.1f')}" for m in ABL_MS))
    a("### Fixed settings (no pick on test)")
    a("One ranking and one m for every layer, so nothing is chosen on test passages. Restored "
      f"= test AUROC >= {R3_AUROC:.2f} at that cell. The controls are the means over draws at "
      "the same cell.")
    for ranking, m, note in FIXED_CELLS:
        tag = f"{ranking}_m{m}"
        a(f"- ranking `{ranking}`, m = {m}{note}:")
        for label, t in groups6:
            if t.empty:
                continue
            auc = t[f"abl_{tag}"]
            a(f"  - {label} (n = {len(t)}): carriers restored at {count(auc >= R3_AUROC)}; "
              f"AUROC min / median / max {spread(auc)}; share of test tokens dropped "
              f"{_rng(t[f'dm_{tag}'])}; train tokens dropped {_rng(t[f'tc_{tag}'], '.0f')}")
            for kind, pre, dm, tc in (("control", "ctl", "cdm", "ctc"),
                                      ("control_mass", "mctl", "mdm", "mtc")):
                c = t[f"{pre}_{tag}_mean"]
                a(f"    - {CONTROL_LABEL[kind]}: restored at {count(c >= R3_AUROC)}; AUROC "
                  f"min / median / max {spread(c)}; share of test tokens dropped "
                  f"{_rng(t[f'{dm}_{tag}'])}; train tokens dropped "
                  f"{_rng(t[f'{tc}_{tag}'], '.0f')}")
    a("")

    # ---------------------------------------------------------------- R4
    a("## 7. R4: passage length")
    for name in present:
        t = length[length["model"] == {v: k for k, v in DISP.items()}.get(name, name)]
        for x in t.itertuples():
            a(f"- {name}, {x.split}: n_p mean {x.mean_n:.1f}, median {x.median_n:.0f}; "
              f"truncated at {MAX_LENGTH} {int(x.n_truncated)}, no token {int(x.n_zero_token)}"
              f" of {int(x.n_passages)}; mean |delta log n| same-directory "
              f"{x.mean_dlog_same:.4f} ({int(x.n_same_pairs)} pairs), different-directory "
              f"{x.mean_dlog_diff:.4f} ({int(x.n_diff_pairs)} pairs); medians "
              f"{x.median_dlog_same:.4f} and {x.median_dlog_diff:.4f}; AUROC of "
              f"-|delta log n| {x.auroc_neg_dlog:.4f}")
    if n_coll:
        v = f"{int(coll['r4'].sum())}/{n_coll}"
        a(f"- **R4 holds at {v} collapsed layers**; |Spearman(s_PC1, log n)| >= {R4_RHO} at "
          f"{count(coll['rho_pc1'].abs() >= R4_RHO)}; same-directory pairs differ more in log "
          f"length than different-directory pairs at {count(coll['len_same'] > coll['len_diff'])}"
          " (a per-model fact)")
        verdicts.append(("R4 length account at collapsed layers", v))
        for m in T5:
            t = coll[coll["m"] == m]
            if len(t):
                a(f"  - {m}: R4 at {count(t['r4'])}; Spearman of the PC1 score with log n "
                  f"{_rng(t['rho_pc1'], '+.3f')} (train {_rng(t['tr_rho_pc1'], '+.3f')}); PC2 "
                  f"{_rng(t['rho_pc2'], '+.3f')}, PC3 {_rng(t['rho_pc3'], '+.3f')}; random "
                  f"directions, mean |rho| {_rng(t['rand_absrho_mean'])}")
    a("")

    # ---------------------------------------------------------------- expectations
    a("## 8. Expectations recorded before the run")
    for m in ("LaTa", "PhilTa"):
        t = coll[coll["m"] == m]
        if t.empty:
            continue
        k = int((t["top_group"] == "frequent").sum())
        v = f"{'MET' if k == len(t) else 'NOT MET'} ({k}/{len(t)})"
        a(f"- {m}: `frequent` has the largest PC1 share: **{v}**")
        verdicts.append((f"{m}: frequent tokens have the largest PC1 share at collapsed layers",
                         v))
        def keeps(pre: str, t=t) -> str:
            k = int((t[f"{pre}_sif_keepspecial"] == "rescue").sum())
            return f"{'MET' if k == len(t) else 'NOT MET'} ({k}/{len(t)})"

        def nospecial(pre: str, t=t) -> str:
            k = int((t[f"{pre}_mean_nospecial"] == "no_rescue").sum())
            ng = int((t[f"{pre}_mean_nospecial"] == "no_sif_gain").sum())
            v = f"{'MET' if k == len(t) else 'NOT MET'} ({k}/{len(t)} not rescued"
            return v + (f"; {ng} with no SIF gain to recover)" if ng else ")")

        both = bool(t["pub_sif"].notna().all())
        v = keeps("r1")
        a(f"- {m}: `sif_keepspecial` rescues: **{v}**"
          + (f"; with the published `sif_only` cell as AUROC_sif: {keeps('r1p')}" if both else ""))
        verdicts.append((f"{m}: sif_keepspecial rescues collapsed layers (R1)", v))
        v = nospecial("r1")
        a(f"- {m}: `mean_nospecial` does not rescue: **{v}**"
          + (f"; with the published `sif_only` cell as AUROC_sif: {nospecial('r1p')}"
             if both else ""))
        verdicts.append((f"{m}: mean_nospecial does not rescue collapsed layers (R1)", v))
    t = coll[coll["m"] == "mT5-base"]
    if len(t):
        def no_arm(pre: str, gain: str) -> str:
            evaluable = (t[gain] >= R1_MIN_SIF_GAIN).to_numpy()
            rescued = np.zeros(len(t), dtype=bool)
            for arm in R1_ARMS:
                rescued |= (t[f"{pre}_{arm}"] == "rescue").to_numpy()
            if not evaluable.any():
                return (f"NOT EVALUABLE BY R1 (SIF gain below {R1_MIN_SIF_GAIN:.2f} at all "
                        f"{len(t)} layers, so there is no SIF gain to recover)")
            k = int((~rescued & evaluable).sum())
            return (f"{'MET' if k == int(evaluable.sum()) else 'NOT MET'} ({k}/"
                    f"{int(evaluable.sum())} evaluable layers with no rescuing arm; "
                    f"{int((~evaluable).sum())} layers with no SIF gain to recover)")

        v = no_arm("r1", "sif_gain")
        a(f"- mT5-base: no arm rescues: **{v}**"
          + (f"; with the published `sif_only` cell as AUROC_sif: {no_arm('r1p', 'sif_gain_pub')}"
             if t["pub_sif"].notna().all() else "")
          + ". Raw changes against mean pooling, median (min "
          "to max): " + "; ".join(f"{arm} {_rng(t[f'r1chg_{arm}'], '+.3f')}" for arm in R1_ARMS)
          + f"; sif {_rng(t['sif_gain'], '+.3f')}")
        verdicts.append(("mT5-base: no arm rescues collapsed layers (R1)", v))
        k = int((t["top_group"] == "other").sum())
        v = f"{'MET' if k == len(t) else 'NOT MET'} ({k}/{len(t)})"
        a(f"- mT5-base: tokens SIF keeps (`other`) have the largest PC1 share: **{v}**")
        verdicts.append(("mT5-base: `other` tokens have the largest PC1 share at collapsed "
                         "layers", v))
    a("")

    # ---------------------------------------------------------------- carriers
    a("## 9. Carrier tokens at the worst layer of each T5 model")
    id_of = {v: k for k, v in DISP.items()}
    for name in [m for m in T5 if m in present]:
        x = worst_layer(w, name)
        a(f"- {name} L{int(x['layer'])}: PC1 share by group " + ", ".join(
            f"{grp} {x[f'sh_{grp}_pc1']:+.3f}" for grp in GROUPS)
          + f"; token-mix EV {f3(x['ev_pc1'])}")
        L.extend(carrier_lines(carriers, id_of.get(name, name), int(x["layer"])))
    a("")

    # ---------------------------------------------------------------- healthy contrast
    a("## 10. Contrast: embedding-trained models and T5 layers that are not collapsed")
    groups = [(m, w[w["m"] == m]) for m in NON_T5 if m in present]
    groups += [(f"{m}, not collapsed", w[(w["m"] == m) & ~w["collapsed"]]) for m in T5
               if m in present]
    for label, t in groups:
        if t.empty:
            continue
        a(f"- {label} ({len(t)} layers; layers {_layers(t)}), median (min to max):")
        a("  - AUROC: " + "; ".join(f"{arm} {_rng(t[f'auc_{arm}'])}" for arm in ARMS))
        a(f"  - token-mix EV: PC1 {_rng(t['ev_pc1'])} (R2 at {count(t['r2'])}); random "
          f"directions, mean {_rng(t['rand_ev_mean'])}; {JOINT_LABEL['pcs1_3']} "
          f"variance-weighted {_rng(t['ev_pcs1_3'])}, equal-weight {_rng(t['eveq_pcs1_3'])}")
        a("  - PC1 share: " + "; ".join(
            f"{grp} {_rng(t[f'sh_{grp}_pc1'], '+.3f')}" for grp in GROUPS)
          + f"; first token {_rng(t['first_pc1'], '+.3f')}; PC1 share of the train variance "
          f"{_rng(t['tr_varshare_pc1'])}")
        a(f"  - Spearman of the PC1 score with log n {_rng(t['rho_pc1'], '+.3f')}; token "
          f"ablation at the cell chosen on train AUROC: test AUROC {_rng(t['sel_abl'])}, "
          f"count-nearest control {_rng(t['sel_ctl'])}, mass-matched control "
          f"{_rng(t['sel_mctl'])}")
    a("")

    a("## 11. Verdicts in one place")
    if gate2_failed:
        a("- " + sif_deviation(gates, cmp))
        a("- The R1 lines below use the job's own `sif` arm as AUROC_sif; sections 3 and 8 "
          "also give the counts with the published `sif_only` cells.")
    for claim, v in verdicts:
        a(f"- {claim}: {v}")
    a("")

    # ---------------------------------------------------------------- per layer
    a("## 12. All layers")
    a("\\* marks a collapsed layer. R1 cells: share of the SIF gain recovered, or `ng` and "
      "the raw change where SIF gains less than "
      f"{R1_MIN_SIF_GAIN:.2f}.")
    for name in present:
        t = w[w["m"] == name].sort_values("layer")

        def lab(x) -> str:
            return f"{int(x['layer'])}{'*' if x['collapsed'] else ''}"

        def r1cell(x, arm) -> str:
            if np.isfinite(x[f"r1frac_{arm}"]):
                return f"{x[f'r1frac_{arm}']:.2f}"
            return f"ng {x[f'r1chg_{arm}']:+.3f}"

        a(f"### {name}: AUROC by pooling arm, and R1")
        L.extend(_md(["L", "published", *ARMS, *(f"R1 {arm}" for arm in R1_ARMS)],
                     [[lab(x), f3(x["base"]) if x["base_published"] else "--",
                       *(f3(x[f"auc_{arm}"]) for arm in ARMS),
                       *(r1cell(x, arm) for arm in R1_ARMS)] for _, x in t.iterrows()]))
        a("")
        a(f"### {name}: token audit of PC1 (test)")
        L.extend(_md(["L", "EV", "r2", "EV train", "EV random mean (min to max)",
                      *(f"share {grp}" for grp in GROUPS), "share first", "rho log n",
                      "rho freq mass", "R2", "R4"],
                     [[lab(x), f3(x["ev_pc1"]), f3(x["r2_pc1"]), f3(x["tr_ev_pc1"]),
                       f"{f3(x['rand_ev_mean'])} ({f3(x['rand_ev_min'])} to "
                       f"{f3(x['rand_ev_max'])})",
                       *(f"{x[f'sh_{grp}_pc1']:+.3f}" for grp in GROUPS),
                       f"{x['first_pc1']:+.3f}", f"{x['rho_pc1']:+.3f}", f"{x['rhof_pc1']:+.3f}",
                       "yes" if x["r2"] else "no", "yes" if x["r4"] else "no"]
                      for _, x in t.iterrows()]))
        a("")
        a(f"### {name}: PC2, PC3 and the subspaces (test)")
        L.extend(_md(["L", "var PC1", "var PC2", "var PC3", "var PC2 / var PC3", "EV PC2",
                      "EV PC3", "EV span(PC2,3)", "eq-weight", "EV span(PC1-3)", "eq-weight",
                      "EV random 2-D", "EV random 3-D",
                      *(f"span(PC2,3) {grp}" for grp in GROUPS),
                      *(f"span(PC1-3) eq-weight {grp}" for grp in GROUPS)],
                     [[lab(x), *(f"{x[f'tr_var_{d}']:.4g}" for d in PC_NAMES),
                       f2(x["var_ratio_pc2_pc3"]), f3(x["ev_pc2"]), f3(x["ev_pc3"]),
                       f3(x["ev_pcs2_3"]), f3(x["eveq_pcs2_3"]), f3(x["ev_pcs1_3"]),
                       f3(x["eveq_pcs1_3"]), f3(x["rand2_ev_mean"]), f3(x["rand3_ev_mean"]),
                       *(f"{x[f'sh_{grp}_pcs2_3']:+.3f}" for grp in GROUPS),
                       *(f"{x[f'sheq_{grp}_pcs1_3']:+.3f}" for grp in GROUPS)]
                      for _, x in t.iterrows()]))
        a("")
        for ranking in RANKINGS:
            a(f"### {name}: token ablation, ranking by {RANK_LABEL[ranking]} (test AUROC: "
              "carriers / count-nearest control / mass-matched control)")
            L.extend(_md(["L", "mean", *(f"m={m}" for m in ABL_MS), "R3 smallest m"],
                         [[lab(x), f3(x["auc_mean"]),
                           *(f"{f3(x[f'abl_{ranking}_m{m}'])} / "
                             f"{f3(x[f'ctl_{ranking}_m{m}_mean'])} / "
                             f"{f3(x[f'mctl_{ranking}_m{m}_mean'])}" for m in ABL_MS),
                           str(int(x[f"r3m_{ranking}"])) if x[f"r3m_{ranking}"] > 0 else "none"]
                          for _, x in t.iterrows()]))
            a("")

            def n0(v) -> str:
                return f"{v:.0f}" if np.isfinite(v) else "--"

            a(f"### {name}: tokens dropped, ranking by {RANK_LABEL[ranking]} (train tokens, "
              "then the share of test tokens in parentheses: carriers / count-nearest control "
              "/ mass-matched control)")
            L.extend(_md(["L", *(f"m={m}" for m in ABL_MS)],
                         [[lab(x), *(
                             f"{n0(x[f'tc_{ranking}_m{m}'])} / {n0(x[f'ctc_{ranking}_m{m}'])} / "
                             f"{n0(x[f'mtc_{ranking}_m{m}'])} ({f3(x[f'dm_{ranking}_m{m}'])} / "
                             f"{f3(x[f'cdm_{ranking}_m{m}'])} / {f3(x[f'mdm_{ranking}_m{m}'])})"
                             for m in ABL_MS)] for _, x in t.iterrows()]))
            a("")
    path.write_text("\n".join(L) + "\n")


def cmd_render(args) -> int:
    arms = read_csv(args.out_dir / ARMS_NAME)
    audit = read_csv(args.out_dir / AUDIT_NAME)
    carriers = read_csv(args.out_dir / CARRIER_NAME)
    abl = read_csv(args.out_dir / ABL_NAME)
    length = read_csv(args.out_dir / LENGTH_NAME)
    res = pd.read_csv(args.results_csv) if Path(args.results_csv).exists() else None
    w = summarize(arms, audit, abl, length, res)
    gates = gates_for(arms, audit, args)
    args.tab_dir.mkdir(parents=True, exist_ok=True)
    omitted = write_table(w, args.tab_dir / TABLE_NAME, gates)
    for name in omitted:
        print(f"omitting {name}: no rows in {args.out_dir / ARMS_NAME}")
    print(f"wrote {args.tab_dir / TABLE_NAME} ({len(ORDER) - len(omitted)} model rows)")
    for name in ORDER:
        x = worst_layer(w, name)
        if x is not None:
            print(f"  worst baseline layer {name}: {int(x['layer'])} (AUROC {x['base']:.3f})")
    if args.facts_md is not None:
        args.facts_md.parent.mkdir(parents=True, exist_ok=True)
        facts(w, arms, carriers, length, gates, args.facts_md, omitted=omitted)
        print(f"wrote {args.facts_md}")
    return 0


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def refs(p) -> None:
        p.add_argument("--out_dir", type=Path, default=None,
                       help=f"default {OUT_DIR}; {OUT_DIR}/smoke_limit<N> under --limit N")
        p.add_argument("--results_csv", type=Path, default=RES_CSV)
        p.add_argument("--tol_auroc", type=float, default=GATE_TOL_AUROC,
                       help="AUROC tolerance of gates 1a and 2a (default 1e-6)")
        p.add_argument("--models", default="",
                       help="comma list of ids or display names (default all six); the "
                            "gates fail if one of them has no rows")
        p.add_argument("--allow_missing", action="store_true",
                       help="gates: do not require every model to have rows")

    p = sub.add_parser("audit", help="forward passes, readouts and the result CSVs (GPU)")
    refs(p)
    p.add_argument("--split_csv", type=Path, default=SPLIT_CSV)
    p.add_argument("--bases_root", type=Path, default=BASES_ROOT)
    p.add_argument("--e1_coords_csv", type=Path, default=E1_COORD_CSV)
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
                   help="metric worker processes (forked before torch is imported)")
    p.add_argument("--limit", type=int, default=0,
                   help="smoke run on the first N train and N test passages; gates skipped")
    p.add_argument("--control_metrics", choices=("auroc", "full"), default="auroc",
                   help="the two random controls: Task A AUROC alone (default) or the full "
                        "metric block and geometry (about 4 times the CPU time)")
    p.add_argument("--check", action="store_true",
                   help="run the gates after writing; exit 3 if one fails")
    p = sub.add_parser("check", help="gates on existing CSVs")
    refs(p)
    p.add_argument("--no_write", action="store_true", help=f"do not rewrite {GATE_NAME}")
    p = sub.add_parser("render", help="paper table and facts file")
    refs(p)
    p.add_argument("--tab_dir", type=Path, default=TAB_DIR)
    p.add_argument("--facts_md", type=Path, default=None)
    p.add_argument("--no_facts", action="store_true")
    args = ap.parse_args(argv)
    if args.out_dir is None:
        limit = getattr(args, "limit", 0)
        args.out_dir = OUT_DIR / f"smoke_limit{limit}" if limit else OUT_DIR
    if args.cmd == "audit":
        return cmd_audit(args)
    if args.cmd == "check":
        return cmd_check(args)
    if args.facts_md is None and not args.no_facts:
        args.facts_md = args.out_dir / FACTS_NAME
    if args.no_facts:
        args.facts_md = None
    return cmd_render(args)


if __name__ == "__main__":
    sys.exit(main())
