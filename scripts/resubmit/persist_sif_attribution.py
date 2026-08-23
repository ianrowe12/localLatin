"""Persist SIF-weighted attribution variants (``sif``, ``sif_abtt``) per pair.

Sibling of ``persist_attribution_methods.py``, which writes the ``baseline``
(raw hidden states) and ``abtt`` (top-D components removed) variants. This
script adds the two SIF variants on top of them:

    pair_matrix_<method>_sif       = SIF-reweighted pair_matrix_<method>_baseline
    pair_matrix_<method>_sif_abtt  = SIF-reweighted pair_matrix_<method>_abtt

The reweighting is the standard SIF token weight [Arora et al. 2017]

    w(t) = a / (a + p(t)),   a = 1e-3

applied to the *token-level aggregation* of an already-computed pair matrix::

    M_sif[i, j] = M[i, j] * sqrt(w_hat_q[i] * w_hat_c[j])

with ``w_hat = w * n / sum(w)`` (mean 1 over the sequence) so a flat weight
vector is the identity, and a final rescale that preserves ``sum |M|`` so the
SIF panel stays on the same colour scale as the panel it was derived from.
For ``ig`` and ``ot`` -- the two methods whose baseline matrices are already an
outer product of L1-normalised |IG| scores -- this is algebraically identical
to feeding SIF-scaled IG scores into the builder and renormalising.

Leak-free protocol: p(t) is a unigram distribution estimated over the TRAIN
split only (default ``runs/active/resubmit/data/phase_resubmit_split.csv``).
Documents whose filename also appears as a query or candidate in the examples
CSV are dropped from the estimation corpus, so no evaluated pair contributes to
its own weights. Special tokens get weight 0, matching
``sif_abtt.sif_weights_from_ids``.

Also written per NPZ (all trimmed to masked length):

    query_sif_weights / candidate_sif_weights   mean-1 normalised w_hat
    query_ig_sif / query_ig_sif_abtt            SIF-scaled per-token IG
    candidate_ig_sif / candidate_ig_sif_abtt

Usage::

    python scripts/resubmit/persist_sif_attribution.py \\
        --examples_csv runs/active/ig_examples/phase12f_examples.csv \\
        --artifacts_dir runs/active/ig_examples/artifacts

    # inspect what SIF does to the frequent-function-word tokens of one pair
    python scripts/resubmit/persist_sif_attribution.py ... \\
        --spot_check_example 6 --spot_check_tokens et in de est --dry_run
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ig"))

from sif_abtt import sif_weights_from_ids, token_probabilities  # noqa: E402

from persist_attribution_methods import (  # noqa: E402
    MAIN_METHODS,
    PRESERVED_SIDECAR_METHODS,
    atomic_write_npz,
    topk_indices,
)
from persist_decoded_tokens import SLUG_TO_HF, decode_ids, get_tokenizer  # noqa: E402

ALL_METHODS: list[str] = MAIN_METHODS + PRESERVED_SIDECAR_METHODS

# (source variant -> new variant) pairs.
SIF_VARIANTS: list[tuple[str, str]] = [("baseline", "sif"), ("abtt", "sif_abtt")]

DEFAULT_SPLIT_CSV = REPO_ROOT / "runs/active/resubmit/data/phase_resubmit_split.csv"


# ---------------------------------------------------------------------------
# Token probabilities (train split only)
# ---------------------------------------------------------------------------

def load_train_texts(
    split_csv: Path,
    repo_root: Path,
    exclude_filenames: set[str],
) -> list[str]:
    split = pd.read_csv(split_csv)
    train = split[split["split"] == "train"]
    texts: list[str] = []
    dropped = 0
    missing = 0
    for _, row in train.iterrows():
        name = str(row.get("filename", ""))
        if name in exclude_filenames:
            dropped += 1
            continue
        raw = str(row["path"])
        path = Path(raw)
        if not path.is_absolute():
            path = repo_root / raw
        if not path.exists():
            missing += 1
            continue
        texts.append(path.read_text(encoding="utf-8", errors="ignore"))
    print(
        f"  train corpus: {len(texts)} docs "
        f"({dropped} excluded as evaluated pairs, {missing} unresolved paths)"
    )
    if not texts:
        raise SystemExit(f"No readable train documents from {split_csv}")
    return texts


@lru_cache(maxsize=8)
def _probs_cache_path(cache_dir: str, slug: str) -> Path:
    return Path(cache_dir) / f"{slug}_train_token_probs.json"


def token_probs_for_model(
    slug: str,
    hf_id: str,
    texts: list[str],
    cache_dir: Path | None,
    trust_remote_code: bool,
    max_length: int,
    refresh: bool,
) -> dict[int, float]:
    cache_path = None
    if cache_dir is not None:
        cache_path = _probs_cache_path(str(cache_dir), slug)
        if cache_path.exists() and not refresh:
            with cache_path.open() as fh:
                return {int(k): float(v) for k, v in json.load(fh).items()}

    tokenizer = get_tokenizer(hf_id, trust_remote_code)
    probs = token_probabilities(tokenizer, texts, max_length=max_length)
    print(f"  {slug}: estimated p(t) over {len(probs)} distinct token ids")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w") as fh:
            json.dump({str(k): v for k, v in probs.items()}, fh)
    return probs


# ---------------------------------------------------------------------------
# Reweighting
# ---------------------------------------------------------------------------

def normalized_sif_weights(
    input_ids: np.ndarray,
    probs: dict[int, float],
    a: float,
    special_ids: list[int],
    seq_len: int,
) -> np.ndarray:
    """Mean-1 normalised SIF weights for the first ``seq_len`` tokens."""
    ids = np.asarray(input_ids)
    if ids.ndim == 1:
        ids = ids[None, :]
    w = sif_weights_from_ids(ids, probs, a=a, special_ids=special_ids)[0, :seq_len]
    w = np.asarray(w, dtype=np.float32)
    total = float(w.sum())
    if total <= 0.0:
        return np.ones(seq_len, dtype=np.float32)
    return (w * (seq_len / total)).astype(np.float32)


def reweight_matrix(mat: np.ndarray, w_q: np.ndarray, w_c: np.ndarray) -> np.ndarray:
    """Apply sqrt(w_q x w_c) and rescale to preserve the L1 mass of ``mat``."""
    scale = np.sqrt(np.outer(w_q, w_c)).astype(np.float32)
    out = mat.astype(np.float32) * scale
    src_mass = float(np.abs(mat).sum())
    out_mass = float(np.abs(out).sum())
    if out_mass > 0.0 and src_mass > 0.0:
        out = out * np.float32(src_mass / out_mass)
    return out.astype(np.float32)


def build_sif_keys(
    data: dict[str, np.ndarray],
    w_q: np.ndarray,
    w_c: np.ndarray,
    topk: int,
) -> tuple[dict[str, np.ndarray], list[str]]:
    new_keys: dict[str, np.ndarray] = {}
    methods: list[str] = []
    q_len = w_q.shape[0]
    c_len = w_c.shape[0]

    for method in ALL_METHODS:
        made_any = False
        for src_variant, dst_variant in SIF_VARIANTS:
            src_key = f"pair_matrix_{method}_{src_variant}"
            if src_key not in data:
                continue
            mat = np.asarray(data[src_key], dtype=np.float32)[:q_len, :c_len]
            if mat.shape != (q_len, c_len):
                print(
                    f"  [WARN] {src_key}: shape {mat.shape} != ({q_len},{c_len}); skipping",
                    file=sys.stderr,
                )
                continue
            out = reweight_matrix(mat, w_q, w_c)
            new_keys[f"pair_matrix_{method}_{dst_variant}"] = out
            q_top, c_top = topk_indices(out, topk)
            new_keys[f"topk_{method}_{dst_variant}_query"] = q_top
            new_keys[f"topk_{method}_{dst_variant}_candidate"] = c_top
            made_any = True
        if made_any:
            methods.append(method)

    new_keys["query_sif_weights"] = w_q
    new_keys["candidate_sif_weights"] = w_c
    for side, weights, length in (("query", w_q, q_len), ("candidate", w_c, c_len)):
        for src_variant, dst_variant in SIF_VARIANTS:
            ig_key = f"{side}_ig_{src_variant}"
            if ig_key in data:
                ig = np.asarray(data[ig_key], dtype=np.float32)[:length]
                new_keys[f"{side}_ig_{dst_variant}"] = (ig * weights).astype(np.float32)
    return new_keys, methods


# ---------------------------------------------------------------------------
# Spot check
# ---------------------------------------------------------------------------

def token_mass(mat: np.ndarray, axis: int) -> np.ndarray:
    """Share of total |M| mass carried by each token on ``axis``."""
    abs_m = np.abs(mat)
    marginal = abs_m.sum(axis=1 - axis)
    total = marginal.sum()
    return marginal / total if total > 0 else marginal


def spot_check(
    path: Path,
    data: dict[str, np.ndarray],
    w_q: np.ndarray,
    w_c: np.ndarray,
    slug: str,
    method: str,
    target_tokens: list[str],
    trust_remote_code: bool,
) -> None:
    hf_id = SLUG_TO_HF[slug]
    tokenizer = get_tokenizer(hf_id, trust_remote_code)
    q_len, c_len = w_q.shape[0], w_c.shape[0]
    q_tokens = decode_ids(tokenizer, data["query_input_ids"])[:q_len]
    c_tokens = decode_ids(tokenizer, data["candidate_input_ids"])[:c_len]

    base = np.asarray(data[f"pair_matrix_{method}_baseline"], dtype=np.float32)[:q_len, :c_len]
    sif = reweight_matrix(base, w_q, w_c)

    print(f"\n=== SIF spot check: {path.name} ({slug}, method={method}) ===")
    wanted = {t.lower() for t in target_tokens}
    for side, tokens, weights, axis in (
        ("query", q_tokens, w_q, 0),
        ("candidate", c_tokens, w_c, 1),
    ):
        base_mass = token_mass(base, axis)
        sif_mass = token_mass(sif, axis)
        rows = [
            (i, tok, float(weights[i]), float(base_mass[i]), float(sif_mass[i]))
            for i, tok in enumerate(tokens)
            if tok.strip().lower() in wanted
        ]
        if not rows:
            print(f"  [{side}] none of {sorted(wanted)} present")
            continue
        print(f"  [{side}] idx  token      sif_w   base_mass   sif_mass   ratio")
        for i, tok, w, bm, sm in rows:
            ratio = (sm / bm) if bm > 0 else float("nan")
            print(f"         {i:3d}  {tok!r:10s} {w:6.3f}  {bm:9.5f}  {sm:9.5f}  {ratio:6.3f}")
        sel = [r[0] for r in rows]
        bm_tot = float(base_mass[sel].sum())
        sm_tot = float(sif_mass[sel].sum())
        print(
            f"         TOTAL over {len(sel)} target tokens: "
            f"base={bm_tot:.5f} -> sif={sm_tot:.5f} "
            f"({(sm_tot / bm_tot - 1) * 100:+.1f}%)"
        )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Persist sif / sif_abtt attribution variants.")
    p.add_argument("--examples_csv", required=True, type=Path)
    p.add_argument("--artifacts_dir", required=True, type=Path)
    p.add_argument("--split_csv", type=Path, default=DEFAULT_SPLIT_CSV)
    p.add_argument("--repo_root", type=Path, default=REPO_ROOT,
                   help="Root for resolving relative paths in the split CSV.")
    p.add_argument("--sif_a", type=float, default=1e-3)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--token_probs_cache", type=Path, default=None,
                   help="Directory to cache per-model train token probabilities.")
    p.add_argument("--refresh_probs", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true", default=True)
    p.add_argument("--no_trust_remote_code", dest="trust_remote_code", action="store_false")
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--spot_check_example", type=int, default=None)
    p.add_argument("--spot_check_method", default="ig")
    p.add_argument("--spot_check_tokens", nargs="*", default=["et", "in", "de", "est"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    examples = pd.read_csv(args.examples_csv)
    if args.models:
        examples = examples[examples["model_name"].isin(args.models)]

    # Filenames evaluated by the artifacts -- excluded from the SIF corpus.
    evaluated: set[str] = set()
    for col in ("query_path", "candidate_path"):
        for raw in examples[col].dropna().astype(str):
            evaluated.add(Path(raw).name)

    print(f"Loading train corpus from {args.split_csv}")
    texts = load_train_texts(args.split_csv, args.repo_root, evaluated)

    counts = {"ok": 0, "missing": 0, "error": 0}
    per_model: dict[str, int] = {}

    for model_name, group in examples.groupby("model_name", sort=False):
        slug = str(model_name).replace("/", "_")
        hf_id = SLUG_TO_HF.get(slug)
        if hf_id is None:
            print(f"[WARN] no HF id for slug {slug}; skipping model", file=sys.stderr)
            continue
        print(f"\n[{slug}] {len(group)} examples")
        probs = token_probs_for_model(
            slug, hf_id, texts, args.token_probs_cache,
            args.trust_remote_code, args.max_length, args.refresh_probs,
        )
        tokenizer = get_tokenizer(hf_id, args.trust_remote_code)
        special_ids = [int(t) for t in getattr(tokenizer, "all_special_ids", []) or []]

        for _, row in group.iterrows():
            path = (
                args.artifacts_dir
                / slug
                / f"example{int(row['example_id']):03d}_{row['candidate_role']}.npz"
            )
            if not path.exists():
                counts["missing"] += 1
                continue
            try:
                data = dict(np.load(path, allow_pickle=False))
                q_len = int(data["query_attention_mask"][0].sum())
                c_len = int(data["candidate_attention_mask"][0].sum())
                w_q = normalized_sif_weights(
                    data["query_input_ids"], probs, args.sif_a, special_ids, q_len
                )
                w_c = normalized_sif_weights(
                    data["candidate_input_ids"], probs, args.sif_a, special_ids, c_len
                )
                new_keys, methods = build_sif_keys(data, w_q, w_c, args.topk)
            except Exception as exc:  # noqa: BLE001
                print(f"  [ERROR] {path.name}: {exc}", file=sys.stderr)
                counts["error"] += 1
                continue

            if args.spot_check_example is not None and int(row["example_id"]) == args.spot_check_example:
                spot_check(
                    path, data, w_q, w_c, slug, args.spot_check_method,
                    args.spot_check_tokens, args.trust_remote_code,
                )

            if args.dry_run:
                counts["ok"] += 1
                continue

            data.update(new_keys)
            atomic_write_npz(path, data)
            counts["ok"] += 1
            per_model[slug] = per_model.get(slug, 0) + 1
            if counts["ok"] % 20 == 0:
                print(f"  ... {counts['ok']} artifacts written")
        if not args.dry_run:
            print(f"  [{slug}] methods with SIF variants: {methods}")

    print("\n=== Summary ===")
    print(f"  artifacts updated: {counts['ok']}")
    print(f"  missing NPZ:       {counts['missing']}")
    print(f"  errors:            {counts['error']}")
    for slug, n in sorted(per_model.items()):
        print(f"    {slug:60s} {n}")
    if counts["error"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
