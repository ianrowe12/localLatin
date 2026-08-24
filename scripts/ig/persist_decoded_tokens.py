"""Persist decoded token strings into the IG example NPZs.

Before this, ``web/services/token_map_svc.py`` decoded ``query_input_ids`` at
serve time using a hardcoded two-entry slug -> HuggingFace map, so LaBSE,
Qwen3-0.6B, mT5-base and KaLM-mini rendered ``[0] [1] [2] ...`` placeholders in
the highlight view. Storing the strings in the artifact removes the tokenizer
from the serving path entirely.

Two string arrays are added per NPZ (trimmed to the attention-mask length, so
they line up 1:1 with the persisted pair matrices)::

    query_token_strings      (q_len,)  <U..
    candidate_token_strings  (c_len,)  <U..

Tokenizer-only, so this runs fine on CPU for every model.

Usage::

    python scripts/ig/persist_decoded_tokens.py \\
        --examples_csv runs/active/ig_examples/phase12f_examples.csv \\
        --artifacts_dir runs/active/ig_examples/artifacts
"""
from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "resubmit"))

from persist_attribution_methods import atomic_write_npz  # noqa: E402

# Every model in the CLAUDE.md model table plus mT5-base, which the attribution
# layer contract also covers. Kept in sync with the fallback map in
# web/services/token_map_svc.py.
SLUG_TO_HF: dict[str, str] = {
    "bowphs_LaTa": "bowphs/LaTa",
    "bowphs_PhilTa": "bowphs/PhilTa",
    "google_mt5-base": "google/mt5-base",
    "sentence-transformers_LaBSE": "sentence-transformers/LaBSE",
    "Qwen_Qwen3-Embedding-0.6B": "Qwen/Qwen3-Embedding-0.6B",
    "Qwen_Qwen3-Embedding-8B": "Qwen/Qwen3-Embedding-8B",
    "KaLM-Embedding_KaLM-embedding-multilingual-mini-instruct-v2.5": (
        "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5"
    ),
}

QUERY_KEY = "query_token_strings"
CANDIDATE_KEY = "candidate_token_strings"


@lru_cache(maxsize=8)
def get_tokenizer(hf_id: str, trust_remote_code: bool):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(hf_id, trust_remote_code=trust_remote_code)


def decode_ids(tokenizer, ids: np.ndarray) -> list[str]:
    """Decode ids one at a time, preserving whitespace cues where possible.

    ``convert_tokens_to_string`` handles both SentencePiece (``_`` prefix) and
    byte-level BPE (``G`` prefix, multi-byte continuation pieces) correctly,
    which a raw ``convert_ids_to_tokens`` does not. Falls back to ``decode``
    for tokenizers that choke on a single piece.
    """
    flat = [int(t) for t in np.asarray(ids).flatten().tolist()]
    pieces = tokenizer.convert_ids_to_tokens(flat)
    out: list[str] = []
    for tid, piece in zip(flat, pieces):
        text = None
        if piece is not None:
            try:
                text = tokenizer.convert_tokens_to_string([piece])
            except Exception:  # noqa: BLE001
                text = None
        if not text:
            try:
                text = tokenizer.decode([tid])
            except Exception:  # noqa: BLE001
                text = str(piece) if piece is not None else f"[{tid}]"
        out.append(text)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--examples_csv", required=True, type=Path)
    p.add_argument("--artifacts_dir", required=True, type=Path)
    p.add_argument("--trust_remote_code", action="store_true", default=True)
    p.add_argument("--no_trust_remote_code", dest="trust_remote_code", action="store_false")
    p.add_argument("--overwrite", action="store_true", help="Redecode NPZs that already carry strings.")
    p.add_argument("--models", nargs="*", default=None, help="Restrict to these model_name strings.")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    examples = pd.read_csv(args.examples_csv)
    if args.models:
        examples = examples[examples["model_name"].isin(args.models)]

    written = skipped = failed = 0
    for _, row in examples.iterrows():
        model_name = str(row["model_name"])
        slug = model_name.replace("/", "_")
        path = (
            args.artifacts_dir
            / slug
            / f"example{int(row['example_id']):03d}_{row['candidate_role']}.npz"
        )
        if not path.exists():
            skipped += 1
            continue

        hf_id = SLUG_TO_HF.get(slug)
        if hf_id is None:
            print(f"  [WARN] no HF id for slug {slug}; skipping", file=sys.stderr)
            failed += 1
            continue

        data = dict(np.load(path, allow_pickle=False))
        if not args.overwrite and QUERY_KEY in data and CANDIDATE_KEY in data:
            skipped += 1
            continue

        try:
            tokenizer = get_tokenizer(hf_id, args.trust_remote_code)
            q_len = int(data["query_attention_mask"][0].sum())
            c_len = int(data["candidate_attention_mask"][0].sum())
            q_tokens = decode_ids(tokenizer, data["query_input_ids"])[:q_len]
            c_tokens = decode_ids(tokenizer, data["candidate_input_ids"])[:c_len]
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERROR] {path.name}: {exc}", file=sys.stderr)
            failed += 1
            continue

        if args.dry_run:
            print(f"  [DRY] {slug}/{path.name}: {q_tokens[:8]} ...")
            written += 1
            continue

        data[QUERY_KEY] = np.asarray(q_tokens, dtype=np.str_)
        data[CANDIDATE_KEY] = np.asarray(c_tokens, dtype=np.str_)
        atomic_write_npz(path, data)
        written += 1

    print("=== Summary ===")
    print(f"  npz written:  {written}")
    print(f"  skipped:      {skipped}")
    print(f"  failed:       {failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
