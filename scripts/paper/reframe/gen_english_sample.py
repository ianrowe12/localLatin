#!/usr/bin/env python3
"""GEN (issue #234): a length-matched English sample for the label-free geometry check.

One English passage is drawn for each of the 1,705 labelled Latin passages, with the
same token length under the mT5-base tokenizer (EOS included, as the encoder sees it).
The English passage inherits its Latin partner's train/test split, so the "train half"
of the English sample has exactly the length profile of the 847 Latin training passages
on which the paper reads top-PC share and effective rank.

Source: US court opinions from the Caselaw Access Project, as redistributed (public
domain only, raw version) by Common Pile v0.1:
  https://huggingface.co/datasets/common-pile/caselaw_access_project
at a pinned revision, seven of its 173 shards spread over the collection (federal and
state reporters from the early 19th century to the 2010s). Legal prose is the nearest
English register to canon law that is openly licensed and on the HF hub. The dataset card
at the pinned revision states that only public-domain documents are included, and each
sampled record's own source and licence fields are written to the output (checked in
docs/research/reframe_gen_ft.md). Nothing is redistributed by this repo.

Selection (deterministic, --seed):
  1. From each shard, a fixed random set of documents is read (--docs_per_shard).
  2. A document's prose is the concatenation of its body paragraphs: lines that start at
     column 0 (CAP indents the caption, docket, date and counsel lines), have at least
     --min_para_words words, and whose words contain a Latin letter at least 80% of the
     time (drops tables and citation strings).
  3. Latin targets are visited in a seeded random order. For each, the first unused
     document (in a seeded order) that can hold the target is used once: a sentence start
     is drawn at random, and whole words are added until the mT5 token count is as close
     to the target as possible. A match is accepted within max(2, 3% of the target)
     tokens; targets above --max_target are capped there (the encoder truncates at 512).
  4. The two empty Latin files (1 token, EOS only) get an empty English partner.

Writes <out_csv> with one row per Latin passage (same order as the split CSV):
  eng_id, latin_filename, latin_split, target_len, len_mt5, shard, doc_id, source, licence,
  start_word, n_words, text   (source / licence: the Common Pile record's own fields)
and <out_csv stem>_lengths.csv: token lengths of both texts under every GEN tokenizer.

Run from the repo root (CPU, a few minutes; shards must be in the HF cache, or online):
  python scripts/paper/reframe/gen_english_sample.py \
      --split_csv runs/active/resubmit/data/phase_resubmit_split.csv \
      --out_csv runs/active/reframe/gen/english_sample.csv
"""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import numpy as np
import pandas as pd

DATASET = "common-pile/caselaw_access_project"
REVISION = "3c2cb5080b3a16a04d8d8d07b28eaec7c1ba7a90"
SHARDS = [
    "cap_00000.jsonl.gz",  # F.2d (federal appeals, 1970s)
    "cap_00030.jsonl.gz",  # Ill. App., Ohio St., Okla. Crim.
    "cap_00060.jsonl.gz",  # Me., Conn., Mich. App., Cowen (N.Y. 1820s)
    "cap_00090.jsonl.gz",  # S. Ct., S.C., Miss., Sandford's Chancery
    "cap_00120.jsonl.gz",  # Ohio St. 3d, N.J., N.J. Super.
    "cap_00150.jsonl.gz",  # Ala. App., Mass. App. Ct., Martin (La. 1810s-1830s)
    "cap_00170.jsonl.gz",  # Ga. App., F.R.D., Neb.
]
# HF model revisions used for GEN (tokenizers here, weights in gen_extract.py).
MODEL_REVISIONS = {
    "google/mt5-base": "2eb15465c5dd7f72a8f7984306ad05ebc3dd1e1f",
    "bowphs/PhilTa": "8572ff520a1a7316fa99ee0c9cb80a30e451c55d",
    "google/t5-v1_1-base": "b5fc947a416ea3cb079532cb3c2bbadeb7f800fc",
}
MATCH_TOKENIZER = "google/mt5-base"
LENGTH_TOKENIZERS = {
    "mt5": "google/mt5-base",
    "philta": "bowphs/PhilTa",
    "t5v11": "google/t5-v1_1-base",
}
SENT_END = re.compile(r"[.?!][\"')\]”’]*$")
HAS_LETTER = re.compile(r"[A-Za-z]")


def prose_paragraphs(text: str, min_words: int = 12, min_alpha: float = 0.8) -> List[str]:
    """Body paragraphs of a CAP opinion (see module docstring, step 2)."""
    out = []
    for line in text.split("\n"):
        if not line or line[0].isspace():
            continue
        words = line.split()
        if len(words) < min_words:
            continue
        if sum(1 for w in words if HAS_LETTER.search(w)) / len(words) < min_alpha:
            continue
        out.append(" ".join(words))
    return out


def sentence_starts(words: Sequence[str]) -> List[int]:
    """Indices of words that open a sentence (first word, or after . ? ! + capital)."""
    starts = [0] if words else []
    for i in range(1, len(words)):
        if SENT_END.search(words[i - 1]) and words[i][:1].isupper():
            starts.append(i)
    return starts


def fit_span(words: Sequence[str], start: int, target: int,
             count: Callable[[str], int]) -> Tuple[int, int]:
    """Number of words from `start` whose token count is closest to `target`.

    Binary search for the smallest k with count >= target (count is non-decreasing in k
    up to tokenizer noise), then compare k with k-1. Returns (k, count(k)); k = 0 if even
    the whole remainder is shorter than target - 0.
    """
    hi = len(words) - start
    if hi <= 0:
        return 0, 0
    full = count(" ".join(words[start:]))
    if full < target:
        return 0, full
    lo = 1
    while lo < hi:
        mid = (lo + hi) // 2
        if count(" ".join(words[start:start + mid])) >= target:
            hi = mid
        else:
            lo = mid + 1
    best_k, best_n = lo, count(" ".join(words[start:start + lo]))
    if lo > 1:
        n_prev = count(" ".join(words[start:start + lo - 1]))
        if abs(n_prev - target) < abs(best_n - target):
            best_k, best_n = lo - 1, n_prev
    return best_k, best_n


def tolerance(target: int) -> int:
    return max(2, int(round(0.03 * target)))


def match_sample(targets: Sequence[int], docs: Sequence[Tuple[str, str, List[str]]],
                 count: Callable[[str], int], seed: int) -> List[dict]:
    """Assign one document span to every target (see module docstring, step 3).

    docs: (shard, doc_id, words) in candidate order. Returns one dict per target, in
    target order.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(targets))
    used = np.zeros(len(docs), dtype=bool)
    starts = [sentence_starts(w) for _, _, w in docs]
    out: List[dict] = [dict() for _ in targets]
    for ti in order:
        t = int(targets[ti])
        if t <= 1:
            out[ti] = dict(shard="", doc_id="", start_word=-1, n_words=0, text="", len_mt5=count(""))
            continue
        tol = tolerance(t)
        placed = False
        for di in range(len(docs)):
            if used[di]:
                continue
            shard, doc_id, words = docs[di]
            # Crude capacity screen before tokenizing: English runs ~1.2-1.6 mT5 tokens/word.
            if 1.8 * len(words) < t:
                continue
            cand = [s for s in starts[di] if 1.8 * (len(words) - s) >= t]
            if not cand:
                continue
            s = int(cand[rng.integers(len(cand))])
            k, n = fit_span(words, s, t, count)
            if k == 0 or abs(n - t) > tol:
                continue
            used[di] = True
            out[ti] = dict(shard=shard, doc_id=doc_id, start_word=s, n_words=k,
                           text=" ".join(words[s:s + k]), len_mt5=n)
            placed = True
            break
        if not placed:
            raise SystemExit(f"no document fits target {t}; raise --docs_per_shard")
    return out


def doc_licence(d: dict) -> Tuple[str, str]:
    """(source, licence) of a Common Pile record; metadata is a dict or its repr string."""
    md = d.get("metadata", {})
    if isinstance(md, str):
        try:
            md = ast.literal_eval(md)
        except (ValueError, SyntaxError):
            md = {}
    return str(d.get("source", "")), str(md.get("license", "")) if isinstance(md, dict) else ""


def load_docs(shard_paths: Sequence[Tuple[str, Path]], docs_per_shard: int, seed: int,
              min_para_words: int, licences: dict | None = None) -> List[Tuple[str, str, List[str]]]:
    """Candidate documents in a seeded order; fills `licences[doc_id] = (source, licence)`."""
    rng = np.random.default_rng(seed + 1)
    docs = []
    for shard, path in shard_paths:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            lines = fh.readlines()
        pick = sorted(rng.choice(len(lines), size=min(docs_per_shard, len(lines)), replace=False))
        for i in pick:
            d = json.loads(lines[i])
            words = " ".join(prose_paragraphs(d["text"], min_para_words)).split()
            if words:
                docs.append((shard, d["id"], words))
                if licences is not None:
                    licences[d["id"]] = doc_licence(d)
        del lines
    perm = np.random.default_rng(seed + 2).permutation(len(docs))
    return [docs[i] for i in perm]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split_csv", default="runs/active/resubmit/data/phase_resubmit_split.csv")
    ap.add_argument("--data_root", default=".", help="Root that the split CSV's paths are relative to.")
    ap.add_argument("--out_csv", default="runs/active/reframe/gen/english_sample.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--docs_per_shard", type=int, default=1500)
    ap.add_argument("--min_para_words", type=int, default=12)
    ap.add_argument("--max_target", type=int, default=1024)
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    split = pd.read_csv(args.split_csv)
    latin = [Path(args.data_root, p).read_text(encoding="utf-8") for p in split["path"]]
    tok = AutoTokenizer.from_pretrained(MATCH_TOKENIZER, revision=MODEL_REVISIONS[MATCH_TOKENIZER])

    def count(text: str) -> int:
        return len(tok(text)["input_ids"])

    lat_len = np.array([count(t) for t in latin])
    targets = np.minimum(lat_len, args.max_target)

    shard_paths = [(s, Path(hf_hub_download(DATASET, s, repo_type="dataset", revision=REVISION)))
                   for s in SHARDS]
    licences: dict = {}
    docs = load_docs(shard_paths, args.docs_per_shard, args.seed, args.min_para_words, licences)
    print(f"{len(docs)} candidate documents from {len(SHARDS)} shards", flush=True)
    rows = match_sample(targets, docs, count, args.seed)

    out = pd.DataFrame(rows)
    out.insert(0, "eng_id", [f"en{i:04d}" for i in range(len(out))])
    out.insert(1, "latin_filename", split["filename"].to_numpy())
    out.insert(2, "latin_split", split["split"].to_numpy())
    out.insert(3, "target_len", targets)
    out["source"] = [licences.get(d, ("", ""))[0] for d in out.doc_id]
    out["licence"] = [licences.get(d, ("", ""))[1] for d in out.doc_id]
    out = out[["eng_id", "latin_filename", "latin_split", "target_len", "len_mt5", "shard",
               "doc_id", "source", "licence", "start_word", "n_words", "text"]]
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    lens = pd.DataFrame({"eng_id": out.eng_id, "latin_filename": out.latin_filename,
                         "split": out.latin_split})
    for key, name in LENGTH_TOKENIZERS.items():
        t = tok if name == MATCH_TOKENIZER else AutoTokenizer.from_pretrained(
            name, revision=MODEL_REVISIONS[name])
        lens[f"latin_{key}"] = [len(t(x)["input_ids"]) for x in latin]
        lens[f"english_{key}"] = [len(t(x)["input_ids"]) for x in out.text]
    lens_csv = out_csv.with_name(out_csv.stem + "_lengths.csv")
    lens.to_csv(lens_csv, index=False)

    digest = hashlib.sha256("\n".join(out.text).encode("utf-8")).hexdigest()
    meta = dict(dataset=DATASET, revision=REVISION, shards=SHARDS, seed=args.seed,
                docs_per_shard=args.docs_per_shard, min_para_words=args.min_para_words,
                max_target=args.max_target, match_tokenizer=MATCH_TOKENIZER,
                n_passages=int(len(out)), n_candidate_docs=len(docs),
                n_distinct_docs=int(out.doc_id[out.doc_id != ""].nunique()),
                text_sha256=digest, tokenizer_revisions=MODEL_REVISIONS,
                sources=out.source[out.doc_id != ""].value_counts().to_dict(),
                licences=out.licence[out.doc_id != ""].value_counts().to_dict(),
                max_abs_len_diff=int((out.len_mt5 - out.target_len).abs().max()))
    out_csv.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
