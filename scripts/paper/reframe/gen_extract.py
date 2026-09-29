#!/usr/bin/env python3
"""GEN (issue #234): mean-pooled encoder hidden states at every layer, on CPU.

Same representation as the paper's cached T5 vectors (src/extract_hidden_cli.py with
--pooling mean --token_filter tokenizer_empty --max_length 512): the encoder of
AutoModelForSeq2SeqLM, hidden_states[1..N] (layer 1 = first block, layer N = after the
final layer norm), mean over attended tokens minus the tokenizer's empty tokens, EOS
kept, float32. The pooling function and token filter are imported from src/ so the two
cannot drift. Differences from the CLI are operational only: one forward pass per batch
for all layers (the CLI reruns the encoder per layer), batches formed after sorting by
length (less padding on CPU), and rows written back in split-CSV order.

Corpora (rows always in the order of --split_csv, one row per labelled Latin passage):
  latin    data/canon_labelled/... files named by the split CSV
  english  the length-matched sample from gen_english_sample.py (row i partners Latin row i)

Writes <out_root>/<model_slug>/<corpus>/hidden_layer{N}_embeddings.npy, meta.csv
(filename, split, in split order) and config.json (model revision, timing).

Run from the repo root:
  python scripts/paper/reframe/gen_extract.py --model_name google/t5-v1_1-base --corpus english
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_english_sample import MODEL_REVISIONS  # noqa: E402


def load_corpus(corpus: str, split: pd.DataFrame, data_root: Path, english_csv: Path) -> list[str]:
    if corpus == "latin":
        return [Path(data_root, p).read_text(encoding="utf-8") for p in split["path"]]
    eng = pd.read_csv(english_csv, keep_default_na=False)
    if list(eng["latin_filename"]) != list(split["filename"]):
        raise SystemExit("english sample rows are not in split-CSV order")
    return [str(t) for t in eng["text"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--corpus", choices=["latin", "english"], required=True)
    ap.add_argument("--split_csv", default="runs/active/resubmit/data/phase_resubmit_split.csv")
    ap.add_argument("--data_root", default=".")
    ap.add_argument("--english_csv", default="runs/active/reframe/gen/english_sample.csv")
    ap.add_argument("--out_root", default="runs/active/reframe/gen/bases")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--token_filter", default="tokenizer_empty")
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0: leave default)")
    ap.add_argument("--limit", type=int, default=0, help="pilot: first N rows only")
    ap.add_argument("--revision", default="", help="HF revision (default: the pinned GEN revision)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from extract_hidden_cli import pool_hidden
    from token_filtering import build_token_keep_lookup

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    split = pd.read_csv(args.split_csv)
    texts = load_corpus(args.corpus, split, Path(args.data_root), Path(args.english_csv))
    if args.limit:
        split, texts = split.iloc[: args.limit], texts[: args.limit]

    # Pinned revision (gen_english_sample.MODEL_REVISIONS); record the snapshot actually loaded.
    from transformers.utils import cached_file
    pinned = args.revision or MODEL_REVISIONS.get(args.model_name)
    revision = Path(cached_file(args.model_name, "config.json", revision=pinned)).parent.name
    if pinned and revision != pinned:
        raise SystemExit(f"loaded snapshot {revision} != pinned revision {pinned}")

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_name, revision=pinned)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, revision=pinned, dtype=torch.float32)
    encoder = model.get_encoder()
    encoder.eval()
    keep = build_token_keep_lookup(tok, args.token_filter)
    n_layers = len(encoder.block)
    t_load = time.time() - t0

    raw_len = np.array([len(tok(t)["input_ids"]) for t in texts])
    order = np.argsort(-np.minimum(raw_len, args.max_length), kind="stable")
    pooled = np.zeros((n_layers, len(texts), encoder.config.d_model), dtype=np.float32)
    t1 = time.time()
    with torch.no_grad():
        for b in range(0, len(order), args.batch_size):
            idx = order[b:b + args.batch_size]
            enc = tok([texts[i] for i in idx], truncation=True, max_length=args.max_length,
                      padding=True, return_tensors="pt")
            out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                          output_hidden_states=True, return_dict=True)
            for layer in range(1, n_layers + 1):
                v = pool_hidden(out.hidden_states[layer], enc["attention_mask"], "mean",
                                input_ids=enc["input_ids"], token_keep_lookup=keep)
                pooled[layer - 1, idx] = v.numpy().astype(np.float32)
            if (b // args.batch_size) % 25 == 0:
                print(f"  {b + len(idx)}/{len(texts)} passages, {time.time() - t1:.0f}s", flush=True)
    t_fwd = time.time() - t1

    slug = args.model_name.replace("/", "_")
    out_dir = Path(args.out_root) / slug / args.corpus
    out_dir.mkdir(parents=True, exist_ok=True)
    for layer in range(1, n_layers + 1):
        np.save(out_dir / f"hidden_layer{layer}_embeddings.npy", pooled[layer - 1])
    split[["filename", "split"]].to_csv(out_dir / "meta.csv", index=False)
    cfg = dict(model_name=args.model_name, revision=revision, corpus=args.corpus,
               n_rows=len(texts), n_layers=n_layers, d_model=int(encoder.config.d_model),
               pooling="mean", token_filter=args.token_filter, max_length=args.max_length,
               batch_size=args.batch_size, order="split_csv", torch_threads=torch.get_num_threads(),
               torch=torch.__version__, load_seconds=round(t_load, 1),
               forward_seconds=round(t_fwd, 1), n_truncated=int((raw_len > args.max_length).sum()),
               layer_indexing="hidden_states[1..N]; 1 = first block, N = after final layer norm")
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
