#!/usr/bin/env python3
"""P2x2 (issue #248): sanity checks around the extraction of the 2x2 panel models.

The panel adds encoders the paper has never extracted: Sentence-T5 and its raw partner
T5-base, T5-v1.1-base (already in the paper through gen_extract.py, re-extracted here
with the CLI), and the encoder-only LaBERTa, PhilBERTa and SPhilBERTa. Their vectors come
from the repo's own CLIs (``src/extract_hidden_cli.py``, ``src/extract_encoder_cli.py``)
with the arguments of the paper's caches, so the extraction code is not new. What is new
is the models, and nothing on record says what their tokenizers do to this corpus or
whether the CLIs load them correctly. The three stages answer that:

  tokenization  per model over the 1,705 corpus files: token counts, the share of
                passages cut at 512, the share above the model's own sentence-transformers
                max_seq_length (Sentence-T5 256, SPhilBERTa 128: these models were trained
                on shorter inputs than we feed them), the unknown-token rate and what
                becomes <unk>, whether the tokenizer lowercases, how many special tokens
                it adds, the share of tokens the pooling filter keeps, how many passages
                keep none (they pool to a zero vector), and the tokens of one fixed
                passage for a by-eye check. No token ids of these models are tracked anywhere, so this
                report is the sanity gate.
                Writes tokenization_report.csv and tokenization_report.md.
  st5_encoder   the hidden CLI loads every T5 through AutoModelForSeq2SeqLM and takes
                get_encoder(), but the Sentence-T5 checkpoint is a T5EncoderModel with no
                decoder weights. Compares the encoder loaded both ways: weights key by
                key, then every hidden state on 16 corpus passages. Expected difference 0.
                Writes st5_encoder_check.csv.
  verify        after extraction: every model has layers 1-12 of shape (1705, 768),
                float32, finite, a meta.csv covering the split's filenames, and no
                all-zero row outside the whitespace-only corpus files and the rows that
                are zero in the existing caches. The
                reproduction pair (LaTa, LaBSE, re-extracted by the same job) is compared
                layer by layer with the existing caches after aligning both by filename
                (``AlignmentResolver``), never by position: max |diff| and min row cosine.
                Writes extraction_check.csv.

Each stage exits non-zero when it finds a problem. The tokenization and st5_encoder
stages load tokenizers or models, so run them through slurm/reframe/p2x2_extract.sbatch,
not on a login node. Run from the repo root (corpus paths in the split CSV are relative
to it); the caches are gitignored, so point the roots at the checkout that holds them:

  python scripts/paper/reframe/p2x2_checks.py --stage verify \
      --bases_root /u/irowerojas/localLatin/runs/active/reframe/p2x2/bases \
      --repro_root /u/irowerojas/localLatin/runs/active/reframe/p2x2/repro_bases \
      --ref_root /u/irowerojas/localLatin/runs/active/resubmit_bases/phase9_bases
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

SPLIT_CSV = Path("runs/active/resubmit/data/phase_resubmit_split.csv")
OUT_DIR = Path("runs/active/reframe/p2x2")
BASES_ROOT = OUT_DIR / "bases"
REPRO_ROOT = OUT_DIR / "repro_bases"
REF_ROOT = Path("runs/active/resubmit_bases/phase9_bases")
SUBDIR = "hidden_mean_tokempty"

# (display, HF id, native sentence-transformers max_seq_length or None). The native
# limits are the checkpoints' sentence_bert_config.json values; the stage re-reads them.
P2X2_MODELS = [
    ("Sentence-T5", "sentence-transformers/sentence-t5-base", 256),
    ("T5-base", "google-t5/t5-base", None),
    ("T5-v1.1-base", "google/t5-v1_1-base", None),
    ("LaBERTa", "bowphs/LaBerta", None),
    ("PhilBERTa", "bowphs/PhilBerta", None),
    ("SPhilBERTa", "bowphs/SPhilBerta", 128),
]
# Re-extracted by the same job and compared with the paper's existing caches.
REPRO_MODELS = [
    ("LaTa", "bowphs/LaTa"),
    ("LaBSE", "sentence-transformers/LaBSE"),
]
ST5_ID = "sentence-transformers/sentence-t5-base"

N_LAYERS = 12
DIM = 768
MAX_LENGTH = 512
LAYER_RE = re.compile(r"^hidden_layer(\d+)_embeddings\.npy$")

# Tokenization: by-eye passage (capitals, e caudata) and the probes for case handling.
PROBE_FILE = "CamCC279.89.1.txt"
PROBE_TOKENS = 60
CASE_PROBE = "AMBROSIUS DICIT Dominus Deus Episcopus"
UNK_WARN = 0.01      # unknown-token rate worth a line in the report
UNK_FAIL = 0.05      # rate at which the extraction should not be trusted
FRAGMENT_WARN = 4.0  # tokens per whitespace word
POOLED_WARN = 0.75   # share of tokens surviving the tokenizer_empty pooling filter

# verify: |new - cached| <= ATOL + RTOL * max|cached| per layer (GPU nondeterminism).
ATOL = 1e-5
RTOL = 1e-5
MIN_COS = 1.0 - 1e-6


def slug(model_id: str) -> str:
    return model_id.replace("/", "_")


# ---------------------------------------------------------------- tokenization


def native_max_seq_length(model_id: str) -> Optional[int]:
    """max_seq_length of the checkpoint's sentence-transformers config, None without one."""
    from transformers.utils import cached_file

    try:
        path = cached_file(model_id, "sentence_bert_config.json",
                           _raise_exceptions_for_missing_entries=False,
                           _raise_exceptions_for_connection_errors=False)
    except Exception:
        path = None
    if path is None:
        return None
    return int(json.loads(Path(path).read_text())["max_seq_length"])


def tokenizer_stats(name: str, model_id: str, native: Optional[int], texts: List[str],
                    probe_text: str) -> Dict:
    """One report row. Counts are over the full, untruncated tokenization with whatever
    special tokens the CLI's tokenizer call adds (T5: </s>; LaBERTa: <s> ... </s>;
    PhilBERTa and SPhilBERTa: none, their tokenizer.json has no RoBERTa post-processor)."""
    from transformers import AutoTokenizer

    from token_filtering import build_token_keep_lookup

    tok = AutoTokenizer.from_pretrained(model_id)
    enc = tok(texts, truncation=False, add_special_tokens=True,
              return_offsets_mapping=tok.is_fast)
    ids = enc["input_ids"]
    lens = np.array([len(x) for x in ids])
    flat = np.concatenate([np.asarray(x, dtype=np.int64) for x in ids])
    n_tokens = int(flat.size)

    unk_id = tok.unk_token_id
    n_unk = int((flat == unk_id).sum()) if unk_id is not None else 0
    # What turns into <unk>: the source characters behind each unknown token.
    unk_sources: Counter = Counter()
    if tok.is_fast and n_unk:
        for text, row, offsets in zip(texts, ids, enc["offset_mapping"]):
            for token_id, (start, end) in zip(row, offsets):
                if token_id == unk_id:
                    unk_sources[text[start:end]] += 1

    # The CLI pools over attended tokens minus the "empty" class of --token_filter
    # tokenizer_empty; the share that survives shows the filter suits this vocabulary.
    keep = build_token_keep_lookup(tok, "tokenizer_empty")
    pooled_share = float(keep[flat].mean())
    # Passages with nothing left to pool come out of the CLI as a zero vector.
    n_unpooled = sum(1 for row in ids if keep[np.asarray(row, dtype=np.int64)].sum() == 0)

    lowercases = tok(CASE_PROBE)["input_ids"] == tok(CASE_PROBE.lower())["input_ids"]
    case_unk = int(sum(t == unk_id for t in tok(CASE_PROBE)["input_ids"]))
    n_words = sum(len(t.split()) for t in texts)
    n_special = len(tok("")["input_ids"])  # special tokens the CLI's tokenizer call adds
    read_native = native_max_seq_length(model_id)
    normalizer = getattr(getattr(tok, "backend_tokenizer", None), "normalizer", None)

    flags: List[str] = []
    unk_rate = n_unk / n_tokens
    if unk_rate > UNK_FAIL:
        flags.append(f"FAIL unknown-token rate {unk_rate:.3f} above {UNK_FAIL}")
    elif unk_rate > UNK_WARN:
        flags.append(f"unknown-token rate {unk_rate:.3f} above {UNK_WARN}")
    if case_unk:
        flags.append(f"{case_unk} <unk> in the capitalized probe")
    if n_special == 0:
        flags.append("adds no special tokens")
    if n_unpooled:
        flags.append(f"{n_unpooled} passages keep no token after the pooling filter and "
                     "pool to a zero vector")
    if pooled_share < POOLED_WARN:
        flags.append(f"only {pooled_share:.0%} of tokens survive the pooling filter")
    if n_tokens / n_words > FRAGMENT_WARN:
        flags.append(f"{n_tokens / n_words:.2f} tokens per word")
    if read_native != native:
        flags.append(f"native max_seq_length read {read_native}, expected {native}")

    probe_ids = tok(probe_text, truncation=True, max_length=PROBE_TOKENS)["input_ids"]
    return {
        "model": name,
        "model_id": model_id,
        "tokenizer_class": type(tok).__name__,
        "n_files": len(texts),
        "tokens_mean": float(lens.mean()),
        "tokens_median": float(np.median(lens)),
        "tokens_max": int(lens.max()),
        "share_truncated_512": float((lens > MAX_LENGTH).mean()),
        "native_max_seq_length": native if native is not None else "",
        "share_above_native": float((lens > native).mean()) if native else "",
        "unk_rate": unk_rate,
        "n_unk": n_unk,
        "lowercases": bool(lowercases),
        "special_tokens_added": n_special,
        "n_passages_nothing_pooled": n_unpooled,
        "unk_in_case_probe": case_unk,
        "tokens_per_word": n_tokens / n_words,
        "pooled_share": pooled_share,
        "top_unk_sources": " ".join(f"{s!r}:{c}" for s, c in unk_sources.most_common(8)),
        "flags": "; ".join(flags),
        # Report-only fields, dropped from the CSV.
        "_normalizer": repr(normalizer)[:120],  # a Precompiled charsmap runs to 100 kB
        "_probe_tokens": tok.convert_ids_to_tokens(probe_ids),
        "_case_tokens": tok.convert_ids_to_tokens(tok(CASE_PROBE)["input_ids"]),
    }


def tokenization_markdown(rows: List[Dict], probe_text: str) -> str:
    out = [
        "# P2x2 tokenization report",
        "",
        "Generated by `scripts/paper/reframe/p2x2_checks.py --stage tokenization` "
        "(issue #248). Counts are over the full tokenization of the "
        f"{rows[0]['n_files']:,} labelled passages, special tokens included; the extraction "
        f"truncates at {MAX_LENGTH}.",
        "",
        "| Model | Tokens mean | median | max | Cut at 512 | Native max | Above native "
        "| Unk rate | Lowercases | Specials | Tokens/word | Pooled share |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        native = r["native_max_seq_length"]
        above = f"{r['share_above_native']:.1%}" if native else "n/a"
        out.append(
            f"| {r['model']} | {r['tokens_mean']:.1f} | {r['tokens_median']:.0f} "
            f"| {r['tokens_max']} | {r['share_truncated_512']:.1%} | {native or 'n/a'} "
            f"| {above} | {r['unk_rate']:.4%} ({r['n_unk']}) "
            f"| {'yes' if r['lowercases'] else 'no'} | {r['special_tokens_added']} "
            f"| {r['tokens_per_word']:.2f} "
            f"| {r['pooled_share']:.1%} |"
        )
    out += [
        "",
        "Columns: *Above native* is the share of passages longer than the "
        "`max_seq_length` the sentence-transformers checkpoint was trained and is served "
        "with; *Unk rate* is the share of all tokens that are the tokenizer's unknown id; "
        "*Specials* is the number of special tokens the tokenizer adds to a passage; "
        "*Pooled share* is the share of tokens the CLI's `tokenizer_empty` filter keeps in "
        "the mean (it drops whitespace-only tokens).",
        "",
        "## Flags",
        "",
    ]
    flagged = [r for r in rows if r["flags"]]
    out += [f"- **{r['model']}**: {r['flags']}" for r in flagged] or ["None."]
    out += ["", "## What becomes `<unk>`", ""]
    out += [f"- **{r['model']}**: {r['top_unk_sources'] or 'nothing'}" for r in rows]
    out += [
        "",
        "## By-eye check",
        "",
        "`▁` (SentencePiece) and `Ġ` (byte-level BPE) mark a preceding space; `Ċ` is a "
        "newline. Byte-level BPE prints non-ASCII characters as their UTF-8 bytes read "
        "as Latin-1 (`Ä` followed by another character for `ę`), which is the expected "
        "rendering, not corruption.",
        "",
        f"Passage `{PROBE_FILE}`, first {PROBE_TOKENS} tokens. Source text:",
        "",
        "> " + " ".join(probe_text.split())[:400],
        "",
    ]
    for r in rows:
        out += [
            f"### {r['model']} (`{r['model_id']}`, {r['tokenizer_class']})",
            "",
            f"Normalizer (first 120 characters): `{r['_normalizer']}`",
            "",
            "```",
            " ".join(r["_probe_tokens"]),
            "```",
            "",
            f"Case probe `{CASE_PROBE}`:",
            "",
            "```",
            " ".join(r["_case_tokens"]),
            "```",
            "",
        ]
    return "\n".join(out)


def stage_tokenization(args: argparse.Namespace) -> int:
    from transformers.utils import logging as hf_logging

    from canon_retrieval import load_texts

    hf_logging.set_verbosity_error()  # silences the "longer than 512" notice of truncation=False
    split = pd.read_csv(args.split_csv)
    texts = load_texts(split["path"].tolist())  # the CLIs' own reader
    probe_rows = split.index[split["filename"] == PROBE_FILE]
    if len(probe_rows) != 1:
        raise SystemExit(f"probe file {PROBE_FILE} matches {len(probe_rows)} split rows")
    probe_text = texts[int(probe_rows[0])]

    rows = []
    for name, model_id, native in P2X2_MODELS:
        row = tokenizer_stats(name, model_id, native, texts, probe_text)
        rows.append(row)
        print(f"{name:13s} tokens mean {row['tokens_mean']:.1f} median {row['tokens_median']:.0f} "
              f"cut@512 {row['share_truncated_512']:.3f} unk {row['unk_rate']:.5f} "
              f"lowercases {row['lowercases']} flags [{row['flags']}]", flush=True)
        print("  probe:", " ".join(row["_probe_tokens"]), flush=True)
        print("  case :", " ".join(row["_case_tokens"]), flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    public = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    pd.DataFrame(public).to_csv(args.out_dir / "tokenization_report.csv", index=False,
                                float_format="%.6g")
    (args.out_dir / "tokenization_report.md").write_text(
        tokenization_markdown(rows, probe_text) + "\n", encoding="utf-8")
    print(f"wrote {args.out_dir / 'tokenization_report.csv'} and tokenization_report.md")
    failed = [r["model"] for r in rows if "FAIL" in r["flags"]]
    if failed:
        print("TOKENIZATION FAIL:", ", ".join(failed))
    return 1 if failed else 0


# ----------------------------------------------------------------- st5_encoder


def stage_st5_encoder(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5EncoderModel

    from canon_retrieval import load_texts

    device = "cuda" if torch.cuda.is_available() else "cpu"
    problems: List[str] = []

    # The CLI's load, with the loader's own account of what the checkpoint lacked. A
    # weight missing from the checkpoint would be initialized the same way in both loads
    # (layer norms to 1), so equal weights alone do not prove it was read from disk.
    def missing_in_encoder(info: Dict) -> List[str]:
        tied = "encoder.embed_tokens.weight"  # shares storage with shared.weight
        return [k for k in info["missing_keys"]
                if k.startswith(("encoder.", "shared.")) and k != tied]

    seq2seq, info = AutoModelForSeq2SeqLM.from_pretrained(ST5_ID, output_loading_info=True)
    ref_model, ref_info = T5EncoderModel.from_pretrained(ST5_ID, output_loading_info=True)
    missing_encoder = missing_in_encoder(info) + missing_in_encoder(ref_info)
    print(f"seq2seq load: {type(seq2seq).__name__}, dtype {seq2seq.dtype}, "
          f"{len(info['missing_keys'])} missing keys ({len(missing_in_encoder(info))} in the "
          f"encoder), {len(info['unexpected_keys'])} unexpected, "
          f"{len(info.get('mismatched_keys', []))} mismatched")
    print(f"encoder-only load: {len(ref_info['missing_keys'])} missing keys, "
          f"{len(ref_info['unexpected_keys'])} unexpected")
    if missing_encoder:
        problems.append(f"encoder keys missing from the checkpoint: {missing_encoder[:5]}")
    if info["unexpected_keys"] or info.get("mismatched_keys"):
        problems.append(f"unexpected {info['unexpected_keys'][:5]} "
                        f"mismatched {info.get('mismatched_keys', [])[:5]}")
    if seq2seq.dtype != torch.float32:
        # The CLI has no dtype flag: it would extract at whatever this load gives.
        problems.append(f"seq2seq load is {seq2seq.dtype}, the other caches are float32")

    enc_cli = seq2seq.get_encoder().to(device).eval()
    enc_ref = ref_model.encoder.to(device).eval()

    sd_cli, sd_ref = enc_cli.state_dict(), enc_ref.state_dict()
    if sd_cli.keys() != sd_ref.keys():
        problems.append(f"state dict keys differ: {sorted(set(sd_cli) ^ set(sd_ref))[:5]}")
    weight_diff = max(float((sd_cli[k].float() - sd_ref[k].float()).abs().max())
                      for k in sd_cli.keys() & sd_ref.keys())
    print(f"encoder weights: {len(sd_cli)} tensors, max |diff| {weight_diff:.3g}")
    if weight_diff != 0.0:
        problems.append(f"encoder weights differ, max |diff| {weight_diff:.3g}")

    split = pd.read_csv(args.split_csv)
    rows_idx = np.linspace(0, len(split) - 1, args.st5_passages).round().astype(int)
    texts = load_texts(split["path"].iloc[rows_idx].tolist())
    tok = AutoTokenizer.from_pretrained(ST5_ID)
    diffs = np.zeros(N_LAYERS + 1)
    scale = np.zeros(N_LAYERS + 1)
    with torch.no_grad():
        for start in range(0, len(texts), 8):  # the CLI's tokenizer call and batch size
            enc = tok(texts[start:start + 8], truncation=True, max_length=MAX_LENGTH,
                      padding=True, return_tensors="pt").to(device)
            kwargs = dict(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                          output_hidden_states=True, return_dict=True)
            hs_cli = enc_cli(**kwargs).hidden_states
            hs_ref = enc_ref(**kwargs).hidden_states
            for layer in range(N_LAYERS + 1):
                diffs[layer] = max(diffs[layer], float((hs_cli[layer] - hs_ref[layer]).abs().max()))
                scale[layer] = max(scale[layer], float(hs_ref[layer].abs().max()))

    out = pd.DataFrame({"layer": range(N_LAYERS + 1), "max_abs_diff": diffs,
                        "max_abs_ref": scale, "n_passages": len(texts),
                        "weights_max_abs_diff": weight_diff, "dtype": str(seq2seq.dtype),
                        "device": device})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_dir / "st5_encoder_check.csv", index=False, float_format="%.6g")
    print(out[["layer", "max_abs_diff", "max_abs_ref"]].to_string(index=False))
    print(f"st5_encoder: hidden states 0-{N_LAYERS} on {len(texts)} passages, "
          f"max |diff| {diffs.max():.3g} (device {device})")
    if (diffs > ATOL + RTOL * scale).any():
        problems.append(f"hidden states differ, max |diff| {diffs.max():.3g}")

    for p in problems:
        print("ST5 ENCODER PROBLEM:", p)
    if not problems:
        print("st5_encoder: the Seq2SeqLM load gives the T5EncoderModel encoder")
    return 1 if problems else 0


# ---------------------------------------------------------------------- verify


def zero_rows(emb: np.ndarray) -> np.ndarray:
    return np.flatnonzero(~np.any(emb != 0, axis=1))


def row_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine; two all-zero rows count as identical."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    cos = (a * b).sum(axis=1) / np.where(na * nb > 0, na * nb, 1.0)
    return np.where((na == 0) & (nb == 0), 1.0, cos)


def check_run_dir(role: str, name: str, model_id: str, run_dir: Path, split: pd.DataFrame,
                  resolver, allowed_zero: Optional[Set[str]], ref_dir: Optional[Path],
                  problems: List[str]) -> List[Dict]:
    """Rows of extraction_check.csv for one cache directory, in split order.

    ``allowed_zero`` is the set of filenames that may have an all-zero row: those of the
    existing caches plus the blank passages (None: do not check, used for the reference
    caches themselves). ``ref_dir`` adds the
    layer-by-layer comparison with an existing cache.
    """
    from embedding_alignment import STATUS_UNVERIFIED

    tag = f"{name} ({role})"
    if not run_dir.is_dir():
        problems.append(f"{tag}: {run_dir} does not exist")
        return []
    found = sorted(int(m.group(1)) for p in run_dir.iterdir() if (m := LAYER_RE.match(p.name)))
    if found != list(range(1, N_LAYERS + 1)):
        problems.append(f"{tag}: layers {found}, expected 1-{N_LAYERS}")
    filenames = split["filename"].to_numpy()
    rows: List[Dict] = []
    for layer in found:
        path = run_dir / f"hidden_layer{layer}_embeddings.npy"
        emb = resolver.load(path)  # raises AlignmentError unless meta.csv covers the split
        status = resolver.aligner_for(path).status
        zeros = zero_rows(emb)
        row = {"role": role, "model": name, "model_id": model_id, "layer": layer,
               "n_rows": emb.shape[0], "dim": emb.shape[1] if emb.ndim == 2 else -1,
               "dtype": str(emb.dtype), "n_nonfinite": int((~np.isfinite(emb)).sum()),
               "n_zero_rows": len(zeros),
               "mean_row_norm": float(np.linalg.norm(emb.astype(np.float64), axis=1).mean()),
               "alignment": status}
        if emb.shape != (len(split), DIM):
            problems.append(f"{tag} L{layer}: shape {emb.shape}, expected {(len(split), DIM)}")
        if emb.dtype != np.float32:
            problems.append(f"{tag} L{layer}: dtype {emb.dtype}, expected float32")
        if row["n_nonfinite"]:
            problems.append(f"{tag} L{layer}: {row['n_nonfinite']} non-finite values")
        if status == STATUS_UNVERIFIED:
            problems.append(f"{tag} L{layer}: no meta.csv, row order unverified")
        if allowed_zero is not None:
            extra = sorted(set(filenames[zeros]) - allowed_zero)
            if extra:
                problems.append(f"{tag} L{layer}: all-zero rows for {extra[:5]} "
                                f"({len(extra)} files) that are neither whitespace-only nor "
                                "zero in the existing caches")
        if ref_dir is not None:
            ref = resolver.load(ref_dir / path.name)
            diff = float(np.abs(emb.astype(np.float64) - ref.astype(np.float64)).max())
            ref_scale = float(np.abs(ref).max())
            min_cos = float(row_cosines(emb, ref).min())
            row.update(repro_max_abs_diff=diff, repro_max_abs_ref=ref_scale,
                       repro_min_row_cos=min_cos)
            if diff > ATOL + RTOL * ref_scale or min_cos < MIN_COS:
                problems.append(f"{tag} L{layer}: differs from the existing cache, "
                                f"max |diff| {diff:.3g} (max |ref| {ref_scale:.3g}), "
                                f"min row cosine {min_cos:.8f}")
        rows.append(row)
    return rows


def stage_verify(args: argparse.Namespace) -> int:
    from embedding_alignment import AlignmentResolver

    split = pd.read_csv(args.split_csv)
    resolver = AlignmentResolver(split)
    problems: List[str] = []
    rows: List[Dict] = []

    # Files that may legitimately pool to a zero vector: the whitespace-only corpus files
    # (a tokenizer that adds no special tokens, as PhilBERTa's and SPhilBERTa's do, leaves
    # them nothing but tokens the tokenizer_empty filter drops), plus whatever is already
    # zero in the existing caches.
    from canon_retrieval import load_texts

    texts = load_texts(split["path"].tolist())
    blank = {f for f, t in zip(split["filename"], texts) if not t.strip()}
    print(f"whitespace-only corpus files: {sorted(blank) or 'none'}")
    ref_rows: List[Dict] = []
    allowed_zero: Set[str] = set(blank)
    for name, model_id in REPRO_MODELS:
        ref_dir = args.ref_root / slug(model_id) / SUBDIR
        ref_rows += check_run_dir("existing", name, model_id, ref_dir, split, resolver,
                                  None, None, problems)
        for path in sorted(ref_dir.glob("hidden_layer*_embeddings.npy")):
            if LAYER_RE.match(path.name):
                allowed_zero |= set(split["filename"].to_numpy()[zero_rows(resolver.load(path))])
    print(f"all-zero rows in the existing caches: {sorted(allowed_zero - blank) or 'none'}")

    for name, model_id, _ in P2X2_MODELS:
        rows += check_run_dir("p2x2", name, model_id, args.bases_root / slug(model_id) / SUBDIR,
                              split, resolver, allowed_zero, None, problems)
    for name, model_id in REPRO_MODELS:
        rows += check_run_dir("repro", name, model_id, args.repro_root / slug(model_id) / SUBDIR,
                              split, resolver, allowed_zero,
                              args.ref_root / slug(model_id) / SUBDIR, problems)
    print(resolver.summary())

    df = pd.DataFrame(rows + ref_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "extraction_check.csv", index=False, float_format="%.8g")
    print(f"wrote {args.out_dir / 'extraction_check.csv'} ({len(df)} rows)")
    for (role, name), g in df.groupby(["role", "model"], sort=False):
        line = (f"{role:8s} {name:13s} layers {len(g):2d} rows {int(g['n_rows'].iloc[0])} "
                f"dim {int(g['dim'].iloc[0])} zero rows {int(g['n_zero_rows'].max())}")
        if role == "repro":
            line += (f"  vs existing cache: max |diff| {g['repro_max_abs_diff'].max():.3g}, "
                     f"min row cosine {g['repro_min_row_cos'].min():.8f}")
        print(line)
    for p in problems:
        print("EXTRACTION PROBLEM:", p)
    if not problems:
        print("verify: all caches complete; the reproduction pair matches the existing caches")
    return 1 if problems else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=["tokenization", "st5_encoder", "verify"], required=True)
    ap.add_argument("--split_csv", type=Path, default=SPLIT_CSV)
    ap.add_argument("--bases_root", type=Path, default=BASES_ROOT)
    ap.add_argument("--repro_root", type=Path, default=REPRO_ROOT)
    ap.add_argument("--ref_root", type=Path, default=REF_ROOT)
    ap.add_argument("--out_dir", type=Path, default=OUT_DIR)
    ap.add_argument("--st5_passages", type=int, default=16)
    args = ap.parse_args(argv)
    stage = {"tokenization": stage_tokenization, "st5_encoder": stage_st5_encoder,
             "verify": stage_verify}[args.stage]
    return stage(args)


if __name__ == "__main__":
    sys.exit(main())
