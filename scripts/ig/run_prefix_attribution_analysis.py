#!/usr/bin/env python
"""Measure how much attribution mass lands on Latin prefix pieces (issue #211).

Two passes:

``--run deployed``
    the webapp artifacts under ``runs/active/ig_examples``.  For each model the
    pass reads the artifacts at the layer that ``deployed_unlabelled_layers.json``
    currently serves for ``raw`` and for ``sif_abtt``, which is what a reviewer
    actually sees.  Bulk artifacts carry IG only; the 20 gallery pairs per model
    also carry the MaRC masks.

``--run pos200``
    the paper's 200-positive-pair run (``runs/active/ig_examples_200pos_v1``),
    three models, IG and MaRC, ``baseline`` and ``abtt``.

Outputs one summary CSV per run plus a per-pair CSV (large, written to
``--per_pair_dir`` only when asked).

CPU only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prefix_attribution import (  # noqa: E402
    LATIN_PREFIXES,
    build_lexicon,
    build_piece_table,
    pair_side_metrics,
)

REPO = Path(__file__).resolve().parents[2]

SLUG_TO_HF = {
    "bowphs_LaTa": "bowphs/LaTa",
    "bowphs_PhilTa": "bowphs/PhilTa",
    "google_mt5-base": "google/mt5-base",
    "sentence-transformers_LaBSE": "sentence-transformers/LaBSE",
    "Qwen_Qwen3-Embedding-0.6B": "Qwen/Qwen3-Embedding-0.6B",
    "KaLM-Embedding_KaLM-embedding-multilingual-mini-instruct-v2.5":
        "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
}
SHORT = {
    "bowphs_LaTa": "LaTa",
    "bowphs_PhilTa": "PhilTa",
    "google_mt5-base": "mT5-base",
    "sentence-transformers_LaBSE": "LaBSE",
    "Qwen_Qwen3-Embedding-0.6B": "Qwen3-0.6B",
    "KaLM-Embedding_KaLM-embedding-multilingual-mini-instruct-v2.5": "KaLM-mini",
}
# The webapp calls the uncorrected variant "raw"; the artifacts call it
# "baseline".
WEBAPP_TO_ARTIFACT = {"raw": "baseline", "abtt": "abtt", "sif": "sif",
                      "sif_abtt": "sif_abtt"}


def load_tokenizer(hf_id: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(hf_id)


def _views(data, variant: str) -> dict[str, tuple[np.ndarray | None, np.ndarray | None]]:
    """Per-token attribution vectors for each view, or None when absent."""
    def get(key):
        return np.asarray(data[key], dtype=np.float64) if key in data else None
    return {
        "ig": (get(f"query_ig_{variant}"), get(f"candidate_ig_{variant}")),
        "marc": (get(f"q_mask_retrieval_mark_{variant}"),
                 get(f"c_mask_retrieval_mark_{variant}")),
    }


def analyse_artifact(path: Path, tokenizer, lex, specials, variants, top_k=5):
    rows = []
    with np.load(path, allow_pickle=True) as data:
        keys = set(data.files)
        q_ids = np.asarray(data["query_input_ids"]).ravel()
        c_ids = np.asarray(data["candidate_input_ids"]).ravel()
        q_mask = np.asarray(data["query_attention_mask"]).ravel() \
            if "query_attention_mask" in keys else np.ones_like(q_ids)
        c_mask = np.asarray(data["candidate_attention_mask"]).ravel() \
            if "candidate_attention_mask" in keys else np.ones_like(c_ids)
        q_len, c_len = int(q_mask.sum()), int(c_mask.sum())
        q_pieces = tokenizer.convert_ids_to_tokens(q_ids[:q_len].tolist())
        c_pieces = tokenizer.convert_ids_to_tokens(c_ids[:c_len].tolist())
        q_tab = build_piece_table(q_pieces, lex, specials)
        c_tab = build_piece_table(c_pieces, lex, specials)
        for variant in variants:
            art = WEBAPP_TO_ARTIFACT[variant]
            for view, (q_attr, c_attr) in _views(data, art).items():
                for side, attr, tab, n in (("query", q_attr, q_tab, q_len),
                                           ("candidate", c_attr, c_tab, c_len)):
                    if attr is None:
                        continue
                    if not np.isfinite(attr[:n]).all():
                        # The deployed Qwen3-0.6B MaRC masks are riddled with
                        # NaN (87 to 97 percent of positions in all 20 gallery
                        # pairs, both sides, for both variants that carry a
                        # mask): the mask optimisation diverged for that model.
                        # Record the gap rather than propagating NaN into the
                        # means. Anything non-finite disqualifies the side.
                        rows.append({"variant": variant, "view": view,
                                     "side": side, "artifact": path.name,
                                     "nonfinite": 1})
                        continue
                    m = pair_side_metrics(attr[:n], tab, top_k=top_k)
                    if not m:
                        continue
                    m.update(variant=variant, view=view, side=side,
                             artifact=path.name, nonfinite=0)
                    rows.append(m)
    return rows


def deployed_rows(registry: Path, layers_json: Path, slug: str, hf_id: str,
                  variants: list[str], limit: int | None) -> pd.DataFrame:
    reg = pd.read_csv(registry, low_memory=False)
    reg = reg[reg["model_name"] == hf_id].copy()
    reg["variants_available"] = reg["variants_available"].fillna("")
    layers = json.loads(layers_json.read_text())["layers"]
    wanted = []
    for v in variants:
        art = WEBAPP_TO_ARTIFACT[v]
        layer = layers[v][hf_id]
        sel = reg[(reg["layer"] == layer)
                  & reg["variants_available"].str.split(",").apply(lambda xs: art in xs)]
        if limit is not None:
            sel = sel.head(limit)
        sel = sel.assign(_variant=v)
        wanted.append(sel)
    # The gallery pairs (labelled queries) carry MaRC; keep them all.
    gallery = reg[reg["bucket"] != "unlabelled_bulk"].assign(_variant="gallery")
    wanted.append(gallery)
    out = pd.concat(wanted, ignore_index=True)
    return out


def run_deployed(args) -> pd.DataFrame:
    root = Path(args.artifacts_root)
    registry = root / "phase12f_examples.csv"
    layers_json = REPO / "scripts/resubmit/deployed_unlabelled_layers.json"
    all_rows = []
    for slug, hf_id in SLUG_TO_HF.items():
        art_dir = root / "artifacts" / slug
        if not art_dir.is_dir():
            print(f"[skip] no artifacts for {slug}", flush=True)
            continue
        tok = load_tokenizer(hf_id)
        specials = set(tok.all_special_tokens)
        print(f"[{slug}] building piece lexicon over {args.corpus}", flush=True)
        lex = build_lexicon(tok, Path(args.corpus), top_pct=args.top_pct,
                            max_files=args.max_corpus_files)
        print(f"[{slug}] {len(lex.piece_counts)} piece types, "
              f"prefix count share {lex.corpus_share_prefix():.4f}, "
              f"top-1% count share {lex.corpus_share_frequent():.4f}", flush=True)
        sel = deployed_rows(registry, layers_json, slug, hf_id,
                            args.variants, args.limit)
        seen: dict[int, dict] = {}
        for _, r in sel.iterrows():
            eid = int(r["example_id"])
            entry = seen.setdefault(eid, {"layer": int(r["layer"]),
                                          "bucket": str(r["bucket"]),
                                          "avail": set()})
            entry["avail"].update(
                [x for x in str(r["variants_available"]).split(",") if x])
        print(f"[{slug}] {len(seen)} artifacts selected", flush=True)
        for i, (eid, entry) in enumerate(sorted(seen.items())):
            path = art_dir / f"example{eid:03d}_pair_example.npz"
            if not path.exists():
                continue
            want = [v for v in args.variants
                    if WEBAPP_TO_ARTIFACT[v] in entry["avail"]]
            if not want:
                continue
            try:
                rows = analyse_artifact(path, tok, lex, specials, want,
                                        top_k=args.top_k)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] {path.name}: {type(e).__name__}: {e}", flush=True)
                continue
            stratum = ("gallery" if entry["bucket"] != "unlabelled_bulk"
                       else "unlabelled_bulk")
            for row in rows:
                row.update(model_slug=slug, model_short=SHORT[slug],
                           run="deployed", example_id=eid,
                           layer=entry["layer"], stratum=stratum)
            all_rows.extend(rows)
            if (i + 1) % 500 == 0:
                print(f"[{slug}] {i+1}/{len(seen)}", flush=True)
        all_rows.append({"model_slug": slug, "model_short": SHORT[slug],
                         "run": "deployed", "view": "_corpus", "variant": "_corpus",
                         "side": "_corpus",
                         "corpus_share_prefix": lex.corpus_share_prefix(),
                         "corpus_share_frequent": lex.corpus_share_frequent(),
                         "n_piece_types": len(lex.piece_counts)})
    return pd.DataFrame(all_rows)


def run_pos200(args) -> pd.DataFrame:
    root = Path(args.pos200_root)
    all_rows = []
    for slug, hf_id in SLUG_TO_HF.items():
        art_dir = root / "artifacts" / slug
        if not art_dir.is_dir():
            continue
        tok = load_tokenizer(hf_id)
        specials = set(tok.all_special_tokens)
        lex = build_lexicon(tok, Path(args.corpus), top_pct=args.top_pct,
                            max_files=args.max_corpus_files)
        print(f"[{slug}] pos200: prefix count share {lex.corpus_share_prefix():.4f}",
              flush=True)
        paths = sorted(art_dir.glob("*_pair_example.npz"))
        if args.limit:
            paths = paths[: args.limit]
        for path in paths:
            try:
                rows = analyse_artifact(path, tok, lex, specials,
                                        ["raw", "abtt"], top_k=args.top_k)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] {path.name}: {type(e).__name__}: {e}", flush=True)
                continue
            for row in rows:
                row.update(model_slug=slug, model_short=SHORT[slug], run="pos200",
                           layer=-1, stratum="positive_pair")
            all_rows.extend(rows)
        all_rows.append({"model_slug": slug, "model_short": SHORT[slug],
                         "run": "pos200", "view": "_corpus", "variant": "_corpus",
                         "side": "_corpus",
                         "corpus_share_prefix": lex.corpus_share_prefix(),
                         "corpus_share_frequent": lex.corpus_share_frequent(),
                         "n_piece_types": len(lex.piece_counts)})
    return pd.DataFrame(all_rows)


METRIC_COLS = [
    "count_share_prefix", "count_share_prefix_fragment",
    "count_share_prefix_wholeword", "count_share_frequent",
    "count_share_fragment",
    "mass_share_prefix", "mass_share_prefix_fragment",
    "mass_share_prefix_wholeword", "mass_share_frequent", "mass_share_whole_word",
    "mass_share_fragment", "top5_share_prefix", "top5_share_frequent",
    "top5_share_fragment", "top5_share_prefix_fragment",
    "top5_share_prefix_wholeword", "top5_share_distinctive_unit",
    "top5_share_distinctive_word", "top5_distinct_words", "top5_mean_word_chars",
    "word_count_share_prefix", "word_mass_share_prefix",
    "word_mass_share_distinctive", "top5word_share_prefix",
    "top5word_share_distinctive", "top5word_mean_chars",
]


def summarise(per_pair: pd.DataFrame) -> pd.DataFrame:
    """Mean of every metric over the pair sides in each reporting cell."""
    body = per_pair[per_pair["view"] != "_corpus"].copy()
    if "nonfinite" not in body.columns:
        body["nonfinite"] = 0
    body["nonfinite"] = body["nonfinite"].fillna(0).astype(int)
    corpus_cols = ["run", "model_slug", "corpus_share_prefix",
                   "corpus_share_frequent", "n_piece_types"]
    have_corpus = all(c in per_pair.columns for c in corpus_cols)
    corpus = (per_pair.loc[per_pair["view"] == "_corpus", corpus_cols]
              if have_corpus else None)

    keys = [k for k in ("run", "model_short", "model_slug", "stratum", "layer",
                        "variant", "view", "side") if k in body.columns]
    metrics = [c for c in METRIC_COLS if c in body.columns]
    scored = body[body["nonfinite"] == 0]
    grouped = scored.groupby(keys, dropna=False)[metrics].mean().reset_index()
    counts = scored.groupby(keys, dropna=False).size().rename("n_sides").reset_index()
    out = grouped.merge(counts, on=keys, how="outer")

    dropped = body[body["nonfinite"] == 1]
    if len(dropped):
        n_bad = dropped.groupby(keys, dropna=False).size().rename("n_nonfinite")
        out = out.merge(n_bad.reset_index(), on=keys, how="outer")
    else:
        out["n_nonfinite"] = 0
    out["n_nonfinite"] = out["n_nonfinite"].fillna(0).astype(int)
    out["n_sides"] = out["n_sides"].fillna(0).astype(int)

    if corpus is not None and len(corpus):
        out = out.merge(corpus, on=["run", "model_slug"], how="left")
    out["lift_prefix"] = out["mass_share_prefix"] / out["count_share_prefix"]
    out["lift_frequent"] = out["mass_share_frequent"] / out["count_share_frequent"]
    out["lift_prefix_fragment"] = (out["mass_share_prefix_fragment"]
                                   / out["count_share_prefix_fragment"])
    return out.round(6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["deployed", "pos200", "both"], default="both")
    ap.add_argument("--artifacts_root",
                    default=str(REPO / "runs/active/ig_examples"))
    ap.add_argument("--pos200_root",
                    default=str(REPO / "runs/active/ig_examples_200pos_v1"))
    ap.add_argument("--corpus", default=str(REPO / "data/canon_labelled"))
    ap.add_argument("--out_dir", default=str(REPO / "docs/research/data"))
    ap.add_argument("--per_pair_dir", default=None,
                    help="write the per-pair rows here (large; not committed)")
    ap.add_argument("--variants", nargs="+", default=["raw", "sif_abtt"])
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--top_pct", type=float, default=0.01)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_corpus_files", type=int, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    if args.run in ("deployed", "both"):
        frames.append(("deployed", run_deployed(args)))
    if args.run in ("pos200", "both"):
        frames.append(("pos200", run_pos200(args)))

    for name, df in frames:
        if df.empty:
            print(f"[{name}] no rows")
            continue
        tag = f"_{args.tag}" if args.tag else ""
        if args.per_pair_dir:
            pp = Path(args.per_pair_dir)
            pp.mkdir(parents=True, exist_ok=True)
            df.to_csv(pp / f"prefix_attribution_per_pair_{name}{tag}.csv", index=False)
        summary = summarise(df)
        path = out_dir / f"prefix_attribution_{name}{tag}.csv"
        summary.to_csv(path, index=False)
        print(f"[{name}] wrote {path} ({len(summary)} rows, "
              f"{len(df)} per-side rows)")
    print("prefixes:", ", ".join(LATIN_PREFIXES))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
