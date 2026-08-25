"""Generate the full-corpus IG attribution artifacts for one model (issue #84).

One invocation = one model = one SLURM chunk. See
``scripts/ig/bulk_attribution.py`` for what a "pair" is, why the unit includes
the layer, and what each NPZ does and does not store.

Resume-safety
-------------
Every artifact is written to a sibling ``.tmp.npz`` and renamed, so a killed job
never leaves a half-written file at a final path and "the file exists" is a
sufficient skip test. ``example_id`` is a pure function of the model and the
sorted pair list, so a rerun lands on exactly the same ids -- including the 8
demo pairs from #86, whose ``(query, dir, model)`` combinations are re-derived
here under new ids at the same layer. Those get built too: the demo artifacts
carry all 7 methods and are not interchangeable with the slim bulk format, and
the resolver prefers the demo rows because they come first in the CSV.

Budget-safety
-------------
``--max_seconds`` stops the loop cleanly and still writes the registry, so a
chunk that runs out of wallclock hands the next one a consistent state rather
than a TIMEOUT. The sbatch derives it from the SLURM time limit.

Usage::

    python scripts/ig/run_bulk_attribution.py \\
        --model bowphs/LaTa \\
        --artifacts_dir runs/active/ig_examples/artifacts \\
        --registry_dir runs/active/ig_examples/bulk_registry \\
        --limit 50
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts" / "resubmit", Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from attribution_targets import get_embedding_layer  # noqa: E402
from retrieval_targets import ABTTCosSimTarget, BaselineCosSimTarget  # noqa: E402
from sif_abtt import token_probabilities  # noqa: E402
from token_filtering import build_token_keep_lookup  # noqa: E402

from run_resubmit_ig_comparison import build_ig_pair_matrix, cosine_matrix  # noqa: E402
from persist_attribution_methods import topk_indices  # noqa: E402
from persist_sif_attribution import reweight_matrix  # noqa: E402

from bulk_attribution import (  # noqa: E402
    ABTT_IG_VARIANTS,
    ARTIFACT_TO_CSV_VARIANT,
    BASELINE_IG_VARIANTS,
    MODEL_TYPES,
    REGISTRY_COLUMNS,
    BulkPair,
    build_layer_context,
    cleaner_sha1,
    enumerate_pairs,
    load_variant_frames,
    registry_row,
    slug_for,
)
from pooling_cleaners import write_variant_provenance  # noqa: E402

try:
    from captum.attr import LayerIntegratedGradients
except ImportError:  # pragma: no cover -- the sbatch installs captum
    LayerIntegratedGradients = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HuggingFace model id.")
    p.add_argument("--data_root", type=Path, default=REPO_ROOT / "data")
    p.add_argument("--unlabelled_root", type=Path,
                   default=REPO_ROOT / "runs/active/resubmit/unlabelled")
    p.add_argument("--labelled_bases", type=Path,
                   default=REPO_ROOT / "runs/active/resubmit_bases/phase9_bases")
    p.add_argument("--split_csv", type=Path,
                   default=REPO_ROOT / "runs/active/resubmit/data/phase_resubmit_split.csv")
    p.add_argument("--artifacts_dir", type=Path,
                   default=REPO_ROOT / "runs/active/ig_examples/artifacts")
    p.add_argument("--pc_root", type=Path,
                   default=REPO_ROOT / "runs/active/ig_examples/bulk_pcs",
                   help="Bulk-private PC cache. Deliberately NOT "
                        "runs/phase12_release/pcs: that root holds two stale mean "
                        "fits (issue #84 caveat 1) and backs 128 shipped artifacts.")
    p.add_argument("--registry_dir", type=Path,
                   default=REPO_ROOT / "runs/active/ig_examples/bulk_registry")
    p.add_argument("--token_probs_cache", type=Path,
                   default=REPO_ROOT / "runs/active/ig_examples/bulk_sif_token_probs",
                   help="Cache for the deployed-recipe SIF token probabilities. "
                        "Separate from the paper set's cache, which excludes each "
                        "evaluated pair's own text.")
    p.add_argument("--max_length", type=int, default=256,
                   help="IG token budget. The deployed pooling used 512; 256 keeps "
                        "the dense QxC grids inside the per-artifact size target "
                        "and matches the 128 existing artifacts. See --report_truncation.")
    p.add_argument("--sif_max_length", type=int, default=512,
                   help="max_length for the train-corpus token-probability estimate. "
                        "512 is what src/extract_hidden_cli.py used for the deployed "
                        "SIF bases; do not change it to match --max_length.")
    p.add_argument("--n_steps", type=int, default=40)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--sif_a", type=float, default=1e-3)
    p.add_argument("--token_filter", default="tokenizer_empty")
    p.add_argument("--matrix_dtype", default="float16", choices=["float16", "float32"],
                   help="Storage dtype for the dense QxC grids. float16 is ~2x "
                        "smaller with a worst-case error of 4e-4 relative to each "
                        "matrix maximum, which no heatmap can show.")
    p.add_argument("--score_tolerance", type=float, default=5e-5,
                   help="Max |recomputed score - rank1_score| before a pair is "
                        "rejected. Same bound as the demo run.")
    p.add_argument("--max_score_mismatch_frac", type=float, default=0.01,
                   help="Abort the chunk if more than this fraction of attempted "
                        "pairs fail the score check. A handful is a degenerate-file "
                        "edge case; a systematic failure means the wrong cache.")
    p.add_argument("--limit", type=int, default=0, help="0 = all pairs; N = first N.")
    p.add_argument("--max_seconds", type=float, default=0.0,
                   help="0 = no limit. Stop cleanly once the pair loop has run this "
                        "long, and still write the registry.")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--half_precision", action="store_true")
    p.add_argument("--report_truncation", action="store_true",
                   help="Also report what fraction of queries hit --max_length.")
    p.add_argument("--dry_run", action="store_true",
                   help="Enumerate, fit cleaners and report the plan; load no model.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model / tokenizer
# ---------------------------------------------------------------------------


def load_model(model_name: str, model_type: str, half_precision: bool,
               trust_remote_code: bool, device: str):
    from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

    kwargs = {"trust_remote_code": trust_remote_code}
    if half_precision:
        kwargs["torch_dtype"] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if model_type == "t5":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs).get_encoder()
        resolved = "t5"
    else:
        model = AutoModel.from_pretrained(model_name, **kwargs)
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            resolved = "decoder_wrapped"
        elif hasattr(model, "layers"):
            resolved = "decoder"
        elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            resolved = "bert"
        else:
            resolved = model_type
    model.to(device).eval()
    model.requires_grad_(False)
    return model, tokenizer, resolved


def sif_weight_lut(
    tokenizer,
    probs: dict[int, float],
    a: float,
    keep_lookup: np.ndarray,
) -> np.ndarray:
    """Vocabulary-sized SIF weight table, identical to ``sif_weights_from_ids``.

    That function loops over every distinct token id in ``probs`` doing a full
    array comparison per id -- fine for 128 artifacts, ~8k passes per sequence
    here. A lookup table gives the same numbers in one gather.
    """
    size = int(keep_lookup.shape[0])
    lut = np.ones(size, dtype=np.float32)
    ids = np.fromiter(probs.keys(), dtype=np.int64, count=len(probs))
    vals = np.fromiter(probs.values(), dtype=np.float64, count=len(probs))
    inside = (ids >= 0) & (ids < size)
    lut[ids[inside]] = (a / (a + vals[inside])).astype(np.float32)
    for special in getattr(tokenizer, "all_special_ids", []) or []:
        if 0 <= int(special) < size:
            lut[int(special)] = 0.0
    return lut * keep_lookup.astype(np.float32)


def normalized_weights(lut: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Mean-1 SIF weights for one sequence; a flat vector stays the identity."""
    w = lut[np.asarray(ids, dtype=np.int64)].astype(np.float32)
    total = float(w.sum())
    if total <= 0.0:
        return np.ones(w.shape[0], dtype=np.float32)
    return (w * (w.shape[0] / total)).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-pair work
# ---------------------------------------------------------------------------


def run_ig(input_ids, attention_mask, target_fn, emb_layer, n_steps: int) -> np.ndarray:
    if LayerIntegratedGradients is None:
        raise ImportError("captum is required: pip install captum")
    lig = LayerIntegratedGradients(target_fn, emb_layer)
    attr = lig.attribute(
        input_ids,
        additional_forward_args=(attention_mask,),
        n_steps=n_steps,
        internal_batch_size=1,
    )
    return attr.sum(dim=-1).squeeze(0).detach().cpu().float().numpy()


@torch.no_grad()
def encode(model, enc, layer: int) -> torch.Tensor:
    """Token hidden states at ``layer``. No attentions: nothing stored uses them."""
    out = model(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        output_hidden_states=True,
        output_attentions=False,
        return_dict=True,
    )
    return out.hidden_states[layer].float().squeeze(0)


def pool(hidden: torch.Tensor, enc, keep_lookup: torch.Tensor) -> torch.Tensor:
    mask = enc["attention_mask"].float() * keep_lookup[enc["input_ids"]]
    pooled = (hidden.unsqueeze(0) * mask.unsqueeze(-1)).sum(dim=1)
    return (pooled / mask.sum(dim=1, keepdim=True).clamp(min=1.0)).squeeze(0)


def build_artifact_arrays(
    pair: BulkPair,
    q_hidden: np.ndarray,
    c_hidden: np.ndarray,
    q_ids: np.ndarray,
    c_ids: np.ndarray,
    q_tokens: list[str],
    c_tokens: list[str],
    ig: dict[str, tuple[np.ndarray, np.ndarray]],
    cleaners,
    w_q: np.ndarray,
    w_c: np.ndarray,
    topk: int,
    matrix_dtype: np.dtype,
) -> dict[str, np.ndarray]:
    """Every key the slim artifact carries, for the variants this pair has."""
    mats: dict[str, np.ndarray] = {}
    if "baseline" in pair.variants or "sif" in pair.variants:
        q_ig_b, c_ig_b = ig["baseline"]
        mats["baseline"] = build_ig_pair_matrix(q_hidden, c_hidden, q_ig_b, c_ig_b)
    if "abtt" in pair.variants:
        q_ig_a, c_ig_a = ig["abtt"]
        mats["abtt"] = build_ig_pair_matrix(
            cleaners["mean"].clean_tokens(q_hidden),
            cleaners["mean"].clean_tokens(c_hidden),
            q_ig_a,
            c_ig_a,
        )
    if "sif" in pair.variants:
        mats["sif"] = reweight_matrix(mats["baseline"], w_q, w_c)
    if "sif_abtt" in pair.variants:
        q_ig_a, c_ig_a = ig["abtt"]
        # Rebuilt in the SIF-pooled subspace, not reweighted from the mean-pooled
        # abtt matrix -- that shortcut is the bug issue #87 fixed.
        raw = build_ig_pair_matrix(
            cleaners["sif"].clean_tokens(q_hidden),
            cleaners["sif"].clean_tokens(c_hidden),
            q_ig_a,
            c_ig_a,
        )
        mats["sif_abtt"] = reweight_matrix(raw, w_q, w_c)

    arrays: dict[str, np.ndarray] = {
        "example_id": np.array([pair.example_id], dtype=np.int64),
        "layer": np.array([pair.layer], dtype=np.int32),
        "D": np.array([cleaners["mean"].D], dtype=np.int32),
        "D_sif": np.array([cleaners["sif"].D], dtype=np.int32),
        "cleaner_sha1_mean": np.array([cleaner_sha1(cleaners["mean"])], dtype="<U40"),
        "cleaner_sha1_sif": np.array([cleaner_sha1(cleaners["sif"])], dtype="<U40"),
        "query_input_ids": np.asarray(q_ids, dtype=np.int32).reshape(1, -1),
        "candidate_input_ids": np.asarray(c_ids, dtype=np.int32).reshape(1, -1),
        "query_token_strings": np.array(q_tokens, dtype=np.str_),
        "candidate_token_strings": np.array(c_tokens, dtype=np.str_),
        "query_sif_weights": w_q,
        "candidate_sif_weights": w_c,
        "similarity_matrix": cosine_matrix(q_hidden, c_hidden).astype(matrix_dtype),
    }
    write_variant_provenance(arrays, pair.variants)

    for variant in pair.variants:
        mat = mats[variant]
        arrays[f"pair_matrix_ig_{variant}"] = mat.astype(matrix_dtype)
        q_top, c_top = topk_indices(mat, topk)
        arrays[f"topk_ig_{variant}_query"] = q_top
        arrays[f"topk_ig_{variant}_candidate"] = c_top
        src = "baseline" if variant in BASELINE_IG_VARIANTS else "abtt"
        q_ig, c_ig = ig[src]
        scale = variant in ("sif", "sif_abtt")
        arrays[f"query_ig_{variant}"] = (q_ig * w_q if scale else q_ig).astype(np.float32)
        arrays[f"candidate_ig_{variant}"] = (c_ig * w_c if scale else c_ig).astype(np.float32)
    return arrays


def atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: PLR0912, PLR0915 -- one linear pipeline, kept together
    args = parse_args()
    model_name = args.model
    if model_name not in MODEL_TYPES:
        raise SystemExit(f"Unknown model {model_name!r}; known: {sorted(MODEL_TYPES)}")
    slug = slug_for(model_name)
    started = time.time()

    split = pd.read_csv(args.split_csv)
    unlab_meta = pd.read_csv(args.unlabelled_root / "meta_unlabelled.csv")
    filename_to_row = {str(r): i for i, r in enumerate(unlab_meta["filename"])}
    file_ids = {str(r["filename"]): int(r["file_id"]) for _, r in unlab_meta.iterrows()}

    frames = load_variant_frames(args.unlabelled_root)
    pairs = enumerate_pairs(frames, model_name, filename_to_row, file_ids)
    layers = sorted({p.layer for p in pairs})
    print(f"=== {slug} ===")
    print(f"pairs enumerated: {len(pairs)} across layers {layers}")
    for layer in layers:
        at = [p for p in pairs if p.layer == layer]
        print(f"  L{layer}: {len(at):6d} pairs, variants {at[0].variants}")

    artifacts_dir = args.artifacts_dir / slug
    todo = [p for p in pairs if not p.artifact_path(args.artifacts_dir, slug).exists()]
    print(f"already on disk: {len(pairs) - len(todo)}; to build: {len(todo)}")
    if args.limit:
        todo = todo[: args.limit]
        print(f"--limit {args.limit}: building {len(todo)}")

    contexts = {
        layer: build_layer_context(
            slug, layer, args.labelled_bases, args.unlabelled_root / "bases",
            split, args.pc_root,
        )
        for layer in layers
    }

    if args.dry_run:
        print("\n[dry-run] cleaners fitted, nothing generated.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, resolved_type = load_model(
        model_name, MODEL_TYPES[model_name], args.half_precision,
        args.trust_remote_code, device,
    )
    emb_layer = get_embedding_layer(model, resolved_type)
    keep_lookup = build_token_keep_lookup(tokenizer, args.token_filter)
    keep_t = torch.as_tensor(keep_lookup, device=device, dtype=torch.float32)

    # SIF token probabilities, estimated exactly as the deployed SIF pooling
    # estimated them: whole labelled train split, max_length 512, same keep
    # lookup. See the module docstring of scripts/ig/bulk_attribution.py.
    args.token_probs_cache.mkdir(parents=True, exist_ok=True)
    probs_path = args.token_probs_cache / f"{slug}_train_token_probs.json"
    if probs_path.exists():
        with probs_path.open() as fh:
            probs = {int(k): float(v) for k, v in json.load(fh).items()}
        print(f"SIF token probs from cache: {probs_path} ({len(probs)} ids)")
    else:
        train = split[split["split"] == "train"]
        texts = []
        for raw in train["path"].astype(str):
            path = Path(raw)
            if not path.is_absolute():
                path = REPO_ROOT / raw
            if path.exists():
                texts.append(path.read_text(encoding="utf-8", errors="ignore"))
        print(f"estimating SIF token probs over {len(texts)} train docs")
        probs = token_probabilities(
            tokenizer, texts, max_length=args.sif_max_length,
            token_keep_lookup=keep_lookup,
        )
        with probs_path.open("w") as fh:
            json.dump({str(k): v for k, v in probs.items()}, fh)
        print(f"wrote {probs_path} ({len(probs)} ids)")
    lut = sif_weight_lut(tokenizer, probs, args.sif_a, keep_lookup)

    matrix_dtype = np.dtype(args.matrix_dtype)
    split_folders = split["folder_id"].to_numpy()
    split_filenames = split["filename"].astype(str).to_numpy()
    split_file_ids = split["file_id"].to_numpy()
    dir_members = {
        d: np.where(split_folders == d)[0]
        for d in {p.candidate_dir for p in todo}
    }

    counts = {"built": 0, "mismatch": 0, "error": 0, "no_dir": 0, "stopped": 0}
    sizes: list[int] = []
    truncated_q = 0
    mismatches: list[str] = []
    loop_started = time.time()

    for i, pair in enumerate(todo):
        if args.max_seconds and (time.time() - loop_started) > args.max_seconds:
            counts["stopped"] = len(todo) - i
            print(f"\n[budget] --max_seconds reached; {counts['stopped']} pairs left for the next run")
            break

        ctx = contexts[pair.layer]
        members = dir_members.get(pair.candidate_dir)
        if members is None or members.size == 0:
            counts["no_dir"] += 1
            continue

        lab_norm, unlab_norm = ctx.vectors(pair.check_variant)
        sims = lab_norm[members] @ unlab_norm[pair.query_row]
        best = int(members[int(np.argmax(sims))])
        score = float(np.max(sims))
        if abs(score - pair.check_score) > args.score_tolerance:
            counts["mismatch"] += 1
            if len(mismatches) < 20:
                mismatches.append(
                    f"{pair.filename}/{pair.candidate_dir}@L{pair.layer}: "
                    f"{score:.6f} vs csv[{pair.check_variant}] {pair.check_score:.6f}"
                )
            continue

        query_path = args.data_root / "canon_unlabelled" / pair.filename
        cand_path = args.data_root / "canon_labelled" / pair.candidate_dir / split_filenames[best]
        if not query_path.is_file() or not cand_path.is_file():
            counts["error"] += 1
            continue

        try:
            encs = {}
            for name, path in (("query", query_path), ("candidate", cand_path)):
                enc = tokenizer(
                    path.read_text(encoding="utf-8", errors="ignore"),
                    truncation=True, max_length=args.max_length,
                    padding=False, return_tensors="pt",
                )
                encs[name] = {
                    "input_ids": enc["input_ids"].to(device),
                    "attention_mask": enc["attention_mask"].to(device),
                }
            if encs["query"]["input_ids"].shape[1] >= args.max_length:
                truncated_q += 1

            hidden = {n: encode(model, encs[n], pair.layer) for n in encs}
            pooled_raw = {n: pool(hidden[n], encs[n], keep_t) for n in encs}

            need_baseline = bool(set(pair.variants) & BASELINE_IG_VARIANTS)
            need_abtt = bool(set(pair.variants) & ABTT_IG_VARIANTS)
            mean_cleaner = ctx.cleaners["mean"]
            pcs_t = torch.from_numpy(mean_cleaner.pcs).to(device).float()
            mean_t = torch.from_numpy(mean_cleaner.mean_vec).to(device).float()

            ig: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            if need_baseline:
                ig["baseline"] = tuple(
                    run_ig(
                        encs[side]["input_ids"], encs[side]["attention_mask"],
                        BaselineCosSimTarget(
                            model, pair.layer, pooled_raw[other],
                            token_keep_lookup=keep_t,
                        ).to(device),
                        emb_layer, args.n_steps,
                    )
                    for side, other in (("query", "candidate"), ("candidate", "query"))
                )
            if need_abtt:
                cleaned = {
                    n: (pooled_raw[n] - mean_t) - ((pooled_raw[n] - mean_t) @ pcs_t.T @ pcs_t)
                    for n in encs
                }
                ig["abtt"] = tuple(
                    run_ig(
                        encs[side]["input_ids"], encs[side]["attention_mask"],
                        ABTTCosSimTarget(
                            model, pair.layer, cleaned[other], pcs_t, mean_t,
                            token_keep_lookup=keep_t,
                        ).to(device),
                        emb_layer, args.n_steps,
                    )
                    for side, other in (("query", "candidate"), ("candidate", "query"))
                )

            q_ids = encs["query"]["input_ids"].squeeze(0).cpu().numpy()
            c_ids = encs["candidate"]["input_ids"].squeeze(0).cpu().numpy()
            arrays = build_artifact_arrays(
                pair,
                hidden["query"].cpu().numpy().astype(np.float32),
                hidden["candidate"].cpu().numpy().astype(np.float32),
                q_ids,
                c_ids,
                [tokenizer.decode([int(t)]) for t in q_ids],
                [tokenizer.decode([int(t)]) for t in c_ids],
                ig,
                ctx.cleaners,
                normalized_weights(lut, q_ids),
                normalized_weights(lut, c_ids),
                args.topk,
                matrix_dtype,
            )
            out_path = pair.artifact_path(args.artifacts_dir, slug)
            atomic_savez(out_path, arrays)
            sizes.append(out_path.stat().st_size)
            counts["built"] += 1
        except Exception as exc:  # noqa: BLE001 -- one bad pair must not kill 8k
            counts["error"] += 1
            print(f"  [ERROR] {pair.filename}/{pair.candidate_dir}@L{pair.layer}: {exc}",
                  file=sys.stderr)
            continue

        if counts["built"] % 100 == 0:
            rate = (time.time() - loop_started) / counts["built"]
            print(f"  {counts['built']}/{len(todo)} built, {rate:.2f} s/pair, "
                  f"mean {np.mean(sizes) / 1024:.0f} KB")

    attempted = counts["built"] + counts["mismatch"] + counts["error"]
    if attempted and counts["mismatch"] / attempted > args.max_score_mismatch_frac:
        print("\n=== SCORE MISMATCHES (first 20) ===", file=sys.stderr)
        for m in mismatches:
            print("  " + m, file=sys.stderr)

    # --- Registry: rebuilt from what is on disk, so reruns are idempotent ---
    rows = []
    for pair in pairs:
        path = pair.artifact_path(args.artifacts_dir, slug)
        if not path.exists():
            continue
        ctx = contexts[pair.layer]
        members = dir_members.get(pair.candidate_dir)
        if members is None:
            members = np.where(split_folders == pair.candidate_dir)[0]
        lab_norm, unlab_norm = ctx.vectors(pair.check_variant)
        best = int(members[int(np.argmax(lab_norm[members] @ unlab_norm[pair.query_row]))])
        rows.append(
            registry_row(
                pair, model_name, best, int(split_file_ids[best]),
                args.data_root / "canon_unlabelled" / pair.filename,
                args.data_root / "canon_labelled" / pair.candidate_dir / split_filenames[best],
                ctx.cleaners["mean"].D,
            )
        )
    args.registry_dir.mkdir(parents=True, exist_ok=True)
    registry_path = args.registry_dir / f"{slug}.csv"
    tmp = registry_path.with_suffix(".tmp.csv")
    pd.DataFrame(rows, columns=list(REGISTRY_COLUMNS)).to_csv(tmp, index=False)
    tmp.replace(registry_path)

    elapsed = time.time() - started
    loop_elapsed = time.time() - loop_started
    print("\n=== Summary ===")
    print(f"  model:              {model_name} ({slug})")
    print(f"  pairs enumerated:   {len(pairs)}")
    print(f"  built this run:     {counts['built']}")
    print(f"  score mismatches:   {counts['mismatch']}")
    print(f"  errors:             {counts['error']}")
    print(f"  dir not in split:   {counts['no_dir']}")
    print(f"  left by --max_seconds: {counts['stopped']}")
    print(f"  artifacts on disk:  {len(rows)} -> {registry_path}")
    if args.report_truncation and counts["built"]:
        print(f"  queries hitting --max_length {args.max_length}: "
              f"{truncated_q} ({100 * truncated_q / counts['built']:.1f}%)")
    if sizes:
        arr = np.array(sizes, dtype=np.float64) / 1024.0
        print(f"  NPZ size KB: mean={arr.mean():.0f} median={np.median(arr):.0f} "
              f"p95={np.percentile(arr, 95):.0f} max={arr.max():.0f} "
              f"total={arr.sum() / 1024:.1f} MB")
    if counts["built"]:
        print(f"  rate: {loop_elapsed / counts['built']:.3f} s/pair "
              f"({counts['built'] / loop_elapsed * 3600:.0f} pairs/GPU-h)")
    print(f"  wallclock: {elapsed / 60:.1f} min (loop {loop_elapsed / 60:.1f} min)")
    print(f"  artifacts dir: {artifacts_dir}")

    if attempted and counts["mismatch"] / attempted > args.max_score_mismatch_frac:
        raise SystemExit(
            f"score mismatch rate {counts['mismatch']}/{attempted} exceeds "
            f"--max_score_mismatch_frac {args.max_score_mismatch_frac}"
        )


if __name__ == "__main__":
    main()
