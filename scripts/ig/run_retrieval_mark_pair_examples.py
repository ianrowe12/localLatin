"""Run Retrieval MarK mask optimization for selected Phase 12f pair examples.

For each row in the phase12f examples CSV, this script:
  1. Reads the canonical IG artifact NPZ for the pair (tokenization, hidden
     states, ABTT PCs/mean_vec, attention masks).
  2. Computes the four partner pooled vectors (raw + ABTT for both query and
     candidate) from the cached hidden states.
  3. Invokes ``compute_pair_masks`` from ``src/retrieval_mask.py`` to learn a
     per-token mask that preserves the pair cosine on both sides.
  4. Writes a sidecar NPZ containing the learned masks, pair matrices, top-k
     indices, convergence traces, and the original cosine values.

The runner NEVER mutates the canonical NPZ — a separate merge script folds
sidecars back into the canonical files.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from canon_retrieval import mean_pool  # noqa: E402
from retrieval_mask import (  # noqa: E402
    MaskOptimConfig,
    RetrievalMaskOptimizer,
    compute_pair_masks,
)
from token_filtering import (  # noqa: E402
    TOKEN_FILTER_CHOICES,
    build_token_keep_lookup,
    numpy_token_keep_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples_csv",
        type=Path,
        default=Path("runs/active/ig_examples/phase12f_examples.csv"),
    )
    parser.add_argument(
        "--artifacts_in_dir",
        type=Path,
        default=Path("runs/active/ig_examples/artifacts"),
        help="Directory containing canonical per-pair NPZ artifacts to read from.",
    )
    parser.add_argument(
        "--artifacts_out_dir",
        type=Path,
        default=Path("runs/active/ig_examples/retrieval_mark/artifacts"),
        help="Directory where sidecar NPZ outputs will be written.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Optional comma-separated list of model_name filters.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=0,
        help="0 = all; N = first N rows after filtering.",
    )
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--half_precision", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--token_filter",
        choices=list(TOKEN_FILTER_CHOICES),
        default="tokenizer_empty",
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lambda_sparsity", type=float, default=0.01)
    parser.add_argument("--gamma_tv", type=float, default=0.001)
    parser.add_argument("--init_mask_logit", type=float, default=2.197)
    parser.add_argument("--early_stop_thresh", type=float, default=0.01)
    parser.add_argument("--early_stop_min_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip examples whose sidecar NPZ already exists.",
    )
    parser.add_argument(
        "--require_cuda",
        action="store_true",
        help="Fail hard if CUDA is not available (for GPU sbatch runs).",
    )
    return parser.parse_args()


def model_slug(name: str) -> str:
    return name.replace("/", "_")


def load_model_and_tokenizer(
    model_name: str,
    half_precision: bool,
    trust_remote_code: bool,
    device: str,
):
    kwargs = {"trust_remote_code": trust_remote_code}
    if half_precision:
        kwargs["torch_dtype"] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )

    # Probe architecture by attempting T5 first, falling back to AutoModel.
    lower = model_name.lower()
    is_t5_hint = "t5" in lower or "lata" in lower or "philta" in lower
    model = None
    if is_t5_hint:
        try:
            full_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)
            model = full_model.get_encoder()
        except Exception:
            model = None
    if model is None:
        model = AutoModel.from_pretrained(model_name, **kwargs)

    # Resolve model_type from the loaded object.
    try:
        from transformers import BertModel, T5EncoderModel
    except ImportError:  # pragma: no cover - transformers always present here
        BertModel = None  # type: ignore
        T5EncoderModel = None  # type: ignore

    if hasattr(model, "get_encoder") or (
        T5EncoderModel is not None and isinstance(model, T5EncoderModel)
    ):
        resolved_type = "t5"
    elif hasattr(model, "block") and hasattr(model, "final_layer_norm"):
        # T5 encoder exposed directly (no get_encoder attribute).
        resolved_type = "t5"
    elif hasattr(model, "pooler") or (
        BertModel is not None and isinstance(model, BertModel)
    ):
        resolved_type = "bert"
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        resolved_type = "decoder_wrapped"
    else:
        resolved_type = "decoder"

    model.to(device).eval()
    model.requires_grad_(False)
    return model, tokenizer, resolved_type


def abtt_clean_np(vec: np.ndarray, pcs: np.ndarray, mean_vec: np.ndarray) -> np.ndarray:
    centered = vec - mean_vec
    return centered - centered @ pcs.T @ pcs


def compute_partners(
    hidden: np.ndarray,
    attention_mask: np.ndarray,
    input_ids: np.ndarray,
    token_keep_lookup: np.ndarray,
    pcs: np.ndarray,
    mean_vec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (raw_partner, abtt_partner) pooled vectors of shape (dim,)."""
    # hidden: (seq, dim); attention_mask: (1, seq); input_ids: (1, seq)
    pool_mask = numpy_token_keep_mask(input_ids, attention_mask, token_keep_lookup)
    # Use the same mean_pool helper shape convention: (batch, seq, dim).
    pooled = np.asarray(mean_pool(hidden[np.newaxis, ...], pool_mask)).reshape(-1)  # (dim,)
    abtt = abtt_clean_np(pooled, pcs, mean_vec)
    return pooled.astype(np.float32), abtt.astype(np.float32)


def main() -> None:
    args = parse_args()
    examples = pd.read_csv(args.examples_csv)

    if args.models:
        wanted = {m.strip() for m in args.models.split(",") if m.strip()}
        examples = examples[examples["model_name"].isin(wanted)].reset_index(drop=True)

    if args.max_examples and args.max_examples > 0:
        examples = examples.head(args.max_examples).reset_index(drop=True)

    cuda_avail = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_avail else 0
    device_name = torch.cuda.get_device_name(0) if cuda_avail and device_count > 0 else "none"
    print(
        f"[env] torch={torch.__version__} cuda_available={cuda_avail} "
        f"device_count={device_count} device_name={device_name} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
        flush=True,
    )
    if args.require_cuda and not cuda_avail:
        raise SystemExit(
            "CUDA required but not available. Check GPU allocation and torch install."
        )
    device = "cuda" if cuda_avail else "cpu"
    print(f"[env] using device={device}", flush=True)
    cfg = MaskOptimConfig(
        lr=args.lr,
        steps=args.steps,
        lambda_sparsity=args.lambda_sparsity,
        gamma_tv=args.gamma_tv,
        init_mask_logit=args.init_mask_logit,
        early_stop_thresh=args.early_stop_thresh,
        early_stop_min_steps=args.early_stop_min_steps,
    )

    model_cache: dict[str, tuple] = {}
    tok_lookup_cache: dict[int, tuple[np.ndarray, torch.Tensor]] = {}

    n_done = 0
    n_skipped_existing = 0
    n_failed = 0
    failures: list[tuple[int, str]] = []

    args.artifacts_out_dir.mkdir(parents=True, exist_ok=True)

    for row in tqdm(list(examples.itertuples(index=False)), desc="pairs"):
        t0 = time.time()
        example_id = int(row.example_id)
        model_name = str(row.model_name)
        slug = model_slug(model_name)
        layer = int(row.layer)
        d_value = int(row.D)

        src_path = (
            args.artifacts_in_dir
            / slug
            / f"example{example_id:03d}_pair_example.npz"
        )
        out_path = (
            args.artifacts_out_dir
            / slug
            / f"example{example_id:03d}_pair_example.npz"
        )

        if not src_path.exists():
            print(f"[WARN] missing canonical NPZ: {src_path}")
            n_failed += 1
            failures.append((example_id, f"missing_canonical:{src_path}"))
            continue

        if args.skip_existing and out_path.exists():
            n_skipped_existing += 1
            continue

        try:
            # Lazy model load per model_name.
            if model_name not in model_cache:
                model, tokenizer, resolved_type = load_model_and_tokenizer(
                    model_name,
                    args.half_precision,
                    args.trust_remote_code,
                    device,
                )
                model_cache[model_name] = (model, tokenizer, resolved_type)
            model, tokenizer, resolved_type = model_cache[model_name]

            tok_key = id(tokenizer)
            if tok_key not in tok_lookup_cache:
                keep_np = build_token_keep_lookup(tokenizer, args.token_filter).astype(
                    np.float32
                )
                keep_t = torch.from_numpy(keep_np).to(device)
                tok_lookup_cache[tok_key] = (keep_np, keep_t)
            keep_np, keep_t = tok_lookup_cache[tok_key]

            with np.load(src_path) as src:
                query_input_ids = np.asarray(src["query_input_ids"])
                query_attention_mask = np.asarray(src["query_attention_mask"])
                candidate_input_ids = np.asarray(src["candidate_input_ids"])
                candidate_attention_mask = np.asarray(src["candidate_attention_mask"])
                query_hidden = np.asarray(src["query_hidden"]).astype(np.float32)
                candidate_hidden = np.asarray(src["candidate_hidden"]).astype(
                    np.float32
                )
                pcs_full = np.asarray(src["pcs"]).astype(np.float32)
                mean_vec = np.asarray(src["mean_vec"]).astype(np.float32)

            pcs = pcs_full[:d_value]

            query_raw_np, query_abtt_np = compute_partners(
                query_hidden,
                query_attention_mask,
                query_input_ids,
                keep_np,
                pcs,
                mean_vec,
            )
            candidate_raw_np, candidate_abtt_np = compute_partners(
                candidate_hidden,
                candidate_attention_mask,
                candidate_input_ids,
                keep_np,
                pcs,
                mean_vec,
            )

            query_input_ids_t = torch.from_numpy(query_input_ids).long().to(device)
            query_attention_mask_t = (
                torch.from_numpy(query_attention_mask).long().to(device)
            )
            candidate_input_ids_t = (
                torch.from_numpy(candidate_input_ids).long().to(device)
            )
            candidate_attention_mask_t = (
                torch.from_numpy(candidate_attention_mask).long().to(device)
            )
            if query_input_ids_t.ndim == 1:
                query_input_ids_t = query_input_ids_t.unsqueeze(0)
                query_attention_mask_t = query_attention_mask_t.unsqueeze(0)
            if candidate_input_ids_t.ndim == 1:
                candidate_input_ids_t = candidate_input_ids_t.unsqueeze(0)
                candidate_attention_mask_t = candidate_attention_mask_t.unsqueeze(0)

            query_raw_partner_t = torch.from_numpy(query_raw_np).to(device)
            query_abtt_partner_t = torch.from_numpy(query_abtt_np).to(device)
            candidate_raw_partner_t = torch.from_numpy(candidate_raw_np).to(device)
            candidate_abtt_partner_t = torch.from_numpy(candidate_abtt_np).to(device)
            pcs_t = torch.from_numpy(pcs).to(device)
            mean_vec_t = torch.from_numpy(mean_vec).to(device)

            opt = RetrievalMaskOptimizer(
                model=model,
                model_type=resolved_type,
                layer_idx=layer,
                token_keep_lookup=keep_t,
                config=cfg,
                device=device,
            )

            result = compute_pair_masks(
                opt,
                query_input_ids_t,
                query_attention_mask_t,
                candidate_input_ids_t,
                candidate_attention_mask_t,
                candidate_raw_partner_t,
                candidate_abtt_partner_t,
                query_raw_partner_t,
                query_abtt_partner_t,
                pcs_t,
                mean_vec_t,
                query_hidden,
                candidate_hidden,
                seed=args.seed,
            )

            hyperparams = {
                "lr": args.lr,
                "steps": args.steps,
                "lambda_sparsity": args.lambda_sparsity,
                "gamma_tv": args.gamma_tv,
                "init_mask_logit": args.init_mask_logit,
                "early_stop_thresh": args.early_stop_thresh,
                "early_stop_min_steps": args.early_stop_min_steps,
                "token_filter": args.token_filter,
                "seed": args.seed,
            }

            save_dict = {k: np.asarray(v) for k, v in result.items()}
            save_dict["example_id"] = np.array(example_id, dtype=np.int32)
            save_dict["layer"] = np.array(layer, dtype=np.int32)
            save_dict["D"] = np.array(d_value, dtype=np.int32)
            save_dict["hyperparams"] = np.array(
                json.dumps(hyperparams), dtype="<U500"
            )

            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out_path, **save_dict)
            n_done += 1
            dt = time.time() - t0
            print(f"[OK] example {example_id} ({slug}) saved -> {out_path} [{dt:.1f}s]")
        except Exception as exc:  # pragma: no cover - runtime path
            n_failed += 1
            failures.append((example_id, f"{type(exc).__name__}: {exc}"))
            traceback.print_exc()
            print(f"[FAIL] example {example_id} ({slug}): {exc}")

    print("\n=== Summary ===")
    print(f"processed: {n_done}")
    print(f"skipped (already existed): {n_skipped_existing}")
    print(f"failed: {n_failed}")
    for eid, reason in failures:
        print(f"  example {eid}: {reason}")


if __name__ == "__main__":
    main()
