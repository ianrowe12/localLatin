"""Phase 12 Step 2: Run Captum token-level attribution for selected files and layers.

For each file and layer, computes 4 attribution vectors:
  - IntegratedGradients  x {pc1_dot, abtt_norm}
  - FeatureAblation      x {pc1_dot, abtt_norm}

Output: runs/phase12/attributions/<model_slug>/layer{L}_attributions.npz
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from attribution_targets import (
    ABTTNormTarget,
    PC1DotTarget,
    get_embedding_layer,
    get_model_for_forward,
)
from canon_retrieval import load_texts
from cli_utils import parse_layers
from token_filtering import TOKEN_FILTER_CHOICES, build_token_keep_lookup

try:
    from captum.attr import FeatureAblation, LayerIntegratedGradients
except ImportError:
    raise ImportError("captum is required: pip install captum")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 12 Captum attribution.")
    parser.add_argument("--model_name", required=True, help="HuggingFace model name.")
    parser.add_argument(
        "--model_type", required=True, choices=["t5", "bert", "decoder"],
        help="Model architecture type.",
    )
    parser.add_argument("--pc_dir", required=True, help="Directory with layer*_pcs.npz files.")
    parser.add_argument("--meta_csv", required=True, help="Path to meta.csv.")
    parser.add_argument("--sample_csv", default="", help="CSV with 'file_id' column selecting files to attribute.")
    parser.add_argument("--layers", required=True, help="Layer list, e.g. '1,4,8,12'.")
    parser.add_argument("--out_dir", required=True, help="Output directory for attributions.")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--D", type=int, default=10, help="Number of PCs for ABTT target.")
    parser.add_argument("--n_steps", type=int, default=50, help="IG integration steps.")
    parser.add_argument(
        "--half_precision", action="store_true",
        help="Load model weights in float16.",
    )
    parser.add_argument(
        "--trust_remote_code", action="store_true",
        help="Trust remote code for model loading.",
    )
    parser.add_argument(
        "--checkpoint_every", type=int, default=100,
        help="Save checkpoint every N files.",
    )
    parser.add_argument(
        "--token_filter",
        choices=list(TOKEN_FILTER_CHOICES),
        default="all",
        help="Token-level pooling filter for attribution targets.",
    )
    return parser.parse_args()


def load_model(model_name: str, model_type: str, half_precision: bool,
               trust_remote_code: bool, device: str):
    """Load model and tokenizer, return (forward_model, tokenizer, model_type_resolved)."""
    kwargs = {"trust_remote_code": trust_remote_code}
    if half_precision:
        kwargs["torch_dtype"] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )

    if model_type == "t5":
        full_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)
        forward_model = full_model.get_encoder()
        forward_model.to(device)
        forward_model.eval()
        forward_model.requires_grad_(False)
        resolved_type = "t5"
    else:
        model = AutoModel.from_pretrained(model_name, **kwargs)
        model.to(device)
        model.eval()
        model.requires_grad_(False)
        # Detect wrapped vs unwrapped decoder
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            resolved_type = "decoder_wrapped"
        elif hasattr(model, "layers"):
            resolved_type = "decoder"
        elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            resolved_type = "bert"
        else:
            resolved_type = model_type
        forward_model = model

    return forward_model, tokenizer, resolved_type


def run_attribution_for_file(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pc1_target: PC1DotTarget,
    abtt_target: ABTTNormTarget,
    emb_layer: torch.nn.Module,
    n_steps: int,
) -> dict:
    """Run IG and FA for both targets on a single file. Returns dict of numpy arrays."""
    results = {}

    for target_name, target_fn in [("pc1_dot", pc1_target), ("abtt_norm", abtt_target)]:
        # --- IntegratedGradients via LayerIntegratedGradients ---
        lig = LayerIntegratedGradients(target_fn, emb_layer)
        attr_ig = lig.attribute(
            input_ids,
            additional_forward_args=(attention_mask,),
            n_steps=n_steps,
            internal_batch_size=1,
        )
        # Sum across embedding dim to get per-token scalar
        ig_scores = attr_ig.sum(dim=-1).squeeze(0).detach().cpu().float().numpy()
        results[f"ig_{target_name}"] = ig_scores  # (seq_len,)

        # --- FeatureAblation at embedding level ---
        # We ablate per-token by using the embedding output.
        # LayerFeatureAblation would be ideal but FeatureAblation on embeddings
        # requires a wrapper that takes embeddings as input.
        # Instead, we use a simple approach: ablate each token position by
        # replacing its input_id with the pad token (or 0).
        seq_len = input_ids.shape[1]
        baseline_ids = torch.zeros_like(input_ids)  # zero token IDs as baseline

        fa = FeatureAblation(target_fn)
        # Create feature mask: each token position is its own feature
        feature_mask = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        attr_fa = fa.attribute(
            input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask,),
            feature_mask=feature_mask,
        )
        fa_scores = attr_fa.squeeze(0).detach().cpu().float().numpy()
        results[f"fa_{target_name}"] = fa_scores  # (seq_len,)

    return results


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layers = parse_layers(args.layers)

    # Load data
    meta = pd.read_csv(args.meta_csv)

    # Filter to selected samples if provided
    if args.sample_csv:
        sample_df = pd.read_csv(args.sample_csv)
        sample_ids = set(sample_df["file_id"].tolist())
        if "file_id" in meta.columns:
            mask = meta["file_id"].isin(sample_ids)
        else:
            # Use positional index
            mask = meta.index.isin(sample_ids)
        meta = meta[mask].reset_index(drop=True)
        print(f"Filtered to {len(meta)} samples from {args.sample_csv}")

    all_paths = meta["path"].tolist()
    n_files = len(all_paths)

    # Load model
    print(f"Loading model: {args.model_name} ({args.model_type})")
    forward_model, tokenizer, resolved_type = load_model(
        args.model_name, args.model_type, args.half_precision,
        args.trust_remote_code, device,
    )
    emb_layer = get_embedding_layer(forward_model, resolved_type)
    token_keep_lookup = build_token_keep_lookup(tokenizer, args.token_filter)
    token_keep_lookup_t = torch.as_tensor(
        token_keep_lookup, device=device, dtype=torch.float32
    )

    pc_dir = Path(args.pc_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-tokenize all files to find max_seq_len for this model
    print("Pre-tokenizing all files...")
    all_encodings = []
    for path in all_paths:
        text = load_texts([path])[0]
        enc = tokenizer(
            text, truncation=True, max_length=args.max_length,
            padding=False, return_tensors="pt",
        )
        all_encodings.append(enc)

    max_seq_len = max(enc["input_ids"].shape[1] for enc in all_encodings)
    print(f"  {n_files} files, max_seq_len={max_seq_len}")

    for layer in layers:
        print(f"\n=== Layer {layer} ===")
        layer_start = time.time()

        # Load pre-computed PCs
        pc_path = pc_dir / f"layer{layer}_pcs.npz"
        if not pc_path.exists():
            print(f"  WARNING: {pc_path} not found, skipping")
            continue
        pc_data = np.load(pc_path)
        pc1 = torch.from_numpy(pc_data["pcs"][0]).to(device)
        pcs = torch.from_numpy(pc_data["pcs"][: args.D]).to(device)
        mean_vec = torch.from_numpy(pc_data["mean_vec"]).to(device)

        # Build target modules
        pc1_target = PC1DotTarget(
            forward_model, layer, pc1, mean_vec, token_keep_lookup=token_keep_lookup_t
        ).to(device)
        abtt_target = ABTTNormTarget(
            forward_model, layer, pcs, mean_vec, token_keep_lookup=token_keep_lookup_t
        ).to(device)

        # Check for existing checkpoint
        checkpoint_path = out_dir / f"layer{layer}_checkpoint.npz"
        out_path = out_dir / f"layer{layer}_attributions.npz"

        if out_path.exists():
            print(f"  Output already exists: {out_path}, skipping")
            continue

        start_idx = 0
        ig_pc1 = np.zeros((n_files, max_seq_len), dtype=np.float32)
        ig_abtt = np.zeros((n_files, max_seq_len), dtype=np.float32)
        fa_pc1 = np.zeros((n_files, max_seq_len), dtype=np.float32)
        fa_abtt = np.zeros((n_files, max_seq_len), dtype=np.float32)
        seq_lengths = np.zeros(n_files, dtype=np.int32)
        all_input_ids = np.zeros((n_files, max_seq_len), dtype=np.int64)

        if checkpoint_path.exists():
            ckpt = np.load(checkpoint_path)
            start_idx = int(ckpt["completed_files"])
            ig_pc1 = ckpt["ig_pc1_dot"]
            ig_abtt = ckpt["ig_abtt_norm"]
            fa_pc1 = ckpt["fa_pc1_dot"]
            fa_abtt = ckpt["fa_abtt_norm"]
            seq_lengths = ckpt["seq_lengths"]
            all_input_ids = ckpt["input_ids"]
            print(f"  Resuming from checkpoint: {start_idx}/{n_files} files done")

        for file_idx in range(start_idx, n_files):
            enc = all_encodings[file_idx]
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            seq_len = input_ids.shape[1]

            seq_lengths[file_idx] = seq_len
            all_input_ids[file_idx, :seq_len] = enc["input_ids"][0].numpy()

            try:
                attr = run_attribution_for_file(
                    input_ids, attention_mask,
                    pc1_target, abtt_target,
                    emb_layer, args.n_steps,
                )
                ig_pc1[file_idx, :seq_len] = attr["ig_pc1_dot"][:seq_len]
                ig_abtt[file_idx, :seq_len] = attr["ig_abtt_norm"][:seq_len]
                fa_pc1[file_idx, :seq_len] = attr["fa_pc1_dot"][:seq_len]
                fa_abtt[file_idx, :seq_len] = attr["fa_abtt_norm"][:seq_len]
            except Exception as e:
                print(f"  ERROR on file {file_idx} ({all_paths[file_idx]}): {e}")
                # Leave zeros for failed files

            if (file_idx + 1) % 50 == 0:
                elapsed = time.time() - layer_start
                rate = (file_idx + 1 - start_idx) / elapsed
                remaining = (n_files - file_idx - 1) / max(rate, 1e-6)
                print(f"  [{file_idx+1}/{n_files}] {rate:.1f} files/s, ~{remaining/60:.0f} min left")

            # Save checkpoint
            if (file_idx + 1) % args.checkpoint_every == 0:
                np.savez(
                    checkpoint_path,
                    ig_pc1_dot=ig_pc1, ig_abtt_norm=ig_abtt,
                    fa_pc1_dot=fa_pc1, fa_abtt_norm=fa_abtt,
                    seq_lengths=seq_lengths, input_ids=all_input_ids,
                    completed_files=np.array(file_idx + 1),
                )

            # Memory cleanup
            if (file_idx + 1) % 20 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        # Save final output
        np.savez(
            out_path,
            ig_pc1_dot=ig_pc1, ig_abtt_norm=ig_abtt,
            fa_pc1_dot=fa_pc1, fa_abtt_norm=fa_abtt,
            seq_lengths=seq_lengths, input_ids=all_input_ids,
        )
        # Remove checkpoint
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        elapsed = time.time() - layer_start
        print(f"  Layer {layer} done in {elapsed/60:.1f} min → {out_path}")

    print("\nAll layers complete.")


if __name__ == "__main__":
    main()
