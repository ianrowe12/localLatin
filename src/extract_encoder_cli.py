"""Extract hidden-state or FFN-intermediate embeddings from encoder-only / decoder-only models.

Supports:
- sentence-transformers/LaBSE (BERT-style)
- Qwen/Qwen2-7B (decoder-only)
- google/gemma-7b (decoder-only)

Two extraction modes:
- hidden: outputs.hidden_states[layer_idx]
- ffn_intermediate: hook into FFN intermediate activation
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from canon_retrieval import l2_normalize, load_texts, save_json
from sif_abtt import sif_weights_from_ids, token_probabilities
from cli_utils import parse_layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract embeddings from encoder-only/decoder-only models."
    )
    parser.add_argument("--meta_csv", required=True, help="Path to meta.csv.")
    parser.add_argument("--runs_root", required=True, help="Root directory for runs.")
    parser.add_argument(
        "--model_name", required=True,
        help="Model name, e.g. sentence-transformers/LaBSE",
    )
    parser.add_argument(
        "--repr", choices=["hidden", "ffn_intermediate"], default="hidden",
        help="Representation type to extract.",
    )
    parser.add_argument("--layers", default="", help="Layer list, e.g. 0-12 or 0,6,12.")
    parser.add_argument("--pooling", choices=["mean", "sif"], default="mean")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--run_dir", default="", help="Optional explicit run directory.")
    parser.add_argument("--sif_a", type=float, default=1e-3)
    parser.add_argument(
        "--split_csv", default="",
        help="Split CSV for train-only P(w) computation.",
    )
    parser.add_argument(
        "--half_precision", action="store_true",
        help="Load model in float16 to reduce memory.",
    )
    parser.add_argument(
        "--trust_remote_code", action="store_true",
        help="Trust remote code for model loading.",
    )
    return parser.parse_args()


def detect_model_type(model) -> str:
    """Detect whether model is BERT-style or decoder-only."""
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return "bert"
    if hasattr(model, "layers"):
        return "decoder"
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return "decoder_wrapped"
    raise ValueError(
        f"Cannot detect model type. Available attributes: {dir(model)}"
    )


def get_num_layers(model, model_type: str) -> int:
    if model_type == "bert":
        return len(model.encoder.layer)
    if model_type == "decoder":
        return len(model.layers)
    if model_type == "decoder_wrapped":
        return len(model.model.layers)
    raise ValueError(f"Unknown model type: {model_type}")


def get_ffn_module(model, layer_idx: int, model_type: str):
    """Get the FFN intermediate module for hooking."""
    if model_type == "bert":
        return model.encoder.layer[layer_idx].intermediate
    if model_type == "decoder":
        return model.layers[layer_idx].mlp
    if model_type == "decoder_wrapped":
        return model.model.layers[layer_idx].mlp
    raise ValueError(f"Unknown model type: {model_type}")


def pool_embeddings(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str,
    input_ids: Optional[torch.Tensor] = None,
    token_probs: Optional[Dict[int, float]] = None,
    sif_a: float = 1e-3,
    special_ids: Optional[set] = None,
    is_decoder: bool = False,
) -> torch.Tensor:
    """Pool token-level representations to sentence-level."""
    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (hidden * mask).sum(dim=1) / denom
    elif pooling == "sif":
        if input_ids is None:
            raise ValueError("input_ids required for SIF pooling.")
        if special_ids is None:
            special_ids = set()
        input_ids_np = input_ids.detach().cpu().numpy()
        weights_np = sif_weights_from_ids(
            input_ids_np, token_probs or {}, a=sif_a, special_ids=list(special_ids)
        )
        weights = torch.from_numpy(weights_np).to(hidden.device)
        mask = attention_mask.float()
        scaled = hidden * (mask * weights).unsqueeze(-1)
        denom = (mask * weights).sum(dim=1, keepdim=True).clamp(min=1.0)
        return scaled.sum(dim=1) / denom
    else:
        raise ValueError(f"Unknown pooling: {pooling}")


@torch.no_grad()
def extract_hidden_embeddings(
    model,
    tokenizer,
    paths: List[str],
    layer_idx: int,
    max_length: int,
    batch_size: int,
    pooling: str,
    token_probs: Optional[Dict[int, float]],
    sif_a: float,
    special_ids: set,
    device: str,
    is_decoder: bool = False,
) -> np.ndarray:
    """Extract hidden state embeddings from a specific layer."""
    embeddings = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start: start + batch_size]
        batch_texts = load_texts(batch_paths)
        enc = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[layer_idx]
        pooled = pool_embeddings(
            hidden, attention_mask, pooling,
            input_ids=input_ids,
            token_probs=token_probs,
            sif_a=sif_a,
            special_ids=special_ids,
            is_decoder=is_decoder,
        )
        embeddings.append(pooled.detach().cpu().float().numpy().astype(np.float32))
    return np.concatenate(embeddings, axis=0)


@torch.no_grad()
def extract_ffn_intermediate_embeddings(
    model,
    tokenizer,
    paths: List[str],
    layer_idx: int,
    model_type: str,
    max_length: int,
    batch_size: int,
    pooling: str,
    token_probs: Optional[Dict[int, float]],
    sif_a: float,
    special_ids: set,
    device: str,
    is_decoder: bool = False,
) -> np.ndarray:
    """Extract FFN intermediate activations via forward hook."""
    ffn_module = get_ffn_module(model, layer_idx, model_type)

    embeddings = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start: start + batch_size]
        batch_texts = load_texts(batch_paths)
        enc = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        captured = []

        def hook_fn(module, inp, out):
            # For BERT intermediate: out is the activated tensor
            # For decoder MLP: out is a tuple or tensor
            if isinstance(out, tuple):
                captured.append(out[0])
            else:
                captured.append(out)

        handle = ffn_module.register_forward_hook(hook_fn)
        _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        handle.remove()

        if not captured:
            raise RuntimeError(
                f"Hook did not capture any output from {ffn_module.__class__.__name__}"
            )
        ff_act = captured[0]

        # Ensure 3D: [batch, seq_len, dim]
        if ff_act.ndim == 2:
            # Single token or flattened — reshape
            ff_act = ff_act.unsqueeze(0)
        if ff_act.ndim != 3:
            raise RuntimeError(
                f"Unexpected FFN activation shape: {ff_act.shape}"
            )

        pooled = pool_embeddings(
            ff_act, attention_mask, pooling,
            input_ids=input_ids,
            token_probs=token_probs,
            sif_a=sif_a,
            special_ids=special_ids,
            is_decoder=is_decoder,
        )
        embeddings.append(pooled.detach().cpu().float().numpy().astype(np.float32))
    return np.concatenate(embeddings, axis=0)


def main() -> None:
    args = parse_args()
    meta = pd.read_csv(args.meta_csv)
    paths = meta["path"].tolist()

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir) if args.run_dir else Path(args.runs_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.half_precision else None

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=args.trust_remote_code
    )
    model_kwargs = {"trust_remote_code": args.trust_remote_code}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModel.from_pretrained(args.model_name, **model_kwargs)
    model.to(device)
    model.eval()

    model_type = detect_model_type(model)
    is_decoder = model_type in ("decoder", "decoder_wrapped")
    num_layers = get_num_layers(model, model_type)

    # For hidden states: layer 0 = embedding output, 1..N = transformer layers
    # Total hidden_states entries = num_layers + 1
    max_hidden_layer = num_layers
    layers = parse_layers(args.layers)
    if not layers:
        if args.repr == "hidden":
            layers = list(range(0, max_hidden_layer + 1))
        else:
            layers = list(range(0, num_layers))

    # Validate layer indices
    if args.repr == "ffn_intermediate":
        max_ffn = num_layers - 1
        invalid = [l for l in layers if l > max_ffn]
        if invalid:
            print(f"WARNING: Skipping FFN layers {invalid} (max={max_ffn} for {args.model_name})")
            layers = [l for l in layers if l <= max_ffn]
    else:
        max_valid = num_layers
        invalid = [l for l in layers if l > max_valid]
        if invalid:
            print(f"WARNING: Skipping hidden layers {invalid} (max={max_valid} for {args.model_name})")
            layers = [l for l in layers if l <= max_valid]

    # Token probs for SIF pooling
    token_probs = None
    if args.pooling == "sif":
        if args.split_csv:
            split_meta = pd.read_csv(args.split_csv)
            train_paths = split_meta.loc[split_meta["split"] == "train", "path"].tolist()
            texts = load_texts(train_paths)
        else:
            texts = load_texts(paths)
        token_probs = token_probabilities(
            tokenizer, texts, batch_size=128, max_length=args.max_length
        )

    special_ids = set(getattr(tokenizer, "all_special_ids", []))

    config = {
        "model_name": args.model_name,
        "model_type": model_type,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "representation": args.repr,
        "layers": layers,
        "pooling": args.pooling,
        "sif_a": args.sif_a,
        "split_csv": args.split_csv,
        "half_precision": args.half_precision,
        "num_model_layers": num_layers,
        "timestamp": run_id,
        "device": device,
    }
    save_json(str(run_dir / "config.json"), config)
    meta.to_csv(run_dir / "meta.csv", index=False)

    repr_prefix = "hidden" if args.repr == "hidden" else "ffn_int"
    suffix = "" if args.pooling == "mean" else "_sif"

    for layer_num in layers:
        print(f"Extracting {args.repr} layer {layer_num} ({args.pooling})...")

        if args.repr == "hidden":
            emb = extract_hidden_embeddings(
                model=model,
                tokenizer=tokenizer,
                paths=paths,
                layer_idx=layer_num,
                max_length=args.max_length,
                batch_size=args.batch_size,
                pooling=args.pooling,
                token_probs=token_probs,
                sif_a=args.sif_a,
                special_ids=special_ids,
                device=device,
                is_decoder=is_decoder,
            )
        else:
            emb = extract_ffn_intermediate_embeddings(
                model=model,
                tokenizer=tokenizer,
                paths=paths,
                layer_idx=layer_num,
                model_type=model_type,
                max_length=args.max_length,
                batch_size=args.batch_size,
                pooling=args.pooling,
                token_probs=token_probs,
                sif_a=args.sif_a,
                special_ids=special_ids,
                device=device,
                is_decoder=is_decoder,
            )

        emb_norm = l2_normalize(emb)
        np.save(run_dir / f"{repr_prefix}_layer{layer_num}_embeddings{suffix}.npy", emb)
        np.save(run_dir / f"{repr_prefix}_layer{layer_num}_embeddings{suffix}_norm.npy", emb_norm)
        print(f"  Layer {layer_num}: shape={emb.shape}")

    print(f"Done. Run dir: {run_dir}")


if __name__ == "__main__":
    main()
