from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from canon_retrieval import l2_normalize, load_texts, save_json
from sif_abtt import sif_weights_from_ids
from cli_utils import parse_layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract encoder hidden state embeddings.")
    parser.add_argument("--meta_csv", required=True, help="Path to meta.csv.")
    parser.add_argument("--runs_root", required=True, help="Root directory for runs.")
    parser.add_argument("--model_name", default="bowphs/LaTa", help="Model name.")
    parser.add_argument("--layers", default="", help="Layer list, e.g. 0-12 or 0,6,12.")
    parser.add_argument("--pooling", choices=["mean", "lasttok", "sif"], default="mean")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--run_dir", default="", help="Optional explicit run directory.")
    parser.add_argument("--sif_a", type=float, default=1e-3)
    parser.add_argument(
        "--sif_prob_source",
        choices=["corpus", "batch"],
        default="corpus",
        help="Estimate token probs from corpus or per-batch fallback.",
    )
    parser.add_argument(
        "--sif_prob_cache",
        default="",
        help="Optional JSON cache path for SIF token probs.",
    )
    parser.add_argument(
        "--split_csv",
        default="",
        help="Split CSV for train-only P(w) computation (Phase 8 leak fix).",
    )
    return parser.parse_args()


def _load_prob_cache(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    token_probs = {int(k): float(v) for k, v in payload.get("token_probs", {}).items()}
    payload["token_probs"] = token_probs
    return payload


def _save_prob_cache(path: Path, token_probs: dict, meta: dict) -> None:
    payload = {
        "token_probs": {str(k): float(v) for k, v in token_probs.items()},
        **meta,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _batch_token_probs(
    input_ids: np.ndarray, special_ids: set[int]
) -> dict[int, float]:
    counts: dict[int, int] = {}
    total = 0
    for seq in input_ids:
        for tok in seq:
            if int(tok) in special_ids:
                continue
            counts[int(tok)] = counts.get(int(tok), 0) + 1
            total += 1
    if total == 0:
        return {}
    return {tok: count / total for tok, count in counts.items()}


def pool_hidden(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str,
    input_ids: torch.Tensor | None = None,
    token_probs: dict[int, float] | None = None,
    sif_a: float = 1e-3,
    special_ids: set[int] | None = None,
) -> torch.Tensor:
    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (hidden * mask).sum(dim=1) / denom
    lengths = attention_mask.sum(dim=1) - 1
    lengths = lengths.clamp(min=0)
    batch_idx = torch.arange(hidden.size(0), device=hidden.device)
    if pooling == "lasttok":
        return hidden[batch_idx, lengths]
    if pooling != "sif":
        raise ValueError(f"Unknown pooling: {pooling}")
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


@torch.no_grad()
def extract_layer_embeddings(
    encoder: torch.nn.Module,
    tokenizer,
    paths: List[str],
    layer_idx: int,
    max_length: int,
    batch_size: int,
    pooling: str,
    token_probs: dict[int, float] | None,
    sif_a: float,
    sif_prob_source: str,
    special_ids: set[int],
    device: str,
) -> np.ndarray:
    embeddings = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
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
        batch_token_probs = token_probs
        if pooling == "sif" and sif_prob_source == "batch":
            batch_token_probs = _batch_token_probs(
                input_ids.detach().cpu().numpy(), special_ids
            )
        outputs = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[layer_idx]
        pooled = pool_hidden(
            hidden,
            attention_mask,
            pooling,
            input_ids=input_ids,
            token_probs=batch_token_probs,
            sif_a=sif_a,
            special_ids=special_ids,
        )
        embeddings.append(pooled.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(embeddings, axis=0)


def main() -> None:
    args = parse_args()
    meta = pd.read_csv(args.meta_csv)
    paths = meta["path"].tolist()

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir) if args.run_dir else Path(args.runs_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    encoder = model.get_encoder() if hasattr(model, "get_encoder") else model.encoder
    encoder.to(device)
    encoder.eval()

    total_layers = len(encoder.block)
    max_layer = total_layers
    layers = parse_layers(args.layers) or list(range(0, max_layer + 1))
    if any(layer < 0 or layer > max_layer for layer in layers):
        raise ValueError(f"Layers must be within 0..{max_layer}: {layers}")

    token_probs = None
    prob_cache_path = Path(args.sif_prob_cache) if args.sif_prob_cache else None
    if args.pooling == "sif" and args.sif_prob_source == "corpus":
        if prob_cache_path and prob_cache_path.exists():
            cache = _load_prob_cache(prob_cache_path)
            token_probs = cache.get("token_probs", {})
        else:
            if args.split_csv:
                split_meta = pd.read_csv(args.split_csv)
                train_paths = split_meta.loc[split_meta["split"] == "train", "path"].tolist()
                texts = load_texts(train_paths)
            else:
                texts = load_texts(paths)
            from sif_abtt import token_probabilities

            token_probs = token_probabilities(
                tokenizer, texts, batch_size=128, max_length=args.max_length
            )
            if prob_cache_path:
                _save_prob_cache(
                    prob_cache_path,
                    token_probs,
                    {
                        "model_name": args.model_name,
                        "max_length": args.max_length,
                        "pooling": "sif",
                    },
                )

    config = {
        "model_name": args.model_name,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "representation": "hidden_states",
        "layers": layers,
        "pooling": args.pooling,
        "sif_a": args.sif_a,
        "sif_prob_source": args.sif_prob_source,
        "sif_prob_cache": str(prob_cache_path) if prob_cache_path else "",
        "layer_indexing": {
            "hidden_states": "0..N where 0 is embedding output, 1..N encoder blocks"
        },
        "timestamp": run_id,
        "device": device,
    }
    save_json(str(run_dir / "config.json"), config)
    meta.to_csv(run_dir / "meta.csv", index=False)

    for layer_num in layers:
        print(f"Extracting hidden layer {layer_num} ({args.pooling})...")
        emb = extract_layer_embeddings(
            encoder=encoder,
            tokenizer=tokenizer,
            paths=paths,
            layer_idx=layer_num,
            max_length=args.max_length,
            batch_size=args.batch_size,
            pooling=args.pooling,
            token_probs=token_probs,
            sif_a=args.sif_a,
            sif_prob_source=args.sif_prob_source,
            special_ids=set(getattr(tokenizer, "all_special_ids", [])),
            device=device,
        )
        emb_norm = l2_normalize(emb)
        if args.pooling == "mean":
            suffix = ""
        elif args.pooling == "lasttok":
            suffix = "_lasttok"
        else:
            suffix = "_sif"
        np.save(run_dir / f"hidden_layer{layer_num}_embeddings{suffix}.npy", emb)
        np.save(run_dir / f"hidden_layer{layer_num}_embeddings{suffix}_norm.npy", emb_norm)

    print(f"Done. Run dir: {run_dir}")


if __name__ == "__main__":
    main()
