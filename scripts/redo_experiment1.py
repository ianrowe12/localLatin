"""Experiment 1 (Redo): Canon embedding analysis with 50/50 split.

Models: LaTa (hidden+FF1), PhilTa (hidden+FF1), LaBSE (hidden+ffn_intermediate)
Methods: baseline_mean, sif_abtt_optimal, whitening
Protocol: SIF weights + PCs fitted on train only, evaluated on test.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from canon_retrieval import l2_normalize, load_texts, similarity_matrix
from pair_evaluation import (
    retrieval_accuracy_at_k,
    safe_auc_roc,
    safe_spearman,
)
from sif_abtt import EmbeddingCleaner, sif_weights_from_ids, token_probabilities


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 1: Canon redo")
    p.add_argument("--meta_csv", required=True, help="meta_split_v2.csv")
    p.add_argument("--canon_root", required=True, help="Path to canon/")
    p.add_argument("--output", required=True, help="Output CSV path")
    p.add_argument("--D_values", default="1,2,3,5,7,10")
    p.add_argument("--sif_a", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--max_length", type=int, default=512)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model detection
# ---------------------------------------------------------------------------

def detect_model_info(model_name: str, model) -> Tuple[str, int]:
    """Detect model type and number of transformer layers.

    Returns (model_type, num_layers) where model_type is one of:
      'seq2seq' — T5-style with encoder.block
      'bert'    — BERT-style with encoder.layer
      'decoder' — Decoder-only with model.layers or model.model.layers
    """
    # Seq2Seq (T5): use the encoder
    if hasattr(model, "get_encoder"):
        encoder = model.get_encoder()
        if hasattr(encoder, "block"):
            return "seq2seq", len(encoder.block)

    # BERT-style
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return "bert", len(model.encoder.layer)

    # Decoder-only
    if hasattr(model, "layers"):
        return "decoder", len(model.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return "decoder", len(model.model.layers)

    # Fallback via config
    if hasattr(model, "config"):
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            if hasattr(model.config, attr):
                return "decoder", getattr(model.config, attr)

    raise ValueError(f"Cannot detect model type for {model_name}: {dir(model)}")


# ---------------------------------------------------------------------------
# Encoding functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_texts_hidden(
    texts: List[str],
    tokenizer,
    model,
    model_type: str,
    layer_idx: int,
    pooling: str,
    token_probs: Optional[Dict[int, float]],
    sif_a: float,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    """Encode texts using hidden states from a specific layer."""
    device = next(model.parameters()).device
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    embeddings: List[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start: start + batch_size]
        enc = tokenizer(
            batch, truncation=True, max_length=max_length,
            padding=True, return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        if model_type == "seq2seq":
            encoder = model.get_encoder()
            outputs = encoder(
                input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True, return_dict=True,
            )
        else:
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True, return_dict=True,
            )

        hidden = outputs.hidden_states[layer_idx]

        if pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = (hidden * mask).sum(dim=1) / denom
        elif pooling == "sif":
            ids_np = input_ids.detach().cpu().numpy()
            weights_np = sif_weights_from_ids(
                ids_np, token_probs or {}, a=sif_a, special_ids=list(special_ids),
            )
            weights = torch.from_numpy(weights_np).to(device)
            mask = attention_mask.float()
            scaled = hidden * (mask * weights).unsqueeze(-1)
            denom = (mask * weights).sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = scaled.sum(dim=1) / denom
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        embeddings.append(pooled.detach().cpu().float().numpy())

    return np.concatenate(embeddings, axis=0)


def _get_ffn_wo_module(model, model_type: str, layer_idx: int):
    """Get the FFN down-projection module for hooking.

    For T5 (seq2seq): encoder.block[i].layer[1].DenseReluDense.wo
    For BERT: encoder.layer[i].intermediate  (captures post-activation)
    """
    if model_type == "seq2seq":
        encoder = model.get_encoder()
        ffn_layer = encoder.block[layer_idx].layer[1]
        if hasattr(ffn_layer, "DenseReluDense"):
            return ffn_layer.DenseReluDense.wo
        if hasattr(ffn_layer, "DenseGatedGeluDense"):
            return ffn_layer.DenseGatedGeluDense.wo
        raise AttributeError(f"Cannot find FFN wo module for block {layer_idx}")
    elif model_type == "bert":
        return model.encoder.layer[layer_idx].intermediate
    else:
        raise ValueError(f"FFN extraction not supported for model_type={model_type}")


@torch.no_grad()
def encode_texts_ffn(
    texts: List[str],
    tokenizer,
    model,
    model_type: str,
    layer_idx: int,
    pooling: str,
    token_probs: Optional[Dict[int, float]],
    sif_a: float,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    """Encode texts using FFN intermediate activations via forward hooks."""
    device = next(model.parameters()).device
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    hook_module = _get_ffn_wo_module(model, model_type, layer_idx)
    embeddings: List[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start: start + batch_size]
        enc = tokenizer(
            batch, truncation=True, max_length=max_length,
            padding=True, return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        captured = []

        if model_type == "seq2seq":
            # T5: pre-hook on wo captures the intermediate activation
            handle = hook_module.register_forward_pre_hook(
                lambda m, inp: captured.append(inp[0])
            )
            encoder = model.get_encoder()
            encoder(input_ids=input_ids, attention_mask=attention_mask)
        elif model_type == "bert":
            # BERT: post-hook on intermediate captures output
            handle = hook_module.register_forward_hook(
                lambda m, inp, out: captured.append(out)
            )
            model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            raise ValueError(f"FFN not supported for {model_type}")

        handle.remove()
        ff_act = captured[0]  # [batch, seq_len, ffn_dim]

        if pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = (ff_act * mask).sum(dim=1) / denom
        elif pooling == "sif":
            ids_np = input_ids.detach().cpu().numpy()
            weights_np = sif_weights_from_ids(
                ids_np, token_probs or {}, a=sif_a, special_ids=list(special_ids),
            )
            weights = torch.from_numpy(weights_np).to(device)
            mask = attention_mask.float()
            scaled = ff_act * (mask * weights).unsqueeze(-1)
            denom = (mask * weights).sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = scaled.sum(dim=1) / denom
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        embeddings.append(pooled.detach().cpu().float().numpy())

    return np.concatenate(embeddings, axis=0)


# ---------------------------------------------------------------------------
# D sweep + evaluation
# ---------------------------------------------------------------------------

def sweep_D_on_train(
    train_emb: np.ndarray,
    train_folder_ids: np.ndarray,
    D_values: List[int],
) -> Tuple[int, float]:
    """Sweep D on train split using Spearman of upper-triangle pairs."""
    from canon_retrieval import upper_triangle, upper_triangle_labels

    best_d = D_values[0]
    best_sp = -1.0

    for D in D_values:
        cleaner = EmbeddingCleaner(num_components=D, center=True)
        cleaner.fit(train_emb)
        cleaned = cleaner.transform(train_emb)
        cleaned_norm = l2_normalize(cleaned)
        sim = similarity_matrix(cleaned_norm)
        sims = upper_triangle(sim)
        labels = upper_triangle_labels(train_folder_ids)
        sp = safe_spearman(sims, labels.astype(np.float32))
        if not np.isnan(sp) and sp > best_sp:
            best_sp = sp
            best_d = D

    return best_d, best_sp


def evaluate_split(
    emb_norm: np.ndarray,
    folder_ids: np.ndarray,
    query_mask: np.ndarray,
) -> Dict:
    """Evaluate embeddings on a split: Spearman, AUC, retrieval acc."""
    from canon_retrieval import upper_triangle, upper_triangle_labels

    sim = similarity_matrix(emb_norm)
    sims = upper_triangle(sim)
    labels = upper_triangle_labels(folder_ids)

    sp = safe_spearman(sims, labels.astype(np.float32))
    auc = safe_auc_roc(sims, labels.astype(int))
    acc1 = retrieval_accuracy_at_k(sim, folder_ids, query_mask, 1)
    acc3 = retrieval_accuracy_at_k(sim, folder_ids, query_mask, 3)
    acc5 = retrieval_accuracy_at_k(sim, folder_ids, query_mask, 5)

    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)

    return {
        "spearman_test": sp,
        "auc_roc_test": auc,
        "acc_at_1_test": acc1,
        "acc_at_3_test": acc3,
        "acc_at_5_test": acc5,
        "n_pos_test_pairs": n_pos,
        "n_neg_test_pairs": n_neg,
    }


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "model", "repr", "layer", "method", "D", "D_source", "sif_a",
    "spearman_test", "auc_roc_test", "acc_at_1_test", "acc_at_3_test", "acc_at_5_test",
    "spearman_train", "n_train_files", "n_test_files",
    "n_pos_test_pairs", "n_neg_test_pairs",
]


def append_rows(path: Path, rows: List[Dict]) -> None:
    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import pandas as pd
    args = parse_args()

    D_values = [int(x) for x in args.D_values.split(",")]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing output to avoid duplicate rows
    if out_path.exists():
        out_path.unlink()

    # Load meta
    meta = pd.read_csv(args.meta_csv)
    train_mask = meta["split"].values == "train"
    test_mask = meta["split"].values == "test"
    all_paths = meta["path"].tolist()
    all_folder_ids = meta["folder_id"].values
    test_query_mask = meta["is_test_query"].values.astype(bool)

    n_train = int(train_mask.sum())
    n_test = int(test_mask.sum())

    # Load all texts once
    print("Loading all canon texts...")
    all_texts = load_texts(all_paths)

    # Train-only texts for P(w)
    train_texts = [all_texts[i] for i in range(len(all_texts)) if train_mask[i]]

    # Define model configs: (model_name, load_func, reprs)
    seq2seq_models = ["bowphs/LaTa", "bowphs/PhilTa"]
    encoder_models = ["sentence-transformers/LaBSE"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Seq2Seq models (LaTa, PhilTa) ----
    for model_name in seq2seq_models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
        model.to(device)
        model.eval()

        model_type, num_layers = detect_model_info(model_name, model)
        print(f"  Type: {model_type}, Layers: {num_layers}")

        # Train-only P(w)
        print("  Computing train-only P(w)...")
        tp = token_probabilities(tokenizer, train_texts, batch_size=args.batch_size, max_length=args.max_length)

        # Hidden states: layers 0..num_layers (num_layers+1 hidden states)
        for layer_idx in range(0, num_layers + 1):
            print(f"\n  --- hidden layer {layer_idx} ---")
            rows = _process_layer(
                all_texts=all_texts, tokenizer=tokenizer, model=model,
                model_type=model_type, model_name=model_name,
                repr_name="hidden", layer_idx=layer_idx,
                train_mask=train_mask, test_mask=test_mask,
                all_folder_ids=all_folder_ids, test_query_mask=test_query_mask,
                tp=tp, D_values=D_values, args=args,
                n_train=n_train, n_test=n_test,
                encode_fn=encode_texts_hidden,
            )
            if rows:
                append_rows(out_path, rows)

        # FF1: layers 0..num_layers-1 (block indices)
        for layer_idx in range(0, num_layers):
            print(f"\n  --- ff1 layer {layer_idx} ---")
            rows = _process_layer(
                all_texts=all_texts, tokenizer=tokenizer, model=model,
                model_type=model_type, model_name=model_name,
                repr_name="ff1", layer_idx=layer_idx,
                train_mask=train_mask, test_mask=test_mask,
                all_folder_ids=all_folder_ids, test_query_mask=test_query_mask,
                tp=tp, D_values=D_values, args=args,
                n_train=n_train, n_test=n_test,
                encode_fn=encode_texts_ffn,
            )
            if rows:
                append_rows(out_path, rows)

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---- Encoder models (LaBSE) ----
    for model_name in encoder_models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        model.to(device)
        model.eval()

        model_type, num_layers = detect_model_info(model_name, model)
        print(f"  Type: {model_type}, Layers: {num_layers}")

        tp = token_probabilities(tokenizer, train_texts, batch_size=args.batch_size, max_length=args.max_length)

        # Hidden states: layers 0..num_layers
        for layer_idx in range(0, num_layers + 1):
            print(f"\n  --- hidden layer {layer_idx} ---")
            rows = _process_layer(
                all_texts=all_texts, tokenizer=tokenizer, model=model,
                model_type=model_type, model_name=model_name,
                repr_name="hidden", layer_idx=layer_idx,
                train_mask=train_mask, test_mask=test_mask,
                all_folder_ids=all_folder_ids, test_query_mask=test_query_mask,
                tp=tp, D_values=D_values, args=args,
                n_train=n_train, n_test=n_test,
                encode_fn=encode_texts_hidden,
            )
            if rows:
                append_rows(out_path, rows)

        # FFN intermediate: layers 0..num_layers-1
        for layer_idx in range(0, num_layers):
            print(f"\n  --- ffn_intermediate layer {layer_idx} ---")
            rows = _process_layer(
                all_texts=all_texts, tokenizer=tokenizer, model=model,
                model_type=model_type, model_name=model_name,
                repr_name="ffn_intermediate", layer_idx=layer_idx,
                train_mask=train_mask, test_mask=test_mask,
                all_folder_ids=all_folder_ids, test_query_mask=test_query_mask,
                tp=tp, D_values=D_values, args=args,
                n_train=n_train, n_test=n_test,
                encode_fn=encode_texts_ffn,
            )
            if rows:
                append_rows(out_path, rows)

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"\nResults saved: {out_path}")


def _process_layer(
    all_texts, tokenizer, model, model_type, model_name,
    repr_name, layer_idx,
    train_mask, test_mask,
    all_folder_ids, test_query_mask,
    tp, D_values, args,
    n_train, n_test,
    encode_fn,
) -> List[Dict]:
    """Process one (model, repr, layer): extract, apply 3 methods, evaluate."""

    rows = []
    test_folder_ids = all_folder_ids[test_mask]
    test_qmask = test_query_mask[test_mask]
    train_folder_ids = all_folder_ids[train_mask]

    # --- Encode with mean pooling ---
    emb_mean = encode_fn(
        all_texts, tokenizer, model, model_type, layer_idx,
        "mean", None, args.sif_a, args.max_length, args.batch_size,
    )
    train_emb_mean = emb_mean[train_mask]
    test_emb_mean = emb_mean[test_mask]

    # --- Encode with SIF pooling ---
    emb_sif = encode_fn(
        all_texts, tokenizer, model, model_type, layer_idx,
        "sif", tp, args.sif_a, args.max_length, args.batch_size,
    )
    train_emb_sif = emb_sif[train_mask]
    test_emb_sif = emb_sif[test_mask]

    # === Method 1: baseline_mean ===
    test_norm = l2_normalize(test_emb_mean)
    metrics = evaluate_split(test_norm, test_folder_ids, test_qmask)
    rows.append({
        "model": model_name, "repr": repr_name, "layer": layer_idx,
        "method": "baseline_mean", "D": 0, "D_source": "none",
        "sif_a": args.sif_a,
        **metrics,
        "spearman_train": float("nan"),
        "n_train_files": n_train, "n_test_files": n_test,
    })
    print(f"    baseline_mean: acc@1={metrics['acc_at_1_test']:.4f} sp={metrics['spearman_test']:.4f}")

    # === Method 2: sif_abtt_optimal ===
    best_d, train_sp = sweep_D_on_train(train_emb_sif, train_folder_ids, D_values)
    cleaner = EmbeddingCleaner(num_components=best_d, center=True)
    cleaner.fit(train_emb_sif)
    cleaned_test = cleaner.transform(test_emb_sif)
    test_norm = l2_normalize(cleaned_test)
    metrics = evaluate_split(test_norm, test_folder_ids, test_qmask)
    rows.append({
        "model": model_name, "repr": repr_name, "layer": layer_idx,
        "method": "sif_abtt_optimal", "D": best_d, "D_source": "train_sweep",
        "sif_a": args.sif_a,
        **metrics,
        "spearman_train": train_sp,
        "n_train_files": n_train, "n_test_files": n_test,
    })
    print(f"    sif_abtt_optimal (D={best_d}): acc@1={metrics['acc_at_1_test']:.4f} sp={metrics['spearman_test']:.4f}")

    # === Method 3: whitening ===
    pca = PCA(whiten=True)
    pca.fit(train_emb_mean)
    whitened_test = pca.transform(test_emb_mean)
    test_norm = l2_normalize(whitened_test)
    metrics = evaluate_split(test_norm, test_folder_ids, test_qmask)
    rows.append({
        "model": model_name, "repr": repr_name, "layer": layer_idx,
        "method": "whitening", "D": 0, "D_source": "none",
        "sif_a": args.sif_a,
        **metrics,
        "spearman_train": float("nan"),
        "n_train_files": n_train, "n_test_files": n_test,
    })
    print(f"    whitening: acc@1={metrics['acc_at_1_test']:.4f} sp={metrics['spearman_test']:.4f}")

    return rows


if __name__ == "__main__":
    main()
