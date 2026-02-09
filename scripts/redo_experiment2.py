"""Experiment 2 (Redo): MUSTS layer-wise sweep with new models.

Models: Qwen3-Embedding-0.6B, KaLM-embedding-multilingual-mini-instruct-v2.5, LaBSE
Languages: English, French, Sinhala, Tamil
Split: forced 50/50 on all MUSTS data
Methods: baseline_mean, sif_abtt_optimal, whitening
Protocol: P(w) + PCs from train only, optimal D swept on train.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from sif_abtt import EmbeddingCleaner, sif_weights_from_ids, token_probabilities


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 2: MUSTS redo")
    p.add_argument(
        "--models",
        default="Qwen/Qwen3-Embedding-0.6B,KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5,sentence-transformers/LaBSE",
    )
    p.add_argument("--languages", default="english,french,sinhala,tamil")
    p.add_argument("--output", required=True)
    p.add_argument("--D_values", default="1,2,3,5,7,10")
    p.add_argument("--sif_a", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--random_seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# MUSTS data loading with forced 50/50 split
# ---------------------------------------------------------------------------

def load_musts_5050(language: str, random_seed: int = 42):
    """Load MUSTS, concatenate all splits, shuffle, split 50/50."""
    from datasets import concatenate_datasets, load_dataset

    dataset_name = f"musts/{language}"
    all_data = []
    for split in ["train", "test", "validation"]:
        try:
            ds = load_dataset(dataset_name, split=split)
            all_data.append(ds)
        except Exception:
            pass
    if not all_data:
        raise ValueError(f"No data found for {dataset_name}")

    combined = concatenate_datasets(all_data).shuffle(seed=random_seed)
    n = len(combined)
    mid = n // 2
    return combined.select(range(mid)), combined.select(range(mid, n))


def extract_sentences(dataset) -> Tuple[List[str], List[str], np.ndarray]:
    """Extract sentences, filtering None values."""
    rows = [
        (str(row["sentence_1"]), str(row["sentence_2"]), float(row["similarity"]))
        for row in dataset
        if row["sentence_1"] is not None and row["sentence_2"] is not None
           and row["similarity"] is not None
    ]
    if not rows:
        return [], [], np.array([])
    s1 = [r[0] for r in rows]
    s2 = [r[1] for r in rows]
    labels = np.array([r[2] for r in rows], dtype=np.float32)
    dropped = len(dataset) - len(rows)
    if dropped:
        print(f"    Dropped {dropped} rows with None values")
    return s1, s2, labels


# ---------------------------------------------------------------------------
# Model detection
# ---------------------------------------------------------------------------

def detect_num_layers(model) -> int:
    """Detect number of transformer layers with config fallback."""
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return len(model.encoder.layer)
    if hasattr(model, "layers"):
        return len(model.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "config"):
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            if hasattr(model.config, attr):
                return getattr(model.config, attr)
    raise ValueError(f"Cannot detect layers for {type(model)}: {dir(model)}")


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_sentences_layer(
    sentences: List[str],
    tokenizer,
    model,
    layer_idx: int,
    pooling: str,
    token_probs: Optional[Dict[int, float]],
    sif_a: float,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    """Encode sentences using hidden states from a specific layer."""
    device = next(model.parameters()).device
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    embeddings: List[np.ndarray] = []

    for start in range(0, len(sentences), batch_size):
        batch = sentences[start: start + batch_size]
        enc = tokenizer(
            batch, truncation=True, max_length=max_length,
            padding=True, return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

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


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def cosine_sim_pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two embedding matrices."""
    a_norm = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b_norm = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return np.sum(a_norm * b_norm, axis=1)


def safe_spearman(sims: np.ndarray, labels: np.ndarray) -> float:
    if len(sims) < 3:
        return float("nan")
    try:
        result = spearmanr(sims, labels)
        corr = result.statistic if hasattr(result, "statistic") else result[0]
        return float(corr)
    except Exception:
        return float("nan")


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


# ---------------------------------------------------------------------------
# D sweep
# ---------------------------------------------------------------------------

def sweep_D_on_train(
    train_emb1: np.ndarray,
    train_emb2: np.ndarray,
    train_labels: np.ndarray,
    D_values: List[int],
) -> Tuple[int, float]:
    """Sweep D on train pairs, return (best_D, best_spearman)."""
    train_all = np.vstack([train_emb1, train_emb2])
    best_d = D_values[0]
    best_sp = -1.0
    n = len(train_emb1)

    for D in D_values:
        cleaner = EmbeddingCleaner(num_components=D, center=True)
        cleaner.fit(train_all)
        cleaned = cleaner.transform(train_all)
        c1 = cleaned[:n]
        c2 = cleaned[n:]
        sims = cosine_sim_pairs(c1, c2)
        sp = safe_spearman(sims, train_labels)
        if not np.isnan(sp) and sp > best_sp:
            best_sp = sp
            best_d = D

    return best_d, best_sp


# ---------------------------------------------------------------------------
# Per-layer evaluation
# ---------------------------------------------------------------------------

def evaluate_layer(
    model, tokenizer, language: str, layer_idx: int,
    sif_a: float, D_values: List[int],
    max_length: int, batch_size: int,
    random_seed: int,
) -> List[Dict]:
    """Evaluate one (model, language, layer) with 3 methods."""

    # Load data
    train_set, test_set = load_musts_5050(language, random_seed=random_seed)

    train_s1, train_s2, train_labels = extract_sentences(train_set)
    test_s1, test_s2, test_labels = extract_sentences(test_set)

    if not train_s1 or not test_s1:
        print(f"    WARNING: {language} has no valid pairs. Skipping.")
        return []

    # P(w) from train only
    train_texts = train_s1 + train_s2
    tp = token_probabilities(tokenizer, train_texts, batch_size=batch_size, max_length=max_length)

    model_name = getattr(model, "name_or_path", str(model.__class__.__name__))
    results = []

    # --- Encode train ---
    train_emb1_mean = encode_sentences_layer(train_s1, tokenizer, model, layer_idx, "mean", None, sif_a, max_length, batch_size)
    train_emb2_mean = encode_sentences_layer(train_s2, tokenizer, model, layer_idx, "mean", None, sif_a, max_length, batch_size)
    train_emb1_sif = encode_sentences_layer(train_s1, tokenizer, model, layer_idx, "sif", tp, sif_a, max_length, batch_size)
    train_emb2_sif = encode_sentences_layer(train_s2, tokenizer, model, layer_idx, "sif", tp, sif_a, max_length, batch_size)

    # --- Encode test ---
    test_emb1_mean = encode_sentences_layer(test_s1, tokenizer, model, layer_idx, "mean", None, sif_a, max_length, batch_size)
    test_emb2_mean = encode_sentences_layer(test_s2, tokenizer, model, layer_idx, "mean", None, sif_a, max_length, batch_size)
    test_emb1_sif = encode_sentences_layer(test_s1, tokenizer, model, layer_idx, "sif", tp, sif_a, max_length, batch_size)
    test_emb2_sif = encode_sentences_layer(test_s2, tokenizer, model, layer_idx, "sif", tp, sif_a, max_length, batch_size)

    base_row = {
        "model": model_name, "language": language, "layer": layer_idx,
        "sif_a": sif_a, "n_train_pairs": len(train_labels), "n_test_pairs": len(test_labels),
    }

    # === Method 1: baseline_mean ===
    sims_train = cosine_sim_pairs(train_emb1_mean, train_emb2_mean)
    sims_test = cosine_sim_pairs(test_emb1_mean, test_emb2_mean)
    results.append({
        **base_row, "method": "baseline_mean", "D": 0, "D_source": "none",
        "spearman_train": safe_spearman(sims_train, train_labels),
        "spearman_test": safe_spearman(sims_test, test_labels),
    })

    # === Method 2: sif_abtt_optimal ===
    best_d, train_sp = sweep_D_on_train(train_emb1_sif, train_emb2_sif, train_labels, D_values)

    # Fit cleaner on train, apply to train+test
    train_all_sif = np.vstack([train_emb1_sif, train_emb2_sif])
    cleaner = EmbeddingCleaner(num_components=best_d, center=True)
    cleaner.fit(train_all_sif)

    n_train = len(train_emb1_sif)
    cleaned_train = cleaner.transform(train_all_sif)
    ct1, ct2 = cleaned_train[:n_train], cleaned_train[n_train:]

    test_all_sif = np.vstack([test_emb1_sif, test_emb2_sif])
    cleaned_test = cleaner.transform(test_all_sif)
    n_test = len(test_emb1_sif)
    ce1, ce2 = cleaned_test[:n_test], cleaned_test[n_test:]

    results.append({
        **base_row, "method": "sif_abtt_optimal", "D": best_d, "D_source": "train_sweep",
        "spearman_train": safe_spearman(cosine_sim_pairs(ct1, ct2), train_labels),
        "spearman_test": safe_spearman(cosine_sim_pairs(ce1, ce2), test_labels),
    })

    # === Method 3: whitening ===
    train_all_mean = np.vstack([train_emb1_mean, train_emb2_mean])
    pca = PCA(whiten=True)
    pca.fit(train_all_mean)

    wt = pca.transform(train_all_mean)
    wt1, wt2 = wt[:n_train], wt[n_train:]

    test_all_mean = np.vstack([test_emb1_mean, test_emb2_mean])
    we = pca.transform(test_all_mean)
    we1, we2 = we[:n_test], we[n_test:]

    results.append({
        **base_row, "method": "whitening", "D": 0, "D_source": "none",
        "spearman_train": safe_spearman(cosine_sim_pairs(wt1, wt2), train_labels),
        "spearman_test": safe_spearman(cosine_sim_pairs(we1, we2), test_labels),
    })

    return results


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "model", "language", "layer", "method", "D", "D_source", "sif_a",
    "spearman_train", "spearman_test", "n_train_pairs", "n_test_pairs",
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
    args = parse_args()

    models = [m.strip() for m in args.models.split(",")]
    languages = [l.strip() for l in args.languages.split(",")]
    D_values = [int(x) for x in args.D_values.split(",")]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        out_path.unlink()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )
        model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True,
        )
        model.to(device)
        model.eval()

        num_layers = detect_num_layers(model)
        print(f"  Detected {num_layers} layers")

        layers = list(range(0, num_layers + 1))

        for language in languages:
            print(f"\n  Language: {language}")
            for layer_idx in layers:
                if layer_idx > num_layers:
                    print(f"    Skipping layer {layer_idx} (max={num_layers})")
                    continue
                print(f"    Layer {layer_idx}...")
                rows = evaluate_layer(
                    model=model, tokenizer=tokenizer,
                    language=language, layer_idx=layer_idx,
                    sif_a=args.sif_a, D_values=D_values,
                    max_length=args.max_length, batch_size=args.batch_size,
                    random_seed=args.random_seed,
                )
                if rows:
                    append_rows(out_path, rows)
                    sp_base = rows[0]["spearman_test"]
                    sp_abtt = rows[1]["spearman_test"]
                    print(f"      baseline={sp_base:.4f}  sif_abtt={sp_abtt:.4f} (D={rows[1]['D']})")

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
