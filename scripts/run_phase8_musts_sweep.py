"""Phase 8: Clean MUSTS layer-wise sweep with R optimization.

Fixes data leakage: P(w) from train only, PCs fitted on train only.
Extends to layer-wise analysis with LaBSE, Qwen2-7B, Gemma-7B.

Methods evaluated per (model, language, layer):
- baseline_mean: no processing
- sif_only: SIF pooling, no PC removal
- sif_abtt_fixed: SIF + D=10, fixed
- sif_abtt_optimal: SIF + D=best from train sweep
- whitening: PCA whitening (Katie's method)
- variance_filter: variance filtering (Katie's method)
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from sif_abtt import (
    EmbeddingCleaner,
    UnigramProbEstimator,
    sif_weights_from_ids,
    token_probabilities,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 8 MUSTS layer-wise sweep.")
    parser.add_argument(
        "--models", default="sentence-transformers/LaBSE",
        help="Comma-separated model names.",
    )
    parser.add_argument(
        "--languages", default="english,french,sinhala,tamil",
        help="Comma-separated language names.",
    )
    parser.add_argument("--sif_a", type=float, default=1e-3)
    parser.add_argument(
        "--D_values", default="1,2,3,5,7,10",
        help="Comma-separated D values for R sweep.",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--layers", default="", help="Layer list, e.g. 0-12.")
    parser.add_argument(
        "--variance_drop_percent", type=float, default=25.0,
        help="Percent of lowest-variance dims to drop.",
    )
    parser.add_argument(
        "--half_precision", action="store_true",
        help="Load model in float16.",
    )
    parser.add_argument(
        "--trust_remote_code", action="store_true",
        help="Trust remote code for model loading.",
    )
    return parser.parse_args()


def parse_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_layers(value: str) -> List[int]:
    if not value:
        return []
    out: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            out.extend(range(int(start_str), int(end_str) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def parse_d_values(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def detect_num_layers(model) -> int:
    """Detect number of transformer layers."""
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return len(model.encoder.layer)
    if hasattr(model, "layers"):
        return len(model.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    raise ValueError(f"Cannot detect model layers. Attributes: {dir(model)}")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b_norm = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return np.sum(a_norm * b_norm, axis=1)


def safe_spearman(sims: np.ndarray, labels: np.ndarray) -> float:
    """Compute Spearman, returning NaN on failure."""
    if len(sims) < 3:
        return float("nan")
    result = spearmanr(sims, labels)
    corr = result.correlation
    if corr is None or np.isnan(corr):
        return float("nan")
    return float(corr)


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
            batch,
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

        if pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = (hidden * mask).sum(dim=1) / denom
        elif pooling == "sif":
            input_ids_np = input_ids.detach().cpu().numpy()
            weights_np = sif_weights_from_ids(
                input_ids_np, token_probs or {}, a=sif_a, special_ids=list(special_ids)
            )
            weights = torch.from_numpy(weights_np).to(device)
            mask = attention_mask.float()
            scaled = hidden * (mask * weights).unsqueeze(-1)
            denom = (mask * weights).sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = scaled.sum(dim=1) / denom
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        embeddings.append(pooled.detach().cpu().float().numpy().astype(np.float32))
    return np.concatenate(embeddings, axis=0)


# --- Katie's methods ---

def fit_whitening(train_emb: np.ndarray) -> PCA:
    """Fit PCA whitening on train embeddings."""
    pca = PCA(whiten=True)
    pca.fit(train_emb)
    return pca


def transform_whitening(pca: PCA, emb: np.ndarray) -> np.ndarray:
    return pca.transform(emb)


def fit_variance_filter(train_emb: np.ndarray, drop_percent: float = 25.0) -> np.ndarray:
    """Compute per-dimension variance on train, return indices of top dims."""
    var = np.var(train_emb, axis=0)
    order = np.argsort(var)[::-1]
    k = max(int(round((1 - drop_percent / 100) * train_emb.shape[1])), 1)
    return order[:k]


def transform_variance_filter(emb: np.ndarray, selected_dims: np.ndarray) -> np.ndarray:
    return emb[:, selected_dims]


# --- R-value sweep ---

def sweep_R_on_train(
    train_emb1: np.ndarray,
    train_emb2: np.ndarray,
    train_labels: np.ndarray,
    D_values: List[int],
    sif_a: float,
) -> Tuple[int, List[Dict]]:
    """For each D, fit EmbeddingCleaner on train, compute Spearman on train.

    Returns (best_D, all_results).
    """
    train_all = np.vstack([train_emb1, train_emb2])
    results = []
    best_d = D_values[0]
    best_spearman = -1.0

    for D in D_values:
        cleaner = EmbeddingCleaner(num_components=D, center=True)
        cleaner.fit(train_all)
        cleaned = cleaner.transform(train_all)
        n = len(train_emb1)
        c1 = cleaned[:n]
        c2 = cleaned[n:]
        sims = cosine_similarity(c1, c2)
        sp = safe_spearman(sims, train_labels)
        results.append({"D": D, "spearman_train": sp})
        if not np.isnan(sp) and sp > best_spearman:
            best_spearman = sp
            best_d = D

    return best_d, results


def evaluate_layer(
    model,
    tokenizer,
    language: str,
    layer_idx: int,
    sif_a: float,
    D_values: List[int],
    max_length: int,
    batch_size: int,
    variance_drop_percent: float,
) -> List[Dict]:
    """Core evaluation for one (model, language, layer) triple.

    Protocol (leak-free):
    1. Load MUSTS train + test via HuggingFace datasets
    2. P(w) from train sentences only
    3. Encode train s1,s2 from target layer
    4. Encode test s1,s2 from target layer
    5. Sweep R on train -> find optimal D
    6. Fit EmbeddingCleaner(D=optimal) on train embeddings
    7. Transform test embeddings with train-fitted cleaner
    8. Compute Spearman on test
    """
    dataset_name = f"musts/{language}"

    try:
        train_set = load_dataset(dataset_name, split="train")
    except Exception:
        # Some MUSTS languages may not have a train split;
        # fall back to validation or create a synthetic split
        try:
            full = load_dataset(dataset_name, split="test")
            n = len(full)
            split_idx = int(n * 0.7)
            train_set = full.select(range(split_idx))
            test_set = full.select(range(split_idx, n))
        except Exception as e:
            print(f"  WARNING: Cannot load {dataset_name}: {e}")
            return []
    else:
        test_set = load_dataset(dataset_name, split="test")

    # Extract sentences, filtering out rows with None values
    raw_train = [
        (str(row["sentence_1"]), str(row["sentence_2"]), float(row["similarity"]))
        for row in train_set
        if row["sentence_1"] is not None and row["sentence_2"] is not None
           and row["similarity"] is not None
    ]
    raw_test = [
        (str(row["sentence_1"]), str(row["sentence_2"]), float(row["similarity"]))
        for row in test_set
        if row["sentence_1"] is not None and row["sentence_2"] is not None
           and row["similarity"] is not None
    ]

    if not raw_train or not raw_test:
        print(f"  WARNING: {language} has no valid sentence pairs after filtering Nones "
              f"(train={len(raw_train)}, test={len(raw_test)}). Skipping.")
        return []

    n_train_dropped = len(train_set) - len(raw_train)
    n_test_dropped = len(test_set) - len(raw_test)
    if n_train_dropped or n_test_dropped:
        print(f"  Dropped {n_train_dropped} train / {n_test_dropped} test rows with None values")

    train_s1 = [t[0] for t in raw_train]
    train_s2 = [t[1] for t in raw_train]
    train_labels = np.array([t[2] for t in raw_train], dtype=np.float32)

    test_s1 = [t[0] for t in raw_test]
    test_s2 = [t[1] for t in raw_test]
    test_labels = np.array([t[2] for t in raw_test], dtype=np.float32)

    # P(w) from train sentences only
    train_texts = train_s1 + train_s2
    token_probs = token_probabilities(
        tokenizer, train_texts, batch_size=batch_size, max_length=max_length
    )

    results = []
    model_name = getattr(model, "name_or_path", str(model.__class__.__name__))

    # --- Encode train ---
    train_emb1_mean = encode_sentences_layer(
        train_s1, tokenizer, model, layer_idx, "mean", None, sif_a, max_length, batch_size
    )
    train_emb2_mean = encode_sentences_layer(
        train_s2, tokenizer, model, layer_idx, "mean", None, sif_a, max_length, batch_size
    )
    train_emb1_sif = encode_sentences_layer(
        train_s1, tokenizer, model, layer_idx, "sif", token_probs, sif_a, max_length, batch_size
    )
    train_emb2_sif = encode_sentences_layer(
        train_s2, tokenizer, model, layer_idx, "sif", token_probs, sif_a, max_length, batch_size
    )

    # --- Encode test ---
    test_emb1_mean = encode_sentences_layer(
        test_s1, tokenizer, model, layer_idx, "mean", None, sif_a, max_length, batch_size
    )
    test_emb2_mean = encode_sentences_layer(
        test_s2, tokenizer, model, layer_idx, "mean", None, sif_a, max_length, batch_size
    )
    test_emb1_sif = encode_sentences_layer(
        test_s1, tokenizer, model, layer_idx, "sif", token_probs, sif_a, max_length, batch_size
    )
    test_emb2_sif = encode_sentences_layer(
        test_s2, tokenizer, model, layer_idx, "sif", token_probs, sif_a, max_length, batch_size
    )

    common = {
        "model": model_name,
        "language": language,
        "layer": layer_idx,
        "sif_a": sif_a,
        "n_train_pairs": len(train_labels),
        "n_test_pairs": len(test_labels),
    }

    # Method 1: baseline_mean
    sims_train = cosine_similarity(train_emb1_mean, train_emb2_mean)
    sims_test = cosine_similarity(test_emb1_mean, test_emb2_mean)
    results.append({
        **common,
        "method": "baseline_mean",
        "D": 0,
        "D_source": "none",
        "spearman_train": safe_spearman(sims_train, train_labels),
        "spearman_test": safe_spearman(sims_test, test_labels),
    })

    # Method 2: sif_only
    sims_train = cosine_similarity(train_emb1_sif, train_emb2_sif)
    sims_test = cosine_similarity(test_emb1_sif, test_emb2_sif)
    results.append({
        **common,
        "method": "sif_only",
        "D": 0,
        "D_source": "none",
        "spearman_train": safe_spearman(sims_train, train_labels),
        "spearman_test": safe_spearman(sims_test, test_labels),
    })

    # Method 3: sif_abtt_fixed (D=10)
    fixed_D = 10
    train_all_sif = np.vstack([train_emb1_sif, train_emb2_sif])
    cleaner_fixed = EmbeddingCleaner(num_components=fixed_D, center=True)
    cleaner_fixed.fit(train_all_sif)

    train_cleaned = cleaner_fixed.transform(train_all_sif)
    t1_c = train_cleaned[:len(train_emb1_sif)]
    t2_c = train_cleaned[len(train_emb1_sif):]
    sp_train_fixed = safe_spearman(cosine_similarity(t1_c, t2_c), train_labels)

    test_all_sif = np.vstack([test_emb1_sif, test_emb2_sif])
    test_cleaned = cleaner_fixed.transform(test_all_sif)
    e1_c = test_cleaned[:len(test_emb1_sif)]
    e2_c = test_cleaned[len(test_emb1_sif):]
    sp_test_fixed = safe_spearman(cosine_similarity(e1_c, e2_c), test_labels)

    results.append({
        **common,
        "method": "sif_abtt_fixed",
        "D": fixed_D,
        "D_source": "fixed",
        "spearman_train": sp_train_fixed,
        "spearman_test": sp_test_fixed,
    })

    # Method 4: sif_abtt_optimal (sweep R on train)
    best_D, r_results = sweep_R_on_train(
        train_emb1_sif, train_emb2_sif, train_labels, D_values, sif_a
    )
    cleaner_opt = EmbeddingCleaner(num_components=best_D, center=True)
    cleaner_opt.fit(train_all_sif)

    train_cleaned_opt = cleaner_opt.transform(train_all_sif)
    t1_o = train_cleaned_opt[:len(train_emb1_sif)]
    t2_o = train_cleaned_opt[len(train_emb1_sif):]
    sp_train_opt = safe_spearman(cosine_similarity(t1_o, t2_o), train_labels)

    test_cleaned_opt = cleaner_opt.transform(test_all_sif)
    e1_o = test_cleaned_opt[:len(test_emb1_sif)]
    e2_o = test_cleaned_opt[len(test_emb1_sif):]
    sp_test_opt = safe_spearman(cosine_similarity(e1_o, e2_o), test_labels)

    results.append({
        **common,
        "method": "sif_abtt_optimal",
        "D": best_D,
        "D_source": "train_sweep",
        "spearman_train": sp_train_opt,
        "spearman_test": sp_test_opt,
    })

    # Method 5: whitening (Katie's method)
    train_all_mean = np.vstack([train_emb1_mean, train_emb2_mean])
    try:
        pca_w = fit_whitening(train_all_mean)
        train_w = transform_whitening(pca_w, train_all_mean)
        t1_w = train_w[:len(train_emb1_mean)]
        t2_w = train_w[len(train_emb1_mean):]
        sp_train_w = safe_spearman(cosine_similarity(t1_w, t2_w), train_labels)

        test_all_mean = np.vstack([test_emb1_mean, test_emb2_mean])
        test_w = transform_whitening(pca_w, test_all_mean)
        e1_w = test_w[:len(test_emb1_mean)]
        e2_w = test_w[len(test_emb1_mean):]
        sp_test_w = safe_spearman(cosine_similarity(e1_w, e2_w), test_labels)
    except Exception as e:
        print(f"  WARNING: whitening failed for layer {layer_idx}: {e}")
        sp_train_w = float("nan")
        sp_test_w = float("nan")

    results.append({
        **common,
        "method": "whitening",
        "D": 0,
        "D_source": "none",
        "spearman_train": sp_train_w,
        "spearman_test": sp_test_w,
    })

    # Method 6: variance_filter (Katie's method)
    try:
        selected_dims = fit_variance_filter(train_all_mean, variance_drop_percent)
        train_vf = transform_variance_filter(train_all_mean, selected_dims)
        t1_vf = train_vf[:len(train_emb1_mean)]
        t2_vf = train_vf[len(train_emb1_mean):]
        sp_train_vf = safe_spearman(cosine_similarity(t1_vf, t2_vf), train_labels)

        test_all_mean = np.vstack([test_emb1_mean, test_emb2_mean])
        test_vf = transform_variance_filter(test_all_mean, selected_dims)
        e1_vf = test_vf[:len(test_emb1_mean)]
        e2_vf = test_vf[len(test_emb1_mean):]
        sp_test_vf = safe_spearman(cosine_similarity(e1_vf, e2_vf), test_labels)
    except Exception as e:
        print(f"  WARNING: variance_filter failed for layer {layer_idx}: {e}")
        sp_train_vf = float("nan")
        sp_test_vf = float("nan")

    results.append({
        **common,
        "method": "variance_filter",
        "D": 0,
        "D_source": "none",
        "spearman_train": sp_train_vf,
        "spearman_test": sp_test_vf,
    })

    return results


def append_rows(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


FIELDNAMES = [
    "model", "language", "layer", "method", "D", "D_source", "sif_a",
    "spearman_train", "spearman_test", "n_train_pairs", "n_test_pairs",
]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = parse_list(args.models)
    languages = parse_list(args.languages)
    D_values = parse_d_values(args.D_values)
    layers_arg = parse_layers(args.layers)

    scoreboard_path = out_dir / "phase8_musts_sweep.csv"

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if args.half_precision else None

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=args.trust_remote_code
        )
        model_kwargs = {"trust_remote_code": args.trust_remote_code}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        model = AutoModel.from_pretrained(model_name, **model_kwargs)
        model.to(device)
        model.eval()

        num_layers = detect_num_layers(model)
        layers = layers_arg if layers_arg else list(range(0, num_layers + 1))

        for language in languages:
            print(f"\n  Language: {language}")
            for layer_idx in layers:
                if layer_idx > num_layers:
                    print(f"    Skipping layer {layer_idx} (max={num_layers})")
                    continue
                print(f"    Layer {layer_idx}...")
                rows = evaluate_layer(
                    model=model,
                    tokenizer=tokenizer,
                    language=language,
                    layer_idx=layer_idx,
                    sif_a=args.sif_a,
                    D_values=D_values,
                    max_length=args.max_length,
                    batch_size=args.batch_size,
                    variance_drop_percent=args.variance_drop_percent,
                )
                if rows:
                    append_rows(scoreboard_path, rows, FIELDNAMES)

        # Free model memory before loading next
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"\nResults saved: {scoreboard_path}")


if __name__ == "__main__":
    main()
