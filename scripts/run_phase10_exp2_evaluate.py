"""Phase 10 Experiment 2: Core layer-wise evaluation on MUSTS.

Reads 50/50 split CSVs (from data_prep), encodes with caching,
evaluates 3 methods: baseline_mean, sif_abtt_optimal, whitening.

Models: Qwen/Qwen3-Embedding-8B (decoder), tencent/KaLM-Embedding-Gemma3-12B-2511 (decoder).

Additions vs Phase 9:
  A. verify_architecture() — prints arch summary after model.eval()
  B. already_evaluated() dedup guard — skips completed (model, lang, layer) triples
  C. --phase9_results_csv arg — consistent across scripts (unused in evaluate)

Outputs:
- results.csv (incremental append per layer)
- Cached embeddings under cache_dir/{slug}/{lang}/
- Per-pair cosine similarities for visualization
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from sif_abtt import (
    EmbeddingCleaner,
    sif_weights_from_ids,
    token_probabilities,
)


# ── CLI ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 10 Exp 2 evaluation.")
    parser.add_argument("--data_dir", required=True, help="Dir with {lang}_split.csv files.")
    parser.add_argument("--out_dir", required=True, help="Output directory.")
    parser.add_argument("--cache_dir", default="", help="Embedding cache dir (defaults to out_dir/cache).")
    parser.add_argument(
        "--models", default="Qwen/Qwen3-Embedding-8B",
        help="Comma-separated model names.",
    )
    parser.add_argument(
        "--languages", default="english,french,spanish,sinhala,tamil,serbian",
        help="Comma-separated language names.",
    )
    parser.add_argument("--sif_a", type=float, default=1e-3)
    parser.add_argument(
        "--D_values", default="1,2,3,5,7,10",
        help="Comma-separated D values for R sweep.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--layers", default="", help="Layer range, e.g. 0-50.")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--half_precision", action="store_true")
    parser.add_argument("--phase9_results_csv", default="",
                        help="Phase 9 results.csv path (unused in evaluate; kept for pipeline consistency).")
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


# ── Utilities ────────────────────────────────────────────────────

def detect_num_layers(model) -> int:
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return len(model.encoder.layer)
    if hasattr(model, "layers"):
        return len(model.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    raise ValueError(f"Cannot detect model layers. Attributes: {dir(model)}")


def verify_architecture(model, model_name: str) -> str:
    """Detect arch type and print summary. Returns arch_type string."""
    if hasattr(model, "encoder") and hasattr(model, "decoder"):
        arch = "encoder_decoder"
        n = len(model.encoder.block) if hasattr(model.encoder, "block") else len(model.encoder.layer)
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        arch = "encoder_only"
        n = len(model.encoder.layer)
    elif hasattr(model, "layers"):
        arch = "decoder_only"
        n = len(model.layers)
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        arch = "decoder_only"
        n = len(model.model.layers)
    else:
        raise ValueError(
            f"Unknown architecture for {model_name}. attrs: "
            f"{[a for a in dir(model) if not a.startswith('_')]}"
        )

    print(f"\n{'='*55}")
    print(f"Architecture Verification: {model_name}")
    print(f"  type       : {arch}")
    print(f"  num_layers : {n}  (index 0=embedding, 1..{n}=transformer)")
    print(f"  pooling    : mean/sif/last-token all valid (decoder)")
    if arch == "encoder_decoder":
        print(f"  WARNING    : encoder_decoder — hidden_states are encoder-side only")
    print(f"{'='*55}\n")
    return arch


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b_norm = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return np.sum(a_norm * b_norm, axis=1)


def safe_spearman(sims: np.ndarray, labels: np.ndarray) -> float:
    if len(sims) < 3:
        return float("nan")
    result = spearmanr(sims, labels)
    corr = result.correlation
    if corr is None or np.isnan(corr):
        return float("nan")
    return float(corr)


def model_slug(name: str) -> str:
    return name.replace("/", "_")


def already_evaluated(results_path: Path, model_name: str, language: str, layer_idx: int) -> bool:
    """True if all 3 methods already logged for this (model, lang, layer)."""
    if not results_path.exists():
        return False
    df = pd.read_csv(results_path)
    sub = df[
        (df["model"] == model_name) &
        (df["language"] == language) &
        (df["layer"] == layer_idx)
    ]
    return len(sub) >= 3


# ── Encoding with caching ───────────────────────────────────────

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


def cached_encode(
    sentences: List[str],
    tokenizer,
    model,
    layer_idx: int,
    pooling: str,
    token_probs: Optional[Dict[int, float]],
    sif_a: float,
    max_length: int,
    batch_size: int,
    cache_path: Path,
) -> np.ndarray:
    """Encode with file-level caching."""
    if cache_path.exists():
        print(f"      [cache hit] {cache_path.name}")
        return np.load(cache_path)
    emb = encode_sentences_layer(
        sentences, tokenizer, model, layer_idx, pooling,
        token_probs, sif_a, max_length, batch_size,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, emb)
    print(f"      [cached] {cache_path.name}")
    return emb


# ── Whitening ────────────────────────────────────────────────────

def fit_whitening(train_emb: np.ndarray) -> PCA:
    pca = PCA(whiten=True)
    pca.fit(train_emb)
    return pca


def transform_whitening(pca: PCA, emb: np.ndarray) -> np.ndarray:
    return pca.transform(emb)


# ── R-value sweep ───────────────────────────────────────────────

def sweep_R_on_train(
    train_emb1: np.ndarray,
    train_emb2: np.ndarray,
    train_labels: np.ndarray,
    D_values: List[int],
    sif_a: float,
) -> Tuple[int, List[Dict]]:
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


# ── CSV append ──────────────────────────────────────────────────

FIELDNAMES = [
    "model", "language", "layer", "method", "D", "D_source", "sif_a",
    "spearman_train", "spearman_test", "n_train_pairs", "n_test_pairs",
]


def append_rows(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


# ── Per-language evaluation ─────────────────────────────────────

def evaluate_language_layer(
    model,
    tokenizer,
    model_name: str,
    language: str,
    layer_idx: int,
    train_s1: List[str],
    train_s2: List[str],
    train_labels: np.ndarray,
    test_s1: List[str],
    test_s2: List[str],
    test_labels: np.ndarray,
    sif_a: float,
    D_values: List[int],
    max_length: int,
    batch_size: int,
    cache_dir: Path,
) -> List[Dict]:
    """Evaluate one (model, language, layer) triple with 3 methods."""
    slug = model_slug(model_name)
    lang_cache = cache_dir / slug / language

    # P(w) from train sentences only
    train_texts = train_s1 + train_s2
    token_probs = token_probabilities(
        tokenizer, train_texts, batch_size=batch_size, max_length=max_length
    )

    # Encode train (mean + sif)
    train_emb1_mean = cached_encode(
        train_s1, tokenizer, model, layer_idx, "mean", None, sif_a,
        max_length, batch_size, lang_cache / f"layer{layer_idx}_mean_train_s1.npy",
    )
    train_emb2_mean = cached_encode(
        train_s2, tokenizer, model, layer_idx, "mean", None, sif_a,
        max_length, batch_size, lang_cache / f"layer{layer_idx}_mean_train_s2.npy",
    )
    train_emb1_sif = cached_encode(
        train_s1, tokenizer, model, layer_idx, "sif", token_probs, sif_a,
        max_length, batch_size, lang_cache / f"layer{layer_idx}_sif_train_s1.npy",
    )
    train_emb2_sif = cached_encode(
        train_s2, tokenizer, model, layer_idx, "sif", token_probs, sif_a,
        max_length, batch_size, lang_cache / f"layer{layer_idx}_sif_train_s2.npy",
    )

    # Encode test (mean + sif)
    test_emb1_mean = cached_encode(
        test_s1, tokenizer, model, layer_idx, "mean", None, sif_a,
        max_length, batch_size, lang_cache / f"layer{layer_idx}_mean_test_s1.npy",
    )
    test_emb2_mean = cached_encode(
        test_s2, tokenizer, model, layer_idx, "mean", None, sif_a,
        max_length, batch_size, lang_cache / f"layer{layer_idx}_mean_test_s2.npy",
    )
    test_emb1_sif = cached_encode(
        test_s1, tokenizer, model, layer_idx, "sif", token_probs, sif_a,
        max_length, batch_size, lang_cache / f"layer{layer_idx}_sif_test_s1.npy",
    )
    test_emb2_sif = cached_encode(
        test_s2, tokenizer, model, layer_idx, "sif", token_probs, sif_a,
        max_length, batch_size, lang_cache / f"layer{layer_idx}_sif_test_s2.npy",
    )

    common = {
        "model": model_name,
        "language": language,
        "layer": layer_idx,
        "sif_a": sif_a,
        "n_train_pairs": len(train_labels),
        "n_test_pairs": len(test_labels),
    }

    results = []

    # Method 1: baseline_mean
    sims_train = cosine_similarity(train_emb1_mean, train_emb2_mean)
    sims_test = cosine_similarity(test_emb1_mean, test_emb2_mean)
    # Save per-pair cosine sims for visualization (test only)
    np.save(lang_cache / f"layer{layer_idx}_baseline_mean_pair_sims.npy", sims_test)
    results.append({
        **common,
        "method": "baseline_mean",
        "D": 0,
        "D_source": "none",
        "spearman_train": safe_spearman(sims_train, train_labels),
        "spearman_test": safe_spearman(sims_test, test_labels),
    })

    # Method 2: sif_abtt_optimal (sweep R on train)
    best_D, _ = sweep_R_on_train(
        train_emb1_sif, train_emb2_sif, train_labels, D_values, sif_a
    )
    train_all_sif = np.vstack([train_emb1_sif, train_emb2_sif])
    cleaner_opt = EmbeddingCleaner(num_components=best_D, center=True)
    cleaner_opt.fit(train_all_sif)

    train_cleaned = cleaner_opt.transform(train_all_sif)
    n_tr = len(train_emb1_sif)
    t1_o = train_cleaned[:n_tr]
    t2_o = train_cleaned[n_tr:]
    sp_train_opt = safe_spearman(cosine_similarity(t1_o, t2_o), train_labels)

    test_all_sif = np.vstack([test_emb1_sif, test_emb2_sif])
    test_cleaned = cleaner_opt.transform(test_all_sif)
    n_te = len(test_emb1_sif)
    e1_o = test_cleaned[:n_te]
    e2_o = test_cleaned[n_te:]
    sims_test_opt = cosine_similarity(e1_o, e2_o)
    np.save(lang_cache / f"layer{layer_idx}_sif_abtt_optimal_pair_sims.npy", sims_test_opt)
    results.append({
        **common,
        "method": "sif_abtt_optimal",
        "D": best_D,
        "D_source": "train_sweep",
        "spearman_train": sp_train_opt,
        "spearman_test": safe_spearman(sims_test_opt, test_labels),
    })

    # Method 3: whitening (PCA whiten=True on mean embeddings)
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
        sims_test_w = cosine_similarity(e1_w, e2_w)
        np.save(lang_cache / f"layer{layer_idx}_whitening_pair_sims.npy", sims_test_w)
        sp_test_w = safe_spearman(sims_test_w, test_labels)
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

    return results


# ── Main ─────────────────────────────────────────────────────────

def load_split_csv(data_dir: Path, language: str):
    """Load language split CSV and return (train_s1, train_s2, train_labels, test_s1, test_s2, test_labels)."""
    df = pd.read_csv(data_dir / f"{language}_split.csv")
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]

    return (
        train["sentence_1"].tolist(),
        train["sentence_2"].tolist(),
        train["similarity"].values.astype(np.float32),
        test["sentence_1"].tolist(),
        test["sentence_2"].tolist(),
        test["similarity"].values.astype(np.float32),
    )


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "cache"

    models = parse_list(args.models)
    languages = parse_list(args.languages)
    D_values = parse_d_values(args.D_values)
    layers_arg = parse_layers(args.layers)

    results_path = out_dir / "results.csv"

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

        # Architecture verification (Addition A)
        verify_architecture(model, model_name)

        num_layers = detect_num_layers(model)
        layers = layers_arg if layers_arg else list(range(0, num_layers + 1))
        print(f"  Detected {num_layers} layers, evaluating: {layers}")

        for language in languages:
            print(f"\n  Language: {language}")
            try:
                train_s1, train_s2, train_labels, test_s1, test_s2, test_labels = \
                    load_split_csv(data_dir, language)
            except FileNotFoundError:
                print(f"    WARNING: {language}_split.csv not found, skipping.")
                continue

            print(f"    Train: {len(train_labels)} pairs, Test: {len(test_labels)} pairs")

            for layer_idx in layers:
                if layer_idx > num_layers:
                    print(f"    Skipping layer {layer_idx} (max={num_layers})")
                    continue
                # Dedup guard (Addition B)
                if already_evaluated(results_path, model_name, language, layer_idx):
                    print(f"    [skip] Layer {layer_idx} already evaluated.")
                    continue
                print(f"    Layer {layer_idx}...")
                rows = evaluate_language_layer(
                    model=model,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    language=language,
                    layer_idx=layer_idx,
                    train_s1=train_s1,
                    train_s2=train_s2,
                    train_labels=train_labels,
                    test_s1=test_s1,
                    test_s2=test_s2,
                    test_labels=test_labels,
                    sif_a=args.sif_a,
                    D_values=D_values,
                    max_length=args.max_length,
                    batch_size=args.batch_size,
                    cache_dir=cache_dir,
                )
                if rows:
                    append_rows(results_path, rows, FIELDNAMES)

        # Free model memory before loading next
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
