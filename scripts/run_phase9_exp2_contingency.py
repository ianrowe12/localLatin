"""Phase 9.5 Contingency: Last-token pooling + Optimal Transport.

Only run if main Exp 2 results warrant it (submitted manually).

5a: EOS/last-token pooling — re-encode at best layer, apply baseline + SIF+ABTT
5b: Optimal Transport (EMD) — use all token embeddings, no pooling

Reports: Performance Delta, Recovery Rate (gap closed toward LaBSE).
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
from transformers import AutoModel, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from sif_abtt import (
    EmbeddingCleaner,
    sif_weights_from_ids,
    token_probabilities,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 9.5 contingency experiments.")
    parser.add_argument("--data_dir", required=True, help="Dir with {lang}_split.csv files.")
    parser.add_argument("--results_csv", required=True, help="results.csv from Exp 2 main.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--sif_a", type=float, default=1e-3)
    parser.add_argument("--D_values", default="1,2,3,5,7,10")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--skip_ot", action="store_true", help="Skip OT (requires pot).")
    return parser.parse_args()


def parse_d_values(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def model_slug(name: str) -> str:
    return name.replace("/", "_")


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


def detect_num_layers(model) -> int:
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return len(model.encoder.layer)
    if hasattr(model, "layers"):
        return len(model.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    raise ValueError(f"Cannot detect model layers.")


# ── Last-token encoding ─────────────────────────────────────────

@torch.no_grad()
def encode_lasttok_layer(
    sentences: List[str],
    tokenizer,
    model,
    layer_idx: int,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    """Encode sentences using the last non-padding token's hidden state."""
    device = next(model.parameters()).device
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

        # Last non-padding token per sequence
        lengths = attention_mask.sum(dim=1) - 1  # 0-indexed
        batch_indices = torch.arange(hidden.size(0), device=device)
        pooled = hidden[batch_indices, lengths]

        embeddings.append(pooled.detach().cpu().float().numpy().astype(np.float32))
    return np.concatenate(embeddings, axis=0)


# ── Token-level encoding (for OT) ───────────────────────────────

@torch.no_grad()
def encode_all_tokens_layer(
    sentences: List[str],
    tokenizer,
    model,
    layer_idx: int,
    max_length: int,
    batch_size: int,
) -> List[np.ndarray]:
    """Return list of [seq_len, dim] arrays (one per sentence, no padding)."""
    device = next(model.parameters()).device
    all_token_embs: List[np.ndarray] = []

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
        hidden = outputs.hidden_states[layer_idx].detach().cpu().float().numpy()
        mask = attention_mask.detach().cpu().numpy()

        for idx in range(hidden.shape[0]):
            length = int(mask[idx].sum())
            all_token_embs.append(hidden[idx, :length].astype(np.float32))

    return all_token_embs


# ── Optimal Transport distance ───────────────────────────────────

def ot_distance(tok_embs_a: np.ndarray, tok_embs_b: np.ndarray) -> float:
    """Compute Earth Mover's Distance between two sets of token embeddings."""
    import ot as pot_lib

    n_a, n_b = tok_embs_a.shape[0], tok_embs_b.shape[0]
    # Uniform weights
    a_weights = np.ones(n_a, dtype=np.float64) / n_a
    b_weights = np.ones(n_b, dtype=np.float64) / n_b

    # Cosine distance matrix
    a_norm = tok_embs_a / np.maximum(np.linalg.norm(tok_embs_a, axis=1, keepdims=True), 1e-12)
    b_norm = tok_embs_b / np.maximum(np.linalg.norm(tok_embs_b, axis=1, keepdims=True), 1e-12)
    cos_sim = a_norm @ b_norm.T
    cost = (1.0 - cos_sim).astype(np.float64)
    cost = np.maximum(cost, 0.0)

    return float(pot_lib.emd2(a_weights, b_weights, cost))


def ot_similarity_pairs(
    tok_embs_s1: List[np.ndarray],
    tok_embs_s2: List[np.ndarray],
) -> np.ndarray:
    """Compute 1 - EMD for each pair."""
    sims = np.zeros(len(tok_embs_s1), dtype=np.float32)
    for i in range(len(tok_embs_s1)):
        dist = ot_distance(tok_embs_s1[i], tok_embs_s2[i])
        sims[i] = 1.0 - dist
    return sims


# ── Sweep D on train ────────────────────────────────────────────

def sweep_R_on_train(
    train_emb1: np.ndarray,
    train_emb2: np.ndarray,
    train_labels: np.ndarray,
    D_values: List[int],
) -> int:
    train_all = np.vstack([train_emb1, train_emb2])
    best_d = D_values[0]
    best_sp = -1.0
    for D in D_values:
        cleaner = EmbeddingCleaner(num_components=D, center=True)
        cleaner.fit(train_all)
        cleaned = cleaner.transform(train_all)
        n = len(train_emb1)
        sims = cosine_similarity(cleaned[:n], cleaned[n:])
        sp = safe_spearman(sims, train_labels)
        if not np.isnan(sp) and sp > best_sp:
            best_sp = sp
            best_d = D
    return best_d


# ── CSV helpers ──────────────────────────────────────────────────

FIELDNAMES = [
    "model", "language", "layer", "method", "pooling", "D",
    "spearman_test", "baseline_spearman", "performance_delta",
    "labse_spearman", "recovery_rate",
    "n_test_pairs",
]


def append_rows(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


# ── Main ─────────────────────────────────────────────────────────

def main() -> None:
    import pandas as pd

    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    D_values = parse_d_values(args.D_values)

    results_df = pd.read_csv(args.results_csv)
    contingency_path = out_dir / "contingency_results.csv"

    # Find best layer per (model, language) from main results (sif_abtt_optimal)
    abtt = results_df[results_df["method"] == "sif_abtt_optimal"]
    best_layers = {}
    for _, row in abtt.iterrows():
        key = (row["model"], row["language"])
        if key not in best_layers or row["spearman_test"] > best_layers[key]["spearman_test"]:
            best_layers[key] = row.to_dict()

    # Get LaBSE baseline Spearman per language (best layer, baseline_mean)
    labse_baseline = {}
    labse_rows = results_df[
        (results_df["model"] == "sentence-transformers/LaBSE") &
        (results_df["method"] == "baseline_mean")
    ]
    for lang in results_df["language"].unique():
        lang_rows = labse_rows[labse_rows["language"] == lang]
        if not lang_rows.empty:
            labse_baseline[lang] = lang_rows["spearman_test"].max()

    # Get baseline_mean per (model, language) at best layer
    mean_baseline = {}
    for (model_name, lang), info in best_layers.items():
        layer = int(info["layer"])
        bl = results_df[
            (results_df["model"] == model_name) &
            (results_df["language"] == lang) &
            (results_df["layer"] == layer) &
            (results_df["method"] == "baseline_mean")
        ]
        if not bl.empty:
            mean_baseline[(model_name, lang)] = bl["spearman_test"].values[0]

    # Only process decoder models (not LaBSE)
    decoder_models = sorted(set(
        m for m, _ in best_layers.keys()
        if m != "sentence-transformers/LaBSE"
    ))

    for model_name in decoder_models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=args.trust_remote_code
        )
        model = AutoModel.from_pretrained(
            model_name, trust_remote_code=args.trust_remote_code
        )
        model.to(device)
        model.eval()

        languages = sorted(set(
            lang for m, lang in best_layers.keys() if m == model_name
        ))

        for language in languages:
            key = (model_name, language)
            if key not in best_layers:
                continue
            best_layer = int(best_layers[key]["layer"])
            print(f"\n  {language} — best layer: {best_layer}")

            # Load data
            csv_path = data_dir / f"{language}_split.csv"
            if not csv_path.exists():
                print(f"    WARNING: {csv_path} not found, skipping.")
                continue
            df = pd.read_csv(csv_path)
            train = df[df["split"] == "train"]
            test = df[df["split"] == "test"]
            train_s1 = train["sentence_1"].tolist()
            train_s2 = train["sentence_2"].tolist()
            train_labels = train["similarity"].values.astype(np.float32)
            test_s1 = test["sentence_1"].tolist()
            test_s2 = test["sentence_2"].tolist()
            test_labels = test["similarity"].values.astype(np.float32)

            baseline_sp = mean_baseline.get(key, float("nan"))
            labse_sp = labse_baseline.get(language, float("nan"))

            # 5a: Last-token pooling
            print(f"    Encoding last-token...")
            train_lt1 = encode_lasttok_layer(
                train_s1, tokenizer, model, best_layer, args.max_length, args.batch_size
            )
            train_lt2 = encode_lasttok_layer(
                train_s2, tokenizer, model, best_layer, args.max_length, args.batch_size
            )
            test_lt1 = encode_lasttok_layer(
                test_s1, tokenizer, model, best_layer, args.max_length, args.batch_size
            )
            test_lt2 = encode_lasttok_layer(
                test_s2, tokenizer, model, best_layer, args.max_length, args.batch_size
            )

            # Baseline on last-token
            sims_lt = cosine_similarity(test_lt1, test_lt2)
            sp_lt_baseline = safe_spearman(sims_lt, test_labels)
            delta_lt = sp_lt_baseline - baseline_sp
            recovery_lt = (sp_lt_baseline - baseline_sp) / (labse_sp - baseline_sp) if labse_sp != baseline_sp else 0.0

            rows = [{
                "model": model_name, "language": language, "layer": best_layer,
                "method": "baseline", "pooling": "lasttok", "D": 0,
                "spearman_test": sp_lt_baseline,
                "baseline_spearman": baseline_sp,
                "performance_delta": delta_lt,
                "labse_spearman": labse_sp,
                "recovery_rate": recovery_lt,
                "n_test_pairs": len(test_labels),
            }]

            # SIF+ABTT on last-token (SIF doesn't apply to single-token, use ABTT only)
            best_D = sweep_R_on_train(train_lt1, train_lt2, train_labels, D_values)
            all_train = np.vstack([train_lt1, train_lt2])
            cleaner = EmbeddingCleaner(num_components=best_D, center=True)
            cleaner.fit(all_train)
            all_test = np.vstack([test_lt1, test_lt2])
            cleaned = cleaner.transform(all_test)
            n = len(test_lt1)
            sims_lt_abtt = cosine_similarity(cleaned[:n], cleaned[n:])
            sp_lt_abtt = safe_spearman(sims_lt_abtt, test_labels)
            delta_lt_abtt = sp_lt_abtt - baseline_sp
            recovery_lt_abtt = (sp_lt_abtt - baseline_sp) / (labse_sp - baseline_sp) if labse_sp != baseline_sp else 0.0

            rows.append({
                "model": model_name, "language": language, "layer": best_layer,
                "method": "abtt_optimal", "pooling": "lasttok", "D": best_D,
                "spearman_test": sp_lt_abtt,
                "baseline_spearman": baseline_sp,
                "performance_delta": delta_lt_abtt,
                "labse_spearman": labse_sp,
                "recovery_rate": recovery_lt_abtt,
                "n_test_pairs": len(test_labels),
            })

            print(f"    Last-token baseline: {sp_lt_baseline:.4f} (delta={delta_lt:+.4f})")
            print(f"    Last-token ABTT(D={best_D}): {sp_lt_abtt:.4f} (delta={delta_lt_abtt:+.4f})")

            # 5b: Optimal Transport
            if not args.skip_ot:
                try:
                    import ot as pot_lib  # noqa: F401
                    print(f"    Computing OT distances (this may be slow)...")
                    test_tok1 = encode_all_tokens_layer(
                        test_s1, tokenizer, model, best_layer, args.max_length, args.batch_size
                    )
                    test_tok2 = encode_all_tokens_layer(
                        test_s2, tokenizer, model, best_layer, args.max_length, args.batch_size
                    )
                    ot_sims = ot_similarity_pairs(test_tok1, test_tok2)
                    sp_ot = safe_spearman(ot_sims, test_labels)
                    delta_ot = sp_ot - baseline_sp
                    recovery_ot = (sp_ot - baseline_sp) / (labse_sp - baseline_sp) if labse_sp != baseline_sp else 0.0

                    rows.append({
                        "model": model_name, "language": language, "layer": best_layer,
                        "method": "optimal_transport", "pooling": "none", "D": 0,
                        "spearman_test": sp_ot,
                        "baseline_spearman": baseline_sp,
                        "performance_delta": delta_ot,
                        "labse_spearman": labse_sp,
                        "recovery_rate": recovery_ot,
                        "n_test_pairs": len(test_labels),
                    })
                    print(f"    OT: {sp_ot:.4f} (delta={delta_ot:+.4f})")
                except ImportError:
                    print("    WARNING: pot not installed, skipping OT.")

            append_rows(contingency_path, rows, FIELDNAMES)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nContingency results saved: {contingency_path}")


if __name__ == "__main__":
    main()
