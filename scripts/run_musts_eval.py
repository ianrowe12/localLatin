from __future__ import annotations

"""Phase 7 MUSTS evaluator (Spearman). [MUSTS Paper, Table 2]"""

import sys

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import spearmanr
from transformers import AutoModel, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from sif_abtt import remove_top_components, sif_weights_from_ids, token_probabilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 7 MUSTS evaluation.")
    parser.add_argument(
        "--languages", default="sinhala,tamil,english,french", help="Comma-separated."
    )
    parser.add_argument("--model_name", default="xlm-roberta-base")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sif_a", type=float, default=1e-3)
    parser.add_argument("--phase7_D", type=int, default=10)
    parser.add_argument("--phase7_use_sif", type=int, default=1)
    return parser.parse_args()


def parse_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@torch.no_grad()
def encode_sentences(
    sentences: List[str],
    tokenizer,
    model,
    pooling: str,
    token_probs: Dict[int, float] | None,
    sif_a: float,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    device = next(model.parameters()).device
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    embeddings: List[np.ndarray] = []
    for start in range(0, len(sentences), batch_size):
        batch = sentences[start : start + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = outputs.last_hidden_state

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

        embeddings.append(pooled.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(embeddings, axis=0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b_norm = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return np.sum(a_norm * b_norm, axis=1)


def run_language(
    language: str,
    tokenizer,
    model,
    sif_a: float,
    phase7_D: int,
    phase7_use_sif: bool,
    max_length: int,
    batch_size: int,
) -> List[Dict[str, object]]:
    dataset_name = f"musts/{language}"
    test_set = load_dataset(dataset_name, split="test")

    s1 = [row["sentence_1"] for row in test_set]
    s2 = [row["sentence_2"] for row in test_set]
    labels = np.array([row["similarity"] for row in test_set], dtype=np.float32)

    all_texts = s1 + s2
    token_probs = token_probabilities(
        tokenizer, all_texts, batch_size=batch_size, max_length=max_length
    )

    results: List[Dict[str, object]] = []

    # Method A: Base mean pooling
    emb1_base = encode_sentences(
        s1, tokenizer, model, "mean", None, sif_a, max_length, batch_size
    )
    emb2_base = encode_sentences(
        s2, tokenizer, model, "mean", None, sif_a, max_length, batch_size
    )
    sims_base = cosine_similarity(emb1_base, emb2_base)
    results.append(
        {
            "language": language,
            "method": "base_mean",
            "spearman": float(spearmanr(sims_base, labels).correlation),
        }
    )

    # Method B: SIF official (weighted + remove top 1 PC, no centering)
    emb1_sif = encode_sentences(
        s1, tokenizer, model, "sif", token_probs, sif_a, max_length, batch_size
    )
    emb2_sif = encode_sentences(
        s2, tokenizer, model, "sif", token_probs, sif_a, max_length, batch_size
    )
    emb_all = np.vstack([emb1_sif, emb2_sif])
    cleaned_all, _, pcs = remove_top_components(emb_all, 1, center=False)
    emb1_sif_clean = cleaned_all[: len(emb1_sif)]
    emb2_sif_clean = cleaned_all[len(emb1_sif) :]
    sims_sif = cosine_similarity(emb1_sif_clean, emb2_sif_clean)
    results.append(
        {
            "language": language,
            "method": "sif_official_r1",
            "spearman": float(spearmanr(sims_sif, labels).correlation),
        }
    )

    # Method C: Phase 7 (weighted optional + remove top D with centering)
    if phase7_use_sif:
        emb1_p7 = emb1_sif
        emb2_p7 = emb2_sif
    else:
        emb1_p7 = emb1_base
        emb2_p7 = emb2_base
    emb_all_p7 = np.vstack([emb1_p7, emb2_p7])
    cleaned_p7, _, _ = remove_top_components(emb_all_p7, phase7_D, center=True)
    emb1_p7_clean = cleaned_p7[: len(emb1_p7)]
    emb2_p7_clean = cleaned_p7[len(emb1_p7) :]
    sims_p7 = cosine_similarity(emb1_p7_clean, emb2_p7_clean)
    results.append(
        {
            "language": language,
            "method": f"phase7_r{phase7_D}",
            "spearman": float(spearmanr(sims_p7, labels).correlation),
        }
    )

    return results


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()

    languages = parse_list(args.languages)
    all_rows: List[Dict[str, object]] = []
    for lang in languages:
        rows = run_language(
            language=lang,
            tokenizer=tokenizer,
            model=model,
            sif_a=args.sif_a,
            phase7_D=args.phase7_D,
            phase7_use_sif=bool(args.phase7_use_sif),
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
        all_rows.extend(rows)

    out_path = out_dir / "phase7_musts_results.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"MUSTS results saved: {out_path}")


if __name__ == "__main__":
    main()
