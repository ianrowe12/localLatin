"""Phase 8: Canon re-validation sweep with clean train/test split.

Enforces leak-free protocol throughout:
- P(w) from train files only (via --split_csv in extraction CLIs)
- EmbeddingCleaner fitted on train embeddings only
- Evaluation on test query files searching against full gallery
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from canon_retrieval import (
    accuracy_at_k,
    l2_normalize,
    sanity_checks,
    similarity_matrix,
    upper_triangle,
    upper_triangle_labels,
)
from sif_abtt import EmbeddingCleaner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 8 canon sweep (clean protocol).")
    parser.add_argument("--split_csv", required=True, help="Locked split CSV.")
    parser.add_argument("--runs_root", required=True, help="Root for run dirs.")
    parser.add_argument("--out_dir", required=True, help="Output directory.")
    parser.add_argument("--models", default="bowphs/LaTa,bowphs/PhilTa")
    parser.add_argument("--reprs", default="hidden,ff1")
    parser.add_argument("--pooling", default="mean")
    parser.add_argument("--layers_hidden", default="0-12")
    parser.add_argument("--layers_ff1", default="1-12")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--D", type=int, default=10)
    parser.add_argument("--sif_a", type=float, default=1e-3)
    parser.add_argument(
        "--encoder_models", default="",
        help="Comma-separated encoder-only models (e.g. sentence-transformers/LaBSE).",
    )
    parser.add_argument("--encoder_layers", default="0-12")
    parser.add_argument(
        "--half_precision", action="store_true",
        help="Use fp16 for encoder models.",
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


def model_slug(name: str) -> str:
    return name.replace("/", "_")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def embedding_path(run_dir: Path, repr_name: str, pooling: str, layer: int) -> Path:
    suffix = "" if pooling == "mean" else "_lasttok" if pooling == "lasttok" else "_sif"
    return run_dir / f"{repr_name}_layer{layer}_embeddings{suffix}.npy"


def norm_path(run_dir: Path, repr_name: str, pooling: str, layer: int) -> Path:
    suffix = "" if pooling == "mean" else "_lasttok" if pooling == "lasttok" else "_sif"
    return run_dir / f"{repr_name}_layer{layer}_embeddings{suffix}_norm.npy"


def ensure_embeddings(run_dir: Path, repr_name: str, pooling: str, layers: List[int]) -> List[int]:
    missing = []
    for layer in layers:
        if not norm_path(run_dir, repr_name, pooling, layer).exists():
            missing.append(layer)
    return missing


def run_seq2seq_extract(
    repr_name: str,
    meta_csv: Path,
    run_dir: Path,
    model_name: str,
    layers: str,
    pooling: str,
    max_length: int,
    batch_size: int,
    sif_a: float,
    split_csv: str,
) -> None:
    cmd = [
        "python", f"src/extract_{repr_name}_cli.py",
        "--meta_csv", str(meta_csv),
        "--runs_root", str(run_dir.parent),
        "--run_dir", str(run_dir),
        "--model_name", model_name,
        "--layers", layers,
        "--pooling", pooling,
        "--max_length", str(max_length),
        "--batch_size", str(batch_size),
    ]
    if pooling == "sif":
        cmd += ["--sif_a", str(sif_a)]
    if split_csv:
        cmd += ["--split_csv", split_csv]
    subprocess.run(cmd, check=True)


def run_encoder_extract(
    meta_csv: Path,
    run_dir: Path,
    model_name: str,
    repr_mode: str,
    layers: str,
    pooling: str,
    max_length: int,
    batch_size: int,
    sif_a: float,
    split_csv: str,
    half_precision: bool,
) -> None:
    cmd = [
        "python", "src/extract_encoder_cli.py",
        "--meta_csv", str(meta_csv),
        "--runs_root", str(run_dir.parent),
        "--run_dir", str(run_dir),
        "--model_name", model_name,
        "--repr", repr_mode,
        "--layers", layers,
        "--pooling", pooling,
        "--max_length", str(max_length),
        "--batch_size", str(batch_size),
    ]
    if pooling == "sif":
        cmd += ["--sif_a", str(sif_a)]
    if split_csv:
        cmd += ["--split_csv", split_csv]
    if half_precision:
        cmd += ["--half_precision"]
    subprocess.run(cmd, check=True)


def evaluate_embeddings(
    emb_norm: np.ndarray,
    folder_ids: np.ndarray,
    query_mask: np.ndarray,
    is_winnable: np.ndarray,
) -> Dict:
    """Run retrieval evaluation metrics."""
    sim = similarity_matrix(emb_norm)
    checks = sanity_checks(sim)

    acc1 = accuracy_at_k(sim, folder_ids, query_mask, k=1)
    acc3 = accuracy_at_k(sim, folder_ids, query_mask, k=3)
    acc5 = accuracy_at_k(sim, folder_ids, query_mask, k=5)
    acc1_win = accuracy_at_k(sim, folder_ids, query_mask & is_winnable, k=1)
    acc3_win = accuracy_at_k(sim, folder_ids, query_mask & is_winnable, k=3)
    acc5_win = accuracy_at_k(sim, folder_ids, query_mask & is_winnable, k=5)

    sim_upper = upper_triangle(sim)
    labels = upper_triangle_labels(folder_ids)
    same_avg = float(sim_upper[labels].mean()) if labels.any() else float("nan")
    diff_avg = float(sim_upper[~labels].mean()) if (~labels).any() else float("nan")

    return {
        "acc@1_all": acc1,
        "acc@3_all": acc3,
        "acc@5_all": acc5,
        "acc@1_winnable": acc1_win,
        "acc@3_winnable": acc3_win,
        "acc@5_winnable": acc5_win,
        "same_avg": same_avg,
        "diff_avg": diff_avg,
        "gap": same_avg - diff_avg,
        "off_diag_mean": checks["off_diag_mean"],
    }


def apply_abtt_clean(
    emb: np.ndarray,
    train_mask: np.ndarray,
    num_components: int,
) -> np.ndarray:
    """Fit EmbeddingCleaner on train, transform all."""
    train_emb = emb[train_mask]
    cleaner = EmbeddingCleaner(num_components=num_components, center=True)
    cleaner.fit(train_emb)
    cleaned = cleaner.transform(emb)
    return cleaned


def append_row(path: Path, row: Dict) -> None:
    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


FIELDNAMES = [
    "model", "repr", "pooling", "layer", "method", "D", "sif_a",
    "split", "n_train", "n_test_query",
    "acc@1_all", "acc@3_all", "acc@5_all",
    "acc@1_winnable", "acc@3_winnable", "acc@5_winnable",
    "same_avg", "diff_avg", "gap", "off_diag_mean",
]


def main() -> None:
    args = parse_args()
    split_csv = Path(args.split_csv)
    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    split_meta = pd.read_csv(split_csv)
    meta_csv = split_csv  # use split CSV as meta (has all needed columns)

    folder_ids = split_meta["folder_id"].to_numpy()
    is_winnable = split_meta["is_winnable"].to_numpy(dtype=bool)
    is_test_query = split_meta["is_test_query"].to_numpy(dtype=bool)
    train_mask = split_meta["split"].values == "train"

    n_train = int(train_mask.sum())
    n_test_query = int(is_test_query.sum())

    scoreboard_path = out_dir / "phase8_canon_sweep.csv"

    models = parse_list(args.models)
    reprs = parse_list(args.reprs)
    layers_hidden = parse_layers(args.layers_hidden)
    layers_ff1 = parse_layers(args.layers_ff1)

    # --- Seq2Seq models (LaTa, PhilTa) ---
    for model_name in models:
        print(f"\nModel: {model_name}")
        base_dir_root = runs_root / "phase8_bases" / model_slug(model_name)
        derived_root = out_dir / "derived" / model_slug(model_name)

        for repr_name in reprs:
            layers = layers_hidden if repr_name == "hidden" else layers_ff1

            for pooling in ["mean", "sif"]:
                run_dir = base_dir_root / f"{repr_name}_{pooling}"
                ensure_dir(run_dir)

                missing = ensure_embeddings(run_dir, repr_name, pooling, layers)
                if missing:
                    print(f"  Extracting {repr_name}/{pooling} layers {missing}...")
                    run_seq2seq_extract(
                        repr_name=repr_name,
                        meta_csv=meta_csv,
                        run_dir=run_dir,
                        model_name=model_name,
                        layers=",".join(str(x) for x in missing),
                        pooling=pooling,
                        max_length=args.max_length,
                        batch_size=args.batch_size,
                        sif_a=args.sif_a,
                        split_csv=str(split_csv),
                    )

                for layer in layers:
                    emb_p = norm_path(run_dir, repr_name, pooling, layer)
                    if not emb_p.exists():
                        print(f"  WARNING: missing {emb_p}, skipping")
                        continue
                    emb_norm = np.load(emb_p)

                    # Baseline
                    metrics = evaluate_embeddings(emb_norm, folder_ids, is_test_query, is_winnable)
                    row = {
                        "model": model_name, "repr": repr_name, "pooling": pooling,
                        "layer": layer, "method": "baseline", "D": 0, "sif_a": args.sif_a,
                        "split": "clean", "n_train": n_train, "n_test_query": n_test_query,
                        **metrics,
                    }
                    append_row(scoreboard_path, row)

                    # ABTT (fit on train only)
                    emb_raw = np.load(embedding_path(run_dir, repr_name, pooling, layer))
                    cleaned = apply_abtt_clean(emb_raw, train_mask, args.D)
                    cleaned_norm = l2_normalize(cleaned)
                    metrics_abtt = evaluate_embeddings(
                        cleaned_norm, folder_ids, is_test_query, is_winnable
                    )
                    row_abtt = {
                        "model": model_name, "repr": repr_name, "pooling": pooling,
                        "layer": layer, "method": "abtt", "D": args.D, "sif_a": args.sif_a,
                        "split": "clean", "n_train": n_train, "n_test_query": n_test_query,
                        **metrics_abtt,
                    }
                    append_row(scoreboard_path, row_abtt)

                    print(f"  {repr_name}/{pooling}/L{layer}: "
                          f"baseline={metrics['acc@1_winnable']:.3f} "
                          f"abtt={metrics_abtt['acc@1_winnable']:.3f}")

    # --- Encoder-only models (LaBSE, etc.) ---
    encoder_models = parse_list(args.encoder_models)
    encoder_layers = parse_layers(args.encoder_layers)

    for model_name in encoder_models:
        print(f"\nEncoder model: {model_name}")
        base_dir_root = runs_root / "phase8_bases" / model_slug(model_name)

        for repr_mode in ["hidden", "ffn_intermediate"]:
            repr_prefix = "hidden" if repr_mode == "hidden" else "ffn_int"

            for pooling in ["mean", "sif"]:
                run_dir = base_dir_root / f"{repr_prefix}_{pooling}"
                ensure_dir(run_dir)

                suffix = "" if pooling == "mean" else "_sif"
                missing = []
                for layer in encoder_layers:
                    p = run_dir / f"{repr_prefix}_layer{layer}_embeddings{suffix}_norm.npy"
                    if not p.exists():
                        missing.append(layer)

                if missing:
                    print(f"  Extracting {repr_mode}/{pooling} layers {missing}...")
                    run_encoder_extract(
                        meta_csv=meta_csv,
                        run_dir=run_dir,
                        model_name=model_name,
                        repr_mode=repr_mode,
                        layers=",".join(str(x) for x in missing),
                        pooling=pooling,
                        max_length=args.max_length,
                        batch_size=args.batch_size,
                        sif_a=args.sif_a,
                        split_csv=str(split_csv),
                        half_precision=args.half_precision,
                    )

                for layer in encoder_layers:
                    emb_norm_p = run_dir / f"{repr_prefix}_layer{layer}_embeddings{suffix}_norm.npy"
                    emb_raw_p = run_dir / f"{repr_prefix}_layer{layer}_embeddings{suffix}.npy"
                    if not emb_norm_p.exists():
                        continue

                    emb_norm = np.load(emb_norm_p)
                    metrics = evaluate_embeddings(emb_norm, folder_ids, is_test_query, is_winnable)
                    row = {
                        "model": model_name, "repr": repr_mode, "pooling": pooling,
                        "layer": layer, "method": "baseline", "D": 0, "sif_a": args.sif_a,
                        "split": "clean", "n_train": n_train, "n_test_query": n_test_query,
                        **metrics,
                    }
                    append_row(scoreboard_path, row)

                    # ABTT
                    if emb_raw_p.exists():
                        emb_raw = np.load(emb_raw_p)
                        cleaned = apply_abtt_clean(emb_raw, train_mask, args.D)
                        cleaned_norm = l2_normalize(cleaned)
                        metrics_abtt = evaluate_embeddings(
                            cleaned_norm, folder_ids, is_test_query, is_winnable
                        )
                        row_abtt = {
                            "model": model_name, "repr": repr_mode, "pooling": pooling,
                            "layer": layer, "method": "abtt", "D": args.D, "sif_a": args.sif_a,
                            "split": "clean", "n_train": n_train, "n_test_query": n_test_query,
                            **metrics_abtt,
                        }
                        append_row(scoreboard_path, row_abtt)

                    print(f"  {repr_mode}/{pooling}/L{layer}: "
                          f"baseline={metrics['acc@1_winnable']:.3f}")

    print(f"\nPhase 8 canon sweep results: {scoreboard_path}")


if __name__ == "__main__":
    main()
