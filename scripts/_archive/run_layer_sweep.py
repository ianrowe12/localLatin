from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from canon_retrieval import l2_normalize
from sif_abtt import EmbeddingCleaner, remove_top_components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 7 all-layers sweep.")
    parser.add_argument("--meta_csv", required=True)
    parser.add_argument("--runs_root", required=True)
    parser.add_argument("--out_dir", required=True)
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
        "--split_csv",
        default="",
        help="Split CSV for train-only ABTT fitting (Phase 8 leak fix).",
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
            start = int(start_str)
            end = int(end_str)
            out.extend(range(start, end + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def model_slug(name: str) -> str:
    return name.replace("/", "_")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_extract(
    repr_name: str,
    meta_csv: Path,
    run_dir: Path,
    model_name: str,
    layers: str,
    pooling: str,
    max_length: int,
    batch_size: int,
    sif_a: float,
    split_csv: str = "",
) -> None:
    cmd = [
        "python",
        f"src/extract_{repr_name}_cli.py",
        "--meta_csv",
        str(meta_csv),
        "--runs_root",
        str(run_dir.parent),
        "--run_dir",
        str(run_dir),
        "--model_name",
        model_name,
        "--layers",
        layers,
        "--pooling",
        pooling,
        "--max_length",
        str(max_length),
        "--batch_size",
        str(batch_size),
    ]
    if pooling == "sif":
        cmd += ["--sif_a", str(sif_a)]
    if split_csv:
        cmd += ["--split_csv", split_csv]
    subprocess.run(cmd, check=True)


def embedding_path(run_dir: Path, repr_name: str, pooling: str, layer: int) -> Path:
    suffix = "" if pooling == "mean" else "_lasttok" if pooling == "lasttok" else "_sif"
    return run_dir / f"{repr_name}_layer{layer}_embeddings{suffix}.npy"


def norm_path(run_dir: Path, repr_name: str, pooling: str, layer: int) -> Path:
    suffix = "" if pooling == "mean" else "_lasttok" if pooling == "lasttok" else "_sif"
    return run_dir / f"{repr_name}_layer{layer}_embeddings{suffix}_norm.npy"


def ensure_embeddings(
    run_dir: Path,
    repr_name: str,
    pooling: str,
    layers: List[int],
) -> List[int]:
    missing = []
    for layer in layers:
        if not norm_path(run_dir, repr_name, pooling, layer).exists():
            missing.append(layer)
    return missing


def run_retrieval_screen(
    run_dir: Path, repr_name: str, pooling: str, layer: int, output_json: str
) -> Dict[str, object]:
    cmd = [
        "python",
        "src/retrieval_screen_cli.py",
        "--run_dir",
        str(run_dir),
        "--repr",
        repr_name,
        "--pooling",
        pooling,
        "--layer",
        str(layer),
        "--output_json",
        output_json,
    ]
    subprocess.run(cmd, check=True)
    with open(run_dir / output_json, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_abtt(
    base_run_dir: Path,
    out_run_dir: Path,
    repr_name: str,
    pooling: str,
    layer: int,
    num_components: int,
    split_meta: pd.DataFrame | None = None,
) -> None:
    ensure_dir(out_run_dir)
    meta_src = base_run_dir / "meta.csv"
    meta_dst = out_run_dir / "meta.csv"
    if not meta_dst.exists():
        meta_dst.write_text(meta_src.read_text(), encoding="utf-8")

    emb = np.load(embedding_path(base_run_dir, repr_name, pooling, layer))

    if split_meta is not None:
        train_mask = split_meta["split"].values == "train"
        train_emb = emb[train_mask]
    else:
        train_emb = emb

    cleaner = EmbeddingCleaner(num_components=num_components, center=True)
    cleaner.fit(train_emb)
    cleaned = cleaner.transform(emb)
    cleaned_norm = l2_normalize(cleaned)

    cleaner.save(out_run_dir / "cleaner.npz")

    out_emb = embedding_path(out_run_dir, repr_name, pooling, layer)
    out_norm = norm_path(out_run_dir, repr_name, pooling, layer)
    np.save(out_emb, cleaned.astype(np.float32))
    np.save(out_norm, cleaned_norm.astype(np.float32))

    config = {
        "method": "abtt",
        "num_components": int(num_components),
        "center": True,
        "base_run_dir": str(base_run_dir),
        "layer": layer,
        "repr": repr_name,
        "pooling": pooling,
        "mean_vector": cleaner.mean_vec.tolist() if cleaner.mean_vec is not None else [],
        "pcs_shape": list(cleaner.pcs.shape),
        "fit_on_train_only": split_meta is not None,
    }
    with open(out_run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def append_row(path: Path, row: Dict[str, object]) -> None:
    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    meta_csv = Path(args.meta_csv)
    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_run_dir = out_dir / run_id
    ensure_dir(out_run_dir)
    derived_root = out_run_dir / "derived"
    ensure_dir(derived_root)

    split_meta = pd.read_csv(args.split_csv) if args.split_csv else None

    models = parse_list(args.models)
    reprs = parse_list(args.reprs)
    base_pooling = args.pooling

    layers_hidden = parse_layers(args.layers_hidden)
    layers_ff1 = parse_layers(args.layers_ff1)

    scoreboard_path = out_run_dir / "phase7_layer_sweep.csv"

    for model in models:
        model_dir = runs_root / "phase7_bases" / model_slug(model)
        for repr_name in reprs:
            layers = layers_hidden if repr_name == "hidden" else layers_ff1
            base_dir = model_dir / f"{repr_name}_{base_pooling}"
            sif_dir = model_dir / f"{repr_name}_sif"
            ensure_dir(base_dir)
            ensure_dir(sif_dir)

            missing = ensure_embeddings(base_dir, repr_name, base_pooling, layers)
            if missing:
                run_extract(
                    repr_name=repr_name,
                    meta_csv=meta_csv,
                    run_dir=base_dir,
                    model_name=model,
                    layers=",".join(str(x) for x in missing),
                    pooling=base_pooling,
                    max_length=args.max_length,
                    batch_size=args.batch_size,
                    sif_a=args.sif_a,
                    split_csv=args.split_csv,
                )

            missing_sif = ensure_embeddings(sif_dir, repr_name, "sif", layers)
            if missing_sif:
                run_extract(
                    repr_name=repr_name,
                    meta_csv=meta_csv,
                    run_dir=sif_dir,
                    model_name=model,
                    layers=",".join(str(x) for x in missing_sif),
                    pooling="sif",
                    max_length=args.max_length,
                    batch_size=args.batch_size,
                    sif_a=args.sif_a,
                    split_csv=args.split_csv,
                )

            for layer in layers:
                base_metrics = run_retrieval_screen(
                    base_dir, repr_name, base_pooling, layer, "retrieval_screen.json"
                )
                base_row = {
                    "model": model,
                    "repr": repr_name,
                    "pooling": base_pooling,
                    "layer": layer,
                    "method": "baseline",
                    "D": 0,
                    "a": args.sif_a,
                    "run_dir": str(base_dir),
                    **base_metrics,
                }
                append_row(scoreboard_path, base_row)

                abtt_dir = derived_root / model_slug(model) / repr_name / "abtt_mean" / f"layer{layer}"
                apply_abtt(
                    base_run_dir=base_dir,
                    out_run_dir=abtt_dir,
                    repr_name=repr_name,
                    pooling=base_pooling,
                    layer=layer,
                    num_components=args.D,
                    split_meta=split_meta,
                )
                abtt_metrics = run_retrieval_screen(
                    abtt_dir, repr_name, base_pooling, layer, "retrieval_screen.json"
                )
                abtt_row = {
                    "model": model,
                    "repr": repr_name,
                    "pooling": base_pooling,
                    "layer": layer,
                    "method": "abtt",
                    "D": args.D,
                    "a": args.sif_a,
                    "run_dir": str(abtt_dir),
                    **abtt_metrics,
                }
                append_row(scoreboard_path, abtt_row)

                sif_metrics = run_retrieval_screen(
                    sif_dir, repr_name, "sif", layer, "retrieval_screen.json"
                )
                sif_row = {
                    "model": model,
                    "repr": repr_name,
                    "pooling": "sif",
                    "layer": layer,
                    "method": "sif_pool",
                    "D": 0,
                    "a": args.sif_a,
                    "run_dir": str(sif_dir),
                    **sif_metrics,
                }
                append_row(scoreboard_path, sif_row)

                sif_abtt_dir = (
                    derived_root
                    / model_slug(model)
                    / repr_name
                    / "abtt_sif"
                    / f"layer{layer}"
                )
                apply_abtt(
                    base_run_dir=sif_dir,
                    out_run_dir=sif_abtt_dir,
                    repr_name=repr_name,
                    pooling="sif",
                    layer=layer,
                    num_components=args.D,
                    split_meta=split_meta,
                )
                sif_abtt_metrics = run_retrieval_screen(
                    sif_abtt_dir, repr_name, "sif", layer, "retrieval_screen.json"
                )
                sif_abtt_row = {
                    "model": model,
                    "repr": repr_name,
                    "pooling": "sif",
                    "layer": layer,
                    "method": "sif_abtt",
                    "D": args.D,
                    "a": args.sif_a,
                    "run_dir": str(sif_abtt_dir),
                    **sif_abtt_metrics,
                }
                append_row(scoreboard_path, sif_abtt_row)

    print(f"Phase 7 sweep outputs: {out_run_dir}")


if __name__ == "__main__":
    main()
