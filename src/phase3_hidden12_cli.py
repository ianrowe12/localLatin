from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_int_list(value: str) -> List[int]:
    if not value:
        return []
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_float_list(value: str) -> List[float]:
    if not value:
        return []
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3 sweep driver for hidden layer 12 (mean pooling)."
    )
    parser.add_argument("--base_run_dir", required=True)
    parser.add_argument("--out_root_dir", required=True)
    parser.add_argument("--run_id", default="", help="Optional run id under out_root_dir.")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--repr", choices=["hidden"], default="hidden")
    parser.add_argument("--pooling", choices=["mean"], default="mean")

    parser.add_argument(
        "--variance_drop_percents",
        default="10,25,50,75,90",
        help="Comma-separated percents to drop (variance filter).",
    )
    parser.add_argument(
        "--corr_thresholds",
        default="0.95,0.975,0.99",
        help="Comma-separated correlation thresholds.",
    )
    parser.add_argument(
        "--pca_components",
        default="32,64,96,128,256,384,512,640",
        help="Comma-separated PCA k values.",
    )
    parser.add_argument(
        "--random_drop_percents",
        default="10,25,50,75,90",
        help="Comma-separated percents to drop for random subsets.",
    )
    parser.add_argument("--random_trials", type=int, default=10)
    parser.add_argument("--random_seed", type=int, default=13)
    return parser.parse_args()


def copy_baseline_inputs(base_run_dir: Path, out_run_dir: Path, layer: int) -> None:
    out_run_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "meta.csv",
        f"hidden_layer{layer}_embeddings.npy",
        f"hidden_layer{layer}_embeddings_norm.npy",
    ]:
        src = base_run_dir / name
        dst = out_run_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing baseline file: {src}")
        shutil.copy2(src, dst)


def write_baseline_summary(base_run_dir: Path, out_run_dir: Path, layer: int) -> None:
    summary_path = base_run_dir / "hidden_layer_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")
    df = pd.read_csv(summary_path)
    row = df[df["layer"] == layer]
    if row.empty:
        raise ValueError(f"Layer {layer} not found in {summary_path}")
    payload = {
        "base_run_dir": str(base_run_dir),
        "layer": layer,
        "row": row.iloc[0].to_dict(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_run_dir / "phase3_baseline_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_filter(
    base_run_dir: Path,
    out_run_dir: Path,
    layer: int,
    method: str,
    params: Dict[str, str],
) -> None:
    import subprocess

    cmd = [
        "python",
        "src/filter_embeddings_cli.py",
        "--base_run_dir",
        str(base_run_dir),
        "--out_run_dir",
        str(out_run_dir),
        "--repr",
        "hidden",
        "--pooling",
        "mean",
        "--layers",
        str(layer),
        "--method",
        method,
    ]
    for key, value in params.items():
        cmd.extend([key, value])
    subprocess.run(cmd, check=True)


def run_gap_screen(run_dir: Path, layer: int) -> Dict[str, object]:
    import subprocess

    cmd = [
        "python",
        "src/gap_screen_cli.py",
        "--run_dir",
        str(run_dir),
        "--repr",
        "hidden",
        "--pooling",
        "mean",
        "--layer",
        str(layer),
        "--output_json",
        "gap_screen.json",
    ]
    subprocess.run(cmd, check=True)
    with open(run_dir / "gap_screen.json", "r", encoding="utf-8") as f:
        return json.load(f)


def read_num_dims(run_dir: Path, layer: int) -> int:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return 0
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return int(config.get("layer_outputs", {}).get(str(layer), {}).get("num_dims", 0))


def append_scoreboard_row(path: Path, row: Dict[str, object]) -> None:
    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    base_run_dir = Path(args.base_run_dir)
    out_root_dir = Path(args.out_root_dir)
    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_run_dir = out_root_dir / run_id
    derived_root = out_run_dir / "derived"

    copy_baseline_inputs(base_run_dir, out_run_dir, args.layer)
    write_baseline_summary(base_run_dir, out_run_dir, args.layer)

    scoreboard_path = out_run_dir / "phase3_screening_scoreboard.csv"

    variance_drops = parse_float_list(args.variance_drop_percents)
    corr_thresholds = parse_float_list(args.corr_thresholds)
    pca_components = parse_int_list(args.pca_components)
    random_drops = parse_float_list(args.random_drop_percents)

    for drop in variance_drops:
        out_dir = derived_root / f"var_hidden_mean_drop{int(drop)}"
        run_filter(
            out_run_dir,
            out_dir,
            args.layer,
            "variance",
            {"--drop_lowest_percent": str(drop)},
        )
        gap = run_gap_screen(out_dir, args.layer)
        row = {
            "method": "variance",
            "params": f"drop_lowest_percent={drop}",
            "dims_kept": read_num_dims(out_dir, args.layer),
            "gap": gap["gap"],
            "off_diag_mean": gap["off_diag_mean"],
            "elapsed_seconds": gap["elapsed_seconds"],
            "derived_run_dir": str(out_dir),
        }
        append_scoreboard_row(scoreboard_path, row)

    for thresh in corr_thresholds:
        out_dir = derived_root / f"corr_hidden_mean_t{str(thresh).replace('.', '_')}"
        run_filter(
            out_run_dir,
            out_dir,
            args.layer,
            "corr",
            {
                "--corr_threshold": str(thresh),
                "--corr_sample_n": "512",
            },
        )
        gap = run_gap_screen(out_dir, args.layer)
        row = {
            "method": "corr",
            "params": f"corr_threshold={thresh}",
            "dims_kept": read_num_dims(out_dir, args.layer),
            "gap": gap["gap"],
            "off_diag_mean": gap["off_diag_mean"],
            "elapsed_seconds": gap["elapsed_seconds"],
            "derived_run_dir": str(out_dir),
        }
        append_scoreboard_row(scoreboard_path, row)

    for k in pca_components:
        out_dir = derived_root / f"pca_hidden_mean_k{k}"
        run_filter(
            out_run_dir,
            out_dir,
            args.layer,
            "pca",
            {"--pca_components": str(k)},
        )
        gap = run_gap_screen(out_dir, args.layer)
        row = {
            "method": "pca",
            "params": f"pca_components={k}",
            "dims_kept": read_num_dims(out_dir, args.layer),
            "gap": gap["gap"],
            "off_diag_mean": gap["off_diag_mean"],
            "elapsed_seconds": gap["elapsed_seconds"],
            "derived_run_dir": str(out_dir),
        }
        append_scoreboard_row(scoreboard_path, row)

    seed_base = args.random_seed
    for drop in random_drops:
        for trial in range(args.random_trials):
            seed = seed_base + trial
            out_dir = derived_root / f"rand_hidden_mean_drop{int(drop)}_seed{seed}"
            run_filter(
                out_run_dir,
                out_dir,
                args.layer,
                "random",
                {
                    "--drop_lowest_percent": str(drop),
                    "--random_seed": str(seed),
                },
            )
            gap = run_gap_screen(out_dir, args.layer)
            row = {
                "method": "random",
                "params": f"drop_lowest_percent={drop},seed={seed}",
                "dims_kept": read_num_dims(out_dir, args.layer),
                "gap": gap["gap"],
                "off_diag_mean": gap["off_diag_mean"],
                "elapsed_seconds": gap["elapsed_seconds"],
                "derived_run_dir": str(out_dir),
            }
            append_scoreboard_row(scoreboard_path, row)

    print(f"Screening scoreboard: {scoreboard_path}")


if __name__ == "__main__":
    main()
