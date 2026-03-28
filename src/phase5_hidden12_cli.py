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
        description="Phase 5 sweep driver for hidden layer 12 (mean pooling)."
    )
    parser.add_argument("--base_run_dir", required=True)
    parser.add_argument("--out_root_dir", required=True)
    parser.add_argument("--run_id", default="", help="Optional run id under out_root_dir.")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--repr", choices=["hidden"], default="hidden")
    parser.add_argument("--pooling", choices=["mean"], default="mean")

    parser.add_argument(
        "--variance_drop_percents",
        default="25,50",
        help="Comma-separated percents to drop (variance filter).",
    )
    parser.add_argument(
        "--corr_thresholds",
        default="0.90,0.85",
        help="Comma-separated correlation thresholds.",
    )
    parser.add_argument(
        "--corr_target_dims",
        default="576",
        help="Comma-separated target dims for corr pruning.",
    )
    parser.add_argument(
        "--pc_remove_components",
        default="1,3,5,10",
        help="Comma-separated number of top PCs to remove.",
    )
    parser.add_argument(
        "--pc_remove_center",
        action="store_true",
        help="Center embeddings before PC removal.",
    )
    parser.add_argument("--include_center", action="store_true")
    parser.add_argument("--include_standardize", action="store_true")
    parser.add_argument(
        "--chain_variance_drop_percents",
        default="25",
        help="Comma-separated variance drops for variance->pc_remove chaining.",
    )
    parser.add_argument(
        "--chain_pc_remove_components",
        default="3",
        help="Comma-separated PC counts for chained pc_remove.",
    )
    parser.add_argument(
        "--chain_center_pc_remove",
        action="store_true",
        help="Also chain center->pc_remove.",
    )
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
    with open(out_run_dir / "phase5_baseline_summary.json", "w", encoding="utf-8") as f:
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
        if value is None:
            cmd.append(key)
        else:
            cmd.extend([key, value])
    subprocess.run(cmd, check=True)


def run_retrieval_screen(run_dir: Path, layer: int) -> Dict[str, object]:
    import subprocess

    cmd = [
        "python",
        "src/retrieval_screen_cli.py",
        "--run_dir",
        str(run_dir),
        "--repr",
        "hidden",
        "--pooling",
        "mean",
        "--layer",
        str(layer),
        "--output_json",
        "retrieval_screen.json",
    ]
    subprocess.run(cmd, check=True)
    with open(run_dir / "retrieval_screen.json", "r", encoding="utf-8") as f:
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

    scoreboard_path = out_run_dir / "phase5_screening_scoreboard.csv"

    variance_drops = parse_float_list(args.variance_drop_percents)
    corr_thresholds = parse_float_list(args.corr_thresholds)
    corr_targets = parse_int_list(args.corr_target_dims)
    pc_remove_components = parse_int_list(args.pc_remove_components)
    chain_variance_drops = parse_float_list(args.chain_variance_drop_percents)
    chain_pc_remove = parse_int_list(args.chain_pc_remove_components)

    for drop in variance_drops:
        out_dir = derived_root / f"var_hidden_mean_drop{int(drop)}"
        run_filter(
            out_run_dir,
            out_dir,
            args.layer,
            "variance",
            {"--drop_lowest_percent": str(drop)},
        )
        screen = run_retrieval_screen(out_dir, args.layer)
        row = {
            "method": "variance",
            "params": f"drop_lowest_percent={drop}",
            "dims_kept": read_num_dims(out_dir, args.layer),
            **screen,
            "derived_run_dir": str(out_dir),
        }
        append_scoreboard_row(scoreboard_path, row)

    for thresh in corr_thresholds:
        for target_dims in corr_targets or [0]:
            label = f"corr_hidden_mean_t{str(thresh).replace('.', '_')}"
            if target_dims:
                label = f"{label}_k{target_dims}"
            out_dir = derived_root / label
            params = {
                "--corr_threshold": str(thresh),
                "--corr_sample_n": "512",
            }
            if target_dims:
                params["--corr_target_dims"] = str(target_dims)
            run_filter(out_run_dir, out_dir, args.layer, "corr", params)
            screen = run_retrieval_screen(out_dir, args.layer)
            row = {
                "method": "corr",
                "params": f"corr_threshold={thresh},corr_target_dims={target_dims}",
                "dims_kept": read_num_dims(out_dir, args.layer),
                **screen,
                "derived_run_dir": str(out_dir),
            }
            append_scoreboard_row(scoreboard_path, row)

    for r in pc_remove_components:
        out_dir = derived_root / f"pc_remove_hidden_mean_r{r}"
        params = {"--pc_remove_components": str(r)}
        if args.pc_remove_center:
            params["--pc_remove_center"] = None
        run_filter(out_run_dir, out_dir, args.layer, "pc_remove", params)
        screen = run_retrieval_screen(out_dir, args.layer)
        row = {
            "method": "pc_remove",
            "params": f"pc_remove_components={r},pc_remove_center={args.pc_remove_center}",
            "dims_kept": read_num_dims(out_dir, args.layer),
            **screen,
            "derived_run_dir": str(out_dir),
        }
        append_scoreboard_row(scoreboard_path, row)

    if args.include_center:
        out_dir = derived_root / "center_hidden_mean"
        run_filter(out_run_dir, out_dir, args.layer, "center", {})
        screen = run_retrieval_screen(out_dir, args.layer)
        row = {
            "method": "center",
            "params": "",
            "dims_kept": read_num_dims(out_dir, args.layer),
            **screen,
            "derived_run_dir": str(out_dir),
        }
        append_scoreboard_row(scoreboard_path, row)

    if args.include_standardize:
        out_dir = derived_root / "standardize_hidden_mean"
        run_filter(out_run_dir, out_dir, args.layer, "standardize", {})
        screen = run_retrieval_screen(out_dir, args.layer)
        row = {
            "method": "standardize",
            "params": "",
            "dims_kept": read_num_dims(out_dir, args.layer),
            **screen,
            "derived_run_dir": str(out_dir),
        }
        append_scoreboard_row(scoreboard_path, row)

    for drop in chain_variance_drops:
        base_chain_dir = derived_root / f"chain_var{int(drop)}"
        run_filter(
            out_run_dir,
            base_chain_dir,
            args.layer,
            "variance",
            {"--drop_lowest_percent": str(drop)},
        )
        for r in chain_pc_remove:
            out_dir = derived_root / f"chain_var{int(drop)}_pc_remove_r{r}"
            params = {"--pc_remove_components": str(r)}
            if args.pc_remove_center:
                params["--pc_remove_center"] = None
            run_filter(base_chain_dir, out_dir, args.layer, "pc_remove", params)
            screen = run_retrieval_screen(out_dir, args.layer)
            row = {
                "method": "chain_var_pc_remove",
                "params": f"drop_lowest_percent={drop},pc_remove_components={r},pc_remove_center={args.pc_remove_center}",
                "dims_kept": read_num_dims(out_dir, args.layer),
                **screen,
                "derived_run_dir": str(out_dir),
            }
            append_scoreboard_row(scoreboard_path, row)

    if args.chain_center_pc_remove:
        center_dir = derived_root / "chain_center"
        run_filter(out_run_dir, center_dir, args.layer, "center", {})
        for r in chain_pc_remove:
            out_dir = derived_root / f"chain_center_pc_remove_r{r}"
            params = {"--pc_remove_components": str(r)}
            if args.pc_remove_center:
                params["--pc_remove_center"] = None
            run_filter(center_dir, out_dir, args.layer, "pc_remove", params)
            screen = run_retrieval_screen(out_dir, args.layer)
            row = {
                "method": "chain_center_pc_remove",
                "params": f"pc_remove_components={r},pc_remove_center={args.pc_remove_center}",
                "dims_kept": read_num_dims(out_dir, args.layer),
                **screen,
                "derived_run_dir": str(out_dir),
            }
            append_scoreboard_row(scoreboard_path, row)

    print(f"Screening scoreboard: {scoreboard_path}")


if __name__ == "__main__":
    main()
