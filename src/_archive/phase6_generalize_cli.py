from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class TargetSpec:
    model_name: str
    repr_name: str
    pooling: str
    layers: List[int]
    base_run_dir: Path


@dataclass(frozen=True)
class ChainStep:
    method: str
    params: Dict[str, object]


def parse_target(value: str) -> TargetSpec:
    parts = [p.strip() for p in value.split("|")]
    if len(parts) != 5:
        raise ValueError(
            "Target must be 'model|repr|pooling|layers|base_run_dir'."
        )
    model_name, repr_name, pooling, layers_raw, base_run_dir = parts
    layers = parse_layers(layers_raw)
    if not layers:
        raise ValueError(f"Invalid target layers: {layers_raw}")
    return TargetSpec(
        model_name=model_name,
        repr_name=repr_name,
        pooling=pooling,
        layers=layers,
        base_run_dir=Path(base_run_dir),
    )


def parse_layers(value: str) -> List[int]:
    if not value:
        return []
    parts = [p.strip() for p in value.split(",") if p.strip()]
    layers: List[int] = []
    for part in parts:
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid layer range: {part}")
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 6 generalization runner for multiple models/layers."
    )
    parser.add_argument("--phase5_run_dir", required=True)
    parser.add_argument("--out_root_dir", required=True)
    parser.add_argument("--run_id", default="", help="Optional run id under out_root_dir.")
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="Target spec: model|repr|pooling|layers|base_run_dir",
    )
    parser.add_argument("--variance_drop_percent", type=float, default=25.0)
    parser.add_argument("--skip_best_phase5", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_best_phase5(phase5_run_dir: Path) -> Dict[str, object]:
    score_path = phase5_run_dir / "phase5_full_eval_scoreboard.csv"
    if not score_path.exists():
        raise FileNotFoundError(f"Missing Phase 5 scoreboard: {score_path}")
    import pandas as pd

    df = pd.read_csv(score_path)
    if df.empty:
        raise ValueError(f"Phase 5 scoreboard is empty: {score_path}")
    best = df.sort_values("acc@5_winnable", ascending=False).iloc[0]
    return best.to_dict()


def resolve_chain(derived_run_dir: Path) -> List[ChainStep]:
    chain: List[ChainStep] = []
    current = derived_run_dir
    seen: set[Path] = set()
    while True:
        if current in seen:
            raise RuntimeError(f"Cycle detected in config chain at {current}")
        seen.add(current)
        config_path = current / "config.json"
        if not config_path.exists():
            break
        config = read_json(config_path)
        method = config.get("method")
        params = config.get("params", {})
        base_run_dir = config.get("base_run_dir", "")
        if not method or not base_run_dir:
            break
        chain.append(ChainStep(method=method, params=params))
        current = Path(base_run_dir)
    chain.reverse()
    return chain


def select_params_for_method(method: str, params: Dict[str, object]) -> List[str]:
    args: List[str] = []
    if method == "variance":
        drop = float(params.get("drop_lowest_percent", 0.0))
        keep_top_k = int(params.get("keep_top_k", 0))
        keep_ratio = float(params.get("keep_ratio", 0.0))
        if drop > 0:
            args += ["--drop_lowest_percent", str(drop)]
        elif keep_top_k > 0:
            args += ["--keep_top_k", str(keep_top_k)]
        elif keep_ratio > 0:
            args += ["--keep_ratio", str(keep_ratio)]
    elif method == "corr":
        args += ["--corr_threshold", str(params.get("corr_threshold", 0.99))]
        args += ["--corr_sample_n", str(int(params.get("corr_sample_n", 512)))]
        target_dims = int(params.get("corr_target_dims", 0))
        if target_dims:
            args += ["--corr_target_dims", str(target_dims)]
        args += ["--random_seed", str(int(params.get("random_seed", 13)))]
    elif method == "pca":
        components = params.get("pca_components", [256])
        if isinstance(components, list):
            components_str = ",".join(str(int(x)) for x in components)
        else:
            components_str = str(components)
        args += ["--pca_components", components_str]
        args += ["--pca_solver", str(params.get("pca_solver", "randomized"))]
        args += ["--random_seed", str(int(params.get("random_seed", 13)))]
    elif method == "pc_remove":
        args += [
            "--pc_remove_components",
            str(int(params.get("pc_remove_components", 1))),
        ]
        if params.get("pc_remove_center", False):
            args.append("--pc_remove_center")
        args += ["--pca_solver", str(params.get("pca_solver", "randomized"))]
        args += ["--random_seed", str(int(params.get("random_seed", 13)))]
    elif method == "center":
        pass
    elif method == "standardize":
        args += ["--standardize_eps", str(params.get("standardize_eps", 1e-6))]
    elif method == "random":
        drop = float(params.get("drop_lowest_percent", 0.0))
        keep_top_k = int(params.get("keep_top_k", 0))
        keep_ratio = float(params.get("keep_ratio", 0.0))
        if drop > 0:
            args += ["--drop_lowest_percent", str(drop)]
        elif keep_top_k > 0:
            args += ["--keep_top_k", str(keep_top_k)]
        elif keep_ratio > 0:
            args += ["--keep_ratio", str(keep_ratio)]
        args += ["--random_seed", str(int(params.get("random_seed", 13)))]
    elif method == "lda":
        args += ["--lda_test_fraction", str(params.get("lda_test_fraction", 0.2))]
        args += ["--lda_solver", str(params.get("lda_solver", "lsqr"))]
        args += ["--lda_shrinkage", str(params.get("lda_shrinkage", "auto"))]
        args += ["--random_seed", str(int(params.get("random_seed", 13)))]
    else:
        raise ValueError(f"Unsupported method in Phase 6 chain: {method}")
    return args


def run_filter(
    base_run_dir: Path,
    out_run_dir: Path,
    repr_name: str,
    pooling: str,
    layer: int,
    method: str,
    params: Dict[str, object],
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
        repr_name,
        "--pooling",
        pooling,
        "--layers",
        str(layer),
        "--method",
        method,
    ]
    cmd += select_params_for_method(method, params)
    subprocess.run(cmd, check=True)


def run_screen(run_dir: Path, repr_name: str, pooling: str, layer: int) -> Dict[str, object]:
    import subprocess

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
        "retrieval_screen.json",
    ]
    subprocess.run(cmd, check=True)
    return read_json(run_dir / "retrieval_screen.json")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def target_label(spec: TargetSpec, layer: int) -> str:
    model_slug = spec.model_name.replace("/", "_")
    return f"{model_slug}_{spec.repr_name}_{spec.pooling}_layer{layer}"


def append_row(path: Path, row: Dict[str, object]) -> None:
    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    phase5_run_dir = Path(args.phase5_run_dir)
    out_root_dir = Path(args.out_root_dir)
    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_run_dir = out_root_dir / run_id
    ensure_dir(out_run_dir)
    derived_root = out_run_dir / "derived"
    ensure_dir(derived_root)

    targets = [parse_target(t) for t in args.target]
    if not targets:
        raise ValueError("At least one --target is required.")

    best_phase5 = None
    best_chain: List[ChainStep] = []
    if not args.skip_best_phase5:
        best_phase5 = find_best_phase5(phase5_run_dir)
        derived_run_dir = Path(best_phase5["derived_run_dir"])
        best_chain = resolve_chain(derived_run_dir)

    config_payload = {
        "phase5_run_dir": str(phase5_run_dir),
        "best_phase5_row": best_phase5,
        "best_phase5_chain": [
            {"method": step.method, "params": step.params} for step in best_chain
        ],
        "targets": [
            {
                "model_name": t.model_name,
                "repr": t.repr_name,
                "pooling": t.pooling,
                "layers": t.layers,
                "base_run_dir": str(t.base_run_dir),
            }
            for t in targets
        ],
        "variance_drop_percent": args.variance_drop_percent,
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_run_dir / "phase6_config.json", "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    scoreboard_path = out_run_dir / "phase6_scoreboard.csv"

    for target in targets:
        for layer in target.layers:
            label = target_label(target, layer)
            base_run_dir = target.base_run_dir
            emb_path = base_run_dir / f"{target.repr_name}_layer{layer}_embeddings_norm.npy"
            if not emb_path.exists():
                row = {
                    "model": target.model_name,
                    "repr": target.repr_name,
                    "pooling": target.pooling,
                    "layer": layer,
                    "transform": "baseline",
                    "status": "missing_embeddings",
                    "run_dir": str(base_run_dir),
                }
                append_row(scoreboard_path, row)
                continue

            base_metrics = run_screen(base_run_dir, target.repr_name, target.pooling, layer)
            base_row = {
                "model": target.model_name,
                "repr": target.repr_name,
                "pooling": target.pooling,
                "layer": layer,
                "transform": "baseline",
                "status": "ok",
                "run_dir": str(base_run_dir),
                **base_metrics,
            }
            append_row(scoreboard_path, base_row)

            var_dir = derived_root / label / "var25"
            ensure_dir(var_dir)
            run_filter(
                base_run_dir=base_run_dir,
                out_run_dir=var_dir,
                repr_name=target.repr_name,
                pooling=target.pooling,
                layer=layer,
                method="variance",
                params={"drop_lowest_percent": args.variance_drop_percent},
            )
            var_metrics = run_screen(var_dir, target.repr_name, target.pooling, layer)
            var_row = {
                "model": target.model_name,
                "repr": target.repr_name,
                "pooling": target.pooling,
                "layer": layer,
                "transform": f"variance_drop_{int(args.variance_drop_percent)}",
                "status": "ok",
                "run_dir": str(var_dir),
                **var_metrics,
            }
            append_row(scoreboard_path, var_row)

            if not args.skip_best_phase5:
                if not best_chain:
                    row = {
                        "model": target.model_name,
                        "repr": target.repr_name,
                        "pooling": target.pooling,
                        "layer": layer,
                        "transform": "best_phase5",
                        "status": "no_chain",
                        "run_dir": "",
                    }
                    append_row(scoreboard_path, row)
                    continue
                chain_base = base_run_dir
                chain_dir = derived_root / label / "best_phase5"
                ensure_dir(chain_dir)
                for idx, step in enumerate(best_chain):
                    step_dir = chain_dir / f"step_{idx+1}_{step.method}"
                    ensure_dir(step_dir)
                    run_filter(
                        base_run_dir=chain_base,
                        out_run_dir=step_dir,
                        repr_name=target.repr_name,
                        pooling=target.pooling,
                        layer=layer,
                        method=step.method,
                        params=step.params,
                    )
                    chain_base = step_dir
                best_metrics = run_screen(chain_base, target.repr_name, target.pooling, layer)
                best_row = {
                    "model": target.model_name,
                    "repr": target.repr_name,
                    "pooling": target.pooling,
                    "layer": layer,
                    "transform": "best_phase5",
                    "status": "ok",
                    "run_dir": str(chain_base),
                    **best_metrics,
                }
                append_row(scoreboard_path, best_row)

    print(f"Phase 6 outputs: {out_run_dir}")


if __name__ == "__main__":
    main()
