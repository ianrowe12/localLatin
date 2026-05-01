"""Build and validate a manifest for a three-model attribution artifact run."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attribution_model_config import DEFAULT_MODELS, model_config, model_slug  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples_csv", required=True, type=Path)
    parser.add_argument("--pc_root", required=True, type=Path)
    parser.add_argument("--artifacts_root", required=True, type=Path)
    parser.add_argument("--retrieval_mark_root", required=True, type=Path)
    parser.add_argument("--metrics_root", required=True, type=Path)
    parser.add_argument("--out_json", required=True, type=Path)
    parser.add_argument("--out_inventory_csv", required=True, type=Path)
    parser.add_argument("--expected_n_per_model", type=int, default=200)
    parser.add_argument("--expected_models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--required_methods", nargs="*", default=["ig", "retrieval_mark"])
    parser.add_argument("--variants", nargs="*", default=["baseline", "abtt"])
    parser.add_argument(
        "--require_complete",
        action="store_true",
        help="Fail if any expected artifact, sidecar, metric JSON, or summary row is missing.",
    )
    return parser.parse_args()


def _path_for(root: Path, slug: str, example_id: int, role: str) -> Path:
    return root / slug / f"example{example_id:03d}_{role}.npz"


def _json_path_for(root: Path, slug: str, example_id: int, role: str) -> Path:
    return root / slug / f"example{example_id:03d}_{role}.json"


def _load_npz_keys(path: Path) -> set[str]:
    with np.load(path, allow_pickle=False) as data:
        return set(data.files)


def _pc_info(path: Path, expected_d: int) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    info: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "pcs_shape": None,
        "mean_vec_shape": None,
    }
    if not path.exists():
        errors.append(f"missing PC file: {path}")
        return info, errors
    with np.load(path, allow_pickle=False) as data:
        if "pcs" not in data.files or "mean_vec" not in data.files:
            errors.append(f"PC file missing pcs/mean_vec arrays: {path}")
            return info, errors
        pcs = np.asarray(data["pcs"])
        mean_vec = np.asarray(data["mean_vec"])
    info["pcs_shape"] = list(pcs.shape)
    info["mean_vec_shape"] = list(mean_vec.shape)
    if pcs.ndim != 2 or pcs.shape[0] < expected_d:
        errors.append(f"PC file has insufficient components for D={expected_d}: {path} shape={pcs.shape}")
    if mean_vec.ndim != 1 or (pcs.ndim == 2 and mean_vec.shape[0] != pcs.shape[1]):
        errors.append(f"PC mean_vec shape does not match pcs: {path} pcs={pcs.shape} mean={mean_vec.shape}")
    return info, errors


def main() -> None:
    args = parse_args()
    examples = pd.read_csv(args.examples_csv)
    expected_models = list(args.expected_models)
    expected_model_set = set(expected_models)
    errors: list[str] = []
    warnings: list[str] = []

    if set(examples["model_name"].unique()) != expected_model_set:
        errors.append(
            "examples model set mismatch: "
            f"expected={sorted(expected_model_set)} actual={sorted(examples['model_name'].unique())}"
        )

    if "gold_similar" not in examples.columns or not (examples["gold_similar"] == 1).all():
        errors.append("examples_csv must contain only gold_similar=1 rows")

    duplicate_ids = examples["example_id"].duplicated().sum()
    if duplicate_ids:
        errors.append(f"example_id values are not unique: {duplicate_ids} duplicates")

    inventory_rows: list[dict[str, Any]] = []
    pc_records: dict[str, Any] = {}
    artifact_method_errors = 0

    for model_name in expected_models:
        cfg = model_config(model_name)
        slug = model_slug(model_name)
        sub = examples[examples["model_name"] == model_name].copy()
        expected_layer = int(cfg["layer"])
        expected_d = int(cfg["D"])
        layer_values = sorted(int(v) for v in sub["layer"].dropna().unique())
        d_values = sorted(int(v) for v in sub["D"].dropna().unique())

        if len(sub) != args.expected_n_per_model:
            errors.append(f"{model_name}: expected {args.expected_n_per_model} examples, found {len(sub)}")
        if layer_values != [expected_layer]:
            errors.append(f"{model_name}: expected layer {expected_layer}, found {layer_values}")
        if d_values != [expected_d]:
            errors.append(f"{model_name}: expected D {expected_d}, found {d_values}")

        pc_path = args.pc_root / slug / f"layer{expected_layer}_pcs.npz"
        pc_info, pc_errors = _pc_info(pc_path, expected_d)
        pc_records[slug] = pc_info
        errors.extend(pc_errors)

        canonical_missing = 0
        sidecar_missing = 0
        metric_missing = 0
        canonical_present = 0
        sidecar_present = 0
        metric_present = 0

        for row in sub.itertuples(index=False):
            example_id = int(row.example_id)
            role = str(row.candidate_role)
            canonical_path = _path_for(args.artifacts_root, slug, example_id, role)
            sidecar_path = _path_for(args.retrieval_mark_root, slug, example_id, role)
            metric_path = _json_path_for(args.metrics_root, slug, example_id, role)

            if canonical_path.exists():
                canonical_present += 1
                keys = _load_npz_keys(canonical_path)
                for method in args.required_methods:
                    for variant in args.variants:
                        key = f"pair_matrix_{method}_{variant}"
                        if key not in keys:
                            artifact_method_errors += 1
                            errors.append(f"missing {key}: {canonical_path}")
            else:
                canonical_missing += 1

            if sidecar_path.exists():
                sidecar_present += 1
            else:
                sidecar_missing += 1

            if metric_path.exists():
                metric_present += 1
            else:
                metric_missing += 1

        inventory_rows.append({
            "model_name": model_name,
            "model_slug": slug,
            "expected_examples": args.expected_n_per_model,
            "examples_csv_rows": len(sub),
            "layer_values": ",".join(str(v) for v in layer_values),
            "d_values": ",".join(str(v) for v in d_values),
            "canonical_npz_present": canonical_present,
            "canonical_npz_missing": canonical_missing,
            "retrieval_mark_sidecar_present": sidecar_present,
            "retrieval_mark_sidecar_missing": sidecar_missing,
            "metric_json_present": metric_present,
            "metric_json_missing": metric_missing,
            "pc_path": str(pc_path),
            "pc_exists": pc_path.exists(),
        })

    summary_path = args.metrics_root / "summary.csv"
    summary_info: dict[str, Any] = {"path": str(summary_path), "exists": summary_path.exists()}
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        summary_info["rows"] = int(len(summary))
        summary_info["models"] = sorted(str(v) for v in summary["model"].unique())
        summary_info["methods"] = sorted(str(v) for v in summary["method"].unique())
        summary_info["variants"] = sorted(str(v) for v in summary["variant"].unique())
        expected_rows = {
            (model, method, variant)
            for model in expected_models
            for method in args.required_methods + ["random", "inverse"]
            for variant in args.variants
        }
        actual_rows = {
            (str(row.model), str(row.method), str(row.variant))
            for row in summary.itertuples(index=False)
        }
        missing_summary = sorted(expected_rows - actual_rows)
        if missing_summary:
            errors.append(f"summary.csv missing required rows: {missing_summary[:20]}")
    else:
        warnings.append(f"summary.csv not found: {summary_path}")

    inventory = pd.DataFrame(inventory_rows)
    args.out_inventory_csv.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(args.out_inventory_csv, index=False)

    if args.require_complete:
        for row in inventory_rows:
            for key in (
                "canonical_npz_missing",
                "retrieval_mark_sidecar_missing",
                "metric_json_missing",
            ):
                if int(row[key]) != 0:
                    errors.append(f"{row['model_name']}: {key}={row[key]}")
        if not summary_path.exists():
            errors.append(f"missing required summary.csv: {summary_path}")

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/ig/build_attribution_run_manifest.py",
        "examples_csv": str(args.examples_csv),
        "pc_root": str(args.pc_root),
        "artifacts_root": str(args.artifacts_root),
        "retrieval_mark_root": str(args.retrieval_mark_root),
        "metrics_root": str(args.metrics_root),
        "expected_models": expected_models,
        "expected_n_per_model": args.expected_n_per_model,
        "required_methods": args.required_methods,
        "variants": args.variants,
        "pc_files": pc_records,
        "summary": summary_info,
        "inventory_csv": str(args.out_inventory_csv),
        "artifact_method_errors": artifact_method_errors,
        "warnings": warnings,
        "errors": errors,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_inventory_csv}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    if errors:
        print("Errors:")
        for error in errors[:50]:
            print(f"  - {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
