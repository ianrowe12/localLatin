from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from canon_retrieval import sanity_checks, similarity_matrix


@dataclass(frozen=True)
class CompareRun:
    label: str
    run_dir: Path


def parse_k_list(value: str) -> List[int]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    ks = sorted({int(p) for p in parts})
    if not ks:
        raise ValueError("k_list must contain at least one integer.")
    if ks[0] <= 0:
        raise ValueError("k_list values must be positive.")
    return ks


def parse_compare_specs(values: Sequence[str]) -> List[Tuple[str, Dict[str, str]]]:
    specs: List[Tuple[str, Dict[str, str]]] = []
    for raw in values:
        if ":" not in raw:
            raise ValueError(f"Invalid compare spec (expected method:params): {raw}")
        method, params_raw = raw.split(":", 1)
        params = {}
        if params_raw:
            for item in params_raw.split(","):
                if not item.strip():
                    continue
                if "=" not in item:
                    raise ValueError(f"Invalid compare param (expected key=value): {item}")
                key, value = item.split("=", 1)
                params[key.strip()] = value.strip()
        specs.append((method.strip(), params))
    return specs


def format_label(method: str, params: Dict[str, str]) -> str:
    if not params:
        return method
    param_str = "_".join(f"{k}{v}" for k, v in params.items())
    return f"{method}_{param_str}".replace(".", "_")


def parse_drop_percent(params: str) -> Optional[float]:
    for chunk in params.split(","):
        if "drop_lowest_percent" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        if key.strip() != "drop_lowest_percent":
            continue
        try:
            return float(value)
        except ValueError:
            return None
    return None


def select_from_screening(
    screening_csv: Path, compare_specs: List[Tuple[str, Dict[str, str]]]
) -> List[CompareRun]:
    df = pd.read_csv(screening_csv)
    if df.empty:
        raise ValueError(f"Screening CSV is empty: {screening_csv}")
    compare_runs: List[CompareRun] = []
    for method, params in compare_specs:
        if method != "variance":
            raise ValueError("Default compare selection only supports variance specs.")
        target = float(params.get("drop_lowest_percent", "nan"))
        if not np.isfinite(target):
            raise ValueError("variance compare spec requires drop_lowest_percent.")
        matches = df[df["method"] == method].copy()
        if matches.empty:
            raise ValueError(f"No rows for method={method} in {screening_csv}")
        matches["drop_val"] = matches["params"].apply(parse_drop_percent)
        matches = matches[matches["drop_val"].notna()]
        matches = matches[np.isclose(matches["drop_val"].astype(float), target)]
        if matches.empty:
            raise ValueError(
                f"No variance row for drop_lowest_percent={target} in {screening_csv}"
            )
        row = matches.iloc[0]
        compare_runs.append(
            CompareRun(label=format_label(method, params), run_dir=Path(row["derived_run_dir"]))
        )
    return compare_runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4 hard-query recovery/regression analysis."
    )
    parser.add_argument("--phase3_run_dir", required=True)
    parser.add_argument("--out_root_dir", default="")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--k_list", default="1,3,5")
    parser.add_argument(
        "--query_mask_col",
        default="",
        help="Optional meta.csv boolean column to restrict queries.",
    )
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        help="Compare spec as method:key=value (e.g. variance:drop_lowest_percent=25).",
    )
    parser.add_argument(
        "--compare_run_dir",
        action="append",
        default=[],
        help="Direct path to a derived run dir (bypasses screening CSV).",
    )
    parser.add_argument(
        "--compare_label",
        action="append",
        default=[],
        help="Label for each compare_run_dir (same order).",
    )
    parser.add_argument(
        "--write_figures",
        action="store_true",
        help="Write optional margin histogram PNGs.",
    )
    return parser.parse_args()


def load_meta(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "meta.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing meta.csv: {path}")
    return pd.read_csv(path)


def ensure_meta_alignment(base: pd.DataFrame, other: pd.DataFrame, run_dir: Path) -> None:
    for col in ["folder_id", "path", "filename"]:
        if col not in base.columns or col not in other.columns:
            raise ValueError(f"Missing required meta column: {col}")
        if not base[col].equals(other[col]):
            raise ValueError(f"Meta mismatch in column {col} for run {run_dir}")


def load_embeddings(run_dir: Path, layer: int) -> np.ndarray:
    path = run_dir / f"hidden_layer{layer}_embeddings_norm.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing embeddings: {path}")
    return np.load(path)


def compute_query_metrics(
    sim: np.ndarray,
    folder_ids: np.ndarray,
    query_mask: np.ndarray,
    k_list: Sequence[int],
) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    n = sim.shape[0]
    max_k = max(k_list)
    hits: Dict[int, np.ndarray] = {k: np.full(n, np.nan) for k in k_list}
    best_same = np.full(n, np.nan)
    best_diff = np.full(n, np.nan)
    margin = np.full(n, np.nan)

    for i in range(n):
        if not query_mask[i]:
            continue
        scores = sim[i].copy()
        scores[i] = -np.inf
        same_mask = folder_ids == folder_ids[i]
        same_mask[i] = False
        if np.any(same_mask):
            best_same[i] = float(np.max(scores[same_mask]))
        diff_mask = ~same_mask
        diff_mask[i] = False
        if np.any(diff_mask):
            best_diff[i] = float(np.max(scores[diff_mask]))
        if np.isfinite(best_same[i]) and np.isfinite(best_diff[i]):
            margin[i] = float(best_same[i] - best_diff[i])

        top_idx = np.argpartition(-scores, max_k)[:max_k]
        top_sorted = top_idx[np.argsort(-scores[top_idx])]
        for k in k_list:
            topk = top_sorted[:k]
            hits[k][i] = 1.0 if np.any(folder_ids[topk] == folder_ids[i]) else 0.0

    return hits, best_same, best_diff, margin


def summarize_array(values: np.ndarray) -> Dict[str, float]:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "std": float("nan"),
        }
    return {
        "count": int(vals.size),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p25": float(np.percentile(vals, 25)),
        "p75": float(np.percentile(vals, 75)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "std": float(np.std(vals)),
    }


def write_figures(
    out_dir: Path,
    margins: Dict[str, np.ndarray],
    compare_labels: Sequence[str],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for --write_figures") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    base = margins["baseline"]
    for label in compare_labels:
        plt.figure(figsize=(7, 4))
        plt.hist(base[np.isfinite(base)], bins=60, alpha=0.6, label="baseline")
        plt.hist(margins[label][np.isfinite(margins[label])], bins=60, alpha=0.6, label=label)
        plt.xlabel("margin (best_same - best_diff)")
        plt.ylabel("count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"margin_hist_{label}.png", dpi=160)
        plt.close()


def main() -> None:
    args = parse_args()
    phase3_run_dir = Path(args.phase3_run_dir)
    if not phase3_run_dir.exists():
        raise FileNotFoundError(f"Phase 3 run dir not found: {phase3_run_dir}")

    out_root_dir = Path(args.out_root_dir) if args.out_root_dir else Path.cwd() / "runs"
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_run_dir = out_root_dir / "phase4_hard_queries" / run_id
    out_run_dir.mkdir(parents=True, exist_ok=True)

    k_list = parse_k_list(args.k_list)

    compare_runs: List[CompareRun] = []
    if args.compare_run_dir:
        labels = args.compare_label or []
        if labels and len(labels) != len(args.compare_run_dir):
            raise ValueError("compare_label count must match compare_run_dir count.")
        for idx, path_str in enumerate(args.compare_run_dir):
            label = labels[idx] if idx < len(labels) else f"compare_{idx+1}"
            compare_runs.append(CompareRun(label=label, run_dir=Path(path_str)))
    else:
        specs = parse_compare_specs(args.compare or ["variance:drop_lowest_percent=25", "variance:drop_lowest_percent=50"])
        screening_csv = phase3_run_dir / "phase3_screening_scoreboard.csv"
        if not screening_csv.exists():
            raise FileNotFoundError(f"Missing screening CSV: {screening_csv}")
        compare_runs = select_from_screening(screening_csv, specs)

    base_meta = load_meta(phase3_run_dir)
    for compare in compare_runs:
        compare_meta = load_meta(compare.run_dir)
        ensure_meta_alignment(base_meta, compare_meta, compare.run_dir)

    if args.query_mask_col:
        if args.query_mask_col not in base_meta.columns:
            raise ValueError(f"Missing query mask column: {args.query_mask_col}")
        query_mask = base_meta[args.query_mask_col].to_numpy(dtype=bool)
        query_mask_col = args.query_mask_col
    else:
        query_mask = base_meta["is_winnable"].to_numpy(dtype=bool)
        query_mask_col = "is_winnable"

    folder_ids = base_meta["folder_id"].to_numpy()

    run_metrics: Dict[str, Dict[str, np.ndarray]] = {}
    run_margins: Dict[str, np.ndarray] = {}
    sanity_payload: Dict[str, dict] = {}

    base_emb = load_embeddings(phase3_run_dir, args.layer)
    base_sim = similarity_matrix(base_emb)
    sanity_payload["baseline"] = sanity_checks(base_sim)
    base_hits, base_best_same, base_best_diff, base_margin = compute_query_metrics(
        base_sim, folder_ids, query_mask, k_list
    )
    run_metrics["baseline"] = {
        "hits": base_hits,
        "best_same": base_best_same,
        "best_diff": base_best_diff,
        "margin": base_margin,
    }
    run_margins["baseline"] = base_margin

    for compare in compare_runs:
        emb = load_embeddings(compare.run_dir, args.layer)
        sim = similarity_matrix(emb)
        sanity_payload[compare.label] = sanity_checks(sim)
        hits, best_same, best_diff, margin = compute_query_metrics(
            sim, folder_ids, query_mask, k_list
        )
        run_metrics[compare.label] = {
            "hits": hits,
            "best_same": best_same,
            "best_diff": best_diff,
            "margin": margin,
        }
        run_margins[compare.label] = margin

    hard_rows: List[Dict[str, object]] = []
    for idx, row in base_meta.iterrows():
        row_payload: Dict[str, object] = {
            "file_id": int(row.get("file_id", idx)),
            "folder_id": row["folder_id"],
            "filename": row["filename"],
            "path": row["path"],
            "query_mask": bool(query_mask[idx]),
        }
        for k in k_list:
            row_payload[f"baseline_hit@{k}"] = run_metrics["baseline"]["hits"][k][idx]
        row_payload["baseline_best_same"] = run_metrics["baseline"]["best_same"][idx]
        row_payload["baseline_best_diff"] = run_metrics["baseline"]["best_diff"][idx]
        row_payload["baseline_margin"] = run_metrics["baseline"]["margin"][idx]

        for compare in compare_runs:
            label = compare.label
            for k in k_list:
                row_payload[f"{label}_hit@{k}"] = run_metrics[label]["hits"][k][idx]
            row_payload[f"{label}_best_same"] = run_metrics[label]["best_same"][idx]
            row_payload[f"{label}_best_diff"] = run_metrics[label]["best_diff"][idx]
            row_payload[f"{label}_margin"] = run_metrics[label]["margin"][idx]
            row_payload[f"{label}_margin_delta"] = (
                run_metrics[label]["margin"][idx] - run_metrics["baseline"]["margin"][idx]
                if np.isfinite(run_metrics[label]["margin"][idx])
                and np.isfinite(run_metrics["baseline"]["margin"][idx])
                else np.nan
            )

        hard_rows.append(row_payload)

    hard_df = pd.DataFrame(hard_rows)
    hard_path = out_run_dir / "hard_queries.csv"
    hard_df.to_csv(hard_path, index=False)

    summary_rows: List[Dict[str, object]] = []
    total_queries = int(np.sum(query_mask))
    for compare in compare_runs:
        label = compare.label
        for k in k_list:
            base_hits = run_metrics["baseline"]["hits"][k]
            comp_hits = run_metrics[label]["hits"][k]
            base_hit_mask = base_hits == 1.0
            comp_hit_mask = comp_hits == 1.0
            recovered = int(np.sum(~base_hit_mask & comp_hit_mask & query_mask))
            regressed = int(np.sum(base_hit_mask & ~comp_hit_mask & query_mask))
            unchanged = int(np.sum((base_hit_mask == comp_hit_mask) & query_mask))
            summary_rows.append(
                {
                    "compare_label": label,
                    "k": int(k),
                    "total_queries": total_queries,
                    "baseline_hits": int(np.nansum(base_hit_mask & query_mask)),
                    "compare_hits": int(np.nansum(comp_hit_mask & query_mask)),
                    "recovered": recovered,
                    "regressed": regressed,
                    "unchanged": unchanged,
                    "net_gain": recovered - regressed,
                    "baseline_hit_rate": float(
                        np.nansum(base_hit_mask & query_mask) / total_queries
                    )
                    if total_queries
                    else 0.0,
                    "compare_hit_rate": float(
                        np.nansum(comp_hit_mask & query_mask) / total_queries
                    )
                    if total_queries
                    else 0.0,
                    "recovered_rate": recovered / total_queries if total_queries else 0.0,
                    "regressed_rate": regressed / total_queries if total_queries else 0.0,
                    "net_gain_rate": (recovered - regressed) / total_queries
                    if total_queries
                    else 0.0,
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_run_dir / "recovery_regression_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    pivot_rows: List[Dict[str, object]] = []
    for compare in compare_runs:
        label = compare.label
        row_payload: Dict[str, object] = {
            "compare_label": label,
            "total_queries": total_queries,
        }
        for k in k_list:
            match = summary_df[
                (summary_df["compare_label"] == label) & (summary_df["k"] == int(k))
            ]
            if match.empty:
                continue
            data = match.iloc[0]
            row_payload[f"baseline_hits@{k}"] = int(data["baseline_hits"])
            row_payload[f"compare_hits@{k}"] = int(data["compare_hits"])
            row_payload[f"baseline_hit_rate@{k}"] = float(data["baseline_hit_rate"])
            row_payload[f"compare_hit_rate@{k}"] = float(data["compare_hit_rate"])
            row_payload[f"recovered@{k}"] = int(data["recovered"])
            row_payload[f"regressed@{k}"] = int(data["regressed"])
            row_payload[f"net_gain@{k}"] = int(data["net_gain"])
            row_payload[f"recovered_rate@{k}"] = float(data["recovered_rate"])
            row_payload[f"regressed_rate@{k}"] = float(data["regressed_rate"])
            row_payload[f"net_gain_rate@{k}"] = float(data["net_gain_rate"])
        pivot_rows.append(row_payload)
    pivot_df = pd.DataFrame(pivot_rows)
    pivot_path = out_run_dir / "recovery_regression_pivot.csv"
    pivot_df.to_csv(pivot_path, index=False)

    margin_rows: List[Dict[str, object]] = []
    base_summary = summarize_array(run_margins["baseline"])
    margin_rows.append({"run_label": "baseline", "metric": "margin", **base_summary})
    for compare in compare_runs:
        label = compare.label
        margin_rows.append(
            {"run_label": label, "metric": "margin", **summarize_array(run_margins[label])}
        )
        delta = run_margins[label] - run_margins["baseline"]
        margin_rows.append(
            {"run_label": label, "metric": "margin_delta", **summarize_array(delta)}
        )
    margin_df = pd.DataFrame(margin_rows)
    margin_path = out_run_dir / "margin_summary.csv"
    margin_df.to_csv(margin_path, index=False)

    config = {
        "phase3_run_dir": str(phase3_run_dir),
        "layer": args.layer,
        "k_list": k_list,
        "query_mask_col": query_mask_col,
        "compare_runs": [
            {"label": c.label, "run_dir": str(c.run_dir)} for c in compare_runs
        ],
        "sanity_checks": sanity_payload,
        "outputs": {
            "hard_queries_csv": hard_path.name,
            "recovery_regression_csv": summary_path.name,
            "recovery_regression_pivot_csv": pivot_path.name,
            "margin_summary_csv": margin_path.name,
        },
    }
    with open(out_run_dir / "phase4_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    if args.write_figures:
        fig_dir = out_run_dir / "figures"
        write_figures(fig_dir, run_margins, [c.label for c in compare_runs])

    print(f"Phase 4 outputs: {out_run_dir}")


if __name__ == "__main__":
    main()
