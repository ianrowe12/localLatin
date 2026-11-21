#!/usr/bin/env python3
"""Fit corrections on decretal excerpts, then score the full canon testbed.

Run from the repo root. Both cache roots contain phase9_bases/ and default to the
Delta layout. The H1 reproduction gate runs before scoring and exits 3 on failure.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path, PurePosixPath
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abtt_subspace_whiten as asw  # noqa: E402

EXT_ROOT = Path("runs/active/decritals_bases")
OUT = asw.OUT_ROOT / "external_fixes"
N_CANON = 1705
N_DECRETALS = 439
ABTT_D = (1, 3, 10)
RANKS = (64, 128, 256)
SD_FLOOR_RATIO = 1e-8
GATE_TOL = 1e-6
RESTORE_AUROC = 0.90
METRICS = ("auroc", "acc1", "mrr")
GROUPS = {
    "none": ("none",),
    "center": ("center",),
    "abtt": tuple(f"abtt_D{d}" for d in ABTT_D),
    "std_coord": ("std_coord",),
    "std_comp": tuple(f"std_comp_r{r}" for r in RANKS),
    "var_comp": tuple(f"var_comp_r{r}" for r in RANKS),
}


def coordinate_scale(fit: np.ndarray) -> tuple[np.ndarray, int]:
    sd = np.asarray(fit, dtype=np.float64).std(axis=0)
    floor = SD_FLOOR_RATIO * sd.max()
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("coordinate standardization requires finite, nonconstant fit data")
    return np.maximum(sd, floor), int((sd < floor).sum())


def pca(centered: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, singular_values, pcs = np.linalg.svd(centered, full_matrices=False)
    return pcs, singular_values**2 / (len(centered) - 1)


def component_standardize(
    scores: np.ndarray, variance: np.ndarray, rank: int, *, by_variance: bool,
) -> np.ndarray:
    scale = variance[:rank] if by_variance else np.sqrt(variance[:rank])
    if not 0 < rank <= min(scores.shape[1], len(variance)) or np.any(~np.isfinite(scale) | (scale <= 0)):
        raise ValueError(f"cannot standardize {rank} principal components")
    return scores[:, :rank] / scale


def corrected(canon: np.ndarray, decretals: np.ndarray) -> Iterator[
    tuple[str, np.ndarray, np.ndarray | None, int]
]:
    """Yield each correction's canon/decretal arrays and the number of floored SDs."""
    yield "none", canon, decretals, 0
    ext = decretals.astype(np.float64)
    mu = ext.mean(axis=0)
    a, b = canon - mu, ext - mu
    yield "center", a, b, 0
    if max(RANKS) >= len(ext):
        raise ValueError("PCA ranks must be smaller than the number of decretal witnesses")
    pcs, var = pca(b)
    a_pc, b_pc = a @ pcs[:max(RANKS)].T, b @ pcs[:max(RANKS)].T
    for d in ABTT_D:
        yield f"abtt_D{d}", a - a_pc[:, :d] @ pcs[:d], b - b_pc[:, :d] @ pcs[:d], 0
    sd, n_floored = coordinate_scale(ext)
    yield "std_coord", a / sd, b / sd, n_floored
    for group in ("std_comp", "var_comp"):
        for rank in RANKS:
            by_variance = group == "var_comp"
            yield (f"{group}_r{rank}", component_standardize(a_pc, var, rank, by_variance=by_variance),
                   component_standardize(b_pc, var, rank, by_variance=by_variance), 0)

    # These references see the scored vectors, so they cannot enter external selection.
    canon64 = canon.astype(np.float64)
    centered = canon64 - canon64.mean(axis=0)
    pcs, _ = pca(centered)
    u = pcs[:3]
    yield "canon_all_abtt_D3", centered - (centered @ u.T) @ u, None, 0
    sd, n_floored = coordinate_scale(canon64)
    yield "canon_all_std_coord", centered / sd, None, n_floored


@dataclass
class ScoreLabels:
    triangle: tuple[np.ndarray, np.ndarray]
    pair: np.ndarray
    partner: np.ndarray
    queries: np.ndarray

    @classmethod
    def from_ids(cls, ids: np.ndarray) -> ScoreLabels:
        triangle = np.triu_indices(len(ids), k=1)
        same = ids[:, None] == ids
        np.fill_diagonal(same, False)
        return cls(triangle, same[triangle], same, same.any(axis=1))


def loo_metrics(sim: np.ndarray, labels: ScoreLabels) -> tuple[float, float]:
    """Rank the first non-self partner, breaking cosine ties by row order."""
    if not labels.queries.any():
        raise ValueError("leave-one-out retrieval needs at least one same-directory pair")
    if not labels.queries.all():
        sim = sim[labels.queries]
        partner = labels.partner[labels.queries]
    else:
        partner = labels.partner
    rows = np.arange(len(sim))
    best = np.where(partner, sim, -np.inf).argmax(axis=1)
    top = sim[rows, best, None]
    ahead = (sim > top) | ((sim == top) & (np.arange(sim.shape[1]) < best[:, None]))
    ahead[rows, np.flatnonzero(labels.queries)] = False
    rank = 1 + ahead.sum(axis=1)
    return float((rank == 1).mean()), float((1 / rank).mean())


def score(emb: np.ndarray, labels: ScoreLabels) -> dict[str, float]:
    sim = asw.similarity_matrix(asw.l2_normalize(emb))
    auroc = roc_auc_score(labels.pair, sim[labels.triangle])
    acc1, mrr = loo_metrics(sim, labels)
    return {"auroc": auroc, "acc1": acc1, "mrr": mrr}


def cache_dir(root: Path, model: str) -> Path:
    return root / "phase9_bases" / asw.slug(model) / asw.SUBDIR


def init_worker(split: pd.DataFrame, ext_root: Path) -> None:
    threadpool_limits(limits=1)
    global _resolver, _test, _test_labels, _canon_labels, _ext_root, _ext_labels
    _resolver = asw.AlignmentResolver(split)
    _test = split["split"].to_numpy() == "test"
    ids = split["folder_id"].to_numpy()
    _test_labels = ScoreLabels.from_ids(ids[_test])
    _canon_labels = ScoreLabels.from_ids(ids)
    _ext_root = ext_root
    _ext_labels = {}


def load_canon(root: Path, model: str, layer: int) -> np.ndarray:
    return _resolver.load(cache_dir(root, model) / f"hidden_layer{layer}_embeddings.npy")


def gate_task(task: tuple[Path, str, int]) -> tuple[str, int, float]:
    _, model, layer = task
    return model, layer, score(load_canon(*task)[_test], _test_labels)["auroc"]


def score_task(task: tuple[Path, str, int]) -> list[dict]:
    _, model, layer = task
    canon = load_canon(*task)
    directory = cache_dir(_ext_root, model)
    decretals = np.load(directory / f"hidden_layer{layer}_embeddings.npy")
    if decretals.shape != (N_DECRETALS, canon.shape[1]):
        raise ValueError(f"unexpected decretal vector shape for {model} layer {layer}: {decretals.shape}")
    if model not in _ext_labels:
        paths = pd.read_csv(directory / "meta.csv")["path"]
        if len(paths) != N_DECRETALS or paths.isna().any():
            raise ValueError(f"expected {N_DECRETALS} decretal paths for {model}")
        ids = np.array([PurePosixPath(path).parent.name for path in paths])
        _ext_labels[model] = ScoreLabels.from_ids(ids)
    rows = []
    for name, a, b, n_floored in corrected(canon, decretals):
        prefix = {"model": model, "layer": layer, "correction": name}
        rows.append({**prefix, "scored_on": "canon", **score(a, _canon_labels), "n_floored": n_floored})
        if b is not None:
            rows.append({**prefix, "scored_on": "decretals", **score(b, _ext_labels[model]),
                         "n_floored": n_floored})
    return rows


def select_settings(scores: pd.DataFrame) -> pd.DataFrame:
    canon = scores[scores.scored_on == "canon"].set_index(["model", "layer", "correction"])
    external = scores[scores.scored_on == "decretals"]
    rows = []
    for model, group in external.groupby("model", sort=False):
        for method, corrections in GROUPS.items():
            candidates = group[group.correction.isin(corrections)].copy()
            candidates["grid_order"] = candidates.correction.map({c: i for i, c in enumerate(corrections)})
            candidates = candidates.sort_values(["layer", "grid_order"])
            picked = candidates.loc[candidates.auroc.idxmax()]
            result = canon.loc[(model, picked.layer, picked.correction)]
            raw = canon.loc[(model, picked.layer, "none")]
            rows.append({"model": model, "method": method, "layer": picked.layer,
                         "correction": picked.correction, "decretals_auroc": picked.auroc,
                         **{f"canon_{k}": result[k] for k in METRICS},
                         **{f"same_layer_none_{k}": raw[k] for k in METRICS}})
    return pd.DataFrame(rows)


def summarize(scores: pd.DataFrame, selected: pd.DataFrame, gate: pd.DataFrame) -> str:
    collapsed = gate[gate.h1_test_auroc < asw.COLLAPSE_AUROC].set_index(["model", "layer"]).index
    canon = scores[scores.scored_on == "canon"].set_index(["model", "layer"])
    lines = [
        "# External corrections on the full canon testbed", "",
        "Corrections use only the 439 decretal embeddings, except the two label-free `canon_all`",
        "references, which fit on the same 1,705 witnesses they score. Neither canon embeddings",
        "nor canon labels enter external fits or selection. Coordinate SD uses population SD;",
        "component scaling uses the sample eigenvalues (or their square roots).", "",
        "Pair AUROC uses all 1,452,660 unordered canon pairs. Acc@1 and MRR exclude singleton",
        "queries, not singleton candidates. Ties in retrieval follow split order (cache metadata",
        "order for decretals). The SD floor count is repeated on both score rows for the same fit.", "",
        f"The raw test-split reproduction gate passed {int(gate.passed.sum())}/{len(gate)} layers "
        f"(maximum absolute AUROC difference {gate.abs_error.max():.3g}; tolerance {GATE_TOL:g}).", "",
        "## Canon scores at decretal-selected settings", "",
        "Each cell shows layer (and D/r where applicable), then AUROC / Acc@1 / MRR.",
        "Each column independently takes the first maximum decretal pair AUROC, with ties",
        "resolved by lowest layer then smallest D/r. `selected.csv` also reports the uncorrected",
        "canon scores at each chosen layer for same-layer comparisons.", "",
        "| Model | None | Center | ABTT | Coordinate SD | Component SD | Component variance |",
        "|---|---|---|---|---|---|---|",
    ]
    for model, group in selected.groupby("model", sort=False):
        cells = []
        for row in group.itertuples(index=False):
            setting = f"L{row.layer}"
            if row.method in ("abtt", "std_comp", "var_comp"):
                setting += "/" + row.correction.rsplit("_", 1)[1]
            cells.append(f"{setting}: {row.canon_auroc:.3f} / {row.canon_acc1:.3f} / {row.canon_mrr:.3f}")
        lines.append(f"| {asw.DISP[model]} | " + " | ".join(cells) + " |")
    lines += [
        "", "## Restoration and medians across all model-layers", "",
        f"Restored means full-canon AUROC >= {RESTORE_AUROC:.2f} on a layer whose raw H1",
        f"test AUROC was < {asw.COLLAPSE_AUROC:.2f}.", "",
        "| Correction | Restored | Median AUROC | Median Acc@1 | Median MRR |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, group in canon.groupby("correction", sort=False):
        restored = (group.loc[collapsed, "auroc"] >= RESTORE_AUROC).sum()
        med = group[list(METRICS)].median()
        lines.append(f"| `{name}` | {restored}/{len(collapsed)} | {med.auroc:.3f} | "
                     f"{med.acc1:.3f} | {med.mrr:.3f} |")
    return "\n".join(lines) + "\n"


def plot_depth(scores: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = (
        ("none", "None", "#000000", "-"),
        ("abtt_D3", "ABTT D=3", "#0072B2", "--"),
        ("std_coord", "Coordinate SD", "#D55E00", "-."),
        ("std_comp_r128", "Component SD r=128", "#009E73", ":"),
    )
    canon = scores[(scores.scored_on == "canon") & scores.correction.isin([s[0] for s in styles])]
    with plt.rc_context({"font.size": 9, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(4, 3, figsize=(10, 8), sharey="row", layout="constrained")
        for i, (model, label, *_) in enumerate(asw.MODELS):
            row, col = 2 * (i // 3), i % 3
            subset = canon[canon.model == model]
            for offset, (metric, ylabel, lower) in enumerate(
                (("auroc", "Pair AUROC", 0.45), ("acc1", "LOO Acc@1", 0)),
            ):
                ax = axes[row + offset, col]
                for name, legend, color, line in styles:
                    data = subset[subset.correction == name].sort_values("layer")
                    ax.plot(data.layer, data[metric], label=legend, color=color, linestyle=line)
                ax.set_ylim(min(lower, canon[metric].min()), 1.01)
                ax.grid(axis="y", alpha=0.25)
                if col == 0:
                    ax.set_ylabel(ylabel)
                if offset == 0:
                    ax.set_title(label)
                    ax.tick_params(labelbottom=False)
                    ax.sharex(axes[row + 1, col])
                else:
                    ax.set_xlabel("Layer")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=len(styles), frameon=False)
        fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
        plt.close(fig)


def main() -> None:
    start = perf_counter()
    cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cli.add_argument("--bases_root", type=Path, default=asw.BASES_ROOT)
    cli.add_argument("--ext_root", type=Path, default=EXT_ROOT)
    cli.add_argument("--workers", type=int, default=8)
    args = cli.parse_args()
    if args.workers < 1:
        cli.error("--workers must be positive")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"

    split = pd.read_csv(asw.SPLIT_CSV)
    if len(split) != N_CANON or set(split["split"]) != {"train", "test"} or split.folder_id.isna().any():
        raise SystemExit(f"expected {N_CANON} labelled canon witnesses in the paper split")
    raw = (pd.read_csv(asw.H1_CSV).query('variant == "raw"')[["model", "layer", "aucroc"]]
           .rename(columns={"aucroc": "h1_test_auroc"}))
    if (raw.h1_test_auroc < asw.COLLAPSE_AUROC).sum() != 26:
        raise SystemExit("expected 26 collapsed layers in the H1 raw rows")
    tasks = [
        (args.bases_root, model, layer)
        for model, *_ in asw.MODELS
        for layer in sorted(set(asw.discover(args.bases_root, model)) & set(asw.discover(args.ext_root, model)))
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers, initializer=init_worker, initargs=(split, args.ext_root)) as pool:
        measured = pd.DataFrame(pool.map(gate_task, tasks), columns=["model", "layer", "test_auroc"])
        gate = raw.merge(measured, on=["model", "layer"], how="outer", validate="one_to_one")
        gate["abs_error"] = (gate.test_auroc - gate.h1_test_auroc).abs()
        gate["passed"] = gate.abs_error <= GATE_TOL
        gate.to_csv(OUT / "gate.csv", index=False)
        status = f"H1 AUROC gate: {int(gate.passed.sum())}/{len(gate)} pass; max error {gate.abs_error.max():.3g}"
        if not gate.passed.all():
            print(f"{status}; mismatches:\n{gate[~gate.passed].to_string(index=False)}", file=sys.stderr)
            raise SystemExit(3)
        print(status, flush=True)
        rows = [row for part in pool.imap(score_task, tasks, chunksize=1) for row in part]

    scores = pd.DataFrame(rows)
    selected = select_settings(scores)
    scores.to_csv(OUT / "scores.csv", index=False, float_format="%.12g")
    selected.to_csv(OUT / "selected.csv", index=False, float_format="%.12g")
    (OUT / "summary.md").write_text(summarize(scores, selected, gate), encoding="utf-8")
    plot_depth(scores, OUT / "fig_external_fixes_depth.pdf")
    print(f"Scored {len(tasks)} model-layers with {args.workers} workers in {perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
