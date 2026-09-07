"""Cross-dataset retrieval: predict directory labels for unlabelled files.

For each model, loads labelled embeddings as database and unlabelled as queries,
computes cosine similarity, and outputs top-K directory predictions per
unlabelled file.

Variants (``--variant``)
------------------------
Each variant is a different embedding pipeline. The variant alone decides which
embedding files are read and whether ABTT (All-But-The-Top / principal-component
removal) is applied:

===========  ============  ==========  ====================================
variant      pooling       ABTT        embedding file loaded
===========  ============  ==========  ====================================
``raw``      mean          no          ``hidden_layer{N}_embeddings.npy``
``abtt``     mean          yes         ``hidden_layer{N}_embeddings.npy``
``sif``      SIF           no          ``hidden_layer{N}_embeddings_sif.npy``
``sif_abtt`` SIF           yes         ``hidden_layer{N}_embeddings_sif.npy``
===========  ============  ==========  ====================================

Layer selection
---------------
The per-model layer is read from the evaluation results CSV
(``runs/active/resubmit/results/phase_resubmit_results.csv``), picking the row
with the highest ``overall_assignment_acc`` *among the methods that correspond
to the requested variant*:

* ``raw``      -> ``baseline`` rows
* ``abtt``     -> ``abtt_fixed`` / ``abtt_optimal`` rows
* ``sif``      -> ``sif_only`` rows
* ``sif_abtt`` -> ``sif_abtt_fixed`` / ``sif_abtt_optimal`` rows

So every variant runs at the best layer *for that variant*, which makes the four
output CSVs directly comparable. ``--layer_overrides`` pins a layer per model
(e.g. ``"Qwen/Qwen3-Embedding-0.6B=5"``) for reproducing an older run.

Note: the ``pooling`` column of the results CSV is deliberately ignored. It is
implied by the variant, and trusting it used to mis-pair the two (the previous
version forced SIF files whenever the winning method was ``abtt_*``, which are
mean-pooled rows).

Sticky layers for the deployed CSVs
-----------------------------------
These CSVs are not a paper artifact. They are what the reviewer pilot serves: a
scholar opens a query and sees the top-1 directory this file predicted. So the
argmax above has a property the paper's tables do not have to care about, which
is that it is a *hard* argmax over near-tied layers, and a hundredth of a point
of movement in the selection metric can swing it. Issue #113's label correction
did exactly that: ``Qwen3-0.6B`` under ``sif_abtt`` moved from
``overall_assignment_acc`` 0.911422 at layer 7 to 0.913753 at layer 1, a 0.23
point rise, and the deployed layer flipped 7 to 1. Different layer, different
embedding space, and 1,276 of that model's 2,238 reviewer-facing top-1 answers
change: 57 percent of the shortlists a scholar may already have looked at, for a
correction that touches two files.

So the deployed layer is sticky. ``--deployed_layers_json`` names a committed
record of the layer each (variant, model) is currently serving, read off the
live prediction CSVs. A newly computed best layer replaces it only when it beats
the deployed layer on the same selection metric by more than
``--sticky_tolerance`` (default 0.005, i.e. half a point of assignment accuracy).
Below that the movement is not distinguishable from the noise a two-file
relabelling introduces, and continuity for the reviewer is worth more than the
last hundredth. Every decision is printed and written to
``sticky_layer_decisions_{variant}.json``.

Two things this rule is deliberately not:

* It is not a claim that the old layer is better. It is a claim that the new one
  is not better *enough* to be worth invalidating a live shortlist over.
* It is not used for the paper. The tables in ``overleaf_drafts/`` come from
  ``build_headline_tables.py`` and friends, which take the pure argmax with no
  stickiness at all. Pass ``--no_sticky_layers`` here to reproduce that.

``--layer_overrides`` still wins over both: an explicit pin is an explicit pin.

Recording the deployed layers
-----------------------------
``--record_deployed_layers`` reads the ``layer`` column of the prediction CSVs in
a directory and writes the JSON, then exits without scoring anything. Point it at
the CSVs that are actually live::

    python scripts/resubmit/run_resubmit_unlabelled_retrieval.py \\
        --record_deployed_layers runs/active/resubmit/unlabelled \\
        --deployed_layers_json scripts/resubmit/deployed_unlabelled_layers.json

Leak-free protocol
------------------
``EmbeddingCleaner`` is fitted on the *train* split only (per ``--split_csv``)
and then applied to the labelled database and the unlabelled queries. The D
value is swept over ``D_VALUES`` and chosen by assignment accuracy on train.

Degenerate-file guard (issue #66)
---------------------------------
A handful of corpus files are empty or whitespace-only. They carry no text to
embed, and the vector the model emits for them is an artefact of the tokenizer's
special tokens. Two failure modes follow:

* Under SIF pooling the weighted sum has no terms, so the embedding is exactly
  zero. ``l2_normalize`` floors the norm at ``eps`` and maps such a row back to
  zero, which looks like "cosine 0.0 with everything" (``sif`` variant). If ABTT
  runs first, mean-centering turns *every* zero row into the same vector
  ``-mean_vec``, so an empty query and an empty reference normalize to identical
  directions and score a spurious cosine of exactly **1.0** (``sif_abtt``).
* Under mean pooling the norm is non-zero, but two empty files tokenize to the
  same special-token sequence and can still score a genuine 1.0 (seen for LaBSE
  and mt5-base in the ``raw`` and ``abtt`` variants).

So the guard is applied at the **source** level, before any embedding is read:
any file whose ``.txt`` is empty or whitespace-only is excluded, and
:func:`canon_retrieval.zero_norm_mask` runs on the pre-ABTT embeddings as a
numeric backstop for anything the text check misses. Excluded *references* are
dropped from their directory's max-cosine; a directory that loses all of its
files is dropped from the candidate set entirely (warned about, and it can no
longer be predicted for any query). Excluded *queries* still get a row, with
every ``rank*`` cell left blank. Every exclusion is logged. Pass
``--no_degenerate_guard`` to reproduce the old, unguarded behaviour.

Output schema
-------------
``{out_dir}/unlabelled_predictions_{variant}.csv`` (all models concatenated) and
``{out_dir}/unlabelled_predictions_{variant}_{model_slug}.csv`` (per model), with
one row per (model, query):

    file_id, filename, file_path,
    rank1_dir, rank1_score, ..., rank{K}_dir, rank{K}_score,
    model, variant, layer, pooling, status

``rank{i}_dir`` is the labelled directory name, ``rank{i}_score`` the max cosine
similarity between the query and any file in that directory.

``status`` is ``ok`` for a scored query, or ``excluded_blank_source`` /
``excluded_zero_norm`` for a query the guard dropped (all ``rank*`` cells blank).
The row count is unchanged by the guard: one row per (model, query) either way.

The legacy ``unlabelled_predictions.csv`` / ``unlabelled_predictions_{slug}.csv``
files (no ``variant`` column) are the frozen webapp input and are no longer
written by this script; every output filename now carries the variant. To refresh
the webapp input, copy ``unlabelled_predictions_sif_abtt.csv`` over it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from canon_retrieval import (
    blank_text_mask,
    build_directory_index,
    l2_normalize,
    similarity_matrix,
    sweep_thresholds,
    top_k_directories,
    upper_triangle,
    upper_triangle_labels,
    zero_norm_mask,
)
from embedding_alignment import AlignmentResolver
from sif_abtt import EmbeddingCleaner

STATUS_OK = "ok"
STATUS_BLANK_SOURCE = "excluded_blank_source"
STATUS_ZERO_NORM = "excluded_zero_norm"

# Fallback layers if the results CSV is missing a model (repr is always "hidden").
FALLBACK_CONFIGS = [
    ("bowphs/LaTa", 6, "hidden"),
    ("bowphs/PhilTa", 1, "hidden"),
    ("sentence-transformers/LaBSE", 12, "hidden"),
    ("Qwen/Qwen3-Embedding-0.6B", 21, "hidden"),
    ("KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5", 1, "hidden"),
    ("google/mt5-base", 1, "hidden"),
]

ALL_MODELS = [c[0] for c in FALLBACK_CONFIGS]
D_VALUES = [1, 2, 3, 5, 7, 10]
TOP_K = 10

# The column the layer is chosen on. Sticky comparisons use the same column, so
# the rule can only ever keep a layer the unmodified selector had already been
# willing to pick.
SELECTION_METRIC = "overall_assignment_acc"

# How much better the new layer has to be before a live shortlist is invalidated.
# 0.005 is half a point of assignment accuracy, comfortably above the 0.0023 that
# the two-file label correction moved Qwen3-0.6B's sif_abtt layer by, and well
# below the 0.02-0.09 spread between a model's good and bad layers. See the
# "Sticky layers" section of the module docstring.
STICKY_TOLERANCE = 0.005

DEFAULT_DEPLOYED_LAYERS_JSON = (
    Path(__file__).resolve().parent / "deployed_unlabelled_layers.json"
)

# variant -> (pooling, apply_abtt, results-CSV methods used for layer selection)
VARIANTS: Dict[str, Tuple[str, bool, Tuple[str, ...]]] = {
    "raw": ("mean", False, ("baseline",)),
    "abtt": ("mean", True, ("abtt_fixed", "abtt_optimal")),
    "sif": ("sif", False, ("sif_only",)),
    "sif_abtt": ("sif", True, ("sif_abtt_fixed", "sif_abtt_optimal")),
}


def model_slug(name: str) -> str:
    return name.replace("/", "_")


def find_best_config_from_results(results_csv: str, methods: Tuple[str, ...]) -> Dict[str, Tuple[int, str]]:
    """Best layer per model by overall_assignment_acc, restricted to `methods`.

    Returns {model_name: (layer, repr)}. The results-CSV `pooling` column is not
    consulted: pooling is determined by the variant (see module docstring).
    """
    df = pd.read_csv(results_csv)
    df = df[df["method"].isin(methods)]
    best = {}
    for model_name in ALL_MODELS:
        mdf = df[df["model"] == model_name]
        if len(mdf) == 0:
            continue
        idx = mdf[SELECTION_METRIC].idxmax()
        row = mdf.loc[idx]
        best[model_name] = (int(row["layer"]), row["repr"])
    return best


def layer_scores_from_results(
    results_csv: str, methods: Tuple[str, ...]
) -> Dict[str, Dict[int, float]]:
    """{model: {layer: best selection-metric value at that layer}}.

    The sticky rule needs the metric at the *deployed* layer, which is not
    generally the argmax row, so it needs the whole per-layer curve rather than
    the single winner :func:`find_best_config_from_results` returns.
    """
    df = pd.read_csv(results_csv)
    df = df[df["method"].isin(methods)]
    scores: Dict[str, Dict[int, float]] = {}
    for model_name in ALL_MODELS:
        mdf = df[df["model"] == model_name]
        if len(mdf) == 0:
            continue
        by_layer = mdf.groupby("layer")[SELECTION_METRIC].max()
        scores[model_name] = {int(k): float(v) for k, v in by_layer.items()}
    return scores


def load_deployed_layers(path) -> Dict[str, Dict[str, int]]:
    """Read the committed record of what each (variant, model) is serving.

    Returns ``{variant: {model: layer}}``, empty when the file is absent. A
    missing file is not an error: it just means nothing is deployed yet and the
    plain argmax stands.
    """
    path = Path(path)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    layers = payload.get("layers", {})
    return {
        variant: {model: int(layer) for model, layer in per_model.items()}
        for variant, per_model in layers.items()
    }


def _clears(gain: float, tolerance: float) -> bool:
    """Is `gain` more than `tolerance`, ignoring floating-point representation?

    ``0.905 - 0.900`` is ``0.005000000000000004`` in binary floating point, which
    would flip a deployment decision on a rounding artefact. The metric here is a
    ratio over a few hundred test files, so its real granularity is around
    ``1e-3``; ``1e-9`` is far below anything meaningful and absorbs only noise.
    A gain exactly at the tolerance does not clear it.
    """
    return gain > tolerance and not math.isclose(gain, tolerance, abs_tol=1e-9)


def apply_sticky_layers(
    best_configs: Dict[str, Tuple[int, str]],
    deployed: Dict[str, int],
    scores: Dict[str, Dict[int, float]],
    tolerance: float,
) -> Tuple[Dict[str, Tuple[int, str]], List[Dict[str, object]]]:
    """Keep the deployed layer unless the new best layer clears `tolerance`.

    Returns the (possibly adjusted) configs and one decision record per model, so
    the caller can print them and write them next to the predictions. Models with
    no deployed layer recorded are passed through untouched.
    """
    out: Dict[str, Tuple[int, str]] = {}
    decisions: List[Dict[str, object]] = []

    for model_name, (new_layer, repr_name) in best_configs.items():
        deployed_layer = deployed.get(model_name)
        model_scores = scores.get(model_name, {})
        new_score = model_scores.get(new_layer)
        deployed_score = (
            None if deployed_layer is None else model_scores.get(deployed_layer)
        )

        if deployed_layer is None:
            chosen, action, why = new_layer, "new", "no deployed layer recorded"
        elif deployed_layer == new_layer:
            chosen, action, why = new_layer, "unchanged", "argmax equals deployed layer"
        elif deployed_score is None:
            chosen, action, why = (
                new_layer,
                "new",
                f"deployed layer {deployed_layer} has no row in the results CSV",
            )
        elif new_score is not None and _clears(new_score - deployed_score, tolerance):
            chosen, action, why = (
                new_layer,
                "switched",
                f"beats deployed layer by {new_score - deployed_score:.6f} "
                f"> tolerance {tolerance}",
            )
        else:
            margin = "unknown" if new_score is None else f"{new_score - deployed_score:.6f}"
            chosen, action, why = (
                deployed_layer,
                "kept",
                f"argmax layer {new_layer} beats deployed layer by {margin}, "
                f"not more than tolerance {tolerance}",
            )

        out[model_name] = (chosen, repr_name)
        decisions.append(
            {
                "model": model_name,
                "deployed_layer": deployed_layer,
                "argmax_layer": new_layer,
                "chosen_layer": chosen,
                "action": action,
                "reason": why,
                f"deployed_{SELECTION_METRIC}": deployed_score,
                f"argmax_{SELECTION_METRIC}": new_score,
            }
        )

    return out, decisions


def _repo_relative(path) -> str:
    """Path relative to the repo root, or the path unchanged if it is outside."""
    repo_root = Path(__file__).resolve().parents[2]
    try:
        return str(Path(path).resolve().relative_to(repo_root))
    except ValueError:
        return str(path)


def record_deployed_layers(
    predictions_dir, out_json, variants: Sequence[str] = tuple(sorted(VARIANTS))
) -> Dict[str, Dict[str, int]]:
    """Read the live prediction CSVs' `layer` column and write the sticky record.

    This is how the JSON is produced in the first place, and how it is refreshed
    after a deliberate layer change ships: the source of truth is what the CSVs
    on disk are actually serving, not what anyone remembers deciding.
    """
    predictions_dir = Path(predictions_dir)
    layers: Dict[str, Dict[str, int]] = {}
    for variant in variants:
        csv_path = predictions_dir / f"unlabelled_predictions_{variant}.csv"
        if not csv_path.exists():
            print(f"  {variant}: no {csv_path.name}, skipping.")
            continue
        df = pd.read_csv(csv_path, usecols=["model", "layer"])
        per_model: Dict[str, int] = {}
        for model_name, group in df.groupby("model"):
            seen = sorted(set(int(x) for x in group["layer"]))
            if len(seen) != 1:
                raise SystemExit(
                    f"{csv_path} has {len(seen)} distinct layers for {model_name}: "
                    f"{seen}. A deployed CSV must serve one layer per model."
                )
            per_model[str(model_name)] = seen[0]
        layers[variant] = per_model
        print(f"  {variant}: {per_model}")

    payload = {
        "description": (
            "Layer each (variant, model) is currently serving in the reviewer "
            "pilot. run_resubmit_unlabelled_retrieval.py keeps these layers "
            "unless a re-run's best layer beats them on "
            f"{SELECTION_METRIC} by more than the sticky tolerance. Paper "
            "tables do not use this file; they take the pure argmax."
        ),
        "selection_metric": SELECTION_METRIC,
        "sticky_tolerance": STICKY_TOLERANCE,
        # Repo-relative when possible: an absolute path here is provenance that
        # stops being true the moment the checkout moves.
        "recorded_from": _repo_relative(predictions_dir),
        "recorded_on": date.today().isoformat(),
        "layers": layers,
    }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_json}")
    return layers


def parse_layer_overrides(spec: str) -> Dict[str, int]:
    """Parse 'model=layer,model=layer' into {model: layer}."""
    overrides: Dict[str, int] = {}
    if not spec:
        return overrides
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Bad --layer_overrides entry {item!r}; expected 'model=layer'.")
        name, layer = item.rsplit("=", 1)
        overrides[name.strip()] = int(layer)
    return overrides


def find_optimal_D(train_emb, train_folder_ids, D_values):
    """Sweep D values on train set, return best D by assignment accuracy."""
    best_D = D_values[0]
    best_score = -1.0
    unique, counts = np.unique(train_folder_ids, return_counts=True)
    multi_folders = set(unique[counts > 1])
    has_partner = np.array([fid in multi_folders for fid in train_folder_ids])

    for D in D_values:
        cleaner = EmbeddingCleaner(num_components=D, center=True)
        cleaner.fit(train_emb)
        cleaned = cleaner.transform(train_emb)
        cleaned_norm = l2_normalize(cleaned)
        sim = similarity_matrix(cleaned_norm)

        sims = upper_triangle(sim)
        labels = upper_triangle_labels(train_folder_ids)
        thresh_df = sweep_thresholds(sims, labels, np.linspace(0, 1, 200))
        best_f1_idx = thresh_df["f1"].idxmax()
        tau = float(thresh_df.loc[best_f1_idx, "threshold"])

        # Compute assignment accuracy
        n = sim.shape[0]
        existing_correct = 0
        existing_total = 0
        new_correct = 0
        new_total = 0
        for i in range(n):
            scores = sim[i].copy()
            scores[i] = -np.inf
            max_sim = float(np.max(scores))
            if has_partner[i]:
                existing_total += 1
                if max_sim >= tau:
                    existing_correct += 1
            else:
                new_total += 1
                if max_sim < tau:
                    new_correct += 1
        overall = (existing_correct + new_correct) / n if n > 0 else 0.0

        print(f"    D={D:>2d}  tau={tau:.4f}  assign_acc={overall:.4f}"
              f"  (exist={existing_correct}/{existing_total},"
              f" new={new_correct}/{new_total})")

        if overall > best_score:
            best_score = overall
            best_D = D

    return best_D


def resolve_source_path(path: str, data_root: Path) -> Path:
    """Locate a meta-CSV ``path`` entry on disk.

    The two meta CSVs disagree on their root: ``unlabelled_meta.csv`` stores
    ``canon_unlabelled/x.txt`` (relative to ``data/``) while
    ``phase_resubmit_split.csv`` stores ``data/canon_labelled/y/z.txt``
    (relative to the repo root). Try both, plus the path as given.

    Raises ``FileNotFoundError`` if none of the candidates exists. This is
    deliberately fatal: a silently unresolvable path would make the guard a
    no-op and quietly reintroduce the bug it exists to prevent.
    """
    p = str(path)
    for candidate in (data_root / p, data_root.parent / p, Path(p)):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not locate source file {p!r} under {data_root} or {data_root.parent}."
    )


def read_blank_source_mask(paths: Sequence[str], data_root: Path) -> np.ndarray:
    """Boolean mask of files whose .txt is empty or whitespace-only."""
    texts: List[str] = []
    for p in paths:
        texts.append(
            resolve_source_path(p, data_root).read_text(encoding="utf-8", errors="replace")
        )
    return blank_text_mask(texts)


def predict_directories(
    labelled_emb_norm: np.ndarray,
    unlabelled_emb_norm: np.ndarray,
    folder_ids: np.ndarray,
    top_k: int = 10,
    exclude_labelled: np.ndarray = None,
    exclude_unlabelled: np.ndarray = None,
) -> List[List[Tuple[str, float]]]:
    """For each unlabelled file, find top-K directories by max cosine similarity.

    Reference files flagged in ``exclude_labelled`` are removed from their
    directory's max; a directory left with no usable file is removed from the
    candidate set. Queries flagged in ``exclude_unlabelled`` get an empty
    prediction list instead of a score. See :func:`build_directory_index`.
    """
    dir_to_indices, dropped_dirs, _ = build_directory_index(
        folder_ids, exclude_refs=exclude_labelled
    )
    if dropped_dirs:
        print(
            f"  WARNING: {len(dropped_dirs)} labelled director"
            f"{'y is' if len(dropped_dirs) == 1 else 'ies are'} unusable "
            "(every file excluded) and cannot be predicted:"
        )
        for d in dropped_dirs:
            print(f"    - {d}")
    print(f"  Candidate directories: {len(dir_to_indices)} usable")

    # Cross-similarity: (n_unlabelled, n_labelled)
    cross_sim = unlabelled_emb_norm @ labelled_emb_norm.T

    n_queries = cross_sim.shape[0]
    if exclude_unlabelled is None:
        exclude_unlabelled = np.zeros(n_queries, dtype=bool)

    predictions: List[List[Tuple[str, float]]] = []
    for q_idx in range(n_queries):
        if exclude_unlabelled[q_idx]:
            predictions.append([])
            continue
        predictions.append(top_k_directories(cross_sim[q_idx], dir_to_indices, top_k))

    return predictions


def resolve_model_configs(
    args: argparse.Namespace, sel_methods: Tuple[str, ...], variant: str
) -> Tuple[List[Tuple[str, int, str]], List[Dict[str, object]]]:
    """(model, layer, repr) per model, plus the sticky-layer decision records.

    Precedence, weakest first: the hardcoded fallback, the results-CSV argmax,
    the sticky rule, and finally ``--layer_overrides``, which is an explicit pin
    and wins over everything.

    ``build_qq_matrices.py`` calls this too. The query-query matrices have to be
    built at exactly the layer the predictions were built at or the webapp scores
    reviewer-created directories in a different space from everything else, so
    the two scripts share one implementation rather than two that agree today.
    """
    results_path = Path(args.results_csv)
    if results_path.exists():
        print(f"Reading best layers from {results_path}")
        best_configs = find_best_config_from_results(args.results_csv, sel_methods)
        scores = layer_scores_from_results(args.results_csv, sel_methods)
    else:
        print("Results CSV not found, using fallback configs.")
        best_configs, scores = {}, {}

    decisions: List[Dict[str, object]] = []
    sticky = not getattr(args, "no_sticky_layers", False)
    if sticky and best_configs:
        deployed_all = load_deployed_layers(args.deployed_layers_json)
        deployed = deployed_all.get(variant, {})
        if not deployed:
            print(
                f"Sticky layers: nothing recorded for variant {variant!r} in "
                f"{args.deployed_layers_json}; taking the plain argmax."
            )
        else:
            print(
                f"Sticky layers: comparing against {args.deployed_layers_json} "
                f"on {SELECTION_METRIC}, tolerance {args.sticky_tolerance}"
            )
            best_configs, decisions = apply_sticky_layers(
                best_configs, deployed, scores, args.sticky_tolerance
            )
            for d in decisions:
                print(
                    f"  [sticky] {d['model']}: {d['action']} -> layer "
                    f"{d['chosen_layer']} ({d['reason']})"
                )
    elif not sticky:
        print("Sticky layers DISABLED (--no_sticky_layers): pure argmax, as the paper uses.")

    layer_overrides = parse_layer_overrides(args.layer_overrides)

    configs: List[Tuple[str, int, str]] = []
    for model_name, fallback_layer, fallback_repr in FALLBACK_CONFIGS:
        if model_name in best_configs:
            layer, repr_name = best_configs[model_name]
            source = "from results"
        else:
            layer, repr_name = fallback_layer, fallback_repr
            source = "fallback"
        if model_name in layer_overrides:
            layer = layer_overrides[model_name]
            source = "override"
        print(f"  {model_name}: layer={layer}, repr={repr_name} ({source})")
        configs.append((model_name, layer, repr_name))

    if args.models != "all":
        selected = [m.strip() for m in args.models.split(",")]
        configs = [c for c in configs if c[0] in selected]

    return configs, decisions


def parse_args():
    parser = argparse.ArgumentParser(description="Unlabelled -> labelled cross-dataset retrieval.")
    parser.add_argument("--labelled_bases", default="runs/active/resubmit_bases/phase9_bases",
                        help="Root for labelled embeddings.")
    parser.add_argument("--unlabelled_bases", default="runs/active/resubmit/unlabelled/bases",
                        help="Root for unlabelled embeddings.")
    parser.add_argument("--split_csv", default="runs/active/resubmit/data/phase_resubmit_split.csv")
    parser.add_argument("--unlabelled_meta", default="runs/active/resubmit/data/unlabelled_meta.csv")
    parser.add_argument("--results_csv", default="runs/active/resubmit/results/phase_resubmit_results.csv",
                        help="Evaluation results CSV for finding the best layer per model.")
    parser.add_argument("--out_dir", default="runs/active/resubmit/unlabelled")
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--models", default="all", help="Comma-separated model names or 'all'")
    parser.add_argument("--variant", default="sif_abtt", choices=sorted(VARIANTS),
                        help="Embedding pipeline: raw | abtt | sif | sif_abtt (see module docstring).")
    parser.add_argument("--layer_overrides", default="",
                        help="Pin layers, e.g. 'Qwen/Qwen3-Embedding-0.6B=5,google/mt5-base=1'.")
    parser.add_argument("--deployed_layers_json", default=str(DEFAULT_DEPLOYED_LAYERS_JSON),
                        help="Committed record of the layer each (variant, model) currently "
                             "serves. Drives the sticky-layer rule; see the module docstring.")
    parser.add_argument("--sticky_tolerance", type=float, default=STICKY_TOLERANCE,
                        help=f"How much better ({SELECTION_METRIC}) a new layer must be before "
                             "it replaces the deployed one.")
    parser.add_argument("--no_sticky_layers", action="store_true",
                        help="Take the pure argmax, ignoring the deployed record. This is what "
                             "the paper tables use.")
    parser.add_argument("--record_deployed_layers", default="",
                        help="Read the layer column of the prediction CSVs in this directory, "
                             "write --deployed_layers_json, and exit without scoring.")
    parser.add_argument("--data_root", default="data",
                        help="Root the `path` columns of the meta CSVs are relative to.")
    parser.add_argument("--no_degenerate_guard", action="store_true",
                        help="Disable the empty-file / zero-norm guard (reproduces pre-#66 output).")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.record_deployed_layers:
        print(f"Recording deployed layers from {args.record_deployed_layers}")
        record_deployed_layers(args.record_deployed_layers, args.deployed_layers_json)
        return

    variant = args.variant
    pooling, apply_abtt, sel_methods = VARIANTS[variant]
    print(f"Variant: {variant} (pooling={pooling}, abtt={apply_abtt}, "
          f"layer selected from methods {list(sel_methods)})")

    # Load metadata
    split_meta = pd.read_csv(args.split_csv)
    lab_aligner = AlignmentResolver(split_meta)
    unlabelled_meta = pd.read_csv(args.unlabelled_meta)

    # --- Source-level degenerate-file guard (issue #66) ---------------------
    # Computed once: it depends only on the .txt files, not on the model.
    data_root = Path(args.data_root)
    if args.no_degenerate_guard:
        print("\nDegenerate-file guard DISABLED (--no_degenerate_guard).")
        lab_blank = np.zeros(len(split_meta), dtype=bool)
        unlab_blank = np.zeros(len(unlabelled_meta), dtype=bool)
    else:
        lab_blank = read_blank_source_mask(split_meta["path"].tolist(), data_root)
        unlab_blank = read_blank_source_mask(unlabelled_meta["path"].tolist(), data_root)
        print(f"\nDegenerate-file guard: scanning sources under {data_root.resolve()}")
        if lab_blank.any():
            print(f"  WARNING: {int(lab_blank.sum())} labelled file(s) are empty/whitespace-only "
                  "and are excluded from their directory's max-cosine:")
            for _, r in split_meta[lab_blank].iterrows():
                print(f"    - {r['path']}  (dir: {r['folder_id']})")
        if unlab_blank.any():
            print(f"  WARNING: {int(unlab_blank.sum())} unlabelled quer(y/ies) are "
                  "empty/whitespace-only and will be emitted with no predictions:")
            for _, r in unlabelled_meta[unlab_blank].iterrows():
                print(f"    - file_id={r['file_id']}  {r['path']}")
        if not lab_blank.any() and not unlab_blank.any():
            print("  No empty/whitespace-only source files found.")

    model_configs, decisions = resolve_model_configs(args, sel_methods, variant)
    if decisions:
        out_path = Path(args.out_dir) / f"sticky_layer_decisions_{variant}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "variant": variant,
                    "selection_metric": SELECTION_METRIC,
                    "tolerance": args.sticky_tolerance,
                    "decisions": decisions,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"  Sticky-layer decisions -> {out_path}")

    all_predictions_dfs = []

    for model_name, opt_layer, repr_name in model_configs:
        slug = model_slug(model_name)
        print(f"\n{'='*60}")
        print(f"Model: {model_name} (layer {opt_layer}, {pooling}, variant {variant})")

        # Determine suffix based on pooling
        suffix = "_sif" if pooling == "sif" else ""
        subdir = f"{repr_name}_{pooling}_tokempty"
        fname = f"{repr_name}_layer{opt_layer}_embeddings{suffix}.npy"

        lab_path = Path(args.labelled_bases) / slug / subdir / fname
        unlab_path = Path(args.unlabelled_bases) / slug / subdir / fname

        if not lab_path.exists():
            print(f"  Labelled embeddings not found: {lab_path}, skipping.")
            continue
        if not unlab_path.exists():
            print(f"  Unlabelled embeddings not found: {unlab_path}, skipping.")
            continue

        lab_emb = lab_aligner.load(lab_path)
        unlab_emb = np.load(unlab_path)
        print(f"  Labelled: {lab_emb.shape}, Unlabelled: {unlab_emb.shape}")

        # Validate shapes
        if lab_emb.shape[0] != len(split_meta):
            print(f"  Shape mismatch: {lab_emb.shape[0]} vs {len(split_meta)} files, skipping.")
            continue
        if unlab_emb.shape[0] != len(unlabelled_meta):
            print(f"  Shape mismatch: {unlab_emb.shape[0]} vs {len(unlabelled_meta)} files, skipping.")
            continue

        # Numeric backstop, evaluated on the PRE-ABTT vectors. After ABTT with
        # center=True every zero row becomes -mean_vec, so they stop looking
        # degenerate and start looking identical to one another -- which is the
        # spurious cosine 1.0 of issue #66. Detect them before that happens.
        if args.no_degenerate_guard:
            lab_zero = np.zeros(lab_emb.shape[0], dtype=bool)
            unlab_zero = np.zeros(unlab_emb.shape[0], dtype=bool)
        else:
            lab_zero = zero_norm_mask(lab_emb)
            unlab_zero = zero_norm_mask(unlab_emb)
            extra_lab = int((lab_zero & ~lab_blank).sum())
            extra_unlab = int((unlab_zero & ~unlab_blank).sum())
            if extra_lab or extra_unlab:
                print(f"  WARNING: zero-norm backstop caught {extra_lab} labelled and "
                      f"{extra_unlab} unlabelled row(s) not flagged by the source check.")
            if lab_zero.any() or unlab_zero.any():
                print(f"  Zero-norm rows: labelled={int(lab_zero.sum())}, "
                      f"unlabelled={int(unlab_zero.sum())}")

        lab_exclude = lab_blank | lab_zero
        unlab_exclude = unlab_blank | unlab_zero
        if lab_exclude.any() or unlab_exclude.any():
            print(f"  Excluding {int(lab_exclude.sum())} labelled reference file(s) and "
                  f"{int(unlab_exclude.sum())} quer(y/ies).")

        if apply_abtt:
            # ABTT: fit the cleaner on the train split only, then apply to all.
            train_mask = split_meta["split"].values == "train"
            train_emb = lab_emb[train_mask]
            train_folder_ids = split_meta.loc[train_mask, "folder_id"].values

            # Degenerate rows are deliberately left in the cleaner's fit: they are
            # 2 of 1,705 and dropping them would perturb mean_vec/pcs, hence every
            # query's scores. The guard acts at scoring time, not at fit time.
            best_D = find_optimal_D(train_emb, train_folder_ids, D_VALUES)
            print(f"  Optimal D: {best_D}")

            cleaner = EmbeddingCleaner(num_components=best_D, center=True)
            cleaner.fit(train_emb)
            lab_emb = cleaner.transform(lab_emb)
            unlab_emb = cleaner.transform(unlab_emb)
        else:
            print("  No ABTT for this variant (cosine on raw pooled vectors).")

        lab_norm = l2_normalize(lab_emb)
        unlab_norm = l2_normalize(unlab_emb)

        # Predict top-K directories
        folder_ids = split_meta["folder_id"].values
        predictions = predict_directories(
            lab_norm,
            unlab_norm,
            folder_ids,
            top_k=args.top_k,
            exclude_labelled=lab_exclude,
            exclude_unlabelled=unlab_exclude,
        )

        # Build output DataFrame
        rows = []
        for i, preds in enumerate(predictions):
            row = {
                "file_id": int(unlabelled_meta.iloc[i]["file_id"]),
                "filename": unlabelled_meta.iloc[i]["filename"],
                "file_path": unlabelled_meta.iloc[i]["path"],
            }
            for rank, (dir_name, score) in enumerate(preds, 1):
                row[f"rank{rank}_dir"] = dir_name
                row[f"rank{rank}_score"] = round(float(score), 6)
            row["model"] = model_name
            row["variant"] = variant
            row["layer"] = opt_layer
            row["pooling"] = pooling
            if unlab_blank[i]:
                row["status"] = STATUS_BLANK_SOURCE
            elif unlab_zero[i]:
                row["status"] = STATUS_ZERO_NORM
            else:
                row["status"] = STATUS_OK
            rows.append(row)

        # Save per-model predictions
        per_model_path = Path(args.out_dir) / f"unlabelled_predictions_{variant}_{slug}.csv"
        per_model_path.parent.mkdir(parents=True, exist_ok=True)
        model_df = pd.DataFrame(rows)
        model_df.to_csv(per_model_path, index=False)
        print(f"  Saved {len(rows)} predictions -> {per_model_path}")

        all_predictions_dfs.append(model_df)

    # Save combined predictions
    if all_predictions_dfs:
        combined = pd.concat(all_predictions_dfs, ignore_index=True)
        combined_path = Path(args.out_dir) / f"unlabelled_predictions_{variant}.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\nCombined predictions ({len(combined)} rows) -> {combined_path}")


if __name__ == "__main__":
    main()
