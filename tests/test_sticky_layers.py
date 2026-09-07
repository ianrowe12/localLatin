"""The deployed unlabelled predictions hold their layer unless clearly beaten.

The prediction CSVs under ``runs/active/resubmit/unlabelled/`` are what the
reviewer pilot serves, so the layer they are built at is user-visible state, not
just an experimental knob. Picking it by a hard argmax over near-tied layers
means a hundredth of a point of movement can change thousands of shortlists: the
issue #113 label correction touched two files and moved Qwen3-0.6B's ``sif_abtt``
layer from 7 to 1, rewriting 1,276 of that model's 2,238 top-1 answers.

So a new layer has to beat the deployed one by more than a tolerance before it
replaces it. These tests pin the rule's edges: the tolerance is exclusive, it
compares on the same metric the selector uses, and it never invents a layer that
the plain argmax would not itself have been willing to pick.
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "resubmit"))

import run_resubmit_unlabelled_retrieval as R  # noqa: E402

MODEL = "Qwen/Qwen3-Embedding-0.6B"
OTHER = "bowphs/LaTa"


def make_results_csv(path: Path, rows) -> Path:
    """A minimal results CSV: (model, layer, method, selection metric)."""
    frame = pd.DataFrame(
        [
            {
                "model": model,
                "repr": "hidden",
                "pooling": "sif",
                "layer": layer,
                "method": "sif_abtt_fixed",
                R.SELECTION_METRIC: score,
            }
            for model, layer, score in rows
        ]
    )
    frame.to_csv(path, index=False)
    return path


def make_deployed_json(path: Path, layers) -> Path:
    path.write_text(
        json.dumps({"selection_metric": R.SELECTION_METRIC, "layers": layers}),
        encoding="utf-8",
    )
    return path


def args_for(results_csv: Path, deployed_json: Path, **overrides) -> Namespace:
    base = dict(
        results_csv=str(results_csv),
        deployed_layers_json=str(deployed_json),
        sticky_tolerance=R.STICKY_TOLERANCE,
        no_sticky_layers=False,
        layer_overrides="",
        models="all",
    )
    base.update(overrides)
    return Namespace(**base)


def layer_of(configs, model=MODEL) -> int:
    return next(layer for name, layer, _ in configs if name == model)


# --- apply_sticky_layers, the rule itself -----------------------------------


def scores_for(mapping):
    return {MODEL: mapping}


def test_keeps_deployed_layer_when_the_gain_is_inside_the_tolerance():
    """The real case: 0.911422 at L7 versus 0.913753 at L1 is a 0.0023 gain."""
    configs, decisions = R.apply_sticky_layers(
        {MODEL: (1, "hidden")},
        {MODEL: 7},
        scores_for({1: 0.913753, 7: 0.911422}),
        tolerance=0.005,
    )
    assert configs[MODEL] == (7, "hidden")
    assert decisions[0]["action"] == "kept"
    assert decisions[0]["argmax_layer"] == 1
    assert decisions[0]["chosen_layer"] == 7


def test_switches_when_the_gain_clears_the_tolerance():
    configs, decisions = R.apply_sticky_layers(
        {MODEL: (1, "hidden")},
        {MODEL: 7},
        scores_for({1: 0.95, 7: 0.90}),
        tolerance=0.005,
    )
    assert configs[MODEL] == (1, "hidden")
    assert decisions[0]["action"] == "switched"


def test_the_tolerance_is_exclusive():
    """Exactly the tolerance is not "more than" it, so the deployed layer holds."""
    configs, _ = R.apply_sticky_layers(
        {MODEL: (1, "hidden")},
        {MODEL: 7},
        scores_for({1: 0.905, 7: 0.900}),
        tolerance=0.005,
    )
    assert configs[MODEL] == (7, "hidden")

    configs, _ = R.apply_sticky_layers(
        {MODEL: (1, "hidden")},
        {MODEL: 7},
        scores_for({1: 0.905001, 7: 0.900}),
        tolerance=0.005,
    )
    assert configs[MODEL] == (1, "hidden")


def test_a_model_with_no_deployed_layer_is_passed_through():
    configs, decisions = R.apply_sticky_layers(
        {MODEL: (4, "hidden")}, {}, scores_for({4: 0.9}), tolerance=0.005
    )
    assert configs[MODEL] == (4, "hidden")
    assert decisions[0]["action"] == "new"


def test_a_deployed_layer_missing_from_the_results_csv_does_not_win():
    """A layer nobody scored cannot be defended, so the argmax stands."""
    configs, decisions = R.apply_sticky_layers(
        {MODEL: (1, "hidden")},
        {MODEL: 99},
        scores_for({1: 0.9}),
        tolerance=0.005,
    )
    assert configs[MODEL] == (1, "hidden")
    assert decisions[0]["action"] == "new"


def test_an_unchanged_argmax_is_reported_as_unchanged():
    configs, decisions = R.apply_sticky_layers(
        {MODEL: (7, "hidden")}, {MODEL: 7}, scores_for({7: 0.9}), tolerance=0.005
    )
    assert configs[MODEL] == (7, "hidden")
    assert decisions[0]["action"] == "unchanged"


# --- the selection metric read off the results CSV --------------------------


def test_layer_scores_takes_the_best_method_at_each_layer(tmp_path):
    csv = make_results_csv(
        tmp_path / "results.csv", [(MODEL, 1, 0.80), (MODEL, 1, 0.90), (MODEL, 7, 0.85)]
    )
    scores = R.layer_scores_from_results(str(csv), ("sif_abtt_fixed",))
    assert scores[MODEL] == {1: 0.90, 7: 0.85}


# --- resolve_model_configs, the wiring --------------------------------------


def test_resolve_keeps_the_deployed_layer_and_no_sticky_takes_the_argmax(tmp_path):
    csv = make_results_csv(
        tmp_path / "results.csv",
        [(MODEL, 1, 0.913753), (MODEL, 7, 0.911422), (OTHER, 1, 0.9)],
    )
    deployed = make_deployed_json(
        tmp_path / "deployed.json", {"sif_abtt": {MODEL: 7, OTHER: 1}}
    )

    sticky = R.resolve_model_configs(
        args_for(csv, deployed), ("sif_abtt_fixed",), "sif_abtt"
    )[0]
    assert layer_of(sticky) == 7

    argmax = R.resolve_model_configs(
        args_for(csv, deployed, no_sticky_layers=True), ("sif_abtt_fixed",), "sif_abtt"
    )[0]
    assert layer_of(argmax) == 1


def test_an_explicit_override_beats_the_sticky_layer(tmp_path):
    """`--layer_overrides` is an explicit pin and outranks both rules."""
    csv = make_results_csv(
        tmp_path / "results.csv", [(MODEL, 1, 0.913753), (MODEL, 7, 0.911422)]
    )
    deployed = make_deployed_json(tmp_path / "deployed.json", {"sif_abtt": {MODEL: 7}})
    configs = R.resolve_model_configs(
        args_for(csv, deployed, layer_overrides=f"{MODEL}=22"),
        ("sif_abtt_fixed",),
        "sif_abtt",
    )[0]
    assert layer_of(configs) == 22


def test_a_variant_with_nothing_recorded_falls_back_to_the_argmax(tmp_path):
    csv = make_results_csv(
        tmp_path / "results.csv", [(MODEL, 1, 0.913753), (MODEL, 7, 0.911422)]
    )
    deployed = make_deployed_json(tmp_path / "deployed.json", {"raw": {MODEL: 28}})
    configs, decisions = R.resolve_model_configs(
        args_for(csv, deployed), ("sif_abtt_fixed",), "sif_abtt"
    )
    assert layer_of(configs) == 1
    assert decisions == []


def test_a_missing_deployed_json_is_not_an_error(tmp_path):
    csv = make_results_csv(
        tmp_path / "results.csv", [(MODEL, 1, 0.913753), (MODEL, 7, 0.911422)]
    )
    configs, _ = R.resolve_model_configs(
        args_for(csv, tmp_path / "absent.json"), ("sif_abtt_fixed",), "sif_abtt"
    )
    assert layer_of(configs) == 1


# --- recording the deployed layers ------------------------------------------


def write_predictions(path: Path, per_model):
    pd.DataFrame(
        [
            {"file_id": i, "model": model, "layer": layer}
            for i, (model, layer) in enumerate(per_model)
        ]
    ).to_csv(path, index=False)


def test_record_reads_the_layer_column_of_the_live_csvs(tmp_path):
    write_predictions(
        tmp_path / "unlabelled_predictions_sif_abtt.csv",
        [(MODEL, 7), (MODEL, 7), (OTHER, 1)],
    )
    out = tmp_path / "deployed.json"
    layers = R.record_deployed_layers(tmp_path, out, variants=("sif_abtt",))
    assert layers == {"sif_abtt": {MODEL: 7, OTHER: 1}}

    payload = json.loads(out.read_text())
    assert payload["selection_metric"] == R.SELECTION_METRIC
    assert payload["sticky_tolerance"] == R.STICKY_TOLERANCE
    assert payload["layers"]["sif_abtt"][MODEL] == 7


def test_record_refuses_a_csv_serving_two_layers_for_one_model(tmp_path):
    """A deployed CSV with mixed layers is corrupt, not a thing to average."""
    write_predictions(
        tmp_path / "unlabelled_predictions_sif_abtt.csv", [(MODEL, 7), (MODEL, 1)]
    )
    with pytest.raises(SystemExit, match="distinct layers"):
        R.record_deployed_layers(
            tmp_path, tmp_path / "deployed.json", variants=("sif_abtt",)
        )


def test_the_committed_record_matches_what_the_scripts_expect():
    """The shipped JSON has to parse, and cover every variant and model."""
    deployed = R.load_deployed_layers(R.DEFAULT_DEPLOYED_LAYERS_JSON)
    assert set(deployed) == set(R.VARIANTS)
    for variant, per_model in deployed.items():
        assert set(per_model) == set(R.ALL_MODELS), variant
        assert all(isinstance(v, int) and v >= 0 for v in per_model.values())
