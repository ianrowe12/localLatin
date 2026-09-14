"""The appendix artefacts select layers on the train split, like the headline tables (#184).

Before #184 the nine per-layer tables bolded the test argmax (LaTa layer 1 in
``tab:taskB_routing_main`` while Table 3 reports layer 8), the last-token
comparison defaulted to the test column ``overall_assignment_acc``, and the
cluster figures took their layers from a test-argmax summary CSV. The synthetic
frames below rig every (model, method) so that the train argmax and the test
argmax sit on different layers; a generator that slips back to a test rule
bolds or reports the wrong layer and fails.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "resubmit"))

import build_headline_tables as bht  # noqa: E402
import build_lasttok_comparison_table as blt  # noqa: E402
import build_per_layer_tables as bplt  # noqa: E402

MODELS = bplt.ALL_MODELS
LAYERS = [1, 2, 3]
METHODS = ["baseline", "sif_only", "sif_abtt_fixed", "abtt_optimal", "sif_abtt_optimal"]


def _train_best(i: int, j: int) -> int:
    return LAYERS[(i + j) % 3]


def _test_best(i: int, j: int) -> int:
    return LAYERS[(i + j + 1) % 3]


def _results_frame() -> pd.DataFrame:
    """Single-seed frame: train and test argmax differ for every (model, method)."""
    rows = []
    for i, model in enumerate(MODELS):
        for j, method in enumerate(METHODS):
            for layer in LAYERS:
                train = 0.90 if layer == _train_best(i, j) else 0.60
                test = 0.90 if layer == _test_best(i, j) else 0.60
                rows.append(
                    {
                        "model": model,
                        "repr": "hidden",
                        "method": method,
                        "layer": layer,
                        "D": 10,
                        "aucroc": test,
                        "gap": test / 2,
                        "overall_assignment_acc": test,
                        "dir_acc_at_1": test - 0.01,
                        "existing_acc": test,
                        "new_acc": test,
                        "train_aucroc": train,
                        "train_dir_acc_at_1": train,
                        "n_test": 858,
                        "n_existing": 535,
                    }
                )
    return pd.DataFrame(rows)


def _bold_layers(tex: str) -> dict[str, int]:
    """model display -> bold layer, for both the longtable and table* emitters."""
    out: dict[str, int] = {}
    current = None
    for line in tex.splitlines():
        if "&" not in line or not line.rstrip().endswith(r"\\"):
            continue
        cells = [c.strip() for c in line.split("&")]
        if cells[0] and "\\" not in cells[0]:
            current = cells[0]
        m = re.fullmatch(r"\\textbf\{(\d+)\}", cells[1])
        if m:
            assert current not in out, f"two bold rows for {current}"
            out[current] = int(m.group(1))
    return out


def _headline_subscripts(frame: pd.DataFrame, method: str, metric: str) -> dict[str, int]:
    """The layer the headline generator prints as the subscript of ``method``."""
    best = bht.best_rows(frame, "hidden", metric)
    return {
        bplt.MODEL_DISPLAY[row["_model_id"]]: int(row["layer"])
        for _, row in best.iterrows()
        if row["_method"] == method
    }


@pytest.mark.parametrize(
    "emit, select_method, metric, metrics",
    [
        (bplt.emit_taskA, "abtt_optimal", "train_aucroc", ["aucroc", "gap"]),
        (bplt.emit_taskA, "sif_abtt_optimal", "train_aucroc", ["aucroc"]),
        (bplt.emit_taskB_routing, "abtt_optimal", "train_dir_acc_at_1",
         ["existing_acc", "new_acc", "overall_assignment_acc"]),
        (bplt.emit_taskB_routing, "sif_abtt_optimal", "train_dir_acc_at_1",
         ["overall_assignment_acc"]),
        (bplt.emit_taskB_ranking_single, "abtt_optimal", "train_dir_acc_at_1",
         ["dir_acc_at_1", "existing_acc", "new_acc"]),
    ],
)
@pytest.mark.parametrize("float_table", [False, True])
def test_bold_row_is_the_headline_subscript(
    tmp_path: Path, emit, select_method, metric, metrics, float_table
):
    frame = _results_frame()
    task = "taskA" if metric == "train_aucroc" else "taskB"
    out = tmp_path / "table.tex"
    emit(
        frame,
        models=MODELS,
        methods=["baseline", select_method],
        metrics=metrics,
        out_tex=out,
        out_audit=tmp_path / "audit.csv",
        caption=bplt._selected_layer_caption(bplt.METHOD_DISPLAY[select_method], task),
        label="tab:x",
        banner="x",
        select_method=select_method,
        select_metric=metric,
        float_table=float_table,
    )
    bold = _bold_layers(out.read_text())
    subscripts = _headline_subscripts(frame, select_method, metric)
    assert len(bold) == len(MODELS)
    assert bold == subscripts
    j = METHODS.index(select_method)
    for i, model in enumerate(MODELS):
        assert bold[bplt.MODEL_DISPLAY[model]] != _test_best(i, j)  # never the test argmax

    audit = pd.read_csv(tmp_path / "audit.csv")
    assert audit["train_selected"].sum() == len(MODELS)
    tex = out.read_text()
    assert bplt.HEADLINE_LABEL[task] in tex
    assert "chosen on the train split" in tex


def _mseed_frame() -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for method in ("baseline", "sif_abtt_optimal"):
            for layer in LAYERS:
                row = {"model": model, "repr": "hidden", "method": method, "layer": layer}
                for stem in ("dir_acc_at_1", "existing_acc", "new_acc"):
                    row[f"{stem}_mean"] = 0.8
                    row[f"{stem}_std"] = 0.01
                rows.append(row)
    return pd.DataFrame(rows)


def test_mseed_table_refuses_a_selected_layer_its_csv_lacks(tmp_path: Path):
    """The five-seed table selects on the single-seed CSV; a layer the five-seed
    CSV never ran must fail loudly instead of leaving that model unbolded."""
    layers = {model: LAYERS[0] for model in MODELS}
    layers[MODELS[0]] = 99
    with pytest.raises(SystemExit, match="absent"):
        bplt.emit_taskB_ranking_mseed(
            _mseed_frame(),
            models=MODELS,
            methods=["baseline", "sif_abtt_optimal"],
            out_tex=tmp_path / "t.tex",
            out_audit=tmp_path / "a.csv",
            caption="c",
            label="l",
            banner="b",
            selected_layers=layers,
        )


# ------------------------------- last-token table ------------------------------


LASTTOK_MODELS = [m for m in blt.MODEL_DISPLAY if m != "google/mt5-base"]


def _lasttok_frame() -> pd.DataFrame:
    rows = []
    for i, model in enumerate(LASTTOK_MODELS + ["Qwen/Qwen3-Embedding-8B"]):
        for k, pooling in enumerate(("mean", "lasttok")):
            for method in ("abtt_fixed", "abtt_optimal"):
                for layer in LAYERS:
                    train = 0.90 if layer == _train_best(i, k) else 0.60
                    test = 0.90 if layer == _test_best(i, k) else 0.60
                    rows.append(
                        {
                            "model": model,
                            "repr": "hidden",
                            "pooling": pooling,
                            "layer": layer,
                            "method": method,
                            "D": 7 if method == "abtt_optimal" else 10,
                            "overall_assignment_acc": test,
                            "dir_acc_at_1": test - 0.01,
                            "gap": test / 2,
                            "train_dir_acc_at_1": train,
                        }
                    )
    return pd.DataFrame(rows)


def test_lasttok_generator_defaults_to_train_selection(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["build_lasttok_comparison_table.py"])
    args = blt.parse_args()
    assert args.select_on == "train_dir_acc_at_1"
    assert args.select_method == "abtt_optimal"
    assert blt.DEFAULT_SELECT_ON.startswith("train_")


def test_lasttok_table_reports_the_train_argmax_for_the_six_paper_models():
    table = blt.build_table(_lasttok_frame())
    assert set(table["Model"]) == {blt.MODEL_DISPLAY[m] for m in LASTTOK_MODELS}
    assert "Qwen3-8B" not in set(table["Model"])
    assert set(table["D"]) == {"7"}  # the abtt_optimal rows, not abtt_fixed
    for _, row in table.iterrows():
        i = LASTTOK_MODELS.index(
            next(m for m, d in blt.MODEL_DISPLAY.items() if d == row["Model"])
        )
        k = blt.POOL_ORDER.index(row["Pool"])
        assert row["Layer"] == _train_best(i, k)
        assert row["Assign Acc"] == "0.600"  # the test-argmax row (0.900) was not taken


def test_lasttok_table_refuses_a_test_column():
    with pytest.raises(SystemExit):
        blt.build_table(_lasttok_frame(), select_on="overall_assignment_acc")


# --------------------------------- cluster figures -----------------------------


def test_cluster_figures_take_the_train_selected_abtt_layer():
    vc = pytest.importorskip("visualize_clusters_2d")  # needs matplotlib
    frame = _results_frame()
    names = [bplt.MODEL_DISPLAY[m] for m in MODELS]
    layers = vc.select_layers(frame, names)
    expected = _headline_subscripts(frame, "abtt_optimal", "train_dir_acc_at_1")
    assert layers == expected
    with pytest.raises(SystemExit):
        vc.select_layers(frame, names, metric="overall_assignment_acc")
    assert vc.HIGHLIGHT_K == 6
