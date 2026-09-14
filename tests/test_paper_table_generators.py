"""Shape guards for the two paper table generators changed by #118 and #120.

Neither test looks at a real number. What they pin is the layout the two issues
asked for, because that is what silently regresses when a generator is edited
later: the reference block under the headline tables (#118), and the absence of
``base -> ABTT`` arrow cells in the main attribution table (#120).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "resubmit"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ig"))

import build_headline_tables as bht  # noqa: E402
import build_main_attribution_artifacts as bmaa  # noqa: E402


def _results_frame() -> pd.DataFrame:
    """Two layers per (model, method) so the train-selected layer is a real choice."""
    rows = []
    for model_id, _ in bht.MODELS:
        for method, _ in bht.METHODS:
            for layer, train, test in ((1, 0.60, 0.61), (2, 0.90, 0.91)):
                rows.append(
                    {
                        "model": model_id,
                        "repr": "hidden",
                        "method": method,
                        "layer": layer,
                        "aucroc": test,
                        "gap": test / 2,
                        "overall_assignment_acc": test,
                        "dir_acc_at_1": test - 0.01,
                        "train_aucroc": train,
                        "train_dir_acc_at_1": train,
                        "n_test": 858,
                        "n_existing": 535,
                    }
                )
    return pd.DataFrame(rows)


def _lexical_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": key,
                "aucroc": 0.9,
                "gap": 0.4,
                "overall_assignment_acc": 0.8,
                "dir_acc_at_1": 0.7,
            }
            for key, _ in bht.LEXICAL_SYSTEMS
        ]
    )


def _finetune_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "system": label,
                "method": method,
                "taskA_layer": 12,
                "taskA_aucroc": 0.98,
                "taskA_cosine_gap": 0.38,
                "taskB_layer": 12,
                "taskB_assignment_acc": 0.83,
                "taskB_dir_acc_at_1": 0.81,
            }
            for label, method in bht.FINETUNE_VARIANTS
        ]
    )


def _render_task_a() -> str:
    results = _results_frame()
    lexical = _lexical_frame()
    best = bht.best_rows(results, "hidden", "train_aucroc")
    return bht.render_table(
        best,
        left_banner="Task A AUROC",
        right_banner="Task A cosine gap",
        left_col="aucroc",
        right_col="gap",
        fmt=".3f",
        scale=1.0,
        caption=bht.task_a_caption(best),
        label="tab:taskA_headline",
        reference_lines=bht.reference_rows(
            lexical,
            _finetune_frame(),
            lexical_left_col="aucroc",
            lexical_right_col="gap",
            finetune_left_col="taskA_aucroc",
            finetune_right_col="taskA_cosine_gap",
            finetune_layer_col="taskA_layer",
            fmt=".3f",
            scale=1.0,
        ),
    )


def test_reference_block_sits_below_the_six_model_rows():
    tex = _render_task_a()
    lines = tex.splitlines()
    model_row = next(i for i, line in enumerate(lines) if line.startswith("KaLM-mini &"))
    ceiling_row = next(
        i for i, line in enumerate(lines) if line.startswith(bht.FINETUNE_ROW_LABEL)
    )
    assert model_row < ceiling_row


def test_lexical_rows_span_their_metric_block():
    tex = _render_task_a()
    row = next(
        line for line in tex.splitlines() if line.startswith("TF-IDF char 3--5 &")
    )
    # One value per metric block, not one per post-processing setting.
    assert row.count(r"\multicolumn{4}{c}") == 2


def test_finetune_row_leaves_the_sif_columns_empty():
    tex = _render_task_a()
    row = next(
        line
        for line in tex.splitlines()
        if line.startswith(bht.FINETUNE_ROW_LABEL + " &")
    )
    cells = [cell.strip() for cell in row.rstrip("\\ ").split("&")]
    # Model, then Base SIF ABTT SIF+ABTT twice.
    assert len(cells) == 9
    assert cells[2] == "--" and cells[4] == "--"
    assert cells[6] == "--" and cells[8] == "--"


def test_caption_does_not_claim_an_embedding_win_over_surface_matching():
    tex = _render_task_a()
    caption = tex[tex.index(r"\caption{") :]
    assert "practitioner's operating point" in caption
    assert "rather than beat it" in caption


def _attribution_summary() -> pd.DataFrame:
    rows = []
    for model, _ in bmaa.MODELS:
        for method, _ in bmaa.METHODS:
            for variant, value in (("baseline", 0.2), ("abtt", 0.5)):
                row = {
                    "model": model,
                    "method": method,
                    "variant": variant,
                    "n": 200,
                    "full_cos_mean": 0.5,
                    f"{bmaa.DEL_GAP_KEY}_n": 200 if variant == "baseline" else 193,
                    f"{bmaa.INS_GAP_KEY}_n": 200 if variant == "baseline" else 193,
                    # Criterion 5 is read off this column rather than asserted,
                    # so the secondary caption needs it present.
                    f"{bmaa.SHUFFLE_GAP_KEY}_mean": 0.1,
                }
                for key in bmaa.METRIC_KEYS:
                    row[f"{key}_mean"] = value
                rows.append(row)
    return pd.DataFrame(rows)


def test_main_attribution_table_has_no_arrow_cells(tmp_path: Path):
    summary = bmaa.select_main_rows(_attribution_summary())
    out = tmp_path / "attribution_metrics_main.tex"
    bmaa.render_table(summary, out)
    tex = out.read_text()
    assert r"\rightarrow" not in tex
    # Paired base/ABTT columns for exactly the two selected metrics.
    assert tex.count(r"\multicolumn{2}{c}") == 2
    assert r"DelAUC gap" in tex and r"\rho_{\text{LOO}}" in tex


def test_secondary_attribution_table_carries_the_demoted_metrics(tmp_path: Path):
    summary = bmaa.select_main_rows(_attribution_summary())
    out = tmp_path / "attribution_metrics_secondary.tex"
    bmaa.render_secondary_table(summary, out)
    tex = out.read_text()
    assert r"\rightarrow" not in tex
    for label in (r"\tau_{\text{LOO}}", "InsAUC gap", r"Suff@25\%", r"Comp@25\%",
                  "MinFrac@0.80"):
        assert label in tex


def test_caption_states_no_tie_when_it_has_no_paired_evidence():
    """A tie claim names a cell, so it may only come from per-pair differences.

    The published caption hardcoded "the LaTa MaRC DelAUC win is a tie at about
    1.2 standard errors". That was true of one sample; issue #141 re-sampled and
    the narrow cells moved. With no per-pair cache to read, the caption says
    nothing about ties rather than repeating the old one.
    """
    summary = bmaa.select_main_rows(_attribution_summary())
    caption = bmaa.main_caption(summary, pairs_root=None)
    assert "standard error" not in caption
    assert "tie" not in caption


def test_caption_names_the_tie_cell_from_paired_differences(tmp_path: Path):
    """A cell whose paired difference is inside the noise is named as a tie."""
    pairs_root = tmp_path / "v2_hidden" / "bowphs_PhilTa"
    pairs_root.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(40):
        rows = []
        for method, _ in bmaa.METHODS:
            # PhilTa/MaRC: a real gain. PhilTa/IG: noise around zero.
            shift = 0.5 if method == "retrieval_mark" else 0.0
            base = float(rng.normal(0.0, 0.05))
            rows.append({"model": "bowphs/PhilTa", "method": method,
                         "variant": "baseline", bmaa.RHO_KEY: base})
            rows.append({"model": "bowphs/PhilTa", "method": method,
                         "variant": "abtt", bmaa.RHO_KEY: base + shift
                         + float(rng.normal(0.0, 0.05))})
        (pairs_root / f"example{i:03d}.json").write_text(json.dumps(rows))

    stats = bmaa.paired_cell_stats(tmp_path / "v2_hidden", bmaa.RHO_KEY)
    ig_mean, ig_se = stats[("bowphs/PhilTa", "ig")]
    marc_mean, marc_se = stats[("bowphs/PhilTa", "retrieval_mark")]
    assert abs(ig_mean / ig_se) < 2.0
    assert abs(marc_mean / marc_se) > 2.0

    summary = bmaa.select_main_rows(_attribution_summary())
    clause = bmaa.tie_sentence(tmp_path / "v2_hidden", summary,
                               bmaa.RHO_KEY, "rho")
    assert "PhilTa IG" in clause
    assert "PhilTa MaRC" not in clause


def test_pair_count_phrase_collapses_when_every_cell_agrees():
    """The baseline count is a range now, but must not print as '200 to 200'."""
    assert bmaa._count_phrase(200, 200) == "200"
    assert bmaa._count_phrase(199, 200) == "199 to 200"


def test_load_main_rows_rejects_a_summary_missing_the_new_columns():
    summary = _attribution_summary().drop(columns=[f"{bmaa.DEL_GAP_KEY}_mean"])
    with pytest.raises(ValueError, match="missing required columns"):
        bmaa.select_main_rows(summary)
