"""The two five-seed SIF+ABTT tables must report the same cell per model (#175).

``tables/taskB_topk.tex`` and ``tables/taskB_ranking_appendix_mseed.tex`` are
written by different generators from the same five-seed CSV. Before #175 one
took a test-set argmax over every method and the other a test-set argmax over
``sif_abtt_optimal`` only, so LaTa and KaLM-mini disagreed. Both now go through
``taskb_mseed_selection``: ``sif_abtt_optimal`` at the layer with the highest
single-seed *train* directory accuracy at rank 1. The synthetic frames below are
built so that the train argmax, the five-seed test argmax, and the best
``sif_abtt_fixed`` row all sit on different layers; any generator that slips
back to a test-set rule fails.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "resubmit"))

import build_per_layer_tables as bplt  # noqa: E402
import taskb_mseed_selection as sel  # noqa: E402
import visualize_taskb_mseed as vtm  # noqa: E402

MODELS = bplt.ALL_MODELS
LAYERS = [1, 2, 3]
# Per model: train-best layer for sif_abtt_optimal. Test means are rigged so the
# five-seed argmax is a different layer, and sif_abtt_fixed beats everything.
TRAIN_BEST = {m: LAYERS[i % 3] for i, m in enumerate(MODELS)}


def _single_seed_frame() -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for method in ("baseline", "sif_abtt_optimal"):
            for layer in LAYERS:
                train = 0.90 if (method == "sif_abtt_optimal" and layer == TRAIN_BEST[model]) else 0.60
                rows.append(
                    {
                        "model": model,
                        "repr": "hidden",
                        "method": method,
                        "layer": layer,
                        "D": 10,
                        "dir_acc_at_1": train - 0.02,
                        "train_dir_acc_at_1": train,
                    }
                )
    return pd.DataFrame(rows)


def _mseed_frame() -> pd.DataFrame:
    rows = []
    for model in MODELS:
        test_best = LAYERS[(LAYERS.index(TRAIN_BEST[model]) + 1) % 3]
        for method, pooling in (
            ("baseline", "mean"),
            ("sif_only", "sif"),
            ("sif_abtt_fixed", "sif"),
            ("sif_abtt_optimal", "sif"),
        ):
            for layer in LAYERS:
                if method == "sif_abtt_fixed":
                    mean = 0.99  # would win any all-method test argmax
                elif method == "sif_abtt_optimal":
                    mean = 0.95 if layer == test_best else 0.80 + 0.01 * layer
                else:
                    mean = 0.50
                row = {
                    "model": model,
                    "method": method,
                    "repr": "hidden",
                    "pooling": pooling,
                    "layer": layer,
                    "D": 10,
                    "n_seeds": 5,
                }
                for k in range(1, 6):
                    row[f"dir_acc_at_{k}_mean"] = min(1.0, mean + 0.01 * (k - 1))
                    row[f"dir_acc_at_{k}_std"] = 0.01
                for stem in ("existing_acc", "new_acc", "overall_assignment_acc", "tau"):
                    row[f"{stem}_mean"] = mean
                    row[f"{stem}_std"] = 0.01
                rows.append(row)
    return pd.DataFrame(rows)


def _bold_rows(tex: str) -> dict[str, tuple[int, str]]:
    """model display -> (bold layer, bold SIF+ABTT Acc@1 cell) from the per-layer table."""
    out: dict[str, tuple[int, str]] = {}
    current = None
    for line in tex.splitlines():
        if not line.rstrip().endswith(r"\\") or "&" not in line:
            continue
        cells = [c.strip() for c in line.split("&")]
        if cells[0] and "\\" not in cells[0]:
            current = cells[0]
        m = re.fullmatch(r"\\textbf\{(\d+)\}", cells[1])
        if m:
            layer = int(m.group(1))
            acc1_sif = re.search(r"\\textbf\{([\d.]+)", cells[3]).group(1)
            out[current] = (layer, acc1_sif)
    return out


def test_selected_layer_is_the_train_argmax_not_the_test_argmax():
    layers = sel.train_selected_layers(_single_seed_frame(), MODELS)
    assert layers == TRAIN_BEST
    rows = sel.select_mseed_rows(_mseed_frame(), layers)
    assert set(rows["method"]) == {"sif_abtt_optimal"}
    for _, row in rows.iterrows():
        assert int(row["layer"]) == TRAIN_BEST[row["model"]]
        assert row["dir_acc_at_1_mean"] < 0.95  # the rigged test-argmax row


def test_missing_model_raises_instead_of_falling_back():
    with pytest.raises(KeyError):
        sel.train_selected_layers(_single_seed_frame(), MODELS + ["nope/model"])


def test_both_generators_report_the_same_cell_per_model(tmp_path: Path):
    single, mseed = _single_seed_frame(), _mseed_frame()

    best_df = vtm.select_train_layer_configs(mseed, single, "hidden")
    topk_path = tmp_path / "taskB_topk.tex"
    vtm.write_paper_topk_table(best_df, topk_path)

    layers = sel.train_selected_layers(single, MODELS)
    mseed_path = tmp_path / "taskB_ranking_appendix_mseed.tex"
    bplt.emit_taskB_ranking_mseed(
        mseed,
        models=MODELS,
        methods=["baseline", sel.MSEED_METHOD],
        out_tex=mseed_path,
        out_audit=tmp_path / "audit.csv",
        caption=sel.SELECTION_RULE_CAPTION,
        label="tab:x",
        banner="x",
        selected_layers=layers,
    )

    bold = _bold_rows(mseed_path.read_text())
    assert len(bold) == len(MODELS)
    topk = {
        line.split("&")[0].strip(): line.split("&")[1].strip()
        for line in topk_path.read_text().splitlines()
        if line.rstrip().endswith(r"\\") and "&" in line and "textbf" not in line
    }
    for model in MODELS:
        name = bplt.MODEL_DISPLAY[model]
        layer, acc1_3dp = bold[name]
        assert layer == TRAIN_BEST[model]
        top1_pct = float(topk[name].split("$")[0])
        assert abs(top1_pct - 100 * float(acc1_3dp)) < 0.06  # 3 dp vs 1 dp of percent

    assert sel.SELECTION_RULE_CAPTION in vtm.TASKB_TOPK_CAPTION
    assert sel.SELECTION_RULE_CAPTION in mseed_path.read_text()


REAL_SINGLE = REPO_ROOT / "runs/active/resubmit/results/phase_resubmit_results.csv"
REAL_MSEED = REPO_ROOT / "runs/active/resubmit/taskb_mseed/aggregated_results.csv"


@pytest.mark.skipif(
    not (REAL_SINGLE.exists() and REAL_MSEED.exists()),
    reason="the result CSVs are tracked but ci.yml sparse-checkouts without runs/",
)
def test_committed_tables_agree_on_the_real_csvs():
    single, mseed = pd.read_csv(REAL_SINGLE), pd.read_csv(REAL_MSEED)
    layers = sel.train_selected_layers(single, MODELS)
    rows = sel.select_mseed_rows(mseed, layers)
    bold = _bold_rows((REPO_ROOT / "overleaf_drafts/tables/taskB_ranking_appendix_mseed.tex").read_text())
    topk_tex = (REPO_ROOT / "overleaf_drafts/tables/taskB_topk.tex").read_text()
    for _, row in rows.iterrows():
        name = bplt.MODEL_DISPLAY[row["model"]]
        assert bold[name][0] == int(row["layer"])
        assert bold[name][1] == f"{row['dir_acc_at_1_mean']:.3f}"
        assert re.search(
            rf"^{re.escape(name)}\s*& {row['dir_acc_at_1_mean'] * 100:.1f} ", topk_tex, re.M
        ), name
