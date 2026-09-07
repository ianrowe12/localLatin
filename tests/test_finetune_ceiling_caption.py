"""The generated ceiling table's caption must describe the run that made it (#138).

The caption used to carry literals: "the 565 positive pairs", "epoch 7, the
terminal epoch". Benchmark v1 changed the dev carve underneath them, so a
re-run would have shipped a table whose numbers were new and whose prose was
old, with nothing to signal the mismatch. ``CeilingFacts`` derives every one of
those statements from the run, and these tests are what keep them derived: each
one changes an input and asserts the caption follows.

The header check guards a separate rule (#117): ``overleaf_drafts/tables/``
ships to Overleaf, so a generated file there must not name a repository path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "resubmit"))

pytest.importorskip("torch", reason="finetune_lata_ceiling imports torch")
pytest.importorskip("transformers", reason="finetune_lata_ceiling imports transformers")

from finetune_pairs import build_pairs  # noqa: E402

import finetune_lata_ceiling as ceiling  # noqa: E402


D_GRID = [1, 2, 3, 5, 7, 10]


def make_split() -> pd.DataFrame:
    """Four train directories of two files each, plus a test query per directory."""
    rows = []
    for d in range(4):
        folder = f"dir{d}"
        for i in range(2):
            rows.append({"folder_id": folder, "filename": f"tr{d}_{i}.txt", "split": "train",
                         "is_test_query": False})
        rows.append({"folder_id": folder, "filename": f"te{d}.txt", "split": "test",
                     "is_test_query": True})
    return pd.DataFrame(rows)


def make_ft_results(D_by_layer: dict) -> pd.DataFrame:
    """One abtt_fixed and one abtt_optimal row per layer, with a chosen D each."""
    rows = []
    for layer, D in D_by_layer.items():
        rows.append({"layer": layer, "method": "abtt_fixed", "D": 10,
                     "aucroc": 0.9, "dir_acc_at_1": 0.8})
        rows.append({"layer": layer, "method": "abtt_optimal", "D": D,
                     "aucroc": 0.9 if D == 10 else 0.95,
                     "dir_acc_at_1": 0.8 if D == 10 else 0.85})
    return pd.DataFrame(rows)


COMPARISON = pd.DataFrame([{
    "system": "LaTa (fine-tuned)", "finetuned": True, "method": "baseline",
    "taskA_layer": 12, "taskA_aucroc": 0.984, "taskA_cosine_gap": 0.384,
    "taskB_layer": 12, "taskB_assignment_acc": 0.826, "taskB_dir_acc_at_1": 0.808,
}])


def build_facts(selection, ft_results=None, epoch_budget=8, dev_frac=0.25):
    split = make_split()
    pair_data = build_pairs(split, dev_frac, 42)
    return ceiling.CeilingFacts(
        pair_data=pair_data,
        split_meta=split,
        D_values=D_GRID,
        selection=selection,
        epoch_budget=epoch_budget,
        ft_results=make_ft_results({12: 10}) if ft_results is None else ft_results,
    ), pair_data, split


def render(tmp_path, facts) -> str:
    out = tmp_path / "finetune_ceiling.tex"
    ceiling.write_tex(COMPARISON, None, out, facts)
    return out.read_text(encoding="utf-8")


def unwrapped(tex: str) -> str:
    """The file with comment markers and line wrapping removed.

    The notes are wrapped to a column width, so a sentence the generator emits
    as one string arrives split across lines. Assert on the sentence, not on
    where the wrapper happened to break it.
    """
    body = " ".join(
        line.lstrip("%").strip() for line in tex.splitlines() if line.startswith("%")
    )
    return " ".join(body.split())


def test_caption_quotes_the_runs_own_pair_count(tmp_path):
    facts, pair_data, _ = build_facts({"selected_epoch": 3, "epochs_run": 5})
    tex = render(tmp_path, facts)
    assert f"the {pair_data.n_all_train_pairs} positive pairs" in tex
    # The literal the caption used to hardcode must not survive a different run.
    assert "565" not in tex


def test_terminal_epoch_is_flagged_as_a_budget_ceiling(tmp_path):
    facts, _, _ = build_facts({"selected_epoch": 7, "epochs_run": 7}, epoch_budget=8)
    tex = render(tmp_path, facts)
    assert "epoch 7, the terminal epoch of the 8-epoch budget" in tex
    assert "ceiling at this training budget" in tex


def test_mid_run_epoch_is_not_called_terminal(tmp_path):
    facts, _, _ = build_facts({"selected_epoch": 4, "epochs_run": 8}, epoch_budget=8)
    tex = render(tmp_path, facts)
    assert "terminal" not in tex
    assert "epoch 4 of 8 run" in tex


def test_epoch_zero_says_training_bought_nothing(tmp_path):
    facts, _, _ = build_facts({"selected_epoch": 0, "epochs_run": 8})
    tex = render(tmp_path, facts)
    assert "kept the pre-trained encoder (epoch 0)" in tex
    assert "PRE-TRAINED encoder" in unwrapped(tex)


def test_boundary_hit_is_reported_as_a_boundary_hit(tmp_path):
    facts, _, _ = build_facts(
        {"selected_epoch": 7, "epochs_run": 7},
        ft_results=make_ft_results({l: 10 for l in range(1, 13)}),
    )
    assert facts.n_optimal_at_max_D == facts.n_optimal_rows == 12
    assert facts.optimal_matches_fixed is True
    tex = render(tmp_path, facts)
    assert "select $D=10$ everywhere" in tex
    assert "Boundary hit, not a tuned optimum" in unwrapped(tex)
    assert "equals abtt_fixed in every row" in unwrapped(tex)


def test_a_genuine_sweep_is_not_reported_as_a_boundary_hit(tmp_path):
    D_by_layer = {l: (10 if l > 6 else 3) for l in range(1, 13)}
    facts, _, _ = build_facts(
        {"selected_epoch": 7, "epochs_run": 7}, ft_results=make_ft_results(D_by_layer)
    )
    assert facts.n_optimal_at_max_D == 6
    tex = render(tmp_path, facts)
    assert "6 of 12 layer rows select the top of the grid" in tex
    assert "selects its top value in 6 of 12 layer rows" in unwrapped(tex)
    assert "Boundary hit" not in unwrapped(tex)
    assert "everywhere" not in tex


def test_near_duplicate_overlap_is_counted_from_the_split(tmp_path):
    facts, pair_data, split = build_facts({"selected_epoch": 7, "epochs_run": 7})
    fit = set(pair_data.fit_dirs)
    expected = int(split[split["is_test_query"]]["folder_id"].isin(fit).sum())
    assert facts.n_test_queries == 4
    assert facts.n_test_queries_touched == expected
    tex = render(tmp_path, facts)
    assert f"{expected} of the 4 test query files" in unwrapped(tex)


def test_header_is_neutral_and_names_no_repository_path(tmp_path):
    facts, _, _ = build_facts({"selected_epoch": 7, "epochs_run": 7})
    tex = render(tmp_path, facts)
    assert tex.splitlines()[0] == "% generated table"
    for leak in ("runs/active", "scripts/", "/u/", "overleaf_drafts", ".py"):
        assert leak not in tex, f"generated table leaks a repository path: {leak}"


def test_missing_selection_drops_the_checkpoint_sentence_rather_than_guessing(tmp_path):
    facts, _, _ = build_facts(None)
    tex = render(tmp_path, facts)
    assert "selected checkpoint" not in tex
    assert "epoch" not in tex.split(r"\caption")[1].split("}")[0]
