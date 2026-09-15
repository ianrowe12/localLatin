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
import attribution_run_of_record as aror  # noqa: E402
import package_attribution_sweep_appendix as pasa  # noqa: E402


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


FT_LATA = "LaTa (fine-tuned)"
FT_QWEN = "Qwen3-0.6B (fine-tuned)"


def _finetune_frame(*labels: str, assignment: float = 0.83,
                    dir_acc: float = 0.81) -> pd.DataFrame:
    """One ceiling per label, each with its baseline and ABTT row."""
    rows = []
    for label in labels or (FT_LATA,):
        for csv_label, method in bht.finetune_variants(label):
            rows.append(
                {
                    "system": csv_label,
                    "method": method,
                    "taskA_layer": 12,
                    "taskA_aucroc": 0.98,
                    "taskA_cosine_gap": 0.38,
                    "taskB_layer": 12,
                    "taskB_assignment_acc": assignment,
                    "taskB_dir_acc_at_1": dir_acc,
                }
            )
    return pd.DataFrame(rows)


def _render_task_a(lexical: pd.DataFrame | None = None) -> str:
    """The paper's default has no lexical rows (issue #197); pass a frame for
    the opt-in rebuttal variant."""
    results = _results_frame()
    best = bht.best_rows(results, "hidden", "train_aucroc")
    return bht.render_table(
        best,
        left_banner="Task A AUROC",
        right_banner="Task A cosine gap",
        left_col="aucroc",
        right_col="gap",
        fmt=".3f",
        scale=1.0,
        caption=bht.task_a_caption(best, lexical, _finetune_frame(), []),
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
        i for i, line in enumerate(lines) if line.startswith(FT_LATA)
    )
    assert model_row < ceiling_row


LEXICAL_TERMS = ("TF-IDF", "BM25", "Levenshtein", "lexical", "surface", "string")


def test_lexical_rows_and_clauses_are_off_by_default():
    """Issue #197: the paper carries no lexical baseline, in rows or caption."""
    tex = _render_task_a()
    for term in LEXICAL_TERMS:
        assert term not in tex, term
    # The reference block is still there and still holds the ceiling.
    assert r"\midrule" in tex
    assert any(
        line.startswith(FT_LATA + " &") for line in tex.splitlines()
    )


def test_lexical_rows_span_their_metric_block_when_opted_in():
    tex = _render_task_a(_lexical_frame())
    row = next(
        line for line in tex.splitlines() if line.startswith("TF-IDF char 3--5 &")
    )
    # One value per metric block, not one per post-processing setting.
    assert row.count(r"\multicolumn{4}{c}") == 2
    caption = tex[tex.index(r"\caption{") :]
    # The rebuttal variant keeps the framing constraint of issues #119 and #176.
    assert "practitioner's operating point" in caption
    assert "rather than beat it" in caption
    # The fine-tune clause opens with an acronym; it must not be lowercased
    # when it follows the lexical clause (review of PR #200).
    assert "and ABTT moves the fine-tuned encoder's AUROC" in caption


def test_finetune_row_leaves_the_sif_columns_empty():
    tex = _render_task_a()
    row = next(
        line for line in tex.splitlines() if line.startswith(FT_LATA + " &")
    )
    cells = [cell.strip() for cell in row.rstrip("\\ ").split("&")]
    # Model, then Base SIF ABTT SIF+ABTT twice.
    assert len(cells) == 9
    assert cells[2] == "--" and cells[4] == "--"
    assert cells[6] == "--" and cells[8] == "--"


@pytest.mark.parametrize("lexical", [None, _lexical_frame()], ids=["paper", "rebuttal"])
def test_caption_never_says_leads_or_reaches(lexical):
    # Issue #176: the comparison sentence is derived from the cells, and a
    # single-seed difference inside the seed spread is never a lead. Issue
    # #197 removed the comparison from the paper; the guard stays for the
    # fine-tuning clause and for the opt-in variant.
    tex = _render_task_a(lexical)
    caption = tex[tex.index(r"\caption{") :]
    assert "leads" not in caption and "reaches" not in caption
    assert "a ceiling at this training budget" in caption


def test_level_word_only_calls_a_lead_outside_the_seed_spread():
    assert bht.level_word(0.1, 1.0) == "level with"
    assert bht.level_word(-0.9, 1.0) == "level with"
    assert bht.level_word(1.5, 1.0) == "above"
    assert bht.level_word(-1.5, 1.0) == "below"


def test_task_b_comparison_states_the_ceiling_as_a_finding():
    results = _results_frame()
    best = bht.best_rows(results, "hidden", "train_dir_acc_at_1")
    # Fixture: ABTT cells at 0.91 assignment / 0.90 dir@1; lexical 0.80 / 0.70;
    # fine-tuned + ABTT 0.83 / 0.81, so below every zero-shot ABTT cell.
    sentence = bht.task_b_comparison(best, None, _finetune_frame())
    assert sentence.startswith("The fine-tuned encoder with ABTT (83.0 and 81.0)")
    assert "is below every zero-shot ABTT cell" in sentence
    assert "TF-IDF" not in sentence
    rebuttal = bht.task_b_comparison(best, _lexical_frame(), _finetune_frame())
    assert "TF-IDF char 3--5 is below the best ABTT cell (80.0 against 91.0" in rebuttal
    assert "is below every zero-shot ABTT cell" in rebuttal


def test_task_a_comparison_covers_the_finetuned_row_alone_by_default():
    results = _results_frame()
    best = bht.best_rows(results, "hidden", "train_aucroc")
    sentence = bht.task_a_comparison(best, None, _finetune_frame())
    assert sentence.startswith("ABTT moves the fine-tuned encoder's AUROC")
    assert "TF-IDF" not in sentence


def test_both_headline_captions_share_the_finetune_pairs_clause():
    """Function level: given the same ``facts``, both caption builders emit the
    same pairs clause. The call site that once dropped ``facts`` on the Task B
    call is covered by ``test_main_writes_the_pairs_clause_into_both_headline_captions``."""
    results = _results_frame()
    finetune = _finetune_frame()
    best_a = bht.best_rows(results, "hidden", "train_aucroc")
    best_b = bht.best_rows(results, "hidden", "train_dir_acc_at_1")
    for facts in (
        [],
        [{"n_fit_pairs": 499, "n_all_train_pairs": 565, "n_dev_dirs": 28}],
    ):
        clause = bht.finetune_pairs_clause(facts)
        cap_a = bht.task_a_caption(best_a, None, finetune, facts)
        cap_b = bht.task_b_caption(results, best_b, None, finetune, facts)
        assert clause in cap_a and clause in cap_b
        assert "a ceiling at this training budget" in cap_a
        assert "a ceiling at this training budget" in cap_b
    assert "499 of the 565 positive train pairs (a 28-directory dev carve" in cap_b
    assert "left by a directory-level dev carve" not in cap_b


def test_main_writes_the_pairs_clause_into_both_headline_captions(tmp_path, monkeypatch):
    """Issue #185 review: ``main()`` once passed ``facts`` only on the Task A
    call, so the committed Task B caption printed the number-free fallback
    while the caption function itself was correct. This drives ``main()`` end
    to end and reads the clause back out of both written files."""
    _results_frame().to_csv(tmp_path / "r.csv", index=False)
    _finetune_frame().to_csv(tmp_path / "f.csv", index=False)
    facts = {"n_fit_pairs": 499, "n_all_train_pairs": 565, "n_dev_dirs": 28}
    (tmp_path / "run_info.json").write_text(json.dumps({"caption_facts": facts}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_headline_tables.py",
            "--results_csv", str(tmp_path / "r.csv"),
            "--finetune_csv", str(tmp_path / "f.csv"),
            "--finetune_run_info", str(tmp_path / "run_info.json"),
            "--out_dir", str(tmp_path / "out"),
        ],
    )
    bht.main()
    clause = bht.finetune_pairs_clause([facts])
    assert "499 of the 565" in clause
    for name in ("taskA_headline.tex", "taskB_headline.tex"):
        tex = (tmp_path / "out" / name).read_text()
        assert clause in tex, name
        assert "left by a directory-level dev carve" not in tex, name
        # Issue #197: a default run must not re-add the lexical rows.
        for term in LEXICAL_TERMS:
            assert term not in tex, (name, term)


def test_main_adds_lexical_rows_only_when_the_csv_is_passed(tmp_path, monkeypatch):
    """The rebuttal variant is still buildable, but only on request."""
    _results_frame().to_csv(tmp_path / "r.csv", index=False)
    _lexical_frame().to_csv(tmp_path / "l.csv", index=False)
    _finetune_frame().to_csv(tmp_path / "f.csv", index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_headline_tables.py",
            "--results_csv", str(tmp_path / "r.csv"),
            "--lexical_csv", str(tmp_path / "l.csv"),
            "--finetune_csv", str(tmp_path / "f.csv"),
            "--finetune_run_info", str(tmp_path / "absent.json"),
            "--out_dir", str(tmp_path / "out"),
        ],
    )
    bht.main()
    for name in ("taskA_headline.tex", "taskB_headline.tex"):
        tex = (tmp_path / "out" / name).read_text()
        for key, display in bht.LEXICAL_SYSTEMS:
            assert any(line.startswith(display + " &") for line in tex.splitlines()), (name, key)
        assert "three lexical baselines" in tex, name


# --- two fine-tuning ceilings in the reference block (#194) ----------------


def _two_model_finetune() -> pd.DataFrame:
    """LaTa below every zero-shot ABTT cell; Qwen3-0.6B above every one of them.

    The fixture's ABTT cells are 0.91 assignment / 0.90 dir@1, so 0.83/0.81 is
    below all of them and 0.95/0.94 is above all of them. If the caption were
    still written once and reused, the two rows would get the same verdict.
    """
    return pd.concat(
        [
            _finetune_frame(FT_LATA),
            _finetune_frame(FT_QWEN, assignment=0.95, dir_acc=0.94),
        ],
        ignore_index=True,
    )


def test_reference_block_gets_one_row_per_fine_tuned_model():
    lines = bht.reference_rows(
        _lexical_frame(),
        _two_model_finetune(),
        lexical_left_col="aucroc",
        lexical_right_col="gap",
        finetune_left_col="taskA_aucroc",
        finetune_right_col="taskA_cosine_gap",
        finetune_layer_col="taskA_layer",
        fmt=".3f",
        scale=1.0,
    )
    starts = [line.split(" &")[0] for line in lines if "&" in line]
    assert starts[0] == FT_LATA
    assert starts[1] == FT_QWEN


def test_each_ceiling_gets_its_own_verdict_from_its_own_cells():
    best = bht.best_rows(_results_frame(), "hidden", "train_dir_acc_at_1")
    sentence = bht.task_b_comparison(best, _lexical_frame(), _two_model_finetune())
    assert "below every zero-shot ABTT cell for LaTa (83.0 and 81.0)" in sentence
    assert "above every zero-shot ABTT cell for Qwen3-0.6B (95.0 and 94.0)" in sentence
    assert "encoders with ABTT sit" in sentence


def test_reference_caption_names_every_fine_tuned_model():
    best = bht.best_rows(_results_frame(), "hidden", "train_aucroc")
    facts = [{"n_fit_pairs": 499, "n_all_train_pairs": 565, "n_dev_dirs": 28}] * 2
    caption = bht.task_a_caption(best, None, _two_model_finetune(), facts)
    assert "LaTa and Qwen3-0.6B fine-tuned contrastively on 499 of the 565" in caption
    assert "Below the rule, reference systems on the same split" in caption
    assert "encoders' AUROC 0.980 to 0.980 for LaTa" in caption
    assert "for Qwen3-0.6B" in caption


def test_a_single_ceiling_keeps_the_published_wording():
    """Adding the second-model machinery must not rewrite a shipped caption."""
    best_a = bht.best_rows(_results_frame(), "hidden", "train_aucroc")
    best_b = bht.best_rows(_results_frame(), "hidden", "train_dir_acc_at_1")
    one = _finetune_frame(FT_LATA)
    assert (
        "ABTT moves the fine-tuned encoder's AUROC from 0.980 to 0.980 while "
        "moving its gap from 0.380 to 0.380."
    ) in bht.task_a_comparison(best_a, _lexical_frame(), one)
    assert (
        "the fine-tuned encoder with ABTT (83.0 and 81.0) is below every "
        "zero-shot ABTT cell (91.0 to 91.0 and 90.0 to 90.0)."
    ) in bht.task_b_comparison(best_b, _lexical_frame(), one)


def test_pairs_clause_drops_the_numbers_when_the_runs_disagree():
    """One split, one seed, so the counts must agree; if they do not, say less."""
    agreeing = [{"n_fit_pairs": 499, "n_all_train_pairs": 565, "n_dev_dirs": 28}] * 2
    assert "499 of the 565" in bht.finetune_pairs_clause(agreeing)
    disagreeing = [
        {"n_fit_pairs": 499, "n_all_train_pairs": 565, "n_dev_dirs": 28},
        {"n_fit_pairs": 480, "n_all_train_pairs": 565, "n_dev_dirs": 30},
    ]
    assert bht.finetune_pairs_clause(disagreeing) == (
        "the positive train pairs left by a directory-level dev carve"
    )


def test_main_accepts_one_finetune_csv_per_model(tmp_path, monkeypatch):
    _results_frame().to_csv(tmp_path / "r.csv", index=False)
    _lexical_frame().to_csv(tmp_path / "l.csv", index=False)
    _finetune_frame(FT_LATA).to_csv(tmp_path / "lata.csv", index=False)
    _finetune_frame(FT_QWEN, assignment=0.95, dir_acc=0.94).to_csv(
        tmp_path / "qwen.csv", index=False
    )
    facts = {"n_fit_pairs": 499, "n_all_train_pairs": 565, "n_dev_dirs": 28}
    for name in ("lata_info.json", "qwen_info.json"):
        (tmp_path / name).write_text(json.dumps({"caption_facts": facts}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_headline_tables.py",
            "--results_csv", str(tmp_path / "r.csv"),
            "--lexical_csv", str(tmp_path / "l.csv"),
            "--finetune_csv", str(tmp_path / "lata.csv"),
            "--finetune_run_info", str(tmp_path / "lata_info.json"),
            "--finetune_csv", str(tmp_path / "qwen.csv"),
            "--finetune_run_info", str(tmp_path / "qwen_info.json"),
            "--out_dir", str(tmp_path / "out"),
        ],
    )
    bht.main()
    for name in ("taskA_headline.tex", "taskB_headline.tex"):
        tex = (tmp_path / "out" / name).read_text()
        assert FT_LATA + " &" in tex, name
        assert FT_QWEN + " &" in tex, name
        assert "LaTa and Qwen3-0.6B fine-tuned contrastively" in tex, name
    task_b = (tmp_path / "out" / "taskB_headline.tex").read_text()
    assert "below every zero-shot ABTT cell for LaTa" in task_b
    assert "above every zero-shot ABTT cell for Qwen3-0.6B" in task_b


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


def test_caption_omits_ties_only_on_an_explicit_opt_out():
    """A tie claim names a cell, so it may only come from per-pair differences.

    The published caption hardcoded "the LaTa MaRC DelAUC win is a tie at about
    1.2 standard errors". That was true of one sample; issue #141 re-sampled and
    the narrow cells moved. ``pairs_root=None`` is the deliberate opt-out
    (``--no_tie_clause``), and then the caption says nothing about ties rather
    than repeating the old claim.
    """
    summary = bmaa.select_main_rows(_attribution_summary())
    caption = bmaa.main_caption(summary, pairs_root=None)
    assert "standard error" not in caption
    assert "tie" not in caption


def test_caption_refuses_a_missing_per_pair_cache(tmp_path: Path):
    """A cache that should be there but is not must fail, not silently degrade.

    The per-pair cache is gitignored, so on a fresh clone the tie clause would
    otherwise vanish and the committed table would not regenerate from the repo.
    """
    summary = bmaa.select_main_rows(_attribution_summary())
    with pytest.raises(FileNotFoundError, match="per-pair metric cache"):
        bmaa.main_caption(summary, pairs_root=tmp_path / "absent")


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


# --- Run of record (issue #201) -------------------------------------------------
#
# The committed attribution tables come from the benchmark v1 re-sample, but the
# generators' defaults stayed on the run 3 sample after #187, so a bare rerun
# rewrote the paper's numbers from the old run. These tests pin the defaults to
# the run of record and check that a bare regeneration is a no-op on the
# committed tables. The byte-identity tests need the run's summary (tracked) and,
# for the main generator, its gitignored per-pair cache, so they skip where that
# material is absent (CI) rather than fail.

TABLES_DIR = REPO_ROOT / "overleaf_drafts" / "tables"


def test_generator_defaults_resolve_under_the_run_of_record(monkeypatch):
    assert aror.RUN_OF_RECORD == "ig_examples_200pos_v1"
    # Issue #206: the 20-draw metrics pass, written beside the 5-draw one.
    assert aror.METRICS_DIR_OF_RECORD == "attribution_metrics_draws20"
    assert aror.RUN_OF_RECORD in bmaa.DEFAULT_SUMMARY.parts
    assert aror.METRICS_DIR_OF_RECORD in bmaa.DEFAULT_SUMMARY.parts
    assert bmaa.DEFAULT_SUMMARY.name == "summary_v2.csv"

    monkeypatch.setattr(sys, "argv", ["package_attribution_sweep_appendix.py"])
    args = pasa.parse_args()
    for raw in (args.summary_csv, args.long_out, args.missing_report_out):
        assert aror.RUN_OF_RECORD in Path(raw).parts, raw
        assert aror.METRICS_DIR_OF_RECORD in Path(raw).parts, raw
        assert "run3" not in raw
    assert Path(args.main_tex_out).parent == TABLES_DIR
    assert Path(args.supplemental_tex_out).parent == TABLES_DIR


def test_stamped_table_refuses_a_different_run(tmp_path: Path):
    """A table built from one run is not silently rewritten from another."""
    stamped = tmp_path / "stamped.tex"
    stamped.write_text("% generated table\n% source run: run_a\n\\begin{table}\n")
    with pytest.raises(SystemExit, match="built from run run_a"):
        aror.refuse_run_change(stamped, "run_b")
    aror.refuse_run_change(stamped, "run_a")
    aror.refuse_run_change(stamped, "run_b", allow=True)
    # First-time stamping and scratch outputs: no stamp, no objection.
    unstamped = tmp_path / "legacy.tex"
    unstamped.write_text("% generated table\n\\begin{table}\n")
    aror.refuse_run_change(unstamped, "run_b")
    aror.refuse_run_change(tmp_path / "absent.tex", "run_b")
    # The stamp names the metrics directory as well as the run (issue #206):
    # one run can hold two metrics passes whose tables differ.
    assert aror.run_name("runs/active/run_c/attribution_metrics/summary_v2.csv") == (
        "run_c/attribution_metrics")
    assert aror.run_name("runs/active/run_c/attribution_metrics_draws20/summary_v2.csv") == (
        "run_c/attribution_metrics_draws20")


def _skip_unless(path: Path, what: str) -> None:
    if not path.exists():
        pytest.skip(f"{what} not present at {path}")


def test_bare_packager_run_reproduces_the_committed_sweep_tables(tmp_path: Path, monkeypatch):
    _skip_unless(aror.DEFAULT_SUMMARY_CSV, "run-of-record summary")
    monkeypatch.setattr(sys, "argv", [
        "package_attribution_sweep_appendix.py",
        "--main_tex_out", str(tmp_path / "main.tex"),
        "--supplemental_tex_out", str(tmp_path / "supp.tex"),
        "--long_out", str(tmp_path / "long.csv"),
        "--missing_report_out", str(tmp_path / "report.json"),
    ])
    pasa.main()
    for name, out in (("attribution_metrics_sweep_main_methods.tex", "main.tex"),
                      ("attribution_metrics_sweep_supplemental_methods.tex", "supp.tex")):
        assert (tmp_path / out).read_bytes() == (TABLES_DIR / name).read_bytes(), name
    assert (tmp_path / "long.csv").read_bytes() == (
        aror.ATTRIBUTION_METRICS_DIR / "summary_v2_sweep_long_appendix.csv").read_bytes()


def test_bare_headline_run_reproduces_the_committed_tables(tmp_path: Path, monkeypatch):
    """Issue #208: a bare run once dropped the Qwen3-0.6B fine-tuned row because
    the defaults named only the LaTa ceiling. Both ceilings are defaults now, and
    this drives ``main()`` with no arguments but ``--out_dir``."""
    for raw in [bht.DEFAULT_RESULTS_CSV, *bht.DEFAULT_FINETUNE_CSVS,
                *bht.DEFAULT_FINETUNE_RUN_INFOS]:
        _skip_unless(REPO_ROOT / raw, "headline table input (gitignored)")
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setattr(sys, "argv", ["build_headline_tables.py",
                                      "--out_dir", str(tmp_path)])
    bht.main()
    for name in ("taskA_headline.tex", "taskB_headline.tex"):
        assert (tmp_path / name).read_bytes() == (TABLES_DIR / name).read_bytes(), name


def test_bare_generator_run_reproduces_the_committed_main_tables(tmp_path: Path, monkeypatch):
    _skip_unless(aror.DEFAULT_SUMMARY_CSV, "run-of-record summary")
    _skip_unless(aror.ATTRIBUTION_METRICS_DIR / "v2_hidden", "per-pair cache (gitignored)")
    monkeypatch.setattr(sys, "argv", [
        "build_main_attribution_artifacts.py",
        "--table_out", str(tmp_path / "main.tex"),
        "--secondary_table_out", str(tmp_path / "secondary.tex"),
        "--fig_out_base", str(tmp_path / "fig_attribution_rho_loo_main"),
    ])
    bmaa.main()
    for name, out in (("attribution_metrics_main.tex", "main.tex"),
                      ("attribution_metrics_secondary.tex", "secondary.tex")):
        assert (tmp_path / out).read_bytes() == (TABLES_DIR / name).read_bytes(), name
    assert (tmp_path / "fig_attribution_rho_loo_main.tex").read_bytes() == (
        REPO_ROOT / "overleaf_drafts/figures/fig_attribution_rho_loo_main.tex").read_bytes()
