"""Guards for the issue #227 appendix table (attribution at the most anisotropic layers).

The synthetic tests pin the verdict rule, the layout and the stamp. The last
test regenerates the committed table from the two runs' summaries and per-pair
caches and checks it byte for byte; those caches are gitignored, so it skips
where they are absent (CI).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ig"))

import attribution_run_of_record as aror  # noqa: E402
import build_aniso_attribution_table as bat  # noqa: E402
import build_main_attribution_artifacts as bmaa  # noqa: E402

TABLE = REPO_ROOT / "overleaf_drafts" / "tables" / "attribution_metrics_aniso.tex"

OPERATIONAL = {"bowphs/LaTa": 7, "bowphs/PhilTa": 1, "google/mt5-base": 1}
ANISOTROPIC = {"bowphs/LaTa": 8, "bowphs/PhilTa": 6, "google/mt5-base": 5}


def _summary(base: float, abtt: float) -> pd.DataFrame:
    rows = []
    for model, _ in bmaa.MODELS:
        for method, _ in bmaa.METHODS:
            for variant, value in (("baseline", base), ("abtt", abtt)):
                row = {
                    "model": model, "method": method, "variant": variant,
                    "n": 200, "full_cos_mean": 0.5,
                    f"{bmaa.DEL_GAP_KEY}_n": 200 if variant == "baseline" else 195,
                    f"{bmaa.INS_GAP_KEY}_n": 200 if variant == "baseline" else 195,
                }
                for key in bmaa.METRIC_KEYS:
                    row[f"{key}_mean"] = value
                for key in bmaa.SHUFFLE_KEYS:
                    row[bmaa._shuffle_gap_col(key, "mean")] = 0.1
                    row[bmaa._shuffle_gap_col(key, "se")] = 0.01
                    row[bmaa._shuffle_gap_col(key, "n")] = 200
                rows.append(row)
    return pd.DataFrame(rows)


def _write_run(root: Path, layers: dict, shifts: dict, *, seed: int) -> Path:
    """A fake run directory: examples CSV, summary and a per-pair cache.

    ``shifts[(model, method)]`` is the true paired ABTT-minus-baseline shift for
    both metrics; 0.0 gives a tie, anything well away from zero a win or loss.
    """
    rng = np.random.default_rng(seed)
    metrics = root / "attribution_metrics_draws20"
    cache = metrics / "v2_hidden"
    cache.mkdir(parents=True)
    examples = []
    example_id = 0
    for model, _ in bmaa.MODELS:
        slug = model.replace("/", "_")
        (cache / slug).mkdir()
        for _ in range(30):
            example_id += 1
            examples.append({"example_id": example_id, "model_name": model,
                             "layer": layers[model]})
            rows = []
            for method, _ in bmaa.METHODS:
                shift = shifts[(model, method)]
                base = {k: float(rng.normal(0.2, 0.05))
                        for k in (bmaa.RHO_KEY, bmaa.DEL_GAP_KEY)}
                abtt = {k: v + shift + float(rng.normal(0.0, 0.05)) for k, v in base.items()}
                rows.append({"model": model, "method": method, "variant": "baseline", **base})
                rows.append({"model": model, "method": method, "variant": "abtt", **abtt})
            (cache / slug / f"example{example_id:03d}_pair_example.json").write_text(
                json.dumps(rows))
    pd.DataFrame(examples).to_csv(root / "positive200_examples.csv", index=False)
    _summary(0.2, 0.5).to_csv(metrics / "summary_v2.csv", index=False)
    return metrics / "summary_v2.csv"


def _all_cells(value: float) -> dict:
    return {(m, k): value for m, _ in bmaa.MODELS for k, _ in bmaa.METHODS}


def test_verdict_follows_the_two_standard_error_rule():
    assert bat.verdict((0.30, 0.02)) == "win"
    assert bat.verdict((-0.30, 0.02)) == "loss"
    assert bat.verdict((0.03, 0.02)) == "tie"
    assert bat.verdict((-0.039, 0.02)) == "tie"
    assert bat.verdict(None) == "missing"


def test_delta_cell_marks_only_ties():
    assert bat._delta_cell((0.367, 0.023)) == "$+0.367$ ($16.0$)"
    assert bat._delta_cell((-0.024, 0.016)).endswith(r"$^\dagger$")
    assert bat._delta_cell(None) == "--"


def test_run_layers_refuses_a_run_with_two_layers_for_one_model(tmp_path: Path):
    csv = tmp_path / "positive200_examples.csv"
    pd.DataFrame({"model_name": ["bowphs/LaTa", "bowphs/LaTa", "bowphs/PhilTa",
                                 "google/mt5-base"],
                  "layer": [7, 8, 1, 1]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="expected one layer"):
        bat.run_layers(csv)


def test_rendered_table_puts_both_layer_sets_under_each_model(tmp_path: Path):
    ref_shifts = _all_cells(0.4)
    ref_shifts[("bowphs/PhilTa", "retrieval_mark")] = 0.0  # a tie
    aniso_shifts = _all_cells(0.4)
    aniso_shifts[("bowphs/LaTa", "ig")] = -0.4  # a loss
    ref = _write_run(tmp_path / "runs/active/ref_run", OPERATIONAL, ref_shifts, seed=1)
    aniso = _write_run(tmp_path / "runs/active/aniso_run", ANISOTROPIC, aniso_shifts, seed=2)

    out = tmp_path / "table.tex"
    bat.main(["--aniso_summary_csv", str(aniso), "--reference_summary_csv", str(ref),
              "--table_out", str(out)])
    tex = out.read_text()

    assert tex.splitlines()[1] == "% source run: aniso_run/attribution_metrics_draws20"
    assert tex.splitlines()[2] == "% reference run: ref_run/attribution_metrics_draws20"
    for label in ("op. 7", "anis. 8", "op. 1", "anis. 6", "anis. 5"):
        assert f"& {label} &" in tex, label
    # 3 models x 2 layer sets x 2 methods.
    body = [line for line in tex.splitlines() if line.endswith(r"\\") and " & " in line]
    assert len(body) == 2 + 12
    rho = bat.paired_cell_stats(ref.parent / "v2_hidden", bat.RHO_KEY)
    assert bat.verdict(rho[("bowphs/PhilTa", "retrieval_mark")]) == "tie"
    assert "ABTT wins, ties and loses 5/1/0 cells" in tex
    assert "and 5/0/1 and" in tex
    assert "LaTa 8, PhilTa 6, and mT5-base 5" in tex
    assert "—" not in tex and "–" not in tex
    assert r"\label{tab:attribution_metrics_aniso}" in tex
    assert "every $\\rho_{\\text{LOO}}$ cell beats a shuffle" in tex


def test_caption_names_the_cells_that_fail_the_shuffle_control():
    summary = bmaa.select_main_rows(_summary(0.2, 0.5))
    col = bmaa._shuffle_gap_col(bat.RHO_KEY, "mean")
    mask = ((summary["model"] == "bowphs/LaTa") & (summary["method"] == "ig")
            & (summary["variant"] == "baseline"))
    summary.loc[mask, col] = -0.014
    layer_set = bat.LayerSet("anis.", summary, {}, {}, ANISOTROPIC)
    assert bat.shuffle_failures(summary, bat.RHO_KEY) == ["LaTa IG baseline"]
    clause = bat._shuffle_clause(layer_set, "most anisotropic")
    assert "in one of the twelve cells (LaTa IG baseline)" in clause


def test_generator_refuses_two_runs_at_the_same_layers(tmp_path: Path):
    ref = _write_run(tmp_path / "runs/active/a", OPERATIONAL, _all_cells(0.4), seed=1)
    other = _write_run(tmp_path / "runs/active/b", OPERATIONAL, _all_cells(0.4), seed=2)
    with pytest.raises(SystemExit, match="same layers"):
        bat.main(["--aniso_summary_csv", str(other), "--reference_summary_csv", str(ref),
                  "--table_out", str(tmp_path / "t.tex")])


def test_generator_refuses_a_missing_per_pair_cache(tmp_path: Path):
    summary = _write_run(tmp_path / "runs/active/a", OPERATIONAL, _all_cells(0.4), seed=1)
    for path in sorted((summary.parent / "v2_hidden").rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    (summary.parent / "v2_hidden").rmdir()
    with pytest.raises(FileNotFoundError, match="per-pair metric cache"):
        bat.load_layer_set(summary, "op.")


def test_defaults_compare_against_the_run_of_record_without_replacing_it():
    assert bat.DEFAULT_REFERENCE_SUMMARY == aror.DEFAULT_SUMMARY_CSV
    assert bat.ANISO_RUN != aror.RUN_OF_RECORD
    assert bat.ANISO_RUN in bat.DEFAULT_ANISO_SUMMARY.parts
    assert aror.RUN_OF_RECORD == "ig_examples_200pos_v1"
    assert bat.DEFAULT_TABLE_OUT == TABLE


def test_bare_run_reproduces_the_committed_aniso_table(tmp_path: Path):
    for path, what in (
        (bat.DEFAULT_ANISO_SUMMARY, "anisotropic-layer summary"),
        (bat.DEFAULT_ANISO_SUMMARY.parent / "v2_hidden", "anisotropic per-pair cache (gitignored)"),
        (aror.DEFAULT_SUMMARY_CSV, "run-of-record summary"),
        (aror.ATTRIBUTION_METRICS_DIR / "v2_hidden", "run-of-record per-pair cache (gitignored)"),
        (TABLE, "committed table"),
    ):
        if not path.exists():
            pytest.skip(f"{what} not present at {path}")
    out = tmp_path / "attribution_metrics_aniso.tex"
    bat.main(["--table_out", str(out)])
    assert out.read_bytes() == TABLE.read_bytes()
