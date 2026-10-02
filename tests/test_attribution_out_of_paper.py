"""Token attribution left the paper (paper spine, 2 October 2026).

The attribution code, its runs and its generated tables stay in the repository;
only the paper stops using them. These tests keep it that way: every attribution
generator writes outside ``overleaf_drafts/`` by default, and the paper tree
holds no attribution table or figure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER = REPO_ROOT / "overleaf_drafts"
ARTIFACTS = REPO_ROOT / "docs" / "analyses" / "attribution_artifacts"

sys.path.insert(0, str(REPO_ROOT / "scripts" / "ig"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "resubmit"))

import build_aniso_attribution_table as bat  # noqa: E402
import build_main_attribution_artifacts as bmaa  # noqa: E402


def _outside_paper(path) -> bool:
    return "overleaf_drafts" not in Path(path).parts


def test_main_and_aniso_generators_default_outside_the_paper():
    for path in (bmaa.DEFAULT_TABLE_OUT, bmaa.DEFAULT_SECONDARY_OUT,
                 bmaa.DEFAULT_SHUFFLE_OUT, bmaa.DEFAULT_FIG_OUT, bat.DEFAULT_TABLE_OUT):
        assert _outside_paper(path), path
        assert ARTIFACTS in Path(path).parents, path


def test_no_attribution_script_names_the_paper_tree():
    """Default output paths are plain strings in argparse; read the sources."""
    scripts = [
        "scripts/ig/build_main_attribution_artifacts.py",
        "scripts/ig/build_aniso_attribution_table.py",
        "scripts/ig/build_delauc_sensitivity_table.py",
        "scripts/ig/package_attribution_sweep_appendix.py",
        "scripts/ig/run_attribution_metrics.py",
        "scripts/resubmit/visualize_retrieval_mark.py",
        "scripts/resubmit/visualize_pair_attribution.py",
    ]
    for rel in scripts:
        assert "overleaf_drafts" not in (REPO_ROOT / rel).read_text(), rel


@pytest.mark.skipif(not PAPER.exists(), reason="overleaf_drafts/ not checked out")
def test_paper_tree_holds_no_attribution_artifact():
    leftovers = sorted(p.name for p in (PAPER / "tables").glob("attribution_*"))
    leftovers += sorted(
        p.name for pattern in ("fig_attribution_*", "fig_pair_matrix_philta*",
                               "fig_attention_philta*", "fig_retrieval_mark_pair_philta*")
        for p in (PAPER / "figures").glob(pattern)
    )
    assert leftovers == []
