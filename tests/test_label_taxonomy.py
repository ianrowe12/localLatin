"""Appendix A label taxonomy: the counts the paper states must reproduce.

The benchmark is frozen (v1), so the two numbers are fixed; the rule is in
``scripts/data/label_taxonomy.py`` and the appendix sentence states it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "data"))

import label_taxonomy as lt  # noqa: E402


def test_rule_admits_annotated_keys_and_rejects_the_rest():
    for label in ("CTOU.567.16", "CTOU.567.16 (15)", "CARL.501?.18",
                  "DSIR.384.255  cap. 11", "CNEO.315.",
                  "CAGD.506.27; anno 506"):
        assert lt.is_key_named(label), label
    for label in ("Can.apost.12", "Ephesians.5:22", "Capit.Martini.14",
                  "Stat.Eccl.Antiq. 20", "Unidentified", "DGEL492.636 cap. 7,8",
                  "Caesarius, Sermo 43.1 (ed. Morin, 190, lines 21-25)"):
        assert not lt.is_key_named(label), label


def test_frozen_benchmark_counts_match_appendix_a():
    root = lt.DEFAULT_ROOT
    if not root.exists():
        pytest.skip(f"benchmark corpus not present at {root}")
    labels = lt.directory_labels(root)
    assert len(labels) == 840
    assert lt.count_taxonomy(labels) == (689, 151)
