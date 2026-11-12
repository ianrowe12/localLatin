"""The CCL importer's corpus digest is the webapp's (issue #282).

`scripts/data/import_ccl_export.py` keeps a stdlib copy of
`feedback_db.corpus_digest` so it runs without the webapp's dependencies. These
tests pin the two together, and pin the importer's file_id assignment to the
one `scripts/resubmit/index_unlabelled.py` makes, so the digest the import
report predicts is the one `adopt_corpus()` will compute.
"""
from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

from web.services.feedback_db import PRE_XML_ID_CORPUS_SHA256, corpus_digest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "data" / "import_ccl_export.py"
_spec = importlib.util.spec_from_file_location("import_ccl_export_web", MODULE_PATH)
imp = importlib.util.module_from_spec(_spec)
sys.modules["import_ccl_export_web"] = imp
_spec.loader.exec_module(imp)


def test_digest_functions_agree():
    rng = random.Random(282)
    for size in (0, 1, 7, 200):
        names = [
            f"MS{rng.randint(1, 999)}.{rng.randint(1, 99)}r.{i}[x](y)é.txt" for i in range(size)
        ]
        mapping = {rng.randint(0, 10_000) + 10_001 * i: name for i, name in enumerate(names)}
        assert imp.corpus_digest(mapping) == corpus_digest(mapping)


def test_listing_matches_index_unlabelled_and_the_pinned_corpus():
    unlabelled = REPO_ROOT / "data" / "canon_unlabelled"
    # index_unlabelled.py: sorted(root.glob("*.txt"), key=name), file_id = position.
    expected = {
        i: p.name for i, p in enumerate(sorted(unlabelled.glob("*.txt"), key=lambda p: p.name))
    }
    listing = imp.query_listing(p.name for p in unlabelled.glob("*.txt"))
    assert listing == expected
    assert corpus_digest(listing) == PRE_XML_ID_CORPUS_SHA256
    assert imp.corpus_digest(listing) == PRE_XML_ID_CORPUS_SHA256
