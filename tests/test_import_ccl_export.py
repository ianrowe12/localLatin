"""Unit tests for the CCL plain-text export importer (issue #282).

Every fixture is synthetic and built in tmp_path: no CCL text, no real keys
beyond invented ones, no network. The real-export run is documented in the PR.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import stat
import sys
import unicodedata
import zipfile
from datetime import datetime
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "import_ccl_export.py"
_spec = importlib.util.spec_from_file_location("import_ccl_export", MODULE_PATH)
imp = importlib.util.module_from_spec(_spec)
sys.modules["import_ccl_export"] = imp
_spec.loader.exec_module(imp)

SENTINEL = "zzsentinelzz"

# A tiny corpus: two keys, one of them with a line break as some of ours have.
BASELINE_LABELLED = {
    "AAAA.100.1": {"MS1.1r.1": f"   alpha {SENTINEL} one ", "MS2.1r.1": "   alpha two "},
    "Long cite,\nB 2": {"MS1.2r.1": "   beta one "},
}
BASELINE_UNLABELLED = {"MS1.3r.1": "   gamma ", "MS1.3r.2": "   delta ", "MS2.9v.4": "   eps "}


def write_tree(root: Path, files: dict) -> None:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text if isinstance(text, bytes) else text.encode("utf-8"))


def make_baseline(tmp_path: Path) -> Path:
    root = tmp_path / "baseline"
    files = {}
    for key, members in BASELINE_LABELLED.items():
        for xml_id, text in members.items():
            files[f"canon_labelled/{key}/{xml_id}.txt"] = text
    for xml_id, text in BASELINE_UNLABELLED.items():
        files[f"canon_unlabelled/{xml_id}.txt"] = text
    write_tree(root, files)
    return root


def export_files(type_="canon") -> dict:
    """The baseline as CCL exports it, with the line break dropped from the long key."""
    files = {}
    for key, members in BASELINE_LABELLED.items():
        key = key.replace("\n", "")
        for xml_id, text in members.items():
            files[f"cclexport/keyed/{type_}/{key}/{xml_id}.txt"] = text
    for xml_id, text in BASELINE_UNLABELLED.items():
        files[f"cclexport/unkeyed/{type_}/{xml_id}.txt"] = text
    return files


def make_folder(tmp_path: Path, files: dict, name: str = "export") -> Path:
    root = tmp_path / name
    write_tree(root, files)
    return root / "cclexport"


def make_zip(tmp_path: Path, files: dict, name: str = "export.zip") -> Path:
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as archive:
        for rel, text in files.items():
            archive.writestr(rel, text if isinstance(text, bytes) else text.encode("utf-8"))
    return path


def run(tmp_path, export, **kwargs):
    out = tmp_path / kwargs.pop("out_name", "out")
    baseline = kwargs.pop("baseline", None) or make_baseline(tmp_path)
    report = imp.run(export, out, baseline_root=baseline, **kwargs)
    return report, out


@pytest.fixture(params=["folder", "zip"])
def same_export(request, tmp_path):
    files = export_files()
    if request.param == "folder":
        return make_folder(tmp_path, files)
    return make_zip(tmp_path, files)


# --- the unchanged case ------------------------------------------------------


def test_unchanged_export_reports_nothing_but_the_whitespace_key(tmp_path, same_export):
    report, out = run(tmp_path, same_export)
    s = report["summary"]
    assert (s["added"], s["removed"], s["text_changed_bytes"], s["key_changed"]) == (0, 0, 0, 0)
    assert (s["became_keyed"], s["became_unkeyed"]) == (0, 0)
    assert report["key_name_cases"] == [{
        "export_key": "Long cite,B 2",
        "directory": "Long cite,\nB 2",
        "baseline_directory": "Long cite,\nB 2",
        "reason": "kept_baseline_whitespace",
    }]
    # Written under the baseline's names, byte for byte.
    baseline = make_baseline(tmp_path / "again")
    for rel in ("canon_labelled/Long cite,\nB 2/MS1.2r.1.txt",
                "canon_labelled/AAAA.100.1/MS1.1r.1.txt",
                "canon_unlabelled/MS2.9v.4.txt"):
        assert (out / rel).read_bytes() == (baseline / rel).read_bytes()
    q = report["webapp"]["query_corpus"]
    assert q["changed"] is False and q["sha256"] == q["baseline_sha256"]
    assert q["sha256"] == imp.corpus_digest(
        {0: "MS1.3r.1.txt", 1: "MS1.3r.2.txt", 2: "MS2.9v.4.txt"}
    )
    assert "unchanged" in report["webapp"]["startup_forecast"][0]
    assert (out / imp.REPORT_MD).is_file()
    with (out / imp.WITNESSES_CSV).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert {r["directory"] for r in rows} == {"AAAA.100.1", "Long cite,\nB 2", ""}


def test_report_holds_no_witness_text(tmp_path, same_export):
    _, out = run(tmp_path, same_export)
    assert SENTINEL not in (out / imp.REPORT_JSON).read_text()
    assert SENTINEL not in (out / imp.REPORT_MD).read_text()
    assert SENTINEL not in (out / imp.WITNESSES_CSV).read_text()


def test_dry_run_writes_only_the_report(tmp_path, same_export):
    _, out = run(tmp_path, same_export, dry_run=True)
    assert sorted(p.name for p in out.iterdir()) == [imp.REPORT_JSON, imp.REPORT_MD]


# --- changes -----------------------------------------------------------------


def test_every_kind_of_change_is_reported(tmp_path):
    files = export_files()
    # text: one whitespace-only change, one content change
    files["cclexport/keyed/canon/AAAA.100.1/MS2.1r.1.txt"] = "   alpha   two"
    files["cclexport/unkeyed/canon/MS1.3r.2.txt"] = "   delta changed "
    # key change
    del files["cclexport/keyed/canon/AAAA.100.1/MS1.1r.1.txt"]
    files["cclexport/keyed/canon/AAAA.200.9/MS1.1r.1.txt"] = f"   alpha {SENTINEL} one "
    # unkeyed -> keyed, and keyed -> unkeyed
    del files["cclexport/unkeyed/canon/MS1.3r.1.txt"]
    files["cclexport/keyed/canon/AAAA.100.1/MS1.3r.1.txt"] = "   gamma "
    del files["cclexport/keyed/canon/Long cite,B 2/MS1.2r.1.txt"]
    files["cclexport/unkeyed/canon/MS1.2r.1.txt"] = "   beta one "
    # removed and added
    del files["cclexport/unkeyed/canon/MS2.9v.4.txt"]
    files["cclexport/unkeyed/canon/MS3.1r.1.txt"] = "   new "
    files["cclexport/keyed/regCanon/AAAA.100.1/MS4.1r.1.txt"] = "   reg "

    report, out = run(tmp_path, make_folder(tmp_path, files), types="canon,regCanon")
    s = report["summary"]
    assert s["added_by_type_status"] == {"canon": {"unkeyed": 1}, "regCanon": {"keyed": 1}}
    assert [r["xml_id"] for r in report["removed"]] == ["MS2.9v.4"]
    assert s["removed_by_status"] == {"unkeyed": 1}
    changed = {r["xml_id"]: r["whitespace_only"] for r in report["text_changed"]}
    assert changed == {"MS2.1r.1": True, "MS1.3r.2": False}
    assert s["text_changed_bytes"] == 2
    assert s["text_changed_after_whitespace_normalisation"] == 1
    assert report["key_changed"] == [{
        "xml_id": "MS1.1r.1", "type": "canon",
        "baseline_directory": "AAAA.100.1", "directory": "AAAA.200.9",
    }]
    assert report["became_keyed"] == [
        {"xml_id": "MS1.3r.1", "type": "canon", "directory": "AAAA.100.1"}
    ]
    assert report["became_unkeyed"] == [
        {"xml_id": "MS1.2r.1", "type": "canon", "baseline_directory": "Long cite,\nB 2"}
    ]
    assert report["output"]["directories_shared_across_types"] == ["AAAA.100.1"]
    assert (out / "canon_labelled/AAAA.100.1/MS4.1r.1.txt").is_file()

    web = report["webapp"]
    assert web["query_corpus"]["changed"] is True
    leaving = {r["xml_id"]: r["reason"] for r in web["leaving_query_set"]}
    assert leaving == {"MS1.3r.1": "became_keyed", "MS2.9v.4": "absent_from_export"}
    assert web["joining_query_set"] == ["MS1.2r.1", "MS3.1r.1"]
    assert web["labelled_pool_changed"] is True
    forecast = " ".join(web["startup_forecast"])
    assert "MissingWitnessThresholdError" in forecast and "5%" in forecast
    assert "verify_release_matches_corpus" in forecast
    new_names = sorted(p.name for p in (out / "canon_unlabelled").iterdir())
    assert web["query_corpus"]["sha256"] == imp.corpus_digest(dict(enumerate(new_names)))


def test_witness_of_an_unselected_type_is_reported_as_such(tmp_path):
    files = export_files()
    text = files.pop("cclexport/unkeyed/canon/MS2.9v.4.txt")
    files["cclexport/unkeyed/regCanon/MS2.9v.4.txt"] = text
    report, _ = run(tmp_path, make_zip(tmp_path, files))
    assert report["removed"] == [{
        "xml_id": "MS2.9v.4", "status": "unkeyed", "baseline_directory": None,
        "in_export_as": {"type": "regCanon", "status": "unkeyed"},
    }]
    assert report["webapp"]["leaving_query_set"][0]["reason"] == "type_not_selected"


def test_new_key_with_a_line_break_is_written_without_it(tmp_path):
    files = export_files()
    files["cclexport/keyed/canon/New cite,\n  C 3/MS5.1r.1.txt"] = "   x "
    report, out = run(tmp_path, make_zip(tmp_path, files))
    assert {"export_key": "New cite,\n  C 3", "directory": "New cite,  C 3",
            "baseline_directory": None, "reason": "whitespace_collapsed"} in report["key_name_cases"]
    assert (out / "canon_labelled/New cite,  C 3/MS5.1r.1.txt").is_file()


def test_types_all_and_unknown_type(tmp_path):
    files = export_files()
    files["cclexport/unkeyed/creed/MS6.1r.1.txt"] = "   credo "
    report, _ = run(tmp_path, make_zip(tmp_path, files), types="all")
    assert report["types_selected"] == ["canon", "creed"]
    with pytest.raises(imp.ExportError, match="not in the export"):
        run(tmp_path, make_zip(tmp_path, files, "b.zip"), types="canon,decretal", out_name="o2")


# --- names -------------------------------------------------------------------


def test_nfd_names_are_normalised_to_nfc(tmp_path):
    files = export_files()
    nfd_key = unicodedata.normalize("NFD", "Éd. key")
    files[f"cclexport/keyed/canon/{nfd_key}/MS7.1r.1.txt"] = "   e "
    report, out = run(tmp_path, make_zip(tmp_path, files))
    assert report["export"]["names_nfc_normalised"] == 1
    assert [p.name for p in (out / "canon_labelled").iterdir() if p.name.startswith("É")] == [
        unicodedata.normalize("NFC", "Éd. key")
    ]


class _NoUtf8Flag(zipfile.ZipInfo):
    """A ZipInfo written the way the 2026-11-12 export was: UTF-8 bytes, no flag."""

    def _encodeFilenameFlags(self):  # noqa: N802 (zipfile's name)
        return self.filename.encode("utf-8"), self.flag_bits & ~0x800


def test_zip_names_without_the_utf8_flag_are_read_as_utf8(tmp_path):
    path = tmp_path / "noflag.zip"
    files = export_files()
    files["cclexport/keyed/canon/Cite “q” – 4/MS8.1r.1.txt"] = "   q "
    with zipfile.ZipFile(path, "w") as archive:
        for rel, text in files.items():
            archive.writestr(_NoUtf8Flag(rel), text.encode("utf-8"))
    with zipfile.ZipFile(path) as archive:
        assert not any(i.flag_bits & 0x800 for i in archive.infolist())
    _, out = run(tmp_path, path)
    assert (out / "canon_labelled/Cite “q” – 4/MS8.1r.1.txt").is_file()


@pytest.mark.parametrize("twin", ["AAAA.100.\n1", "AAAA.100.\r\n1"])
def test_keys_that_differ_only_by_a_line_break_are_refused(tmp_path, twin):
    files = export_files()
    files[f"cclexport/keyed/canon/{twin}/MS9.1r.1.txt"] = "   x "
    with pytest.raises(imp.ExportError, match="differ only by line breaks"):
        run(tmp_path, make_zip(tmp_path, files))


def test_match_form_deletes_line_breaks_and_collapses_spaces_only():
    assert imp.key_match_form("Gildas,\n82") == imp.key_match_form("Gildas,82")
    assert imp.key_match_form("trans.\r\nWINTER") == "trans.WINTER"
    assert imp.key_match_form(" A  B\t C ") == "A B C"
    assert imp.key_match_form("Ep. 1 23") != imp.key_match_form("Ep. 12 3")
    assert imp.key_match_form("Ep. 1 23") != imp.key_match_form("Ep. 123")


def test_keys_differing_by_an_inter_word_space_are_not_merged_with_the_baseline(tmp_path):
    """Review #284 round 1: `Ep. 1 23` in the baseline is not `Ep. 12 3` in the export."""
    baseline = tmp_path / "baseline"
    write_tree(baseline, {
        "canon_labelled/Ep. 1 23/A.1r.1.txt": "   a ",
        "canon_unlabelled/U.1r.1.txt": "   u ",
    })
    export = make_zip(tmp_path, {
        "cclexport/keyed/canon/Ep. 12 3/A.1r.1.txt": "   a ",
        "cclexport/keyed/canon/Ep. 12 3/B.1r.1.txt": "   b ",
        "cclexport/unkeyed/canon/U.1r.1.txt": "   u ",
    })
    report = imp.run(export, tmp_path / "out", baseline_root=baseline)
    assert report["key_changed"] == [{
        "xml_id": "A.1r.1", "type": "canon",
        "baseline_directory": "Ep. 1 23", "directory": "Ep. 12 3",
    }]
    assert report["added"] == [
        {"xml_id": "B.1r.1", "type": "canon", "status": "keyed", "directory": "Ep. 12 3"}
    ]
    assert report["key_name_cases"] == []
    assert sorted(p.name for p in (tmp_path / "out" / "canon_labelled").iterdir()) == ["Ep. 12 3"]


def test_export_holding_keys_differing_by_an_inter_word_space_is_imported(tmp_path):
    files = export_files()
    files["cclexport/keyed/canon/Ep. 1 23/C.1r.1.txt"] = "   c "
    files["cclexport/keyed/canon/Ep. 12 3/D.1r.1.txt"] = "   d "
    report, out = run(tmp_path, make_zip(tmp_path, files))
    assert (out / "canon_labelled/Ep. 1 23/C.1r.1.txt").is_file()
    assert (out / "canon_labelled/Ep. 12 3/D.1r.1.txt").is_file()
    assert report["output"]["directories"] == 4
    assert {c["export_key"] for c in report["key_name_cases"]} == {"Long cite,B 2"}


def test_new_keys_differing_by_line_break_versus_space_get_two_directories(tmp_path):
    """Review #284 round 2, case 1: `New A\nB` and `New A B` are two keys, two directories."""
    files = export_files()
    files["cclexport/keyed/canon/New A\nB/N1.1r.1.txt"] = "   n1 "
    files["cclexport/keyed/canon/New A B/N2.1r.1.txt"] = "   n2 "
    report, out = run(tmp_path, make_zip(tmp_path, files))
    added = {r["xml_id"]: r["directory"] for r in report["added"]}
    assert added == {"N1.1r.1": "New AB", "N2.1r.1": "New A B"}
    assert sorted(p.name for p in (out / "canon_labelled" / "New AB").iterdir()) == ["N1.1r.1.txt"]
    assert sorted(p.name for p in (out / "canon_labelled" / "New A B").iterdir()) == ["N2.1r.1.txt"]


def test_line_break_variant_of_a_baseline_key_does_not_join_its_directory(tmp_path):
    """Review #284 round 2, case 2: baseline `Old A B`; export `Old A B` and `Old A\nB`."""
    baseline = tmp_path / "baseline"
    write_tree(baseline, {
        "canon_labelled/Old A B/W1.1r.1.txt": "   w1 ",
        "canon_unlabelled/U.1r.1.txt": "   u ",
    })
    export = make_zip(tmp_path, {
        "cclexport/keyed/canon/Old A B/W1.1r.1.txt": "   w1 ",
        "cclexport/keyed/canon/Old A\nB/W2.1r.1.txt": "   w2 ",
        "cclexport/unkeyed/canon/U.1r.1.txt": "   u ",
    })
    report = imp.run(export, tmp_path / "out", baseline_root=baseline)
    assert report["key_changed"] == []
    assert report["added"] == [
        {"xml_id": "W2.1r.1", "type": "canon", "status": "keyed", "directory": "Old AB"}
    ]
    labelled = tmp_path / "out" / "canon_labelled"
    assert sorted(p.name for p in (labelled / "Old A B").iterdir()) == ["W1.1r.1.txt"]
    assert sorted(p.name for p in (labelled / "Old AB").iterdir()) == ["W2.1r.1.txt"]


def test_plan_refuses_two_keys_in_one_directory_whatever_the_naming_rule(monkeypatch):
    """Rule 4 holds on the finished plan, even if the naming rule drifts from the match form."""
    monkeypatch.setattr(imp, "fresh_directory_name", lambda key: " ".join(key.split()))
    with pytest.raises(imp.ExportError, match="resolve to the same directory"):
        imp.plan_directories(["New A\nB", "New A B"], [])
    with pytest.raises(imp.ExportError, match="without matching it"):
        imp.plan_directories(["Old A B", "Old A\nB"], ["Old A B"])
    with pytest.raises(imp.ExportError, match="without matching it"):
        imp.plan_directories(["Old A\nB"], ["Old A B"])


@pytest.mark.parametrize("key", [
    "New A\nB", "New A B", "A  B", " A\tB ", "x\r\ny", "Gildas,\n82", "a\u00a0b", "q \n r",
])
def test_fresh_name_keeps_the_match_form(key):
    assert imp.key_match_form(imp.fresh_directory_name(key)) == imp.key_match_form(key)
    assert "\n" not in imp.fresh_directory_name(key)


def test_fresh_name_keeps_repeated_spaces_and_deletes_line_breaks():
    assert imp.fresh_directory_name("A  B") == "A  B"
    assert imp.fresh_directory_name("New A\nB") == "New AB"
    assert imp.fresh_directory_name("New cite,\n  C 3") == "New cite,  C 3"


def test_plan_gives_every_key_its_own_directory():
    plan, _ = imp.plan_directories(
        ["New A\nB", "New A B", "Old A B", "Old A\nB", "Long cite,B 2"],
        ["Old A B", "Long cite,\nB 2"],
    )
    assert len(set(plan.values())) == len(plan)
    assert plan["Long cite,B 2"] == "Long cite,\nB 2"
    assert plan["Old A B"] == "Old A B"


def test_directories_that_collide_by_case_are_refused(tmp_path):
    files = export_files()
    files["cclexport/keyed/regCanon/aaaa.100.1/MS9.1r.1.txt"] = "   x "
    with pytest.raises(imp.ExportError, match="differ only in case"):
        run(tmp_path, make_zip(tmp_path, files), types="all")


# --- stripped key names -------------------------------------------------------


@pytest.mark.parametrize("kind", ["folder", "zip"])
@pytest.mark.parametrize(("known_key", "stripped_key"), [
    ("eds de Example\u2013 Editor", "eds de Example  Editor"),
    ("E\u0301d. Example", "E d. Example"),
    ("\u00c9d. Example", " d. Example"),
    ("E\u0301d. \u00dcber", "E d.  ber"),
    ("Newline cite,\r\nB 2", "Newline cite,  B 2"),
])
def test_stripped_key_restored_from_same_xml_id(tmp_path, kind, known_key, stripped_key):
    baseline = tmp_path / "baseline"
    write_tree(baseline, {
        f"canon_labelled/{known_key}/A.1r.1.txt": "   a ",
        "canon_unlabelled/U.1r.1.txt": "   u ",
    })
    files = {f"cclexport/keyed/canon/{stripped_key}/A.1r.1.txt": "   a "}
    export = make_folder(tmp_path, files) if kind == "folder" else make_zip(tmp_path, files)
    report, out = run(tmp_path, export, baseline=baseline)
    directory = unicodedata.normalize("NFC", known_key)
    assert (out / f"canon_labelled/{directory}/A.1r.1.txt").read_bytes() == b"   a "
    assert report["key_changed"] == []
    assert report["key_names_restored"] == [{
        "export_key": stripped_key, "directory": directory, "reason": "xml_id",
        "xml_ids": ["A.1r.1"],
    }]
    assert report["summary"]["key_names_restored"] == 1
    assert "## Key names restored (1)" in (out / imp.REPORT_MD).read_text()


def test_same_xml_id_disambiguates_stripped_keys(tmp_path):
    baseline = tmp_path / "baseline"
    write_tree(baseline, {
        "canon_labelled/Cite \u2013 4/A.1r.1.txt": "   a ",
        "canon_labelled/Cite \u2212 4/B.1r.1.txt": "   b ",
        "canon_unlabelled/U.1r.1.txt": "   u ",
    })
    report, out = run(tmp_path, make_zip(tmp_path, {
        "cclexport/keyed/canon/Cite   4/A.1r.1.txt": "   a ",
    }), baseline=baseline)
    assert (out / "canon_labelled/Cite \u2013 4/A.1r.1.txt").is_file()
    assert report["key_names_restored"][0]["reason"] == "xml_id"
    assert "key_names_ambiguous" not in report


def test_unique_stripped_key_restored_for_new_xml_id(tmp_path):
    baseline = tmp_path / "baseline"
    write_tree(baseline, {
        "canon_labelled/Cite \u2013 4/A.1r.1.txt": "   a ",
        "canon_unlabelled/U.1r.1.txt": "   u ",
    })
    report, out = run(tmp_path, make_folder(tmp_path, {
        "cclexport/keyed/canon/Cite   4/B.1r.1.txt": "   b ",
    }), baseline=baseline)
    assert (out / "canon_labelled/Cite \u2013 4/B.1r.1.txt").is_file()
    assert report["key_names_restored"][0]["reason"] == "unique_key"


def test_reference_exports_are_repeatable_and_use_the_existing_reader(tmp_path, capsys):
    reference_zip = make_zip(tmp_path, {
        "cclexport/keyed/regCanon/E\u0301d. \u00dcber/REF.1r.1.txt": "   ref ",
    }, "reference.zip")
    reference_folder = make_folder(tmp_path, {
        "cclexport/keyed/canon/Cite \u2013 4/REF.2r.1.txt": "   ref ",
    }, "reference-folder")
    export = make_zip(tmp_path, {
        "cclexport/keyed/canon/E d.  ber/REF.1r.1.txt": "   a ",
        "cclexport/keyed/canon/Cite   4/NEW.1r.1.txt": "   b ",
    })
    out = tmp_path / "out"
    assert imp.main([
        str(export), "--out-dir", str(out), "--baseline", str(make_baseline(tmp_path)),
        "--reference-export", str(reference_zip),
        "--reference-export", str(reference_folder),
    ]) == 0
    report = json.loads((out / imp.REPORT_JSON).read_text())
    assert (out / "canon_labelled/\u00c9d. \u00dcber/REF.1r.1.txt").is_file()
    assert (out / "canon_labelled/Cite \u2013 4/NEW.1r.1.txt").is_file()
    assert {r["reason"] for r in report["key_names_restored"]} == {"xml_id", "unique_key"}
    assert "Key names restored" in capsys.readouterr().out


@pytest.mark.parametrize("ids", [
    ["NEW.1r.1"], ["A.1r.1", "B.1r.1"], ["A.1r.1", "NEW.1r.1"],
])
def test_ambiguous_stripped_key_is_kept_and_reported(tmp_path, ids):
    baseline = tmp_path / "baseline"
    write_tree(baseline, {
        "canon_labelled/Cite \u2013 4/A.1r.1.txt": "   a ",
        "canon_labelled/Cite \u2212 4/B.1r.1.txt": "   b ",
        "canon_unlabelled/U.1r.1.txt": "   u ",
    })
    report, out = run(tmp_path, make_zip(tmp_path, {
        f"cclexport/keyed/canon/Cite   4/{xml_id}.txt": "   x " for xml_id in ids
    }), baseline=baseline)
    assert sorted(p.name for p in (out / "canon_labelled").iterdir()) == ["Cite   4"]
    assert "key_names_restored" not in report
    assert report["key_names_ambiguous"] == [{
        "export_key": "Cite   4", "candidates": ["Cite \u2013 4", "Cite \u2212 4"],
        "xml_ids": ids,
    }]
    assert report["summary"]["key_names_ambiguous"] == 1
    assert "## Ambiguous key names (1)" in (out / imp.REPORT_MD).read_text()


def test_conflicting_reference_keys_for_the_same_xml_id_are_not_guessed(tmp_path):
    baseline = tmp_path / "custom-baseline"
    write_tree(baseline, {
        "canon_labelled/Cite \u2013 4/A.1r.1.txt": "   a ",
        "canon_unlabelled/U.1r.1.txt": "   u ",
    })
    reference = make_zip(tmp_path, {
        "cclexport/keyed/canon/Cite \u2212 4/A.1r.1.txt": "   a ",
    }, "reference.zip")
    report, out = run(tmp_path, make_zip(tmp_path, {
        "cclexport/keyed/canon/Cite   4/A.1r.1.txt": "   a ",
    }), baseline=baseline, reference_exports=[reference])
    assert (out / "canon_labelled/Cite   4/A.1r.1.txt").is_file()
    assert report["key_names_ambiguous"][0]["candidates"] == ["Cite \u2013 4", "Cite \u2212 4"]


@pytest.mark.parametrize("source", ["baseline", "reference"])
def test_empty_known_key_directories_can_restore_new_witnesses(tmp_path, source):
    baseline = make_baseline(tmp_path)
    if source == "baseline":
        (baseline / "canon_labelled/Empty \u2013 key").mkdir()
        references = []
    else:
        reference = make_folder(tmp_path, {
            "cclexport/unkeyed/canon/REF.1r.1.txt": "   ref ",
        }, "reference")
        (reference / "keyed/canon/Empty \u2013 key").mkdir(parents=True)
        references = [reference]
    report, out = run(tmp_path, make_zip(tmp_path, {
        "cclexport/keyed/canon/Empty   key/NEW.1r.1.txt": "   x ",
    }), baseline=baseline, reference_exports=references)
    assert (out / "canon_labelled/Empty \u2013 key/NEW.1r.1.txt").is_file()
    assert report["key_names_restored"][0]["reason"] == "unique_key"


def test_reference_whitespace_variants_are_the_same_known_key(tmp_path):
    references = [
        make_zip(tmp_path, {
            f"cclexport/keyed/canon/{key}/REF.1r.1.txt": "   ref ",
        }, f"reference-{i}.zip")
        for i, key in enumerate(["\u00c9d.  Example", "\u00c9d. Example"])
    ]
    report, out = run(tmp_path, make_zip(tmp_path, {
        "cclexport/keyed/canon/E d. Example/NEW.1r.1.txt": "   x ",
    }), reference_exports=references)
    assert (out / "canon_labelled/\u00c9d.  Example/NEW.1r.1.txt").is_file()
    assert report["key_names_restored"][0]["reason"] == "unique_key"


def test_reference_exports_are_validated_and_protected_from_output_overlap(tmp_path):
    export = make_zip(tmp_path, export_files())
    reference = make_folder(tmp_path, {
        "cclexport/readme.txt": "   not a witness ",
    }, "reference")
    with pytest.raises(imp.ExportError, match="below 'cclexport/'"):
        run(tmp_path, export, reference_exports=[reference])
    with pytest.raises(imp.ExportError, match="overlaps"):
        run(tmp_path, export, reference_exports=[reference], out_name="reference/cclexport/out")
    assert not (tmp_path / "out").exists()


def test_unknown_ascii_key_and_all_ascii_export_are_flagged(tmp_path):
    report, out = run(tmp_path, make_zip(tmp_path, {
        "cclexport/keyed/canon/Unknown new key/NEW.1r.1.txt": "   x ",
    }))
    assert (out / "canon_labelled/Unknown new key/NEW.1r.1.txt").is_file()
    assert report["key_names_unchecked"] == [{
        "export_key": "Unknown new key", "xml_ids": ["NEW.1r.1"],
    }]
    assert report["summary"]["key_names_unchecked"] == 1
    assert len(report["warnings"]) == 1
    assert "no non-ASCII path characters" in report["warnings"][0]
    markdown = (out / imp.REPORT_MD).read_text()
    assert "## Key names not checked (1)" in markdown
    assert "## Warnings (1)" in markdown


def test_new_ascii_key_is_not_listed_when_export_has_non_ascii_paths(tmp_path):
    report, out = run(tmp_path, make_zip(tmp_path, {
        "cclexport/keyed/canon/NEWK.500.1/NEW.1r.1.txt": "   x ",
        "cclexport/keyed/canon/Cite \u2013 4/OLD.1r.1.txt": "   y ",
    }))
    assert (out / "canon_labelled/NEWK.500.1/NEW.1r.1.txt").is_file()
    assert "key_names_unchecked" not in report
    assert "key_names_unchecked" not in report["summary"]
    assert "warnings" not in report
    assert "Key names not checked" not in (out / imp.REPORT_MD).read_text()


def test_export_with_non_ascii_paths_is_never_restored(tmp_path):
    baseline = tmp_path / "own-baseline"
    write_tree(baseline, {
        "canon_labelled/Ep. 1\u201323/A.1r.1.txt": "   a ",
        "canon_unlabelled/U.1r.1.txt": "   u ",
    })
    report, out = run(tmp_path, make_zip(tmp_path, {
        "cclexport/keyed/canon/Ep. 1 23/A.1r.1.txt": "   a ",
        "cclexport/keyed/canon/Ep. 1 23/NEW.1r.1.txt": "   n ",
        "cclexport/keyed/canon/Other \u00e9/B.1r.1.txt": "   b ",
    }), baseline=baseline)
    assert (out / "canon_labelled/Ep. 1 23/A.1r.1.txt").is_file()
    assert (out / "canon_labelled/Ep. 1 23/NEW.1r.1.txt").is_file()
    assert not (out / "canon_labelled/Ep. 1\u201323").exists()
    assert [r["xml_id"] for r in report["key_changed"]] == ["A.1r.1"]
    assert "key_names_restored" not in report
    assert "warnings" not in report


def test_reference_only_key_gets_the_name_importing_the_reference_gives(tmp_path):
    reference = make_zip(tmp_path, {
        "cclexport/keyed/canon/Long cite,\r\nB\u2013 2/R.1r.1.txt": "   r ",
        "cclexport/keyed/canon/Tab\tcite \u2013 3/T.1r.1.txt": "   t ",
    }, "reference.zip")
    stripped = make_zip(tmp_path, {
        "cclexport/keyed/canon/Long cite,  B  2/R.1r.1.txt": "   r ",
        "cclexport/keyed/canon/Tab cite   3/T.1r.1.txt": "   t ",
    })
    report, out = run(tmp_path, stripped, reference_exports=[reference])
    _, direct = run(tmp_path, reference, out_name="direct")
    listing = lambda root: sorted(
        p.relative_to(root).as_posix() for p in (root / "canon_labelled").rglob("*.txt")
    )
    assert listing(out) == listing(direct)
    assert "canon_labelled/Long cite,B\u2013 2/R.1r.1.txt" in listing(out)
    assert "canon_labelled/Tab cite \u2013 3/T.1r.1.txt" in listing(out)
    assert sorted(r["directory"] for r in report["key_names_restored"]) == [
        "Long cite,B\u2013 2", "Tab cite \u2013 3",
    ]


def test_exact_known_key_is_kept_even_when_it_is_another_keys_stripped_form(tmp_path):
    baseline = tmp_path / "own-baseline"
    write_tree(baseline, {
        "canon_labelled/Cite 4/A.1r.1.txt": "   a ",
        "canon_labelled/Cite \u2013 4/B.1r.1.txt": "   b ",
        "canon_unlabelled/U.1r.1.txt": "   u ",
    })
    # B's own xml:id points at "Cite \u2013 4", but its exported key matches the
    # known key "Cite 4" exactly, so it moves there rather than being restored.
    report, out = run(tmp_path, make_zip(tmp_path, {
        "cclexport/keyed/canon/Cite 4/B.1r.1.txt": "   b ",
    }), baseline=baseline)
    assert (out / "canon_labelled/Cite 4/B.1r.1.txt").is_file()
    assert not (out / "canon_labelled/Cite \u2013 4").exists()
    assert [r["xml_id"] for r in report["key_changed"]] == ["B.1r.1"]
    assert "key_names_restored" not in report


def test_all_ascii_warning_considers_unselected_paths_not_witness_text(tmp_path):
    report, _ = run(tmp_path, make_zip(tmp_path, {
        "cclexport/keyed/canon/AAAA.100.1/MS1.1r.1.txt": "   \u00e9 ",
        "cclexport/unkeyed/regCanon/\u00c9X.1r.1.txt": "   x ",
    }))
    assert "warnings" not in report
    report, _ = run(tmp_path, make_zip(tmp_path, {
        "cclexport/keyed/canon/AAAA.100.1/MS1.1r.1.txt": "   \u00e9 ",
    }, "ascii.zip"), out_name="ascii-out")
    assert "no non-ASCII path characters" in report["warnings"][0]


def test_unstripped_unknown_unicode_key_is_not_restored(tmp_path):
    baseline = tmp_path / "baseline"
    write_tree(baseline, {
        "canon_labelled/Cite \u2013 4/A.1r.1.txt": "   a ",
        "canon_unlabelled/U.1r.1.txt": "   u ",
    })
    report, out = run(tmp_path, make_zip(tmp_path, {
        "cclexport/keyed/canon/Cite \u00e9 4/A.1r.1.txt": "   a ",
    }), baseline=baseline)
    assert (out / "canon_labelled/Cite \u00e9 4/A.1r.1.txt").is_file()
    assert report["key_changed"][0]["directory"] == "Cite \u00e9 4"
    assert "key_names_restored" not in report


@pytest.mark.parametrize("other_key", ["Cite \u2013 4", "cite \u2013 4"])
def test_restored_keys_still_pass_the_final_directory_collision_check(other_key):
    # The safety net itself, whatever restoration decided.
    with pytest.raises(imp.ExportError, match="refusing to merge two keys"):
        imp.plan_directories(
            ["Cite   4", other_key], ["Cite \u2013 4"],
            restored_keys={"Cite   4": "Cite \u2013 4"},
        )


def test_restored_keys_differing_only_in_case_are_refused(tmp_path):
    baseline = tmp_path / "own-baseline"
    write_tree(baseline, {
        "canon_labelled/Cite \u2013 4/A.1r.1.txt": "   a ",
        "canon_unlabelled/U.1r.1.txt": "   u ",
    })
    reference = make_zip(tmp_path, {
        "cclexport/keyed/canon/cite \u2013 4/B.1r.1.txt": "   b ",
    }, "reference.zip")
    export = make_zip(tmp_path, {
        "cclexport/keyed/canon/Cite   4/A.1r.1.txt": "   a ",
        "cclexport/keyed/canon/cite   4/B.1r.1.txt": "   b ",
    })
    with pytest.raises(imp.ExportError, match="refusing to merge two keys"):
        run(tmp_path, export, baseline=baseline, reference_exports=[reference])
    assert not (tmp_path / "out").exists()


def test_unstripped_export_is_byte_identical_to_the_original_importer(tmp_path, monkeypatch):
    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 11, 16, 12, 0, tzinfo=tz)

    monkeypatch.setattr(imp, "datetime", FrozenDatetime)
    monkeypatch.chdir(tmp_path)
    write_tree(Path("baseline"), {
        "canon_labelled/\u00c9d. \u2013 4/A.1r.1.txt": "   a ",
        "canon_labelled/Long cite,\nB 2/B.1r.1.txt": "   b ",
        "canon_unlabelled/U.1r.1.txt": "   u ",
    })
    export = make_folder(Path("."), {
        "cclexport/keyed/canon/E\u0301d. \u2013 4/A.1r.1.txt": "   a ",
        "cclexport/keyed/canon/Long cite,B 2/B.1r.1.txt": "   b ",
        "cclexport/unkeyed/canon/U.1r.1.txt": "   u ",
    })
    imp.run(export, Path("out"), baseline_root=Path("baseline"))
    digests = {
        p.relative_to("out").as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path("out").rglob("*")) if p.is_file()
    }
    # Captured from the pre-restoration importer with the synthetic fixture above.
    assert digests == {
        "canon_labelled/Long cite,\nB 2/B.1r.1.txt":
            "af9ae1bbcb6e68c73dac852a3cbdd30608a6ac39aa79d29b8c349bdd9aa51df0",
        "canon_labelled/\u00c9d. \u2013 4/A.1r.1.txt":
            "eeff1d16e3212d10a9b6e3c5b18580f86e72aa844a986b68ab48441a863706c1",
        "canon_unlabelled/U.1r.1.txt":
            "c2f56631c0adf75e7944ec16f21c1ac652902634afe7bd9f5b6bc55c2f12a985",
        "import_report.json":
            "102cca7322c6e92a764a70aab58de609aca275f833e178a791b30b564dc9cfdf",
        "import_report.md":
            "90508e779061dbef0787b118eed31f5f0fe272b171fb38aad971955aa0a8b2d2",
        "witnesses.csv":
            "a9551f511986f2b60371655dada7ad33269de97e563be76c3012fe28d9a6aafb",
    }


# --- refusals: layout and untrusted input ------------------------------------


@pytest.mark.parametrize("extra, message", [
    ({"cclexport/keyed/regCanon/K/MS1.1r.1.txt": "dup"}, "appears twice in the export"),
    ({"cclexport/readme.txt": "x"}, "below 'cclexport/'"),
    ({"other/x.txt": "x"}, "outside 'cclexport/'"),
    ({"cclexport/keyed/canon/MS9.txt": "x"}, "unexpected depth"),
    ({"cclexport/unkeyed/canon/sub/MS9.txt": "x"}, "unexpected depth"),
    ({"cclexport/unkeyed/canon/MS9.xml": "x"}, "not a .txt file"),
    ({"cclexport/unkeyed/canon/MS 9.txt": "x"}, "whitespace or control character in xml_id"),
    ({"cclexport/unkeyed/canon/MS9.txt": b"\xff\xfe bad"}, "not valid UTF-8"),
    ({"cclexport/keyed/canon/.hidden/MS9.txt": "x"}, "starts with '.'"),
    ({"cclexport/keyed/canon/bad\\key/MS9.txt": "x"}, "backslash"),
    ({"cclexport/keyed/canon/bad\x07key/MS9.txt": "x"}, "control character"),
    ({"cclexport/unkeyed/can on/MS9.txt": "x"}, "unexpected unit type"),
])
def test_unexpected_layouts_are_refused(tmp_path, extra, message):
    files = export_files()
    files.update(extra)
    with pytest.raises(imp.ExportError, match=message):
        run(tmp_path, make_zip(tmp_path, files))


@pytest.mark.parametrize("name, message", [
    ("cclexport/unkeyed/canon/../../../evil.txt", "'.' or '..'"),
    ("/cclexport/unkeyed/canon/abs.txt", "absolute path"),
    ("cclexport//unkeyed/canon/x.txt", "empty"),
])
def test_hostile_zip_paths_are_refused(tmp_path, name, message):
    path = tmp_path / "evil.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for rel, text in export_files().items():
            archive.writestr(rel, text)
        info = zipfile.ZipInfo("placeholder")
        info.filename = name  # bypass ZipInfo's own sanitising, as a hostile archiver would
        archive.writestr(info, "x")
    with pytest.raises(imp.ExportError, match=message):
        run(tmp_path, path)
    assert not (tmp_path / "evil.txt").exists()


def test_zip_symlink_entry_is_refused(tmp_path):
    path = tmp_path / "link.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for rel, text in export_files().items():
            archive.writestr(rel, text)
        info = zipfile.ZipInfo("cclexport/unkeyed/canon/MS9.1r.1.txt")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, "/etc/passwd")
    with pytest.raises(imp.ExportError, match="symlink entry"):
        run(tmp_path, path)


def test_oversized_zip_entry_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(imp, "MAX_FILE_BYTES", 8)
    with pytest.raises(imp.ExportError, match="larger than 8 bytes"):
        run(tmp_path, make_zip(tmp_path, export_files()))


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symlinks")
@pytest.mark.parametrize("target_kind", ["file", "dir"])
def test_folder_symlinks_are_refused(tmp_path, target_kind):
    root = make_folder(tmp_path, export_files())
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "MS9.1r.1.txt").write_text("secret")
    link_dir = root / "unkeyed" / "canon"
    if target_kind == "file":
        os.symlink(outside / "MS9.1r.1.txt", link_dir / "MS9.1r.1.txt")
    else:
        os.symlink(outside, root / "keyed" / "canon" / "LINKED")
    with pytest.raises(imp.ExportError, match="symlink"):
        run(tmp_path, root)


def test_os_artefacts_are_skipped_and_listed(tmp_path):
    files = export_files()
    files["cclexport/keyed/.DS_Store"] = b"\x00\x00"
    files["__MACOSX/cclexport/._x"] = b"\x00"
    report, _ = run(tmp_path, make_zip(tmp_path, files))
    assert report["export"]["ignored_entries"] == ["cclexport/keyed/.DS_Store", "__MACOSX/cclexport/._x"]


def test_folder_may_be_the_parent_of_cclexport(tmp_path):
    root = make_folder(tmp_path, export_files())
    report, _ = run(tmp_path, root.parent)
    assert report["summary"]["added"] == 0
    (root.parent / "stray.txt").write_text("x")
    with pytest.raises(imp.ExportError, match="holding only it"):
        run(tmp_path, root.parent, out_name="o2")


# --- output directory --------------------------------------------------------


def test_out_dir_inside_a_git_tree_is_refused(tmp_path):
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    export = make_zip(tmp_path, export_files())
    with pytest.raises(imp.ExportError, match="inside the git working tree"):
        imp.run(export, tmp_path / "repo" / "corpus", baseline_root=make_baseline(tmp_path))


def test_out_dir_in_this_repo_is_refused(tmp_path):
    export = make_zip(tmp_path, export_files())
    with pytest.raises(imp.ExportError, match="inside the git working tree"):
        imp.run(export, imp.REPO_ROOT / "runs" / "ccl_import_test",
                baseline_root=make_baseline(tmp_path))
    assert not (imp.REPO_ROOT / "runs" / "ccl_import_test").exists()


def test_non_empty_out_dir_is_refused_and_failed_runs_leave_nothing(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    with pytest.raises(imp.ExportError, match="not an empty directory"):
        run(tmp_path, make_zip(tmp_path, export_files()))
    files = export_files()
    files["cclexport/unkeyed/canon/MS9.txt"] = b"\xff"
    with pytest.raises(imp.ExportError):
        run(tmp_path, make_zip(tmp_path, files, "bad.zip"), out_name="fresh")
    assert not (tmp_path / "fresh").exists()
    assert not [p for p in tmp_path.iterdir() if ".partial-" in p.name]


def test_main_exits_2_on_refusal(tmp_path, capsys):
    files = export_files()
    files["cclexport/readme.txt"] = "x"
    code = imp.main([str(make_zip(tmp_path, files)), "--out-dir", str(tmp_path / "o"),
                     "--baseline", str(make_baseline(tmp_path))])
    assert code == 2
    assert "refused" in capsys.readouterr().err


def test_main_writes_report(tmp_path, capsys):
    code = imp.main([str(make_zip(tmp_path, export_files())), "--out-dir", str(tmp_path / "o"),
                     "--baseline", str(make_baseline(tmp_path))])
    assert code == 0
    report = json.loads((tmp_path / "o" / imp.REPORT_JSON).read_text())
    assert report["report_schema_version"] == imp.REPORT_SCHEMA_VERSION
    assert "CCL export import report" in capsys.readouterr().out


# --- shared rules ------------------------------------------------------------


def test_whitespace_rule_matches_tei_to_canon():
    path = MODULE_PATH.parent / "tei_to_canon.py"
    spec = importlib.util.spec_from_file_location("tei_to_canon_for_import", path)
    tei = importlib.util.module_from_spec(spec)
    sys.modules["tei_to_canon_for_import"] = tei  # dataclasses look their module up
    spec.loader.exec_module(tei)
    for text in ["  a  b\n c\t", "", "x", "  y z "]:
        assert imp.normalise_whitespace(text) == tei.normalise(text)


def test_repo_query_listing_digest_is_the_pre_xml_id_corpus():
    """The repo corpus listing hashes to the digest feedback_db pins (#273)."""
    corpus = imp.REPO_ROOT / "data" / "canon_unlabelled"
    if not corpus.is_dir():
        pytest.skip("data/ is not checked out (CI uses a sparse checkout)")
    names = [p.name for p in corpus.glob("*.txt")]
    assert imp.corpus_digest(imp.query_listing(names)) == (
        "23e913f50d96be43e8b412d393302f4b7515586cd9f6b40aecbd0269f81188af"
    )
