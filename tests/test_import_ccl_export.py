"""Unit tests for the CCL plain-text export importer (issue #282).

Every fixture is synthetic and built in tmp_path: no CCL text, no real keys
beyond invented ones, no network. The real-export run is documented in the PR.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import stat
import sys
import unicodedata
import zipfile
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
    report = imp.run(export, out, baseline_root=kwargs.pop("baseline", make_baseline(tmp_path)), **kwargs)
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


def test_new_key_with_a_line_break_is_written_with_a_space(tmp_path):
    files = export_files()
    files["cclexport/keyed/canon/New cite,\n  C 3/MS5.1r.1.txt"] = "   x "
    report, out = run(tmp_path, make_zip(tmp_path, files))
    assert {"export_key": "New cite,\n  C 3", "directory": "New cite, C 3",
            "baseline_directory": None, "reason": "whitespace_collapsed"} in report["key_name_cases"]
    assert (out / "canon_labelled/New cite, C 3/MS5.1r.1.txt").is_file()


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


def test_directories_that_collide_by_case_are_refused(tmp_path):
    files = export_files()
    files["cclexport/keyed/regCanon/aaaa.100.1/MS9.1r.1.txt"] = "   x "
    with pytest.raises(imp.ExportError, match="differ only in case"):
        run(tmp_path, make_zip(tmp_path, files), types="all")


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
