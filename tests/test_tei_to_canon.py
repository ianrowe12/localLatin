"""Unit tests for the CCL TEI to canon plain-text converter.

Everything here runs on inline fixture markup: no corpus files, no network, no
conda env. The reproduction check against the real BN2123 export is documented in
docs/research/data_derivation.md and run by hand on the cluster.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "tei_to_canon.py"
_spec = importlib.util.spec_from_file_location("tei_to_canon", MODULE_PATH)
tei_to_canon = importlib.util.module_from_spec(_spec)
sys.modules["tei_to_canon"] = tei_to_canon
_spec.loader.exec_module(tei_to_canon)


TEI_HEADER = '<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body>'
TEI_FOOTER = "</body></text></TEI>"


def tei(*units: str) -> str:
    return TEI_HEADER + "".join(units) + TEI_FOOTER


LABELLED_UNIT = """
<div type="canon" n="1" xml:id="MS1.1r.1">
  <div type="msSource">
    <p> CANONES AFRICANENSIS . Hira XXIIII . </p>
  </div>
  <div type="scholSource">
    <p>CAFR.100.1</p>
  </div>
  <div type="canonText">
    <p> Baptizandi nomen suum dent </p>
  </div>
</div>
"""

UNLABELLED_UNIT = """
<div type="canon" n="2" xml:id="MS1.1r.2">
  <div type="msSource">
    <p> Hira XXIII </p>
  </div>
  <div type="canonText">
    <p> Ne offiti et lautioribus epulis </p>
  </div>
</div>
"""


def parse_one(markup: str, **kwargs):
    units, warnings = tei_to_canon.parse_units(tei(markup), **kwargs)
    assert len(units) == 1, warnings
    return units[0]


def test_labelled_unit_keys_directory_and_strips_scholsource():
    unit = parse_one(LABELLED_UNIT)
    assert unit.unit_id == "MS1.1r.1"
    assert unit.filename == "MS1.1r.1.txt"
    assert unit.key == "CAFR.100.1"
    assert unit.directory == "CAFR.100.1"
    assert unit.siglum == "MS1"
    assert "CAFR.100.1" not in unit.text


def test_rubric_precedes_canon_text_with_corpus_whitespace():
    unit = parse_one(LABELLED_UNIT)
    assert unit.text == (
        "    CANONES AFRICANENSIS . Hira XXIIII .    Baptizandi nomen suum dent "
    )


def test_unit_without_scholsource_is_unlabelled():
    unit = parse_one(UNLABELLED_UNIT)
    assert unit.key is None
    assert unit.directory is None
    assert tei_to_canon.normalise(unit.text) == "Hira XXIII Ne offiti et lautioribus epulis"


def test_editorial_note_is_rendered_inline():
    unit = parse_one(
        '<div type="canon" xml:id="MS1.2r.1"><div type="canonText">'
        '<p>Si quis iuxta praeceptum <note type="editorial">right margin rubrics: SPE</note>'
        " domini non baptizauerit</p></div></div>"
    )
    assert tei_to_canon.normalise(unit.text) == (
        "Si quis iuxta praeceptum right margin rubrics: SPE domini non baptizauerit"
    )


def test_deleted_text_is_dropped_but_addition_is_kept():
    unit = parse_one(
        '<div type="canon" xml:id="MS1.2r.2"><div type="canonText">'
        '<p>Liberti qui ex man<seg type="corr"><del>t</del><add>c</add></seg>ipiis</p>'
        "</div></div>"
    )
    normalised = tei_to_canon.normalise(unit.text)
    assert normalised == "Liberti qui ex man c ipiis"
    assert "man t" not in normalised


def test_page_break_splits_the_word_it_interrupts():
    unit = parse_one(
        '<div type="canon" xml:id="MS1.2r.3"><div type="canonText">'
        '<p>annorum peni<pb n="3r"/>tentiam communicent</p></div></div>'
    )
    assert tei_to_canon.normalise(unit.text) == "annorum peni tentiam communicent"


def test_sic_and_unclear_content_is_kept():
    unit = parse_one(
        '<div type="canon" xml:id="MS1.2r.4"><div type="canonText">'
        "<p>Mulier <sic>praesummat</sic> XL<unclear><gap/></unclear>III</p>"
        "</div></div>"
    )
    normalised = tei_to_canon.normalise(unit.text)
    assert "praesummat" in normalised
    assert normalised.endswith("XL III")


def test_number_and_rubric_divs_are_rendered_like_a_rubric():
    unit = parse_one(
        '<div type="regCanon" n="1" xml:id="MS1.4r.1">'
        '<div type="number"><p> I </p></div>'
        '<div type="rubric"><p> DE EPISCOPIS </p></div></div>'
    )
    assert tei_to_canon.normalise(unit.text) == "I DE EPISCOPIS"


def test_divs_without_xml_id_are_not_units():
    units, _ = tei_to_canon.parse_units(
        tei('<div type="canon"><div type="canonText"><p>no id here</p></div></div>')
    )
    assert units == []


def test_empty_unit_is_skipped_with_a_warning():
    units, warnings = tei_to_canon.parse_units(
        tei('<div type="canon" xml:id="MS1.9r.1"><div type="canonText"><p> </p></div></div>')
    )
    assert units == []
    assert any("no text content" in w for w in warnings)


def test_duplicate_xml_id_keeps_the_first_and_warns():
    markup = (
        '<div type="canon" xml:id="MS1.5r.1"><div type="scholSource"><p>KEY.1</p></div>'
        '<div type="canonText"><p>first</p></div></div>'
        '<div type="canon" xml:id="MS1.5r.1"><div type="scholSource"><p>KEY.2</p></div>'
        '<div type="canonText"><p>second</p></div></div>'
    )
    units, warnings = tei_to_canon.parse_units(tei(markup))
    assert [u.key for u in units] == ["KEY.1"]
    assert any("duplicate xml:id" in w for w in warnings)


def test_types_filter_selects_unit_types():
    markup = LABELLED_UNIT + (
        '<div type="regCanon" xml:id="MS1.4r.2"><div type="rubric"><p>DE EPISCOPIS</p>'
        "</div></div>"
    )
    units, _ = tei_to_canon.parse_units(tei(markup), types=["canon"])
    assert [u.unit_id for u in units] == ["MS1.1r.1"]


def test_write_units_uses_the_corpus_layout(tmp_path):
    units, _ = tei_to_canon.parse_units(tei(LABELLED_UNIT + UNLABELLED_UNIT))
    n_labelled, n_unlabelled = tei_to_canon.write_units(units, tmp_path)
    assert (n_labelled, n_unlabelled) == (1, 1)
    labelled = tmp_path / "canon_labelled" / "CAFR.100.1" / "MS1.1r.1.txt"
    unlabelled = tmp_path / "canon_unlabelled" / "MS1.1r.2.txt"
    assert labelled.read_text(encoding="utf-8").startswith("    CANONES AFRICANENSIS")
    assert unlabelled.exists()


@pytest.fixture()
def corpus(tmp_path):
    root = tmp_path / "data"
    (root / "canon_labelled" / "CAFR.100.1").mkdir(parents=True)
    (root / "canon_unlabelled").mkdir(parents=True)
    return root


def statuses(rows):
    return {row.unit_id: row.status for row in rows}


def test_diff_reports_identical_whitespace_changed_moved_and_new(corpus):
    units, _ = tei_to_canon.parse_units(tei(LABELLED_UNIT + UNLABELLED_UNIT))
    labelled, unlabelled = units

    (corpus / "canon_labelled" / "CAFR.100.1" / "MS1.1r.1.txt").write_text(
        labelled.text, encoding="utf-8"
    )
    (corpus / "canon_unlabelled" / "MS1.1r.2.txt").write_text(
        "  " + " ".join(unlabelled.text.split()) + " ", encoding="utf-8"
    )
    rows = tei_to_canon.diff_units(units, tei_to_canon.index_corpus(corpus))
    assert statuses(rows) == {"MS1.1r.1": "identical", "MS1.1r.2": "whitespace"}


def test_diff_flags_moved_and_changed_and_absent(corpus):
    units, _ = tei_to_canon.parse_units(tei(LABELLED_UNIT))
    (corpus / "canon_labelled" / "OLD.KEY.9").mkdir(parents=True)
    (corpus / "canon_labelled" / "OLD.KEY.9" / "MS1.1r.1.txt").write_text(
        units[0].text, encoding="utf-8"
    )
    (corpus / "canon_unlabelled" / "MS1.7v.1.txt").write_text("dropped", encoding="utf-8")
    (corpus / "canon_unlabelled" / "OTHER.1r.1.txt").write_text("other ms", encoding="utf-8")

    rows = tei_to_canon.diff_units(units, tei_to_canon.index_corpus(corpus))
    by_id = statuses(rows)
    assert by_id["MS1.1r.1"] == "moved"
    assert by_id["MS1.7v.1"] == "absent_from_export"
    # A file from another manuscript is out of scope for this export.
    assert "OTHER.1r.1" not in by_id

    moved = next(row for row in rows if row.unit_id == "MS1.1r.1")
    assert (moved.existing_dir, moved.export_dir) == ("OLD.KEY.9", "CAFR.100.1")


def test_diff_flags_text_change(corpus):
    units, _ = tei_to_canon.parse_units(tei(LABELLED_UNIT))
    (corpus / "canon_labelled" / "CAFR.100.1" / "MS1.1r.1.txt").write_text(
        units[0].text.replace("Baptizandi", "Baptizandum"), encoding="utf-8"
    )
    rows = tei_to_canon.diff_units(units, tei_to_canon.index_corpus(corpus))
    assert statuses(rows) == {"MS1.1r.1": "changed"}
    assert "Baptizand" in rows[0].detail


def test_cli_dry_run_writes_report(tmp_path, capsys):
    xml_path = tmp_path / "export.xml"
    xml_path.write_text(tei(LABELLED_UNIT), encoding="utf-8")
    data_root = tmp_path / "data"
    (data_root / "canon_labelled" / "CAFR.100.1").mkdir(parents=True)
    report = tmp_path / "diff.csv"

    exit_code = tei_to_canon.main(
        [str(xml_path), "--dry-run", "--data-root", str(data_root), "--report", str(report)]
    )
    assert exit_code == 0
    assert "new" in capsys.readouterr().out
    assert "MS1.1r.1" in report.read_text(encoding="utf-8")
    # --dry-run writes no corpus files.
    assert list((data_root / "canon_labelled" / "CAFR.100.1").iterdir()) == []


def test_cli_requires_an_output_mode(tmp_path):
    xml_path = tmp_path / "export.xml"
    xml_path.write_text(tei(LABELLED_UNIT), encoding="utf-8")
    with pytest.raises(SystemExit):
        tei_to_canon.main([str(xml_path)])
