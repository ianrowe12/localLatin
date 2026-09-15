"""Unit tests for the issue #211 prefix/frequency attribution analysis.

Everything here runs on synthetic pieces and a synthetic NPZ: no tokenizer
download, no artifacts, no corpus. `transformers` is never imported, which
matters because CI installs only the webapp requirements.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "ig"))

from prefix_attribution import (  # noqa: E402
    PieceLexicon,
    TokenizerScheme,
    aggregate_to_words,
    build_piece_table,
    iter_corpus_files,
    normalise_word,
    pair_side_metrics,
)

run_analysis = pytest.importorskip(
    "run_prefix_attribution_analysis",
    reason="pandas is required for the driver module",
)

SP = "▁"
BPE = "Ġ"


def make_lexicon(scheme_name: str, counts: dict[str, int],
                 words: dict[str, int] | None = None,
                 top_pct: float = 0.01) -> PieceLexicon:
    lex = PieceLexicon(scheme=TokenizerScheme(scheme_name),
                       piece_counts=Counter(counts),
                       word_counts=Counter(words or {}),
                       top_pct=top_pct)
    return lex.finalise()


# --------------------------------------------------------------------------
# piece classification and word grouping
# --------------------------------------------------------------------------

def test_normalise_word_strips_punctuation_and_case():
    assert normalise_word("Prae,") == "prae"
    assert normalise_word("...") == ""
    assert normalise_word("ÆDE") == "æde"  # no ASCII-folding of ligatures
    assert normalise_word("ádiuvo") == "adiuvo"


@pytest.mark.parametrize(
    "scheme,pieces,expected_starts",
    [
        ("sentencepiece", [f"{SP}de", f"{SP}prae", "dest", f"{SP}ab"],
         [True, True, False, True]),
        ("wordpiece", ["prae", "##dest", "##ination", "ab"],
         [True, False, False, True]),
        ("bpe", ["de", f"{BPE}prae", "dest", f"{BPE}ab"],
         [False, True, False, True]),
    ],
)
def test_scheme_word_starts(scheme, pieces, expected_starts):
    s = TokenizerScheme(scheme)
    assert [s.starts_word(p) for p in pieces] == expected_starts


def test_piece_table_groups_words_and_flags_prefixes():
    pieces = [f"{SP}de", f"{SP}prae", "dest", "inatione", f"{SP}episcopo"]
    lex = make_lexicon("sentencepiece", {p: 1 for p in pieces})
    tab = build_piece_table(pieces, lex, specials=set())

    assert tab.words == ["de", "praedestinatione", "episcopo"]
    assert tab.word_id.tolist() == [0, 1, 1, 1, 2]
    # "de" and "prae" are both prefix pieces; only "de" is a whole word here.
    assert tab.is_prefix.tolist() == [True, True, False, False, False]
    assert tab.is_whole_word.tolist() == [True, False, False, False, True]
    assert tab.word_is_prefix.tolist() == [True, False, False]


def test_bpe_pieces_are_decoded_back_to_text():
    # "Quae" written with the ae ligature, as byte-level BPE stores it.
    pieces = [f"{BPE}Qu", "\u00c3\u00a6", f"{BPE}ad"]
    lex = make_lexicon("bpe", {p: 1 for p in pieces})
    tab = build_piece_table(pieces, lex, specials=set())
    assert tab.words == ["Qu\u00e6", "ad"]
    assert tab.is_prefix.tolist() == [False, False, True]


def test_piece_table_drops_specials_and_empty_pieces():
    pieces = ["<s>", f"{SP}ad", SP, "</s>"]
    lex = make_lexicon("sentencepiece", {f"{SP}ad": 1})
    tab = build_piece_table(pieces, lex, specials={"<s>", "</s>"})
    assert tab.keep.tolist() == [False, True, False, False]
    assert tab.words == ["ad"]


def test_dropped_boundary_piece_still_breaks_the_word():
    """SentencePiece's bare "\u2581" marks a space and must not glue two words."""
    pieces = [f"{SP}consonante", "r", SP, "capitulum", "<sep>", "xv"]
    lex = make_lexicon("sentencepiece", {f"{SP}consonante": 1, "r": 1,
                                         "capitulum": 1, "xv": 1})
    tab = build_piece_table(pieces, lex, specials={"<sep>"})
    assert tab.words == ["consonanter", "capitulum", "xv"]
    assert tab.word_id.tolist() == [0, 0, -1, 1, -1, 2]


def test_top_one_percent_is_taken_over_observed_types():
    counts = {f"p{i}": 1 for i in range(200)}
    counts["common"] = 10_000
    counts["alsocommon"] = 5_000
    lex = make_lexicon("bpe", counts, top_pct=0.01)
    # 202 types -> round(2.02) == 2 most frequent types.
    assert lex.top_frequent == frozenset({"common", "alsocommon"})
    assert lex.corpus_share_frequent() == pytest.approx(15_000 / 15_200)


def test_corpus_share_prefix_counts_tokens_not_types():
    lex = make_lexicon("sentencepiece", {f"{SP}ad": 30, f"{SP}episcopus": 70})
    assert lex.corpus_share_prefix() == pytest.approx(0.3)


# --------------------------------------------------------------------------
# attribution shares
# --------------------------------------------------------------------------

def test_mass_share_uses_positive_part_only():
    pieces = [f"{SP}ad", f"{SP}episcopo"]
    lex = make_lexicon("sentencepiece", {p: 1 for p in pieces})
    tab = build_piece_table(pieces, lex, specials=set())
    m = pair_side_metrics(np.array([1.0, -3.0]), tab)
    # The negative candidate contributes nothing to the denominator.
    assert m["mass_share_prefix"] == pytest.approx(1.0)
    assert m["count_share_prefix"] == pytest.approx(0.5)


def test_prefix_lift_is_visible_when_mass_concentrates_on_prefixes():
    pieces = [f"{SP}prae", "destinatione", f"{SP}episcopo", f"{SP}canonum"]
    lex = make_lexicon("sentencepiece", {p: 1 for p in pieces})
    tab = build_piece_table(pieces, lex, specials=set())
    m = pair_side_metrics(np.array([9.0, 1.0, 0.0, 0.0]), tab)
    assert m["count_share_prefix"] == pytest.approx(0.25)
    assert m["mass_share_prefix"] == pytest.approx(0.9)
    # Both scoring pieces belong to the two-piece word "praedestinatione", so
    # all of the mass sits on fragments rather than on whole words.
    assert m["mass_share_fragment"] == pytest.approx(1.0)
    assert m["mass_share_whole_word"] == pytest.approx(0.0)


def test_prefix_mass_splits_whole_words_from_word_internal_fragments():
    # "in" stands alone (a preposition); "prae" opens "praedestinatione".
    pieces = [f"{SP}in", f"{SP}prae", "destinatione"]
    lex = make_lexicon("sentencepiece", {p: 1 for p in pieces})
    tab = build_piece_table(pieces, lex, specials=set())
    m = pair_side_metrics(np.array([3.0, 6.0, 1.0]), tab)
    assert m["count_share_prefix"] == pytest.approx(2 / 3)
    assert m["count_share_prefix_wholeword"] == pytest.approx(1 / 3)
    assert m["count_share_prefix_fragment"] == pytest.approx(1 / 3)
    assert m["mass_share_prefix"] == pytest.approx(0.9)
    assert m["mass_share_prefix_wholeword"] == pytest.approx(0.3)
    assert m["mass_share_prefix_fragment"] == pytest.approx(0.6)


def test_aggregate_to_words_sums_pieces():
    pieces = [f"{SP}prae", "destinatione", f"{SP}ad"]
    lex = make_lexicon("sentencepiece", {p: 1 for p in pieces})
    tab = build_piece_table(pieces, lex, specials=set())
    assert aggregate_to_words(np.array([2.0, 3.0, 4.0]), tab).tolist() == [5.0, 4.0]


def test_word_aggregation_moves_the_top_unit_off_the_prefix():
    """A prefix piece outranks every piece, but its word loses to a real word."""
    pieces = [f"{SP}prae", "dest", f"{SP}episcopus", f"{SP}et"]
    lex = make_lexicon("sentencepiece", {p: 1 for p in pieces},
                       words={"et": 500})
    tab = build_piece_table(pieces, lex, specials=set())
    m = pair_side_metrics(np.array([5.0, 0.5, 4.0, 0.1]), tab, top_k=1)
    # Piece level: the winner is the prefix fragment "prae".
    assert m["top5_share_prefix"] == pytest.approx(1.0)
    assert m["top5_share_fragment"] == pytest.approx(1.0)
    # Word level: praedest = 5.5 still wins, but it is now a whole word and the
    # unit reported is the word, not the prefix.
    assert m["top5word_share_prefix"] == pytest.approx(0.0)
    assert m["top5word_share_distinctive"] == pytest.approx(1.0)
    assert m["top5word_mean_chars"] == pytest.approx(len("praedest"))


def test_frequent_words_are_not_distinctive():
    pieces = [f"{SP}et", f"{SP}praedestinatione"]
    lex = make_lexicon("sentencepiece", {p: 1 for p in pieces},
                       words={"et": 900, "praedestinatione": 1}, top_pct=0.5)
    tab = build_piece_table(pieces, lex, specials=set())
    assert tab.word_is_distinctive.tolist() == [False, True]


def test_empty_side_returns_no_metrics():
    lex = make_lexicon("sentencepiece", {f"{SP}ad": 1})
    tab = build_piece_table(["<pad>"], lex, specials={"<pad>"})
    assert pair_side_metrics(np.array([1.0]), tab) == {}


# --------------------------------------------------------------------------
# corpus walk
# --------------------------------------------------------------------------

def test_iter_corpus_files_handles_newlines_in_directory_names(tmp_path):
    weird = tmp_path / "CONC.813.Ar\nles.4"
    weird.mkdir()
    (weird / "a.txt").write_text("de praedestinatione", encoding="utf-8")
    (tmp_path / "plain").mkdir()
    (tmp_path / "plain" / "b.txt").write_text("ad episcopum", encoding="utf-8")
    (tmp_path / "plain" / "c.md").write_text("ignored", encoding="utf-8")

    found = iter_corpus_files(tmp_path)
    assert [p.name for p in found] == ["a.txt", "b.txt"]


# --------------------------------------------------------------------------
# end to end on a synthetic NPZ
# --------------------------------------------------------------------------

class StubTokenizer:
    """Just enough of a HuggingFace tokenizer for analyse_artifact."""

    def __init__(self, vocab: list[str], specials: list[str]):
        self._vocab = vocab
        self.all_special_tokens = specials

    def convert_ids_to_tokens(self, ids):
        return [self._vocab[i] for i in ids]


def _write_synthetic_npz(path: Path) -> None:
    # ids index into the stub vocab below.
    q_ids = np.array([[1, 2, 3, 4, 0]])           # prae|dest|episcopo|ad|<pad>
    c_ids = np.array([[1, 2, 5, 0, 0]])           # prae|dest|canonum|<pad>|<pad>
    np.savez(
        path,
        query_input_ids=q_ids,
        candidate_input_ids=c_ids,
        query_attention_mask=np.array([[1, 1, 1, 1, 0]]),
        candidate_attention_mask=np.array([[1, 1, 1, 0, 0]]),
        query_ig_baseline=np.array([0.8, 0.1, 0.05, 0.05, 0.0], dtype=np.float32),
        candidate_ig_baseline=np.array([0.7, 0.2, 0.1, 0.0, 0.0], dtype=np.float32),
        query_ig_sif_abtt=np.array([0.1, 0.1, 0.7, 0.1, 0.0], dtype=np.float32),
        candidate_ig_sif_abtt=np.array([0.1, 0.1, 0.8, 0.0, 0.0], dtype=np.float32),
        q_mask_retrieval_mark_sif_abtt=np.array([0.9, 0.9, 0.1, 0.1, 0.0],
                                                dtype=np.float32),
        c_mask_retrieval_mark_sif_abtt=np.array([0.9, 0.9, 0.1, 0.0, 0.0],
                                                dtype=np.float32),
    )


def test_analyse_artifact_end_to_end(tmp_path):
    vocab = ["<pad>", f"{SP}prae", "dest", f"{SP}episcopo", f"{SP}ad",
             f"{SP}canonum"]
    tok = StubTokenizer(vocab, ["<pad>"])
    lex = make_lexicon("sentencepiece", {p: 1 for p in vocab[1:]})
    npz = tmp_path / "example001_pair_example.npz"
    _write_synthetic_npz(npz)

    rows = run_analysis.analyse_artifact(npz, tok, lex, {"<pad>"},
                                         ["raw", "sif_abtt"])
    got = {(r["variant"], r["view"], r["side"]): r for r in rows}
    # MaRC exists only for sif_abtt in this fixture; raw must not be invented.
    assert ("raw", "marc", "query") not in got
    assert ("sif_abtt", "marc", "query") in got
    assert ("raw", "ig", "query") in got

    raw_q = got[("raw", "ig", "query")]
    assert raw_q["n_pieces"] == 4          # <pad> dropped by the mask
    assert raw_q["n_words"] == 3           # praedest | episcopo | ad
    # raw puts 0.8 of 1.0 on the prefix piece "prae" and 0.05 on the word "ad".
    assert raw_q["mass_share_prefix"] == pytest.approx(0.85)
    assert raw_q["count_share_prefix"] == pytest.approx(0.5)

    sif_q = got[("sif_abtt", "ig", "query")]
    assert sif_q["mass_share_prefix"] == pytest.approx(0.2, abs=1e-6)
    assert sif_q["top5word_mean_chars"] > 0

    marc_q = got[("sif_abtt", "marc", "query")]
    assert marc_q["mass_share_prefix"] == pytest.approx(
        (0.9 + 0.1) / (0.9 + 0.9 + 0.1 + 0.1), abs=1e-6)


def test_top5_counts_distinct_words_and_prefix_fragments():
    # "prae" and "dest" are two pieces of one word; "ad" is a whole word.
    pieces = [f"{SP}prae", "dest", f"{SP}ad", f"{SP}canonum"]
    lex = make_lexicon("sentencepiece", {p: 1 for p in pieces})
    tab = build_piece_table(pieces, lex, specials=set())
    m = pair_side_metrics(np.array([9.0, 8.0, 7.0, 0.1]), tab, top_k=3)
    assert m["top5_distinct_words"] == pytest.approx(2.0)
    assert m["top5_share_prefix_fragment"] == pytest.approx(1 / 3)
    assert m["top5_share_prefix_wholeword"] == pytest.approx(1 / 3)
    assert m["top5_share_distinctive_unit"] == pytest.approx(0.0)


def test_nonfinite_attribution_is_recorded_not_averaged(tmp_path):
    """The deployed Qwen3-0.6B MaRC masks are all NaN; they must not pollute."""
    vocab = ["<pad>", f"{SP}prae", "dest", f"{SP}episcopo", f"{SP}ad",
             f"{SP}canonum"]
    tok = StubTokenizer(vocab, ["<pad>"])
    lex = make_lexicon("sentencepiece", {p: 1 for p in vocab[1:]})
    npz = tmp_path / "example002_pair_example.npz"
    np.savez(
        npz,
        query_input_ids=np.array([[1, 2, 3]]),
        candidate_input_ids=np.array([[1, 2, 5]]),
        query_attention_mask=np.array([[1, 1, 1]]),
        candidate_attention_mask=np.array([[1, 1, 1]]),
        query_ig_raw_unused=np.zeros(3, dtype=np.float32),
        q_mask_retrieval_mark_baseline=np.array([np.nan, np.nan, 0.0],
                                                dtype=np.float32),
        c_mask_retrieval_mark_baseline=np.array([0.2, 0.3, 0.5],
                                                dtype=np.float32),
    )
    rows = run_analysis.analyse_artifact(npz, tok, lex, {"<pad>"}, ["raw"])
    by_side = {(r["view"], r["side"]): r for r in rows}
    assert by_side[("marc", "query")]["nonfinite"] == 1
    assert "mass_share_prefix" not in by_side[("marc", "query")]
    assert by_side[("marc", "candidate")]["nonfinite"] == 0

    import pandas as pd

    df = pd.DataFrame(rows)
    df["run"] = "deployed"
    df["model_short"] = "Qwen3-0.6B"
    df["model_slug"] = "Qwen_Qwen3-Embedding-0.6B"
    df["stratum"] = "gallery"
    df["layer"] = 23
    out = run_analysis.summarise(df)
    q = out[out["side"] == "query"].iloc[0]
    assert q["n_nonfinite"] == 1
    assert q["n_sides"] == 0
    c = out[out["side"] == "candidate"].iloc[0]
    assert c["n_nonfinite"] == 0
    assert c["n_sides"] == 1


def test_summarise_adds_lift_columns(tmp_path):
    import pandas as pd

    per_pair = pd.DataFrame([
        {"run": "deployed", "model_short": "LaTa", "model_slug": "bowphs_LaTa",
         "variant": "raw", "view": "ig", "side": "query",
         "count_share_prefix": 0.2, "mass_share_prefix": 0.4,
         "count_share_frequent": 0.5, "mass_share_frequent": 0.6},
        {"run": "deployed", "model_short": "LaTa", "model_slug": "bowphs_LaTa",
         "variant": "_corpus", "view": "_corpus", "side": "_corpus",
         "corpus_share_prefix": 0.19, "corpus_share_frequent": 0.55,
         "n_piece_types": 1234},
    ])
    for col in run_analysis.METRIC_COLS:
        if col not in per_pair:
            per_pair[col] = np.nan
    out = run_analysis.summarise(per_pair)
    assert len(out) == 1
    assert out.iloc[0]["lift_prefix"] == pytest.approx(2.0)
    assert out.iloc[0]["lift_frequent"] == pytest.approx(1.2)
    assert out.iloc[0]["corpus_share_prefix"] == pytest.approx(0.19)
    assert out.iloc[0]["n_sides"] == 1
