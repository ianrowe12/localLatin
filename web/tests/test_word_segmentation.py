"""Word grouping and aggregation for the token maps (issue #211).

Synthetic pieces, one case per tokenizer family, because the rules being tested
are the families' own conventions rather than any one model's vocabulary. The
boundary logic is a port of ``scripts/ig/prefix_attribution.py``, which is what
the analysis in ``docs/research/prefix_attribution_analysis.md`` measured with.
"""

from __future__ import annotations

import numpy as np
import pytest

from web.services import word_segmentation as ws


def words_of(seg: ws.Segmentation) -> list[str]:
    return [w.text for w in seg.words]


def pieces_of(seg: ws.Segmentation) -> list[list[int]]:
    return [list(w.piece_indices) for w in seg.words]


# --- scheme detection -------------------------------------------------------


@pytest.mark.parametrize(
    "pieces,expected",
    [
        (["▁de", "▁prae", "destinatione"], ws.SCHEME_SENTENCEPIECE),
        (["Epi", "##scop", "##us"], ws.SCHEME_WORDPIECE),
        (["ĠDE", "ĠDISP", "ONS"], ws.SCHEME_BPE),
        # Byte-level BPE after `tokenizer.decode([id])`: the marker is a real space.
        ([" DE", " DISP", "ONS"], ws.SCHEME_SPACED),
        # Decoded SentencePiece: the boundary marker is gone entirely.
        (["de", "episcopi", "s"], ws.SCHEME_PLAIN),
    ],
)
def test_scheme_is_read_off_the_piece_stream(pieces, expected) -> None:
    assert ws.detect_scheme(pieces) == expected


# --- one family per case ----------------------------------------------------


def test_sentencepiece_groups_a_prefix_with_its_stem() -> None:
    """`praedestinatione` split as `prae` + `destinatione` is ONE word.

    This is the case from issue #211: the reviewer sees `prae` outlined and
    reads it as the model matching on a Latin prefix, when it is half a stem.
    """
    pieces = ["▁de", "▁prae", "destinatione", "▁scriptum", "</s>"]
    seg = ws.segment_by_markers(pieces)

    assert words_of(seg) == ["de", "praedestinatione", "scriptum"]
    assert pieces_of(seg) == [[0], [1, 2], [3]]
    assert seg.max_pieces_per_word == 2


def test_sentencepiece_bare_marker_closes_the_word() -> None:
    """A lone `▁` is the space in front of the next word, not part of this one.

    Without the pending break, `consonante` + `r` + `▁` + `capitulum` reads back
    as one impossible word (the trap called out in prefix_attribution.py).
    """
    seg = ws.segment_by_markers(["▁consonante", "r", "▁", "capitulum"])

    assert words_of(seg) == ["consonanter", "capitulum"]


def test_wordpiece_continuations_join_their_head() -> None:
    seg = ws.segment_by_markers(["[CLS]", "Epi", "##scop", "##us", "aut", "[SEP]"])

    assert words_of(seg) == ["Episcopus", "aut"]
    assert pieces_of(seg) == [[1, 2, 3], [4]]


def test_byte_level_bpe_marker_and_mojibake() -> None:
    """`Ġ` opens a word, and the GPT-2 byte alphabet is decoded back to text."""
    seg = ws.segment_by_markers(["ĠQu", "Ã¦", "Ġsit", "Ċ", "Ġfinis"])

    assert words_of(seg) == ["Quæ", "sit", "finis"]
    assert pieces_of(seg) == [[0, 1], [2], [4]]


def test_decoded_byte_level_bpe_uses_the_leading_space() -> None:
    """What the artifacts actually store for Qwen and KaLM."""
    seg = ws.segment_by_markers(["   ", " Cap", ".", " XX", "III", "."])

    assert words_of(seg) == ["Cap", "XXIII"]
    assert pieces_of(seg) == [[1], [3, 4]]


def test_punctuation_pieces_join_no_word_and_break_the_run() -> None:
    seg = ws.segment_by_markers(["▁supra", ".", "▁capitulum"])

    assert words_of(seg) == ["supra", "capitulum"]
    assert pieces_of(seg) == [[0], [2]]


def test_a_stream_with_no_boundary_evidence_is_reported_as_pieces() -> None:
    """Decoded SentencePiece: one word per piece, and the response says so."""
    seg = ws.segment_by_markers(["de", "episcopi", "s"])

    assert seg.method == ws.SEGMENTATION_PIECES
    assert words_of(seg) == ["de", "episcopi", "s"]


# --- alignment to the original text ----------------------------------------


def test_alignment_recovers_the_words_a_reader_sees() -> None:
    """Decoded T5 pieces carry no marker, so the file's own words supply them."""
    pieces = ["de", "prae", "destinatione", "capitulum", "xii"]
    seg = ws.segment(pieces, text="De praedestinatione. Capitulum XII")

    assert seg.method == ws.SEGMENTATION_TEXT
    # Punctuation stays attached, because that is how the word is on the page.
    assert words_of(seg) == ["De", "praedestinatione.", "Capitulum", "XII"]
    assert pieces_of(seg) == [[0], [1, 2], [3], [4]]


def test_alignment_survives_truncation_and_specials() -> None:
    """The artifact stops at the model's maximum length; the file runs on."""
    pieces = ["[CLS]", "Epi", "##scop", "##us", "aut"]
    seg = ws.segment(pieces, text="Episcopus aut presbiter aut diaconus")

    assert seg.method == ws.SEGMENTATION_TEXT
    assert words_of(seg)[:2] == ["Episcopus", "aut"]
    assert pieces_of(seg)[0] == [1, 2, 3]
    # Words past the truncation are present and simply carry no pieces.
    assert pieces_of(seg)[2:] == [[], [], []]


def test_alignment_refuses_a_text_that_is_not_this_pair() -> None:
    """A wrong text must fall back to the markers, not outline arbitrary words."""
    pieces = ["▁de", "▁prae", "destinatione"]
    seg = ws.segment(pieces, text="Nihil omnino simile huic fragmento inest")

    assert seg.method == ws.SEGMENTATION_MARKERS
    assert words_of(seg) == ["de", "praedestinatione"]


# --- aggregation ------------------------------------------------------------


def test_sum_aggregation_reports_positive_and_negative_separately() -> None:
    seg = ws.segment_by_markers(["▁prae", "destinatione", "▁est"])
    score, pos, neg = ws.aggregate_vector([0.4, -0.1, 0.2], seg)

    assert score == pytest.approx([0.3, 0.2])
    assert pos == pytest.approx([0.4, 0.2])
    assert neg == pytest.approx([-0.1, 0.0])


def test_max_aggregation_keeps_the_strongest_piece_with_its_sign() -> None:
    seg = ws.segment_by_markers(["▁prae", "destinatione"])
    score, _pos, _neg = ws.aggregate_vector([0.4, -0.9], seg, mode=ws.AGGREGATION_MAX)

    assert score == pytest.approx([-0.9])


def test_matrix_aggregation_sums_the_block() -> None:
    q = ws.segment_by_markers(["▁prae", "destinatione", "▁est"])
    c = ws.segment_by_markers(["▁qui", "▁di", "u"])
    matrix = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=np.float32,
    )
    # Query words: {prae+destinatione}, {est}; candidate words: {qui}, {di+u}.
    out = ws.aggregate_matrix(matrix, q, c)

    assert np.allclose(out, [[5.0, 16.0], [7.0, 17.0]])


def test_matrix_aggregation_max_keeps_the_largest_magnitude_cell() -> None:
    q = ws.segment_by_markers(["▁prae", "destinatione"])
    c = ws.segment_by_markers(["▁qui", "▁di", "u"])
    matrix = np.array([[0.1, -0.9, 0.2], [0.3, 0.4, 0.5]], dtype=np.float32)
    out = ws.aggregate_matrix(matrix, q, c, mode=ws.AGGREGATION_MAX)

    assert np.allclose(out, [[0.3, -0.9]], atol=1e-6)
