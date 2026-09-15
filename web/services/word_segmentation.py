"""Group a model's subword pieces into the words of the original text (issue #211).

The webapp renders one highlight per subword piece. Measured over 89,896 pair
sides, 35 to 93 percent of the top five highlighted pieces are fragments of a
longer word, and only 2 to 56 percent are a whole word a reader would quote;
summing each word's pieces raises the whole-word share to 65 to 97 percent
(``docs/research/prefix_attribution_analysis.md``). So the fix is a display
step: group the pieces, sum their attributions, outline the word.

The boundary rules here are a port of ``scripts/ig/prefix_attribution.py``,
which is the module the analysis used, including its two traps: the bare
SentencePiece ``▁`` that closes a word without opening one, and the GPT-2 byte
alphabet that renders ``Quæ`` as ``QuÃ¦`` until it is decoded. Two things are
new, because the analysis had a tokenizer in hand and the serving path does not:

* **Scheme detection from the piece stream.** The artifacts persist
  ``query_token_strings`` produced by ``tokenizer.decode([id])``, one id at a
  time. WordPiece keeps its ``##`` and byte-level BPE keeps a real leading
  space, so both are still segmentable; SentencePiece loses the ``▁`` entirely,
  so a decoded T5 stream carries no boundary evidence at all.
* **Alignment to the original text**, which is what rescues that case and is
  also what issue #211 asks for in as many words ("with the word boundary from
  the original text"). The concatenated pieces differ from the file only in
  case, spacing and the odd special token, so a character-level diff maps each
  piece onto a whitespace-delimited word of the text. Where the text is not
  available the marker rules still run, and a stream with no boundary evidence
  degrades to one word per piece, reported as ``segmentation="pieces"`` rather
  than passed off as words.

Nothing here writes to an artifact: this is a read-time view of vectors the
artifacts already carry.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

# Word-boundary markers by tokenizer family (see prefix_attribution.py).
SP_MARKER = "▁"   # SentencePiece (LaTa, PhilTa, mT5)
BPE_MARKER = "Ġ"  # GPT-2 byte-level BPE (Qwen, KaLM)
BPE_NEWLINE = "Ċ"
WP_CONTINUATION = "##"  # WordPiece (LaBSE)

# Scheme names. "spaced" is decoded byte-level BPE (" XI", " DE"), "plain" is a
# stream with no boundary evidence left in it -- decoded SentencePiece.
SCHEME_WORDPIECE = "wordpiece"
SCHEME_SENTENCEPIECE = "sentencepiece"
SCHEME_BPE = "bpe"
SCHEME_SPACED = "spaced"
SCHEME_PLAIN = "plain"

# How the words in a response were arrived at, reported to the client so a
# degraded segmentation is never displayed as if it were exact.
SEGMENTATION_TEXT = "text"      # aligned to the original file text
SEGMENTATION_MARKERS = "markers"  # from the tokenizer's boundary markers
SEGMENTATION_PIECES = "pieces"  # no boundary evidence; one word per piece

_PUNCT_RE = re.compile(r"[^\w]+", re.UNICODE)
# `[CLS]`, `[SEP]`, `[PAD]`, `</s>`, `<pad>`, `<unk>`, `<|endoftext|>`.
_SPECIAL_RE = re.compile(r"^(\[[A-Za-z]{2,12}\]|<\|?[A-Za-z_/|]{1,20}\|?>)$")


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte-to-unicode map (byte-level BPE alphabet).

    Rebuilt here rather than imported from ``transformers``: the webapp's
    requirements deliberately exclude the model stack, and this is ten lines of
    table building.
    """
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_BYTE_DECODER = {v: k for k, v in _bytes_to_unicode().items()}


def bpe_bytes_to_text(piece: str) -> str:
    """Undo the byte-level BPE alphabet, so ``QuÃ¦`` reads back as ``Quæ``.

    A piece can hold a partial UTF-8 sequence, so decoding one piece is lossy
    by design; an assembled word decodes cleanly.
    """
    try:
        raw = bytearray(_BYTE_DECODER[ch] for ch in piece)
    except KeyError:
        return piece
    return raw.decode("utf-8", errors="replace")


def normalise_word(text: str) -> str:
    """Lowercase, strip punctuation and diacritics; '' when nothing is left."""
    stripped = _PUNCT_RE.sub("", text).lower()
    if not stripped:
        return ""
    decomposed = unicodedata.normalize("NFD", stripped)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def is_special_piece(piece: str) -> bool:
    """Is this one of the tokenizer's special markers rather than text?

    The serving path has no tokenizer and therefore no ``all_special_tokens``,
    so this matches the shapes every model in SLUG_TO_HF uses. It is deliberately
    anchored: a lone ``[`` or ``<`` in a manuscript is not a special token.
    """
    return bool(_SPECIAL_RE.match(piece.strip()))


def detect_scheme(pieces: list[str]) -> str:
    """Pick the boundary convention from the piece strings themselves."""
    has_space = False
    for piece in pieces:
        if piece.startswith(SP_MARKER):
            return SCHEME_SENTENCEPIECE
        if piece.startswith(BPE_MARKER) or piece.startswith(BPE_NEWLINE):
            return SCHEME_BPE
        if piece.startswith(WP_CONTINUATION) and len(piece) > len(WP_CONTINUATION):
            return SCHEME_WORDPIECE
        if piece[:1].isspace():
            has_space = True
    return SCHEME_SPACED if has_space else SCHEME_PLAIN


def piece_core(piece: str, scheme: str) -> str:
    """The piece's surface text with its boundary marker removed."""
    if scheme == SCHEME_SENTENCEPIECE:
        return piece.lstrip(SP_MARKER)
    if scheme == SCHEME_WORDPIECE:
        return piece[len(WP_CONTINUATION):] if piece.startswith(WP_CONTINUATION) else piece
    if scheme == SCHEME_BPE:
        return bpe_bytes_to_text(piece.lstrip(BPE_MARKER).lstrip(BPE_NEWLINE))
    return piece.strip()


def _starts_word(piece: str, scheme: str) -> bool:
    if scheme == SCHEME_SENTENCEPIECE:
        return piece.startswith(SP_MARKER)
    if scheme == SCHEME_WORDPIECE:
        return not piece.startswith(WP_CONTINUATION)
    if scheme == SCHEME_BPE:
        return piece.startswith(BPE_MARKER) or piece.startswith(BPE_NEWLINE)
    if scheme == SCHEME_SPACED:
        return piece[:1].isspace()
    # No evidence either way: every piece opens its own word.
    return True


@dataclass
class WordGroup:
    """One word of the display, and the pieces that make it up."""

    idx: int
    text: str
    piece_indices: list[int] = field(default_factory=list)


@dataclass
class Segmentation:
    """Words, plus how they were arrived at."""

    words: list[WordGroup]
    scheme: str
    method: str  # SEGMENTATION_TEXT | SEGMENTATION_MARKERS | SEGMENTATION_PIECES

    @property
    def word_of_piece(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for word in self.words:
            for pi in word.piece_indices:
                out[pi] = word.idx
        return out

    @property
    def max_pieces_per_word(self) -> int:
        return max((len(w.piece_indices) for w in self.words), default=0)


def segment_by_markers(pieces: list[str], scheme: str | None = None) -> Segmentation:
    """Group pieces using the tokenizer's own boundary convention.

    Ported from ``build_piece_table`` in ``scripts/ig/prefix_attribution.py``,
    including the pending-break rule: a special token, or a piece that is pure
    punctuation or an empty string, closes the current word even though it joins
    no word itself. Without it ``consonante`` + ``r`` + ``▁`` + ``capitulum``
    reads back as one word.
    """
    scheme = scheme or detect_scheme(pieces)
    words: list[WordGroup] = []
    current = -1
    pending_break = False
    for i, piece in enumerate(pieces):
        core = piece_core(piece, scheme)
        if is_special_piece(piece) or not normalise_word(core):
            # Dropped, but still a boundary: specials sit between the two texts
            # and a bare "▁" is the space in front of the next word.
            if is_special_piece(piece) or _starts_word(piece, scheme) or not core.strip():
                pending_break = True
            continue
        if _starts_word(piece, scheme) or pending_break or current < 0:
            words.append(WordGroup(idx=len(words), text=core.strip()))
            current = len(words) - 1
        else:
            words[current].text += core
        pending_break = False
        words[current].piece_indices.append(i)
    method = SEGMENTATION_PIECES if scheme == SCHEME_PLAIN else SEGMENTATION_MARKERS
    return Segmentation(words=words, scheme=scheme, method=method)


def _normalised_stream(items: list[str]) -> tuple[str, list[int]]:
    """Concatenate normalised text, remembering which item each character came from."""
    chars: list[str] = []
    owner: list[int] = []
    for idx, item in enumerate(items):
        for ch in normalise_word(item):
            chars.append(ch)
            owner.append(idx)
    return "".join(chars), owner


def split_text_words(text: str) -> list[str]:
    """The words of the original text: whitespace-delimited, punctuation attached.

    This is what a reader sees, so ``supra.`` is one word rather than two, and
    the frontend can line these up with the words it already renders.
    """
    return [w for w in text.split() if w]


def align_pieces_to_text(
    pieces: list[str],
    text: str,
    scheme: str | None = None,
    min_match_ratio: float = 0.6,
) -> Segmentation | None:
    """Map pieces onto the words of ``text`` with a character-level diff.

    Returns ``None`` when the two streams do not agree well enough to trust the
    mapping -- a wrong text, a different manuscript, an empty file -- so the
    caller can fall back to the marker rules rather than outline arbitrary words.

    The diff, rather than a running character offset, is what absorbs the parts
    that do not correspond: special tokens carry no text, ``[UNK]`` swallows a
    word, and an artifact is truncated at the model's maximum length while the
    file runs on.
    """
    scheme = scheme or detect_scheme(pieces)
    words = split_text_words(text)
    if not words or not pieces:
        return None

    cores = [
        "" if is_special_piece(p) else piece_core(p, scheme)
        for p in pieces
    ]
    piece_stream, piece_owner = _normalised_stream(cores)
    text_stream, text_owner = _normalised_stream(words)
    if not piece_stream or not text_stream:
        return None

    matcher = difflib.SequenceMatcher(None, piece_stream, text_stream, autojunk=False)
    votes: dict[int, Counter] = {}
    matched = 0
    for a0, b0, size in matcher.get_matching_blocks():
        for offset in range(size):
            pi = piece_owner[a0 + offset]
            wi = text_owner[b0 + offset]
            votes.setdefault(pi, Counter())[wi] += 1
            matched += 1

    if matched < min_match_ratio * len(piece_stream):
        return None

    groups = [WordGroup(idx=i, text=w) for i, w in enumerate(words)]
    last_word = -1
    for pi in sorted(votes):
        wi = votes[pi].most_common(1)[0][0]
        # The diff is monotone in aggregate but a majority vote is not, so pin
        # the assignment to be non-decreasing: a piece can never be attributed
        # to a word earlier than the piece before it.
        wi = max(wi, last_word)
        groups[wi].piece_indices.append(pi)
        last_word = wi
    return Segmentation(words=groups, scheme=scheme, method=SEGMENTATION_TEXT)


def segment(
    pieces: list[str],
    text: str | None = None,
    scheme: str | None = None,
) -> Segmentation:
    """Best available word grouping for one side of a pair.

    Prefers the original text, because those are the words the reviewer is
    reading; falls back to the tokenizer's markers, and finally to one word per
    piece for a stream that carries no boundary evidence.
    """
    scheme = scheme or detect_scheme(pieces)
    if text:
        aligned = align_pieces_to_text(pieces, text, scheme=scheme)
        if aligned is not None:
            return aligned
    return segment_by_markers(pieces, scheme=scheme)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

AGGREGATION_SUM = "sum"
AGGREGATION_MAX = "max"


def aggregate_vector(
    values: list[float],
    seg: Segmentation,
    mode: str = AGGREGATION_SUM,
) -> tuple[list[float], list[float], list[float]]:
    """Aggregate a per-piece attribution to per-word.

    Returns ``(score, positive, negative)``, one entry per word.

    Positive and negative are summed separately and always reported, because a
    word whose pieces argue in both directions is a different thing from a word
    nothing lands on, and a net sum cannot tell the two apart. ``score`` is the
    net sum under ``sum`` and the single largest-magnitude piece under ``max``,
    which is the option for a reader who wants "this word contains the strongest
    evidence" rather than "this word carries the most evidence in total".
    """
    pos = [0.0] * len(seg.words)
    neg = [0.0] * len(seg.words)
    best = [0.0] * len(seg.words)
    for word in seg.words:
        for pi in word.piece_indices:
            if pi >= len(values):
                continue
            v = float(values[pi])
            if v >= 0:
                pos[word.idx] += v
            else:
                neg[word.idx] += v
            if abs(v) > abs(best[word.idx]):
                best[word.idx] = v
    if mode == AGGREGATION_MAX:
        score = list(best)
    else:
        score = [p + n for p, n in zip(pos, neg)]
    return score, pos, neg


def _index_map(seg: Segmentation, n_pieces: int) -> tuple[np.ndarray, np.ndarray]:
    """(piece positions, their word ids) for every piece that belongs to a word."""
    rows: list[int] = []
    words: list[int] = []
    for word in seg.words:
        for pi in word.piece_indices:
            if 0 <= pi < n_pieces:
                rows.append(pi)
                words.append(word.idx)
    return np.asarray(rows, dtype=np.int64), np.asarray(words, dtype=np.int64)


def aggregate_matrix(
    matrix,
    row_seg: Segmentation,
    col_seg: Segmentation,
    mode: str = AGGREGATION_SUM,
) -> list[list[float]]:
    """Aggregate a piece x piece matrix to word x word.

    ``sum`` adds the cells of the block, which is the right reading for an
    attribution matrix whose cells are additive contributions to one score.
    ``max`` keeps the largest-magnitude cell, which is the right reading for a
    cosine grid, where adding cells would reward a long word for being long.
    """
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2:
        return []
    n_rows, n_cols = len(row_seg.words), len(col_seg.words)
    out = np.zeros((n_rows, n_cols), dtype=np.float32)
    if n_rows == 0 or n_cols == 0 or arr.size == 0:
        return out.tolist()

    r_pieces, r_words = _index_map(row_seg, arr.shape[0])
    c_pieces, c_words = _index_map(col_seg, arr.shape[1])
    if r_pieces.size == 0 or c_pieces.size == 0:
        return out.tolist()

    block = arr[np.ix_(r_pieces, c_pieces)]
    if mode == AGGREGATION_MAX:
        # Largest magnitude, sign preserved: reduce |v| and read the value back.
        rows_by_word = np.zeros((n_rows, block.shape[1]), dtype=np.float32)
        for wi in range(n_rows):
            sel = block[r_words == wi]
            if sel.size:
                pick = np.argmax(np.abs(sel), axis=0)
                rows_by_word[wi] = sel[pick, np.arange(sel.shape[1])]
        for wj in range(n_cols):
            sel = rows_by_word[:, c_words == wj]
            if sel.size:
                pick = np.argmax(np.abs(sel), axis=1)
                out[:, wj] = sel[np.arange(sel.shape[0]), pick]
    else:
        rows_by_word = np.zeros((n_rows, block.shape[1]), dtype=np.float32)
        np.add.at(rows_by_word, r_words, block)
        cols_by_word = np.zeros((n_rows, n_cols), dtype=np.float32)
        np.add.at(cols_by_word.T, c_words, rows_by_word.T)
        out = cols_by_word
    return out.tolist()
