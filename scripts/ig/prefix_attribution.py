"""Shared logic for the prefix/frequency attribution analysis (issue #211).

Prof. Firey's observation is that the webapp highlight lands on Latin prefixes
(prae, sub, pro, per, ad, ab) rather than on the distinctive words a reader uses
to confirm a match.  The models tokenise words into pieces, so the highlight is
per piece.  This module turns that observation into numbers:

* it classifies every subword piece of a tokenizer as a *prefix piece*, a
  *high-frequency piece* (top 1 percent by corpus count), a *whole word* or a
  *word fragment*;
* it aggregates per-piece attributions to whole words using the tokenizer's own
  word-boundary convention;
* it reports attribution mass shares against the piece-count baseline, so
  "disproportionate" means "above the share of pieces that are prefixes".

The heavy lifting (walking artifacts, writing CSVs) lives in
``run_prefix_attribution_analysis.py``; everything testable lives here.
"""

from __future__ import annotations

import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Common Latin prefixes named in issue #211 (prae, sub, pro, per, ad, ab) plus
# the rest of the productive set a Latin reader would recognise.  Several of
# these (ad, ab, de, ex, in, per, pro, sub) are also standalone prepositions,
# which is why the report separates "prefix piece used as a whole word" from
# "prefix piece used as a word-initial fragment".
LATIN_PREFIXES: tuple[str, ...] = (
    "prae", "pre", "sub", "pro", "per", "ad", "ab", "con", "com",
    "de", "ex", "in", "re", "dis", "trans",
)

# Word-boundary markers by tokenizer family.
SP_MARKER = "▁"   # SentencePiece (T5: LaTa, PhilTa, mT5)
BPE_MARKER = "Ġ"  # GPT-2 byte-level BPE (Qwen, KaLM)
BPE_NEWLINE = "Ċ"
WP_CONTINUATION = "##"  # WordPiece (LaBSE)

_PUNCT_RE = re.compile(r"[^\w]+", re.UNICODE)
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte-to-unicode map (byte-level BPE alphabet).

    Reimplemented here rather than imported from ``transformers``: CI installs
    only the webapp requirements, and this is ten lines of table building.
    """
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\u00a1"), ord("\u00ac") + 1))
          + list(range(ord("\u00ae"), ord("\u00ff") + 1)))
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
    """Undo the byte-level BPE alphabet, so "QuÃ¦" reads back as "Quae"-like text.

    A piece can hold a partial UTF-8 sequence, so decoding is lossy by design;
    assembled words decode cleanly.
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


@dataclass(frozen=True)
class TokenizerScheme:
    """How one tokenizer family marks word boundaries."""

    name: str  # "sentencepiece" | "wordpiece" | "bpe"

    def starts_word(self, piece: str) -> bool:
        if self.name == "sentencepiece":
            return piece.startswith(SP_MARKER)
        if self.name == "wordpiece":
            return not piece.startswith(WP_CONTINUATION)
        # byte-level BPE: a leading space/newline marker opens a new word.
        return piece.startswith(BPE_MARKER) or piece.startswith(BPE_NEWLINE)

    def core(self, piece: str) -> str:
        """The piece's surface text with the boundary marker removed."""
        if self.name == "sentencepiece":
            return piece.lstrip(SP_MARKER)
        if self.name == "wordpiece":
            return piece[len(WP_CONTINUATION):] if piece.startswith(WP_CONTINUATION) else piece
        return piece.lstrip(BPE_MARKER).lstrip(BPE_NEWLINE)

    def to_text(self, assembled: str) -> str:
        """Surface text of an assembled word (byte-level BPE needs decoding)."""
        return bpe_bytes_to_text(assembled) if self.name == "bpe" else assembled


def detect_scheme(tokenizer) -> TokenizerScheme:
    """Pick the boundary convention from a HuggingFace tokenizer instance."""
    cls = type(tokenizer).__name__.lower()
    if "bert" in cls and "albert" not in cls:
        return TokenizerScheme("wordpiece")
    if "t5" in cls or "albert" in cls or "xlmroberta" in cls:
        return TokenizerScheme("sentencepiece")
    return TokenizerScheme("bpe")


@dataclass
class PieceLexicon:
    """Per-model classification of subword pieces, fitted on the corpus.

    ``piece_counts`` are raw corpus counts keyed by the tokenizer's piece
    string (with its boundary marker).  ``top_frequent`` is the top 1 percent of
    *observed piece types* by count; ``word_counts`` backs the "distinctive
    word" test used for the top-5 tables.
    """

    scheme: TokenizerScheme
    piece_counts: Counter = field(default_factory=Counter)
    word_counts: Counter = field(default_factory=Counter)
    top_pct: float = 0.01
    top_frequent: frozenset[str] = frozenset()
    frequent_words: frozenset[str] = frozenset()

    def finalise(self) -> "PieceLexicon":
        n_types = len(self.piece_counts)
        k = max(1, int(round(n_types * self.top_pct))) if n_types else 0
        self.top_frequent = frozenset(p for p, _ in self.piece_counts.most_common(k))
        n_words = len(self.word_counts)
        kw = max(1, int(round(n_words * self.top_pct))) if n_words else 0
        self.frequent_words = frozenset(w for w, _ in self.word_counts.most_common(kw))
        return self

    # -- piece-level predicates -------------------------------------------
    def is_prefix_piece(self, piece: str) -> bool:
        core = self.scheme.to_text(self.scheme.core(piece))
        return normalise_word(core) in LATIN_PREFIXES

    def is_frequent_piece(self, piece: str) -> bool:
        return piece in self.top_frequent

    def is_distinctive_word(self, word: str) -> bool:
        w = normalise_word(word)
        return bool(w) and w not in self.frequent_words and w not in LATIN_PREFIXES

    # -- corpus baselines --------------------------------------------------
    def corpus_share_prefix(self) -> float:
        total = sum(self.piece_counts.values())
        if not total:
            return 0.0
        hit = sum(c for p, c in self.piece_counts.items() if self.is_prefix_piece(p))
        return hit / total

    def corpus_share_frequent(self) -> float:
        total = sum(self.piece_counts.values())
        if not total:
            return 0.0
        return sum(self.piece_counts[p] for p in self.top_frequent) / total


def iter_corpus_files(root: Path) -> list[Path]:
    """Null-safe walk of the labelled corpus.

    ``data/canon_labelled`` has directory names that contain newlines, so the
    walk never goes through a shell or a newline-delimited listing.
    """
    out: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".txt"):
                out.append(Path(dirpath) / fn)
    return sorted(out)


def build_lexicon(tokenizer, corpus_root: Path, top_pct: float = 0.01,
                  max_files: int | None = None) -> PieceLexicon:
    scheme = detect_scheme(tokenizer)
    lex = PieceLexicon(scheme=scheme, top_pct=top_pct)
    files = iter_corpus_files(corpus_root)
    if max_files is not None:
        files = files[:max_files]
    specials = set(tokenizer.all_special_tokens)
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        ids = tokenizer(text, add_special_tokens=False,
                        truncation=False)["input_ids"]
        for piece in tokenizer.convert_ids_to_tokens(ids):
            if piece in specials:
                continue
            if not normalise_word(scheme.core(piece)):
                continue
            lex.piece_counts[piece] += 1
        for word in _WORD_RE.findall(text):
            lex.word_counts[normalise_word(word)] += 1
    lex.word_counts.pop("", None)
    return lex.finalise()


@dataclass
class PieceTable:
    """One side of one pair: pieces, their word grouping, and their flags."""

    pieces: list[str]
    keep: np.ndarray        # bool, False for specials / empty pieces
    word_id: np.ndarray     # int, -1 for dropped pieces
    words: list[str]        # surface form per word id
    is_prefix: np.ndarray
    is_frequent: np.ndarray
    is_whole_word: np.ndarray   # this piece alone spans its whole word
    word_is_prefix: np.ndarray  # per word
    word_is_distinctive: np.ndarray


def build_piece_table(pieces: list[str], lex: PieceLexicon,
                      specials: set[str]) -> PieceTable:
    scheme = lex.scheme
    n = len(pieces)
    keep = np.zeros(n, dtype=bool)
    word_id = np.full(n, -1, dtype=np.int64)
    words: list[str] = []
    current = -1
    # A dropped piece can still be a word boundary. SentencePiece emits a bare
    # "\u2581" for the space in front of a word whose first character is also a
    # separate piece, and specials sit between the two texts; without the
    # pending break the next piece would glue onto the previous word and the
    # table would report "concordantiumepiscoporum" as one word.
    pending_break = False
    for i, piece in enumerate(pieces):
        if piece in specials or not normalise_word(scheme.core(piece)):
            if piece in specials or scheme.starts_word(piece):
                pending_break = True
            continue
        keep[i] = True
        if scheme.starts_word(piece) or pending_break or current < 0:
            words.append(scheme.core(piece))
            current = len(words) - 1
        else:
            words[current] += scheme.core(piece)
        pending_break = False
        word_id[i] = current

    words = [scheme.to_text(w) for w in words]
    sizes = Counter(word_id[keep].tolist())
    is_prefix = np.array([keep[i] and lex.is_prefix_piece(p)
                          for i, p in enumerate(pieces)], dtype=bool)
    is_frequent = np.array([keep[i] and lex.is_frequent_piece(p)
                            for i, p in enumerate(pieces)], dtype=bool)
    is_whole = np.array([keep[i] and sizes[int(word_id[i])] == 1
                         for i in range(n)], dtype=bool)
    word_is_prefix = np.array([normalise_word(w) in LATIN_PREFIXES for w in words],
                              dtype=bool)
    word_is_distinctive = np.array([lex.is_distinctive_word(w) for w in words],
                                   dtype=bool)
    return PieceTable(pieces=pieces, keep=keep, word_id=word_id, words=words,
                      is_prefix=is_prefix, is_frequent=is_frequent,
                      is_whole_word=is_whole, word_is_prefix=word_is_prefix,
                      word_is_distinctive=word_is_distinctive)


def _share(mass: np.ndarray, flag: np.ndarray) -> float:
    total = float(mass.sum())
    return float(mass[flag].sum()) / total if total > 0 else float("nan")


def aggregate_to_words(attr: np.ndarray, table: PieceTable) -> np.ndarray:
    """Sum piece attributions within each word."""
    out = np.zeros(len(table.words), dtype=np.float64)
    if len(table.words):
        np.add.at(out, table.word_id[table.keep], attr[table.keep])
    return out


def pair_side_metrics(attr: np.ndarray, table: PieceTable, top_k: int = 5) -> dict:
    """Attribution shares for one side of one pair.

    ``attr`` is the raw per-token attribution.  Mass shares use the positive
    part (a negative IG score argues against the match, so folding it into the
    denominator would make the shares uninterpretable).  Top-k uses ``|attr|``,
    which is what ``web/services/token_map_svc.py`` ranks auto-highlights by.
    """
    keep = table.keep
    n_keep = int(keep.sum())
    if n_keep == 0:
        return {}
    pos = np.clip(attr, 0.0, None)[keep]
    idx = np.flatnonzero(keep)
    flags = {
        "prefix": table.is_prefix[keep],
        "frequent": table.is_frequent[keep],
        "whole_word": table.is_whole_word[keep],
        "fragment": ~table.is_whole_word[keep],
    }
    # Most "prefix pieces" in this corpus are standalone prepositions (in, de,
    # ad, ex, per, ab), not word-internal prefixes. Split them, because only
    # the second kind is the thing issue #211 describes.
    flags["prefix_fragment"] = flags["prefix"] & flags["fragment"]
    flags["prefix_wholeword"] = flags["prefix"] & flags["whole_word"]
    out = {
        "n_pieces": n_keep,
        "n_words": len(table.words),
        "count_share_prefix": float(flags["prefix"].mean()),
        "count_share_frequent": float(flags["frequent"].mean()),
        "count_share_fragment": float(flags["fragment"].mean()),
        "count_share_prefix_fragment": float(flags["prefix_fragment"].mean()),
        "count_share_prefix_wholeword": float(flags["prefix_wholeword"].mean()),
        "mass_share_prefix": _share(pos, flags["prefix"]),
        "mass_share_prefix_fragment": _share(pos, flags["prefix_fragment"]),
        "mass_share_prefix_wholeword": _share(pos, flags["prefix_wholeword"]),
        "mass_share_frequent": _share(pos, flags["frequent"]),
        "mass_share_whole_word": _share(pos, flags["whole_word"]),
        "mass_share_fragment": _share(pos, flags["fragment"]),
    }

    k = min(top_k, n_keep)
    order = idx[np.argsort(-np.abs(attr[keep]), kind="stable")][:k]
    top_words = table.word_id[order]
    frag = ~table.is_whole_word[order]
    out["top5_share_prefix"] = float(table.is_prefix[order].mean())
    out["top5_share_frequent"] = float(table.is_frequent[order].mean())
    out["top5_share_fragment"] = float(frag.mean())
    # The exact thing issue #211 describes: a prefix highlighted as a fragment
    # of a longer word ("prae" inside "praedestinatione").
    out["top5_share_prefix_fragment"] = float(
        (table.is_prefix[order] & frag).mean())
    out["top5_share_prefix_wholeword"] = float(
        (table.is_prefix[order] & ~frag).mean())
    # Is the highlighted unit itself a word a reader would quote?
    out["top5_share_distinctive_unit"] = float(
        (~frag & table.word_is_distinctive[top_words]).mean())
    # Parent word of the highlighted piece, distinctive or not.
    out["top5_share_distinctive_word"] = float(
        table.word_is_distinctive[top_words].mean())
    # How many different words the k highlight slots actually cover: pieces of
    # one long word can occupy several slots.
    out["top5_distinct_words"] = float(len(set(top_words.tolist())))
    out["top5_mean_word_chars"] = float(
        np.mean([len(table.words[w]) for w in top_words]))

    # Word level: same questions after summing pieces within a word.
    word_attr = aggregate_to_words(attr, table)
    wpos = np.clip(word_attr, 0.0, None)
    out["word_count_share_prefix"] = float(table.word_is_prefix.mean())
    out["word_mass_share_prefix"] = _share(wpos, table.word_is_prefix)
    out["word_mass_share_distinctive"] = _share(wpos, table.word_is_distinctive)
    kw = min(top_k, len(table.words))
    worder = np.argsort(-np.abs(word_attr), kind="stable")[:kw]
    out["top5word_share_prefix"] = float(table.word_is_prefix[worder].mean())
    out["top5word_share_distinctive"] = float(table.word_is_distinctive[worder].mean())
    out["top5word_mean_chars"] = float(np.mean([len(table.words[w]) for w in worder]))
    return out
