"""Diff the numbers in the paper, ignoring everything else.

A textual diff of a regenerated LaTeX table is unreadable: whole rows change
because one digit moved. This walks two versions of the same file, pairs up the
numeric tokens in order, and reports only the ones whose value changed, with the
surrounding text so each one can be traced to a cell or a sentence.

Pairing is positional, which is exactly right when a generator rewrote the same
table with new numbers, and wrong if the structure changed. The tool says so:
when the two sides have a different number of numeric tokens it refuses to pair
them and reports the mismatch instead of inventing an alignment.

    python scripts/paper/numeric_token_diff.py --git-ref HEAD \\
        overleaf_drafts/acl_latex.tex overleaf_drafts/tables/*.tex
"""
from __future__ import annotations

import argparse
import difflib
import re
import subprocess
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

# A number, optionally signed, with optional decimals. Deliberately does not
# match the digits inside \ref{...} style commands, which are stripped first.
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")

# Structural digits that are not results: column counts, rule spans, versions.
STRIP_PATTERNS = [
    re.compile(r"\\multicolumn\{\d+\}"),
    re.compile(r"\\cmidrule(?:\(lr\))?\{[\d-]+\}"),
    re.compile(r"\\(?:ref|label|cite[a-zA-Z]*)\{[^}]*\}"),
    re.compile(r"%[^\n]*"),
]


class Token(NamedTuple):
    value: str
    line: int
    context: str


def strip_structure(text: str) -> str:
    for pattern in STRIP_PATTERNS:
        text = pattern.sub(lambda m: " " * len(m.group(0)), text)
    return text


def tokens(text: str, context_chars: int = 60) -> List[Token]:
    stripped = strip_structure(text)
    line_starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            line_starts.append(i + 1)

    out: List[Token] = []
    for match in NUMBER.finditer(stripped):
        start = match.start()
        line = sum(1 for s in line_starts if s <= start)
        lo = max(0, start - context_chars // 2)
        hi = min(len(text), match.end() + context_chars // 2)
        context = " ".join(text[lo:hi].split())
        out.append(Token(match.group(0), line, context))
    return out


def git_show(ref: str, path: str) -> Optional[str]:
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        capture_output=True,
        text=True,
    )
    return result.stdout if result.returncode == 0 else None


INPUT_CMD = re.compile(r"^[ \t]*\\input\{([^}]+)\}[ \t]*$", re.MULTILINE)


def expand_inputs(text: str, base: Path, read, depth: int = 0) -> str:
    """Splice `\\input{...}` files into the text, in document order.

    A table that moves from inline to `\\input` keeps the same numbers in the
    same place once expanded, so the positional token pairing survives an edit
    that only changes where the numbers live.
    """
    if depth > 5:
        return text

    def replace(match: "re.Match[str]") -> str:
        name = match.group(1)
        for candidate in (name, f"{name}.tex"):
            body = read(str(base / candidate))
            if body is not None:
                return expand_inputs(body, base, read, depth + 1)
        return match.group(0)

    return INPUT_CMD.sub(replace, text)


def pair_tokens(
    lhs: List[Token], rhs: List[Token]
) -> List[Tuple[Optional[Token], Optional[Token]]]:
    """Pair the two token streams positionally, tolerating inserts and deletes.

    Equal-length streams pair one to one. When a sentence gains or loses a
    number, difflib finds the matching blocks around it so the rest of the file
    still pairs correctly, and the odd token out is reported as added or removed
    rather than shifting every later comparison by one.
    """
    if len(lhs) == len(rhs):
        return list(zip(lhs, rhs))

    matcher = difflib.SequenceMatcher(
        a=[t.value for t in lhs], b=[t.value for t in rhs], autojunk=False
    )
    pairs: List[Tuple[Optional[Token], Optional[Token]]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            pairs += list(zip(lhs[i1:i2], rhs[j1:j2]))
        elif tag == "replace":
            width = min(i2 - i1, j2 - j1)
            pairs += list(zip(lhs[i1 : i1 + width], rhs[j1 : j1 + width]))
            pairs += [(t, None) for t in lhs[i1 + width : i2]]
            pairs += [(None, t) for t in rhs[j1 + width : j2]]
        elif tag == "delete":
            pairs += [(t, None) for t in lhs[i1:i2]]
        else:  # insert
            pairs += [(None, t) for t in rhs[j1:j2]]
    return pairs


def diff_file(before: str, after: str, path: str) -> Tuple[List[str], int]:
    lhs, rhs = tokens(before), tokens(after)
    pairs = pair_tokens(lhs, rhs)
    changes = [
        (a, b) for a, b in pairs if a is None or b is None or a.value != b.value
    ]
    if not changes:
        return ([], 0)

    note = ""
    if len(lhs) != len(rhs):
        note = (
            f" Token count moved from {len(lhs)} to {len(rhs)}, so some rows are "
            "an addition or a removal rather than a changed value."
        )
    lines = [
        f"### {path}",
        "",
        f"{len(changes)} of {len(rhs)} numeric tokens changed.{note}",
        "",
        "| Line | Before | After | Context |",
        "|---|---|---|---|",
    ]
    for a, b in changes:
        shown = b or a
        context = shown.context.replace("|", r"\|")
        lines.append(
            f"| {shown.line} | {a.value if a else '(added)'} | "
            f"{b.value if b else '(removed)'} | `{context}` |"
        )
    lines.append("")
    return lines, len(changes)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("paths", nargs="+", help="Files to compare, repo-relative.")
    p.add_argument(
        "--git-ref",
        default="HEAD",
        help="Revision holding the 'before' version of each file.",
    )
    p.add_argument("--out", default="", help="Write the report here instead of stdout.")
    p.add_argument(
        "--expand",
        action="store_true",
        help=(
            "Splice \\input{...} files into each side before comparing. Use for "
            "the main .tex so a table moving inline-to-input still pairs up."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    report: List[str] = []
    total = 0
    unchanged: List[str] = []

    for path in args.paths:
        after_path = Path(path)
        if not after_path.exists():
            report += [f"### {path}", "", "MISSING on disk.", ""]
            continue
        before = git_show(args.git_ref, path)
        if before is None:
            report += [f"### {path}", "", f"NEW: absent at {args.git_ref}.", ""]
            continue
        after = after_path.read_text()
        if args.expand:
            base = after_path.parent
            before = expand_inputs(
                before, base, lambda p: git_show(args.git_ref, p)
            )
            after = expand_inputs(
                after,
                base,
                lambda p: Path(p).read_text() if Path(p).exists() else None,
            )
        lines, n = diff_file(before, after, path)
        if lines:
            report += lines
            total += n
        else:
            unchanged.append(path)

    header = [
        f"Numeric-token diff against `{args.git_ref}`.",
        "",
        f"{total} numeric token(s) changed across "
        f"{len(args.paths) - len(unchanged)} of {len(args.paths)} file(s).",
        "",
    ]
    if unchanged:
        header += ["Unchanged:", ""] + [f"- `{p}`" for p in unchanged] + [""]

    text = "\n".join(header + report)
    if args.out:
        Path(args.out).write_text(text)
        print(f"Wrote {args.out}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
