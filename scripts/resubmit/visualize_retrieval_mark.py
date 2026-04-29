"""Render the paper figure ``fig_retrieval_mark_pair_philta`` showing the
Retrieval-MarK learned token masks for one query-candidate pair.

Produces a 2x2 (or 4x1 stacked) panel of horizontal heatmap-bars:

    rows    = {mask on query (candidate fixed), mask on candidate (query fixed)}
    columns = {baseline (raw cosine target), ABTT (cleaned cosine target)}

Each panel shows the learned mask value per token position in [0, 1]. The
x-axis tick labels are the decoded subwords. Frozen non-content positions
(mask == 0 by construction) are rendered in a muted gray so the reader can
visually distinguish them from positions that the optimizer drove to zero.

Input contract (NPZ keys consumed):

    query_input_ids              (1, q_len) int64
    candidate_input_ids          (1, c_len) int64
    q_mask_retrieval_mark_baseline  (q_len,) float32
    q_mask_retrieval_mark_abtt      (q_len,) float32
    c_mask_retrieval_mark_baseline  (c_len,) float32
    c_mask_retrieval_mark_abtt      (c_len,) float32

Optional scalar keys (rendered as a top-right annotation if present):

    cos_orig_baseline  float32
    cos_orig_abtt      float32

The script supports two fallback paths for finding the NPZ:

    1. Canonical merged NPZ under --artifacts_dir/<slug>/exampleNNN_pair_example.npz
       (written by the IG artifact merger for a given example/model).

    2. Sidecar NPZ under --sidecar_dir/<slug>/exampleNNN_pair_example.npz
       (written by the retrieval-MarK SLURM job before merging into the
       canonical artifacts). When --sidecar_only is passed, the script reads
       the sidecar NPZ *and* the canonical NPZ (for input_ids / cos_orig)
       and merges them in memory for rendering.

If the primary example NPZ lacks retrieval_mark keys, the script falls back
to --fallback_example_id (typically the LaBSE pilot).

LaTeX inclusion:

    \\begin{figure*}[t]
      \\centering
      \\includegraphics[width=\\textwidth]{figures/fig_retrieval_mark_pair_philta.pdf}
      \\caption{Learned Retrieval-MarK masks for a PhilTa query-candidate pair.
               Each row shows the mask optimized on one side of the pair while
               the partner embedding is held fixed; columns compare the raw
               cosine target (left) with the ABTT-cleaned target (right).
               Brighter cells indicate token positions whose embedding
               contributes more to preserving the original cosine similarity.}
      \\label{fig:retrieval-mark-pair}
    \\end{figure*}
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# ---------------------------------------------------------------------------
# Defaults / style
# ---------------------------------------------------------------------------
MODEL_SHORT = {
    "bowphs/PhilTa": "PhilTa",
    "bowphs/LaTa": "LaTa",
    "google/mt5-base": "mT5-base",
    "sentence-transformers/LaBSE": "LaBSE",
    "Qwen/Qwen3-Embedding-0.6B": "Qwen3-0.6B",
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5": "KaLM-mini",
    "Qwen/Qwen3-Embedding-8B": "Qwen3-8B",
}

RETRIEVAL_MARK_KEYS = (
    "q_mask_retrieval_mark_baseline",
    "q_mask_retrieval_mark_abtt",
    "c_mask_retrieval_mark_baseline",
    "c_mask_retrieval_mark_abtt",
)

SUBWORD_MARKERS = ("\u2581", "##", "\u0120", "_")  # "▁", "##", "Ġ", "_"
EMPTY_TOKEN_PLACEHOLDER = "\u23b5"  # "⎵"

FROZEN_EPS = 1e-6
FROZEN_GRAY = "#c8c8c8"

# Font sizes — mirrored against visualize_resubmit.py conventions
TITLE_FONTSIZE = 11
TICK_FONTSIZE = 8
ANNOT_FONTSIZE = 7
SUPTITLE_FONTSIZE = 13


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Retrieval-MarK pair figure for the paper."
    )
    parser.add_argument(
        "--example_id",
        type=int,
        default=41,
        help="Primary example id to render (default: 41, first PhilTa).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bowphs/PhilTa",
        help="HuggingFace model id used for tokenization (default: bowphs/PhilTa).",
    )
    parser.add_argument(
        "--artifacts_dir",
        type=str,
        default="runs/active/ig_examples/artifacts",
        help="Root directory of canonical (merged) pair NPZs.",
    )
    parser.add_argument(
        "--sidecar_dir",
        type=str,
        default="runs/active/ig_examples/retrieval_mark/artifacts",
        help=(
            "Optional sidecar directory with per-pair retrieval_mark NPZs "
            "(used if the canonical NPZ lacks retrieval_mark keys)."
        ),
    )
    parser.add_argument(
        "--sidecar_only",
        action="store_true",
        help=(
            "Read the retrieval_mark keys from the sidecar NPZ and merge with "
            "the canonical NPZ for input_ids / cosines in memory."
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="overleaf_drafts/figures",
        help="Directory to write the PDF + PNG figure.",
    )
    parser.add_argument(
        "--out_name",
        type=str,
        default="fig_retrieval_mark_pair_philta",
        help="Output filename stem (no extension).",
    )
    parser.add_argument(
        "--layout",
        type=str,
        choices=("2x2", "stacked"),
        default="stacked",
        help="2x2 grid or a 4-row stacked layout (default: stacked).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="PNG DPI (PDF is vector).",
    )
    parser.add_argument(
        "--fallback_example_id",
        type=int,
        default=1,
        help=(
            "If the primary example is unavailable or missing retrieval_mark "
            "keys, fall back to this example id (default: 1, LaBSE pilot)."
        ),
    )
    parser.add_argument(
        "--fallback_model_name",
        type=str,
        default="sentence-transformers/LaBSE",
        help="Model to use for the fallback example (default: LaBSE).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _slug(model_name: str) -> str:
    return model_name.replace("/", "_")


def _canonical_npz_path(artifacts_dir: Path, model_name: str, example_id: int) -> Path:
    return (
        artifacts_dir
        / _slug(model_name)
        / f"example{example_id:03d}_pair_example.npz"
    )


def _sidecar_npz_path(sidecar_dir: Path, model_name: str, example_id: int) -> Path:
    return (
        sidecar_dir
        / _slug(model_name)
        / f"example{example_id:03d}_pair_example.npz"
    )


def _has_retrieval_mark(data: dict) -> bool:
    return all(k in data for k in RETRIEVAL_MARK_KEYS)


def load_merged_data(
    model_name: str,
    example_id: int,
    artifacts_dir: Path,
    sidecar_dir: Optional[Path],
    sidecar_only: bool,
) -> tuple[Optional[dict], Optional[Path], str]:
    """Return (merged_dict, canonical_path, source_description) or (None, None, reason).

    Strategy:
        * If sidecar_only: load canonical (for input_ids/cos_orig) + sidecar
          (for retrieval_mark keys) and merge.
        * Else: load canonical; if it has retrieval_mark keys use it directly;
          otherwise try to patch in from sidecar.
    """
    canonical_path = _canonical_npz_path(artifacts_dir, model_name, example_id)
    sidecar_path = (
        _sidecar_npz_path(sidecar_dir, model_name, example_id)
        if sidecar_dir is not None
        else None
    )

    canonical_data: Optional[dict] = None
    if canonical_path.exists():
        canonical_data = dict(np.load(canonical_path))

    sidecar_data: Optional[dict] = None
    if sidecar_path is not None and sidecar_path.exists():
        sidecar_data = dict(np.load(sidecar_path))

    if sidecar_only:
        if canonical_data is None:
            return None, None, f"canonical NPZ missing: {canonical_path}"
        if sidecar_data is None or not _has_retrieval_mark(sidecar_data):
            return None, None, f"sidecar NPZ missing retrieval_mark keys: {sidecar_path}"
        merged = dict(canonical_data)
        for k in RETRIEVAL_MARK_KEYS:
            merged[k] = sidecar_data[k]
        # Carry cos_orig_* from sidecar if present (more authoritative).
        for k in ("cos_orig_baseline", "cos_orig_abtt"):
            if k in sidecar_data:
                merged[k] = sidecar_data[k]
        return merged, canonical_path, f"sidecar+canonical merged in memory"

    # Default path: try canonical, then patch from sidecar if needed.
    if canonical_data is None:
        return None, None, f"canonical NPZ missing: {canonical_path}"
    if _has_retrieval_mark(canonical_data):
        return canonical_data, canonical_path, "canonical"
    if sidecar_data is not None and _has_retrieval_mark(sidecar_data):
        merged = dict(canonical_data)
        for k in RETRIEVAL_MARK_KEYS:
            merged[k] = sidecar_data[k]
        for k in ("cos_orig_baseline", "cos_orig_abtt"):
            if k in sidecar_data:
                merged[k] = sidecar_data[k]
        return merged, canonical_path, "canonical patched from sidecar"
    return None, canonical_path, "retrieval_mark keys not found in canonical or sidecar"


# ---------------------------------------------------------------------------
# Token decoding
# ---------------------------------------------------------------------------
def _clean_token(raw: str) -> str:
    """Strip common subword markers (▁, ##, Ġ, leading '_') but keep content.

    Returns an em-space placeholder if the decoded token is empty/whitespace.
    """
    s = raw
    for marker in SUBWORD_MARKERS:
        if s.startswith(marker):
            s = s[len(marker):]
            break
    if s.strip() == "":
        return EMPTY_TOKEN_PLACEHOLDER
    return s


def decode_all_tokens(tokenizer, input_ids_1d: np.ndarray) -> list[str]:
    out: list[str] = []
    for tid in input_ids_1d.tolist():
        try:
            raw = tokenizer.decode([int(tid)])
        except Exception:
            raw = f"<{int(tid)}>"
        out.append(_clean_token(raw))
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _render_panel(
    ax: plt.Axes,
    mask_values: np.ndarray,
    token_labels: list[str],
    title: str,
    norm: Normalize,
    cmap,
) -> None:
    """Render one 1xN horizontal heatmap-bar with token labels below."""
    n = mask_values.shape[0]
    frozen = mask_values <= FROZEN_EPS

    # Base image shows mask values with the sequential colormap.
    base = mask_values.reshape(1, n)
    ax.imshow(
        base,
        cmap=cmap,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
    )

    # Over-paint frozen (non-content) cells with a muted gray so the reader
    # can distinguish them from "content but optimizer drove to 0".
    if frozen.any():
        # Build an RGBA overlay: gray where frozen, transparent elsewhere.
        overlay = np.zeros((1, n, 4), dtype=np.float32)
        # matplotlib expects floats in [0,1]. Parse FROZEN_GRAY hex.
        r = int(FROZEN_GRAY[1:3], 16) / 255.0
        g = int(FROZEN_GRAY[3:5], 16) / 255.0
        b = int(FROZEN_GRAY[5:7], 16) / 255.0
        overlay[0, frozen, 0] = r
        overlay[0, frozen, 1] = g
        overlay[0, frozen, 2] = b
        overlay[0, frozen, 3] = 1.0  # fully opaque gray over the frozen cells
        ax.imshow(overlay, aspect="auto", interpolation="nearest")

    ax.set_yticks([])
    ax.set_xticks(np.arange(n))
    # Pick label rotation based on max label length for readability.
    max_len = max((len(t) for t in token_labels), default=1)
    rotation = 90 if max_len > 2 or n > 30 else 0
    ax.set_xticklabels(
        token_labels,
        rotation=rotation,
        fontsize=TICK_FONTSIZE,
        fontfamily="monospace",
    )
    ax.tick_params(axis="x", length=2, pad=1)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=4)

    # Thin light gridlines between tokens for readability.
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.4, alpha=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)


def render_figure(
    data: dict,
    tokenizer,
    out_dir: Path,
    out_name: str,
    layout: str,
    dpi: int,
    example_id: int,
    model_name: str,
) -> tuple[Path, Path, tuple[float, float]]:
    """Render the 2x2 (or stacked) figure and save PDF + PNG."""
    q_ids = np.asarray(data["query_input_ids"]).reshape(-1)
    c_ids = np.asarray(data["candidate_input_ids"]).reshape(-1)

    q_mask_base = np.asarray(data["q_mask_retrieval_mark_baseline"], dtype=np.float32)
    q_mask_abtt = np.asarray(data["q_mask_retrieval_mark_abtt"], dtype=np.float32)
    c_mask_base = np.asarray(data["c_mask_retrieval_mark_baseline"], dtype=np.float32)
    c_mask_abtt = np.asarray(data["c_mask_retrieval_mark_abtt"], dtype=np.float32)

    # Mask arrays may be sidecar-truncated to attention length while
    # input_ids keep the padded length. Align by truncation.
    q_ids = q_ids[: q_mask_base.shape[0]]
    c_ids = c_ids[: c_mask_base.shape[0]]

    q_tokens = decode_all_tokens(tokenizer, q_ids)
    c_tokens = decode_all_tokens(tokenizer, c_ids)

    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.get_cmap("viridis")

    panels = [
        ("Baseline: query mask", q_mask_base, q_tokens),
        ("ABTT: query mask", q_mask_abtt, q_tokens),
        ("Baseline: candidate mask", c_mask_base, c_tokens),
        ("ABTT: candidate mask", c_mask_abtt, c_tokens),
    ]

    if layout == "2x2":
        fig, axes = plt.subplots(2, 2, figsize=(13, 5.2), constrained_layout=False)
        order = [0, 1, 2, 3]  # row-major: row0=query baseline/abtt, row1=cand baseline/abtt
        flat = axes.ravel()
        for idx, ax_i in zip(order, flat):
            title, vals, toks = panels[idx]
            _render_panel(ax_i, vals, toks, title, norm, cmap)
    else:  # stacked: 4 rows x 1 column, full width for token legibility
        max_n = max(q_mask_base.shape[0], c_mask_base.shape[0])
        width = max(10.0, min(18.0, 0.22 * max_n + 4.0))
        fig, axes = plt.subplots(4, 1, figsize=(width, 6.2), constrained_layout=False)
        for ax_i, (title, vals, toks) in zip(axes, panels):
            _render_panel(ax_i, vals, toks, title, norm, cmap)

    model_short = MODEL_SHORT.get(model_name, _slug(model_name).split("_")[-1])
    fig.suptitle(
        f"Retrieval-MarK Attribution: {model_short}, Example {example_id}",
        fontsize=SUPTITLE_FONTSIZE,
        fontweight="bold",
    )

    # Shared colorbar on the right side.
    fig.subplots_adjust(right=0.88, top=0.90, bottom=0.22, hspace=0.95)
    cbar_ax = fig.add_axes([0.905, 0.18, 0.015, 0.70])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("mask value", fontsize=TICK_FONTSIZE + 1)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

    # Cosine annotation in the top-right corner of the whole figure.
    cos_base = data.get("cos_orig_baseline")
    cos_abtt = data.get("cos_orig_abtt")

    def _scalar(val) -> Optional[float]:
        if val is None:
            return None
        arr = np.asarray(val)
        if arr.size == 0:
            return None
        return float(arr.reshape(-1)[0])

    cos_base_s = _scalar(cos_base)
    cos_abtt_s = _scalar(cos_abtt)
    if cos_base_s is not None or cos_abtt_s is not None:
        pieces = []
        if cos_base_s is not None:
            pieces.append(f"cos_baseline = {cos_base_s:.3f}")
        if cos_abtt_s is not None:
            pieces.append(f"cos_abtt = {cos_abtt_s:.3f}")
        fig.text(
            0.885,
            0.965,
            ", ".join(pieces),
            ha="right",
            va="top",
            fontsize=ANNOT_FONTSIZE + 1,
            family="monospace",
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{out_name}.pdf"
    png_path = out_dir / f"{out_name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    size = fig.get_size_inches()
    plt.close(fig)
    return pdf_path, png_path, (float(size[0]), float(size[1]))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _try_load(
    model_name: str,
    example_id: int,
    artifacts_dir: Path,
    sidecar_dir: Path,
    sidecar_only: bool,
) -> tuple[Optional[dict], str]:
    data, _path, reason = load_merged_data(
        model_name=model_name,
        example_id=example_id,
        artifacts_dir=artifacts_dir,
        sidecar_dir=sidecar_dir,
        sidecar_only=sidecar_only,
    )
    return data, reason


def main() -> int:
    args = parse_args()

    # Import tokenizer lazily so --help works without torch/transformers.
    from transformers import AutoTokenizer

    artifacts_dir = Path(args.artifacts_dir)
    sidecar_dir = Path(args.sidecar_dir) if args.sidecar_dir else None
    out_dir = Path(args.out_dir)

    data, reason = _try_load(
        model_name=args.model_name,
        example_id=args.example_id,
        artifacts_dir=artifacts_dir,
        sidecar_dir=sidecar_dir,
        sidecar_only=args.sidecar_only,
    )

    rendered_example_id = args.example_id
    rendered_model_name = args.model_name
    if data is None:
        print(
            f"[WARN] Primary example {args.example_id} "
            f"({args.model_name}) unavailable: {reason}"
        )
        print(
            f"[WARN] Falling back to example {args.fallback_example_id} "
            f"({args.fallback_model_name})."
        )
        data, reason = _try_load(
            model_name=args.fallback_model_name,
            example_id=args.fallback_example_id,
            artifacts_dir=artifacts_dir,
            sidecar_dir=sidecar_dir,
            sidecar_only=args.sidecar_only,
        )
        if data is None:
            print(
                "[ERROR] Fallback example also unavailable: "
                f"{reason}. Nothing to render."
            )
            return 2
        rendered_example_id = args.fallback_example_id
        rendered_model_name = args.fallback_model_name

    print(f"[INFO] Rendering from source: {reason}")
    print(
        f"[INFO] example_id={rendered_example_id} "
        f"model={rendered_model_name} layout={args.layout}"
    )

    tokenizer = AutoTokenizer.from_pretrained(rendered_model_name)

    pdf_path, png_path, (w_in, h_in) = render_figure(
        data=data,
        tokenizer=tokenizer,
        out_dir=out_dir,
        out_name=args.out_name,
        layout=args.layout,
        dpi=args.dpi,
        example_id=rendered_example_id,
        model_name=rendered_model_name,
    )

    print(f"[OK] Wrote {pdf_path} ({w_in:.2f} x {h_in:.2f} in)")
    print(f"[OK] Wrote {png_path} (dpi={args.dpi})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
