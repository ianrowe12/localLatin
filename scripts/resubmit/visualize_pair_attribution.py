"""Render the paper's integrated-gradient pair matrix and attention figures.

Writes ``fig_pair_matrix_philta`` (``fig:pairmatrix``) and
``fig_attention_philta`` (``fig:attention``) for one retrieval pair, read from
the canonical pair artifact that ``visualize_retrieval_mark.py`` renders the
MaRC masks from, so all three qualitative figures of the attribution appendix
show the same pair (issue #235 item 9). The two figures were first made by the
phase 12e visualiser (``scripts/_archive/run_phase12e_visualize.py``), whose
inputs no longer exist and whose example was a different pair from the MaRC
figure's; this script ports its matrix construction unchanged.

Where the pair comes from: example 11 of the webapp IG gallery
(``runs/active/ig_examples/phase12f_examples.csv``), a same-directory PhilTa
pair (CANT.328.15) at layer 6, PhilTa's most anisotropic layer and the layer of
the attribution mechanism check, not the operational attribution layer. It is a
demonstration pair, picked because ABTT reverses the baseline decision (baseline
predicts different, ABTT same), and it is not one of the 200 positive pairs the
attribution metrics are computed on. The paper's captions say all of this.

Pair matrix: for the ``--max_tokens`` query and candidate positions with the
largest integrated-gradient magnitude under either variant, each cell is the
token-token cosine times the geometric mean of the two tokens' L1-normalised
|IG| scores, signed by the product of the IG signs. The ABTT panel uses token
states with the training mean and the top-D training components removed (the
artifact's ``mean_vec`` and ``pcs``) and the ABTT integrated gradients.

Attention: the encoder self-attention at the artifact's layer, restricted to
the same positions, for the query and for the candidate. ABTT acts after
encoding, so there is one attention map per side.

Both figures are drawn at their printed width (0.94 of the text width, about
5.9 in) with 8 pt tick labels and Type 42 fonts.

    python scripts/resubmit/visualize_pair_attribution.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

SUBWORD_MARKERS = ("▁", "##", "Ġ")
EMPTY_TOKEN_PLACEHOLDER = "_"
TICK_FONTSIZE = 8.3  # 8 pt after the 0.97 print scale
TITLE_FONTSIZE = 9
LABEL_FONTSIZE = 8.5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--artifacts_dir", default="runs/active/ig_examples/artifacts")
    p.add_argument("--model_name", default="bowphs/PhilTa")
    p.add_argument(
        "--example_id", type=int, default=11,
        help=(
            "Gallery pair to render. 11 is the paper's demonstration pair: PhilTa "
            "layer 6, ABTT reverses the baseline decision, outside the 200 "
            "evaluated pairs; fig_retrieval_mark_pair_philta shows the same pair."
        ),
    )
    p.add_argument("--max_tokens", type=int, default=18,
                   help="Query and candidate positions shown per side.")
    p.add_argument("--out_dir", default="overleaf_drafts/figures")
    p.add_argument("--pair_stem", default="fig_pair_matrix_philta")
    p.add_argument("--attention_stem", default="fig_attention_philta")
    return p.parse_args()


def artifact_path(artifacts_dir: Path, model_name: str, example_id: int) -> Path:
    return (artifacts_dir / model_name.replace("/", "_")
            / f"example{example_id:03d}_pair_example.npz")


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-12)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-12)
    return a_norm @ b_norm.T


def clean_tokens(hidden: np.ndarray, pcs: np.ndarray, mean_vec: np.ndarray) -> np.ndarray:
    """ABTT on token states: subtract the training mean, remove the top-D PCs."""
    centered = hidden - mean_vec
    return centered - centered @ pcs.T @ pcs


def select_positions(ig_base: np.ndarray, ig_abtt: np.ndarray, seq_len: int,
                     max_tokens: int) -> np.ndarray:
    importance = np.maximum(np.abs(ig_base[:seq_len]), np.abs(ig_abtt[:seq_len]))
    return np.sort(np.argsort(importance)[-max_tokens:])


def build_pair_matrix(query_hidden: np.ndarray, candidate_hidden: np.ndarray,
                      query_ig: np.ndarray, candidate_ig: np.ndarray) -> np.ndarray:
    # L1-normalise IG per sequence before the outer product so the matrix
    # reflects the attribution pattern rather than its absolute magnitude.
    cos = cosine_matrix(query_hidden, candidate_hidden)
    q_abs, c_abs = np.abs(query_ig), np.abs(candidate_ig)
    q_norm = q_abs / (q_abs.sum() + 1e-12)
    c_norm = c_abs / (c_abs.sum() + 1e-12)
    weight = np.sqrt(q_norm[:, None] * c_norm[None, :])
    sign = np.sign(query_ig)[:, None] * np.sign(candidate_ig)[None, :]
    return cos * weight * sign


def token_label(raw: str) -> str:
    for marker in SUBWORD_MARKERS:
        if raw.startswith(marker):
            raw = raw[len(marker):]
            break
    return raw if raw.strip() else EMPTY_TOKEN_PLACEHOLDER


def decode(tokenizer, input_ids: np.ndarray, positions: np.ndarray) -> list[str]:
    ids = np.asarray(input_ids).reshape(-1)[positions].tolist()
    return [token_label(tokenizer.decode([int(i)])) for i in ids]


def _label_axes(ax, x_labels, y_labels=None) -> None:
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=90, fontsize=TICK_FONTSIZE, fontfamily="monospace")
    if y_labels is None:
        ax.set_yticks(range(ax.get_images()[0].get_array().shape[0]))
        ax.set_yticklabels([])
    else:
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=TICK_FONTSIZE, fontfamily="monospace")
    ax.tick_params(length=1.5, pad=1)


def save(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(out_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_pair_matrix(data: dict, tokenizer, max_tokens: int, out_dir: Path, stem: str) -> None:
    q_len = int(data["query_attention_mask"][0].sum())
    c_len = int(data["candidate_attention_mask"][0].sum())
    q_pos = select_positions(data["query_ig_baseline"], data["query_ig_abtt"], q_len, max_tokens)
    c_pos = select_positions(data["candidate_ig_baseline"], data["candidate_ig_abtt"],
                             c_len, max_tokens)
    q_tokens = decode(tokenizer, data["query_input_ids"], q_pos)
    c_tokens = decode(tokenizer, data["candidate_input_ids"], c_pos)

    q_abtt = clean_tokens(data["query_hidden"], data["pcs"], data["mean_vec"])[q_pos]
    c_abtt = clean_tokens(data["candidate_hidden"], data["pcs"], data["mean_vec"])[c_pos]
    baseline = build_pair_matrix(data["query_hidden"][q_pos], data["candidate_hidden"][c_pos],
                                 data["query_ig_baseline"][q_pos],
                                 data["candidate_ig_baseline"][c_pos])
    abtt = build_pair_matrix(q_abtt, c_abtt, data["query_ig_abtt"][q_pos],
                             data["candidate_ig_abtt"][c_pos])
    vmax = max(float(np.percentile(np.abs(np.concatenate([baseline.ravel(), abtt.ravel()])), 95)),
               1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.4), constrained_layout=True)
    for i, (ax, matrix, title) in enumerate(zip(axes, (baseline, abtt), ("Baseline", "ABTT"))):
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        _label_axes(ax, c_tokens, q_tokens if i == 0 else None)
        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=3)
        ax.set_xlabel("Candidate", fontsize=LABEL_FONTSIZE)
        if i == 0:
            ax.set_ylabel("Query", fontsize=LABEL_FONTSIZE)
    cbar = fig.colorbar(im, ax=list(axes), shrink=0.85, pad=0.02)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    save(fig, out_dir, stem)


def render_attention(data: dict, tokenizer, max_tokens: int, out_dir: Path, stem: str) -> None:
    q_len = int(data["query_attention_mask"][0].sum())
    c_len = int(data["candidate_attention_mask"][0].sum())
    q_pos = select_positions(data["query_ig_baseline"], data["query_ig_abtt"], q_len, max_tokens)
    c_pos = select_positions(data["candidate_ig_baseline"], data["candidate_ig_abtt"],
                             c_len, max_tokens)
    q_tokens = decode(tokenizer, data["query_input_ids"], q_pos)
    c_tokens = decode(tokenizer, data["candidate_input_ids"], c_pos)
    q_attn = data["query_attention"][np.ix_(q_pos, q_pos)]
    c_attn = data["candidate_attention"][np.ix_(c_pos, c_pos)]
    vmax = float(max(q_attn.max(), c_attn.max()))

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.3), constrained_layout=True)
    for ax, matrix, labels, title in zip(
        axes, (q_attn, c_attn), (q_tokens, c_tokens),
        ("Query self-attention", "Candidate self-attention"),
    ):
        im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=vmax, aspect="auto")
        _label_axes(ax, labels, labels)
        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=3)
    cbar = fig.colorbar(im, ax=list(axes), shrink=0.85, pad=0.02)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    save(fig, out_dir, stem)


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    path = artifact_path(Path(args.artifacts_dir), args.model_name, args.example_id)
    if not path.exists():
        raise SystemExit(f"no pair artifact at {path}")
    data = dict(np.load(path))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    out_dir = Path(args.out_dir)
    render_pair_matrix(data, tokenizer, args.max_tokens, out_dir, args.pair_stem)
    render_attention(data, tokenizer, args.max_tokens, out_dir, args.attention_stem)
    layer = int(np.asarray(data["layer"]).reshape(-1)[0])
    print(f"Rendered {args.model_name} example {args.example_id} (layer {layer}) to {out_dir}")


if __name__ == "__main__":
    main()
