"""Leiden demo: curated 6-method comparison examples with HTML report.

Selects diverse example pairs from phase12f artifacts and generates:
- 6-panel comparison figures (PNG + PDF)
- BERTScore and OT detail figures
- Self-contained HTML summary report with embedded images
- Metrics CSV
"""
from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

# Import rendering and metric functions from the comparison script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_resubmit_ig_comparison import (  # noqa: E402
    load_artifact,
    model_slug,
    render_comparison,
    render_bertscore_detail,
    render_ot_detail,
    METHODS,
    METHOD_LABELS,
    METRIC_SUFFIXES,
    METRIC_LABELS,
)

# 10 examples: one per model per bucket (4 models × 4 buckets = 16; pick 10)
# IDs assume 4 models sorted: LaBSE, LaTa, PhilTa, Qwen3-0.6B; 5 per bucket per model
# correct_similar: LaBSE(1), Qwen(16); correct_not_similar: LaTa(26), PhilTa(31)
# wrong_similar: LaBSE(41), LaTa(46), PhilTa(53), Qwen(56)
# wrong_not_similar: LaBSE(62), Qwen(76)
DEFAULT_IDS = [1, 16, 26, 31, 41, 46, 53, 56, 62, 76]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Leiden demo report.")
    p.add_argument(
        "--examples_csv", required=True,
        help="Path to phase12f_examples.csv",
    )
    p.add_argument(
        "--artifacts_dir", required=True,
        help="Directory containing model-slug NPZ subdirectories",
    )
    p.add_argument(
        "--out_dir", default="runs/active/resubmit/leiden_demo",
        help="Output directory (default: runs/active/resubmit/leiden_demo)",
    )
    p.add_argument("--max_tokens", type=int, default=20)
    p.add_argument(
        "--example_ids",
        default=",".join(str(i) for i in DEFAULT_IDS),
        help="Comma-separated example IDs to include (defaults assume 4-model CSV)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# HTML report builder
# ---------------------------------------------------------------------------

METHOD_DESCRIPTIONS = {
    "ig_abtt": "Integrated Gradients with ABTT cleaning — our primary method. "
               "Positive (blue) = both tokens contribute to similarity; "
               "negative (red) = opposition.",
    "bertscore": "BERTScore greedy alignment — each query token matched to its "
                 "most similar candidate token (and vice versa). No gradient signal.",
    "ot_emd": "Optimal Transport (Earth Mover's Distance) — distributes IG-weighted "
              "mass from query to candidate using cosine cost.",
    "attn_crosssim": "Attention-weighted cross-similarity (Ditto-inspired) — "
                     "self-attention diagonal as token importance weight.",
    "dla": "Direct Logit Attribution — norm-weighted cosine alignment. "
           "No gradients needed; single-pass geometric decomposition.",
    "attn_standalone": "Standalone Attention Score — column-mean of attention matrix "
                       "as importance (tokens that many other tokens attend to).",
}

CSS = """
body { font-family: 'Segoe UI', Helvetica, Arial, sans-serif; max-width: 1400px;
       margin: 0 auto; padding: 20px; background: #fafafa; color: #333; }
h1 { border-bottom: 2px solid #2c3e50; padding-bottom: 8px; }
.example { background: #fff; border: 1px solid #ddd; border-radius: 6px;
           padding: 20px; margin: 24px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.example h2 { margin-top: 0; color: #2c3e50; }
table.meta { border-collapse: collapse; margin-bottom: 12px; }
table.meta td { padding: 3px 12px 3px 0; font-size: 0.9em; }
table.meta td:first-child { font-weight: bold; color: #555; }
table.metrics { border-collapse: collapse; width: 100%; margin-top: 12px; font-size: 0.85em; }
table.metrics th, table.metrics td { border: 1px solid #ddd; padding: 4px 8px; text-align: right; }
table.metrics th { background: #f0f0f0; text-align: center; }
table.metrics td:first-child { text-align: left; font-weight: bold; }
img.comparison { width: 100%; margin: 8px 0; border: 1px solid #eee; }
.legend { background: #f8f8f0; border: 1px solid #e0e0d0; border-radius: 4px;
          padding: 12px 16px; margin: 16px 0; font-size: 0.85em; }
.legend h3 { margin-top: 0; }
.legend dt { font-weight: bold; color: #2c3e50; }
.legend dd { margin: 0 0 6px 0; color: #555; }
.bucket-tag { display: inline-block; padding: 2px 8px; border-radius: 3px;
              font-size: 0.8em; font-weight: bold; color: #fff; }
.correct_similar { background: #27ae60; }
.correct_not_similar { background: #2980b9; }
.wrong_similar { background: #e67e22; }
.wrong_not_similar { background: #c0392b; }
"""


def _embed_png(path: Path) -> str:
    """Read PNG and return base64 data-URI."""
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def _bucket_tag(bucket: str) -> str:
    label = bucket.replace("_", " ").title()
    css_class = bucket.replace(" ", "_")
    return f'<span class="bucket-tag {css_class}">{label}</span>'


def _metrics_table(metrics: dict) -> str:
    """Build an HTML table row for each method's 4 metrics."""
    header = "<tr><th>Method</th>"
    for s in METRIC_SUFFIXES:
        header += f"<th>{METRIC_LABELS[s]}</th>"
    header += "</tr>"

    rows = []
    for m in METHODS:
        cells = f"<td>{METHOD_LABELS[m]}</td>"
        for s in METRIC_SUFFIXES:
            key = f"{m}_{s}"
            val = metrics.get(key)
            cells += f"<td>{val:.3f}</td>" if val is not None else "<td>--</td>"
        rows.append(f"<tr>{cells}</tr>")
    return f'<table class="metrics">{header}{"".join(rows)}</table>'


def _method_legend() -> str:
    items = "".join(
        f"<dt>Panel {chr(65 + i)}: {METHOD_LABELS[m]}</dt><dd>{METHOD_DESCRIPTIONS[m]}</dd>"
        for i, m in enumerate(METHODS)
    )
    return f'<div class="legend"><h3>Method Legend</h3><dl>{items}</dl></div>'


def build_html_report(
    examples_df: pd.DataFrame,
    metrics_list: list[dict],
    fig_dir: Path,
    out_path: Path,
) -> None:
    """Build a self-contained HTML report with embedded images."""
    blocks = []
    for metrics in metrics_list:
        eid = metrics["example_id"]
        row = examples_df[examples_df["example_id"] == eid].iloc[0]
        slug = model_slug(row["model_name"])
        tag = f"example{eid:03d}_{row['candidate_role']}"
        fig_path = fig_dir / f"{slug}_{tag}_comparison.png"

        img_src = _embed_png(fig_path) if fig_path.exists() else ""
        bucket = row.get("bucket", "unknown")
        gold = "Similar" if row.get("gold_similar", -1) == 1 else "Not similar"

        blocks.append(f"""
<div class="example">
  <h2>Example {eid}: {row.get('model_short', row['model_name'])} &mdash; {_bucket_tag(bucket)}</h2>
  <table class="meta">
    <tr><td>Model</td><td>{row['model_name']}</td></tr>
    <tr><td>Layer</td><td>{int(row['layer'])}</td></tr>
    <tr><td>Gold label</td><td>{gold}</td></tr>
    <tr><td>ABTT score</td><td>{row.get('abtt_score', 'N/A'):.4f}</td></tr>
    <tr><td>ABTT prediction</td><td>{int(row.get('abtt_pred', -1))}</td></tr>
    <tr><td>Query folder</td><td><code>{row.get('query_folder_id', '')}</code></td></tr>
    <tr><td>Candidate folder</td><td><code>{row.get('candidate_folder_id', '')}</code></td></tr>
  </table>
  <img class="comparison" src="{img_src}" alt="Example {eid} comparison"/>
  {_metrics_table(metrics)}
</div>""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Leiden Demo &mdash; Phase 12f IG Method Comparison</title>
<style>{CSS}</style>
</head>
<body>
<h1>Phase 12f: 6-Method Attribution Comparison</h1>
<p>Curated examples for Leiden collaborator meeting.
   {len(metrics_list)} examples from {examples_df['model_name'].nunique()} models,
   covering {examples_df['bucket'].nunique()} outcome buckets.</p>
<p><b>Red rectangles</b> on each panel mark the top-5 token pairs by absolute weight.</p>
{_method_legend()}
{"".join(blocks)}
</body>
</html>"""

    out_path.write_text(html)
    print(f"HTML report: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    selected_ids = [int(x) for x in args.example_ids.split(",")]

    examples = pd.read_csv(args.examples_csv)
    examples = examples[examples["example_id"].isin(selected_ids)].copy()
    if examples.empty:
        print(f"No examples matched IDs {selected_ids}")
        return

    artifacts_dir = Path(args.artifacts_dir)
    out_dir = Path(args.out_dir)
    comp_dir = out_dir / "comparisons"
    detail_dir = out_dir / "details"
    comp_dir.mkdir(parents=True, exist_ok=True)
    detail_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tokenizers: dict[str, object] = {}
    all_metrics: list[dict] = []

    for _, row in examples.iterrows():
        eid = int(row["example_id"])
        slug = model_slug(row["model_name"])
        artifact_path = (
            artifacts_dir / slug
            / f"example{eid:03d}_{row['candidate_role']}.npz"
        )
        if not artifact_path.exists():
            print(f"[SKIP] Artifact missing: {artifact_path}")
            continue

        model_name = row["model_name"]
        if model_name not in tokenizers:
            tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)

        data = load_artifact(artifacts_dir, row)
        tag = f"example{eid:03d}_{row['candidate_role']}"

        comp_path = comp_dir / f"{slug}_{tag}_comparison.png"
        metrics = render_comparison(row, data, tokenizers[model_name], comp_path, args.max_tokens)
        all_metrics.append(metrics)

        bs_path = detail_dir / f"{slug}_{tag}_bertscore.png"
        render_bertscore_detail(row, data, tokenizers[model_name], bs_path, args.max_tokens)

        ot_path = detail_dir / f"{slug}_{tag}_ot.png"
        render_ot_detail(row, data, tokenizers[model_name], ot_path, args.max_tokens)

        print(f"[OK] {slug} example {eid} ({row.get('bucket', 'N/A')})")

    if not all_metrics:
        print("No examples were processed.")
        return

    # Save metrics CSV
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(out_dir / "leiden_demo_metrics.csv", index=False)
    print(f"\nMetrics: {out_dir / 'leiden_demo_metrics.csv'}")

    # Build HTML report
    build_html_report(examples, all_metrics, comp_dir, out_dir / "leiden_demo_report.html")

    print(f"\nLeiden demo complete: {len(all_metrics)} examples in {out_dir}")


if __name__ == "__main__":
    main()
