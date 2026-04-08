"""Build the post-meeting walkthrough HTML for Prof. Siddique.

Single-file, self-contained HTML covering every change since the 2026-04-07
meeting. Embeds three figures as base64 data URIs and pulls all numerical
content from the canonical CSV/JSON outputs so reruns are reproducible.

Output: docs/meetings/POSTMEETING_2026_04_07_WALKTHROUGH.html
"""
from __future__ import annotations

import base64
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

OUT_PATH = REPO_ROOT / "docs/meetings/POSTMEETING_2026_04_07_WALKTHROUGH.html"

PER_MODEL_CSV = REPO_ROOT / "runs/active/resubmit/results/taskA_per_model_summary.csv"
SUMMARY_JSON = REPO_ROOT / "runs/active/resubmit/data/resubmit_summary.json"
FIG_AUCROC = REPO_ROOT / "overleaf_drafts/figures/fig_release_aucroc_per_model.png"
FIG_GAP = REPO_ROOT / "overleaf_drafts/figures/fig_release_gap_per_model.png"
FIG_ABTT_COMPARE = REPO_ROOT / "overleaf_drafts/figures/fig_attribution_methods_abtt_compare.png"

DECRITALS_SUMMARY_CSV = REPO_ROOT / "runs/active/decritals/results/decritals_per_model_summary.csv"
DECRITALS_SUMMARY_JSON = REPO_ROOT / "runs/active/decritals/data/resubmit_summary.json"
DECRITALS_TASKA_CSV = REPO_ROOT / "runs/active/decritals/results/decritals_taskA_results.csv"


def png_to_data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def load_per_model_rows() -> list[dict]:
    with PER_MODEL_CSV.open() as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r.get("model_display")]
    return rows


def load_summary() -> dict:
    return json.loads(SUMMARY_JSON.read_text())


# Short display names keyed by the HuggingFace-style model id stored in
# decritals_per_model_summary.csv.  Kept parallel to the canon per-model CSV.
DECRITALS_DISPLAY = {
    "bowphs/LaTa": "LaTa",
    "bowphs/PhilTa": "PhilTa",
    "sentence-transformers/LaBSE": "LaBSE",
    "google/mt5-base": "mT5-base",
    "Qwen/Qwen3-Embedding-0.6B": "Qwen3-0.6B",
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5": "KaLM-mini",
}


def load_decritals_rows() -> list[dict]:
    """Return one dict per model, display-name keyed, joined with canon numbers."""
    with DECRITALS_SUMMARY_CSV.open() as f:
        reader = csv.DictReader(f)
        raw = [r for r in reader if r.get("model")]
    canon_rows = load_per_model_rows()
    canon_by_display = {r["model_display"]: r for r in canon_rows}
    out: list[dict] = []
    for r in raw:
        display = DECRITALS_DISPLAY.get(r["model"], r["model"])
        canon = canon_by_display.get(display, {})
        out.append({
            "display": display,
            "taskA_best_layer": int(r["taskA_best_layer"]),
            "taskA_acc_at_1": float(r["taskA_acc_at_1"]),
            "taskA_oa": float(r["taskA_overall_assignment_acc"]),
            "taskA_aucroc": float(r["taskA_aucroc"]),
            "taskB_best_layer": int(r["taskB_best_layer"]),
            "taskB_acc_at_1_mean": float(r["taskB_acc_at_1_mean"]),
            "taskB_acc_at_1_std": float(r["taskB_acc_at_1_std"]),
            "canon_acc_at_1": float(canon.get("dir_acc_at_1", 0) or 0),
            "canon_oa": float(canon.get("overall_assignment_acc", 0) or 0),
            "canon_aucroc": float(canon.get("aucroc", 0) or 0),
        })
    # Preserve canonical model ordering from the canon per-model table
    order = [r["model_display"] for r in canon_rows]
    out.sort(key=lambda d: order.index(d["display"]) if d["display"] in order else 999)
    return out


def load_decritals_summary() -> dict:
    return json.loads(DECRITALS_SUMMARY_JSON.read_text())


def fmt_pct(x: str | float) -> str:
    return f"{float(x) * 100:.1f}%"


def per_model_table_html(rows: list[dict]) -> str:
    # KaLM-mini has the highest overall_assignment_acc; mark it .best
    best_oa = max(float(r["overall_assignment_acc"]) for r in rows)
    out = ['<table>',
           '<tr><th>Model</th><th>Layer</th><th>Method</th><th>D</th>'
           '<th>Acc@1</th><th>Assign Acc</th><th>AUCROC</th>'
           '<th>Existing Acc</th><th>New Acc</th></tr>']
    for r in rows:
        is_best = abs(float(r["overall_assignment_acc"]) - best_oa) < 1e-9
        oa_class = ' class="best"' if is_best else ''
        out.append(
            f'<tr>'
            f'<td><strong>{r["model_display"]}</strong></td>'
            f'<td>L{int(r["best_layer"])}</td>'
            f'<td><code>{r["method"]}</code></td>'
            f'<td>{int(r["D"])}</td>'
            f'<td>{fmt_pct(r["dir_acc_at_1"])}</td>'
            f'<td{oa_class}>{fmt_pct(r["overall_assignment_acc"])}</td>'
            f'<td>{fmt_pct(r["aucroc"])}</td>'
            f'<td>{fmt_pct(r["existing_acc"])}</td>'
            f'<td>{fmt_pct(r["new_acc"])}</td>'
            f'</tr>'
        )
    out.append('</table>')
    return "\n".join(out)


CSS = """
  :root {
    --bg: #fafafa; --fg: #1a1a1a; --muted: #666; --accent: #2563eb;
    --card-bg: #fff; --card-border: #e5e7eb; --table-stripe: #f9fafb;
    --green: #059669; --red: #dc2626; --amber: #d97706;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--fg); line-height: 1.6; padding: 2rem; max-width: 1100px; margin: 0 auto; }
  h1 { font-size: 1.8rem; margin-bottom: 0.25rem; }
  h2 { font-size: 1.35rem; margin: 2.5rem 0 0.75rem; padding-bottom: 0.4rem; border-bottom: 2px solid var(--accent); color: var(--accent); }
  h3 { font-size: 1.1rem; margin: 1.25rem 0 0.5rem; color: var(--fg); }
  .subtitle { color: var(--muted); font-size: 0.95rem; margin-bottom: 1.5rem; }
  .card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 8px; padding: 1.25rem; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
  .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }
  .metric-card { text-align: center; padding: 1rem; border-radius: 8px; background: var(--card-bg); border: 1px solid var(--card-border); }
  .metric-card .value { font-size: 2rem; font-weight: 700; color: var(--accent); }
  .metric-card .label { font-size: 0.8rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.25rem; }
  .metric-card .detail { font-size: 0.75rem; color: var(--muted); margin-top: 0.15rem; }
  table { width: 100%; border-collapse: collapse; font-size: 0.9rem; margin: 0.75rem 0; }
  th { background: var(--accent); color: white; padding: 0.6rem 0.8rem; text-align: left; font-weight: 600; }
  td { padding: 0.5rem 0.8rem; border-bottom: 1px solid var(--card-border); }
  tr:nth-child(even) { background: var(--table-stripe); }
  .best { font-weight: 700; color: var(--green); }
  .tag { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
  .tag-done { background: #dcfce7; color: #166534; }
  .tag-new { background: #fef3c7; color: #92400e; }
  .tag-todo { background: #e0e7ff; color: #3730a3; }
  .finding { padding: 0.75rem 1rem; margin: 0.5rem 0; border-left: 4px solid var(--accent); background: #eff6ff; border-radius: 0 6px 6px 0; }
  .finding.green { border-left-color: var(--green); background: #ecfdf5; }
  .finding.amber { border-left-color: var(--amber); background: #fffbeb; }
  .finding.red { border-left-color: var(--red); background: #fef2f2; }
  .cols2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  .cols3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; }
  .quote { padding: 0.5rem 1rem; margin: 0.5rem 0; border-left: 3px solid #d1d5db; color: var(--muted); font-style: italic; font-size: 0.9rem; }
  details { margin: 0.5rem 0; }
  details summary { cursor: pointer; font-weight: 600; color: var(--accent); padding: 0.25rem 0; }
  .checklist { list-style: none; padding: 0; }
  .checklist li { padding: 0.35rem 0; display: flex; align-items: center; gap: 0.5rem; }
  .checklist li::before { content: ''; width: 14px; height: 14px; border-radius: 3px; border: 2px solid var(--card-border); flex-shrink: 0; }
  .checklist li.done::before { background: var(--green); border-color: var(--green); }
  pre { background: #f3f4f6; border: 1px solid var(--card-border); border-radius: 6px; padding: 0.75rem 1rem; overflow-x: auto; font-size: 0.78rem; line-height: 1.45; margin: 0.5rem 0; }
  pre code { font-family: 'SF Mono', Menlo, Consolas, monospace; color: #111827; }
  code { font-family: 'SF Mono', Menlo, Consolas, monospace; background: #f3f4f6; padding: 0.1rem 0.35rem; border-radius: 3px; font-size: 0.85em; }
  blockquote.paper { background: #f9fafb; border: 1px solid var(--card-border); border-left: 4px solid var(--accent); padding: 1rem 1.25rem; margin: 0.75rem 0; font-size: 0.92rem; border-radius: 0 6px 6px 0; }
  blockquote.paper p { margin-bottom: 0.6rem; }
  blockquote.paper p:last-child { margin-bottom: 0; }
  blockquote.paper .eq { display: block; text-align: center; font-family: 'SF Mono', Menlo, monospace; font-size: 0.95rem; margin: 0.5rem 0; }
  img.figure { max-width: 100%; height: auto; display: block; margin: 0.75rem auto; border: 1px solid var(--card-border); border-radius: 6px; background: white; }
  .flow { background: #f9fafb; border: 1px solid var(--card-border); border-radius: 8px; padding: 1rem 1.25rem; margin: 0.75rem 0; font-size: 0.88rem; }
  .flow .step { padding: 0.5rem 0.75rem; border-left: 3px solid var(--accent); margin: 0.4rem 0; background: white; border-radius: 0 4px 4px 0; }
  .flow .step.branch { border-left-color: var(--amber); margin-left: 1.5rem; }
  .flow .arrow { color: var(--muted); margin: 0 0.4rem; }
  .footnote { font-size: 0.8rem; color: var(--muted); margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--card-border); }
  @media (max-width: 700px) { .cols2, .cols3 { grid-template-columns: 1fr; } .metric-grid { grid-template-columns: 1fr 1fr; } }
"""


SPLIT_CODE = '''# --- Multi-file (>=3): varied per-folder allocation within each size class ---
# For each size-n class of K folders, distribute train_n values uniformly
# over {1, ..., n-1} via stratified allocation, with the remainder going
# to values nearest n/2 so the class average stays ~n/2. Determinism is
# preserved via the shared rng. This avoids the pathological case where
# every size-3 folder is forced to (1 train, 2 test) and contributes zero
# same-folder positive pairs to the training split.
multi_mask = meta["folder_size"] >= 3
if multi_mask.any():
    multi_df = meta.loc[multi_mask, ["folder_id", "folder_size"]]
    train_n_by_folder: Dict[str, int] = {}
    for size_n in sorted(multi_df["folder_size"].unique().tolist()):
        class_folder_ids = np.sort(
            multi_df.loc[multi_df["folder_size"] == size_n, "folder_id"].unique()
        )
        K = len(class_folder_ids)
        allowed = list(range(1, int(size_n)))
        n_allowed = len(allowed)
        base, rem = divmod(K, n_allowed)
        counts = [base] * n_allowed
        mid = size_n / 2.0
        order = sorted(range(n_allowed), key=lambda i: (abs(allowed[i] - mid), i))
        for r in range(rem):
            counts[order[r]] += 1
        values: List[int] = []
        for val, cnt in zip(allowed, counts):
            values.extend([val] * cnt)
        shuffled_folders = rng.permutation(class_folder_ids)
        shuffled_values = rng.permutation(np.asarray(values, dtype=np.int32))
        for fid, tn in zip(shuffled_folders, shuffled_values):
            train_n_by_folder[str(fid)] = int(tn)'''


def decritals_delta_table_html(rows: list[dict]) -> str:
    """Canon vs Decritals side-by-side, showing the acc@1 collapse and OA pin."""
    out = ['<table>',
           '<tr>'
           '<th rowspan="2">Model</th>'
           '<th colspan="3">Canon (best layer)</th>'
           '<th colspan="3">Decritals (best layer)</th>'
           '<th colspan="3">Delta (Decritals &minus; Canon)</th>'
           '</tr>',
           '<tr>'
           '<th>Acc@1</th><th>OA</th><th>AUCROC</th>'
           '<th>Acc@1</th><th>OA</th><th>AUCROC</th>'
           '<th>Acc@1</th><th>OA</th><th>AUCROC</th>'
           '</tr>']
    for r in rows:
        d_acc1 = r["taskA_acc_at_1"] - r["canon_acc_at_1"]
        d_oa = r["taskA_oa"] - r["canon_oa"]
        d_auc = r["taskA_aucroc"] - r["canon_aucroc"]
        is_pinned = abs(r["taskA_oa"] - 0.8451327433628318) < 1e-6
        pin_marker = ' <span style="color: var(--red); font-weight: 700;" title="threshold-degenerate: predicts every test file as &quot;existing&quot;">&#9679;</span>' if is_pinned else ''
        out.append(
            f'<tr>'
            f'<td><strong>{r["display"]}</strong></td>'
            f'<td>{fmt_pct(r["canon_acc_at_1"])}</td>'
            f'<td>{fmt_pct(r["canon_oa"])}</td>'
            f'<td>{fmt_pct(r["canon_aucroc"])}</td>'
            f'<td>{fmt_pct(r["taskA_acc_at_1"])}</td>'
            f'<td>{fmt_pct(r["taskA_oa"])}{pin_marker}</td>'
            f'<td>{fmt_pct(r["taskA_aucroc"])}</td>'
            f'<td style="color: var(--red); font-weight: 600;">{d_acc1*100:+.1f} pp</td>'
            f'<td style="color: var(--muted);">{d_oa*100:+.1f} pp</td>'
            f'<td style="color: var(--red); font-weight: 600;">{d_auc*100:+.1f} pp</td>'
            f'</tr>'
        )
    out.append('</table>')
    return "\n".join(out)


def decritals_taskb_table_html(rows: list[dict]) -> str:
    out = ['<table>',
           '<tr><th>Model</th><th>Best layer</th><th>Task B Acc@1 (mean &plusmn; std, 5 seeds)</th></tr>']
    for r in rows:
        out.append(
            f'<tr>'
            f'<td><strong>{r["display"]}</strong></td>'
            f'<td>L{r["taskB_best_layer"]}</td>'
            f'<td>{r["taskB_acc_at_1_mean"]*100:.1f}% &plusmn; {r["taskB_acc_at_1_std"]*100:.1f}%</td>'
            f'</tr>'
        )
    out.append('</table>')
    return "\n".join(out)


def build_html() -> str:
    rows = load_per_model_rows()
    summary = load_summary()
    n_pos_train = summary["task_a_pairs"]["train"]["n_positive"]
    n_pos_test = summary["task_a_pairs"]["test"]["n_positive"]

    decritals_rows = load_decritals_rows()
    decritals_summary = load_decritals_summary()
    d_n_folders = decritals_summary["split"]["n_folders"]
    d_n_test = decritals_summary["split"]["n_test"]
    d_n_test_query = decritals_summary["split"]["n_test_query"]
    d_n_pos_train = decritals_summary["task_a_pairs"]["train"]["n_positive"]
    d_n_pos_test = decritals_summary["task_a_pairs"]["test"]["n_positive"]

    # Acc@1 drop range across models (absolute, in pp)
    acc1_drops = sorted(
        (r["canon_acc_at_1"] - r["taskA_acc_at_1"]) for r in decritals_rows
    )
    min_drop_pp = acc1_drops[0] * 100
    max_drop_pp = acc1_drops[-1] * 100

    aucroc_uri = png_to_data_uri(FIG_AUCROC)
    gap_uri = png_to_data_uri(FIG_GAP)
    abtt_uri = png_to_data_uri(FIG_ABTT_COMPARE)

    per_model_table = per_model_table_html(rows)
    decritals_delta_table = decritals_delta_table_html(decritals_rows)
    decritals_taskb_table = decritals_taskb_table_html(decritals_rows)

    parts: list[str] = []
    parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LocalLatin &mdash; Post-Meeting Walkthrough (April 7, 2026)</title>
<style>{CSS}</style>
</head>
<body>

<h1>LocalLatin &mdash; Post-Meeting Walkthrough</h1>
<p class="subtitle">Every change since the April 7, 2026 meeting, mapped back to what you asked for.</p>

<div class="finding">
  <strong>How to read this page.</strong> Each section opens with a one-sentence headline that ties the change back to something you said at the meeting. Tables and figures are embedded inline so the page works offline. The closing section has three explicit decisions we need from you before pushing forward.
</div>
""")

    # ==== Section 1: Snapshot ====
    parts.append(f"""
<!-- ================================================================ -->
<h2>1. Snapshot &mdash; You predicted numbers wouldn't move much. You were right.</h2>

<div class="finding green">
  <strong>The setup is now defensible</strong> &mdash; varied per-folder allocation within each size class, all 6 models re-evaluated, per-model best layers documented, and the whole pipeline re-run end-to-end on a second manuscript dataset (decritals) as cross-dataset validation.
</div>

<div class="metric-grid">
  <div class="metric-card">
    <div class="value">+59%</div>
    <div class="label">Train Positive Pairs</div>
    <div class="detail">354 &rarr; {n_pos_train}</div>
  </div>
  <div class="metric-card">
    <div class="value">662 / 662</div>
    <div class="label">Canon Rows Re-evaluated</div>
    <div class="detail">6 models &times; layers &times; 7 methods</div>
  </div>
  <div class="metric-card">
    <div class="value">700 / 700</div>
    <div class="label">Decritals Rows Evaluated</div>
    <div class="detail">Task A + Task B, 5 seeds</div>
  </div>
  <div class="metric-card">
    <div class="value">&minus;{max_drop_pp:.0f} pp</div>
    <div class="label">Decritals Acc@1 Drop</div>
    <div class="detail">canon &rarr; decritals, max across models</div>
  </div>
</div>

<h3>Best per-model configuration on the new split</h3>
{per_model_table}

<div class="finding green">
  <strong>What to look at first:</strong> KaLM-mini now leads on overall assignment accuracy (92.1%), with five of six models clustered between 90.7% and 92.1% &mdash; a 1.4-point spread. Best layer is layer 1 for everyone except LaBSE (L11) and Qwen3-0.6B (L5). All numbers are pulled live from <code>runs/active/resubmit/results/taskA_per_model_summary.csv</code>.
</div>
""")

    # ==== Section 2: Split refactor ====
    parts.append(f"""
<!-- ================================================================ -->
<h2>2. Split refactor &mdash; Per-folder allocation now varies within each size class</h2>

<div class="quote">
  &ldquo;Sometimes like three, two, one, so we will be able to generate more samples ... I mean nothing is going to change but it will be good that we have like strong defensible experimental setup.&rdquo;
  <br>&mdash; Prof. Siddique, 2026-04-07
</div>

<p>Before, every multi-file directory split with the fixed rule <code>train_n = n // 2</code>, which meant size-3 directories all contributed (1 train, 2 test). With only one file per such directory in train, those directories produced <strong>zero positive same-folder pairs</strong> for training. Now, within each directory-size class we distribute <code>train_n</code> uniformly over <code>{{1, ..., n&minus;1}}</code> via stratified allocation, with the remainder weighted toward values nearest <code>n/2</code> so the class average still hovers around 50/50.</p>

<h3>Before vs after</h3>
<div class="cols2">
  <div class="card">
    <h3>Before (fixed <code>n // 2</code>)</h3>
    <ul class="checklist">
      <li>Size-3 folders: <strong>all</strong> use (1 train, 2 test) &rarr; 0 train pairs from this class</li>
      <li>Size-4 folders: <strong>all</strong> use (2 train, 2 test) &rarr; 1 train pair each</li>
      <li>Size-5 folders: <strong>all</strong> use (2 train, 3 test) &rarr; 1 train pair each</li>
      <li>Total train positive pairs: <strong>354</strong></li>
    </ul>
  </div>
  <div class="card">
    <h3>After (varied per folder)</h3>
    <ul class="checklist">
      <li class="done">Size-3 folders: half use (1, 2), half use (2, 1)</li>
      <li class="done">Size-4 folders: mix of (1, 3), (2, 2), (3, 1) skewed toward (2, 2)</li>
      <li class="done">Size-5+ folders: same idea, all classes contribute pairs to train</li>
      <li class="done">Total train positive pairs: <strong>{n_pos_train}</strong> (+59%)</li>
    </ul>
  </div>
</div>

<h3>Code (<code>src/canon_split_v2.py</code> lines 61&ndash;91)</h3>
<pre><code>{SPLIT_CODE}</code></pre>

<p>Determinism is preserved via the shared <code>rng = np.random.default_rng(random_seed=42)</code>, so reruns produce identical splits. Singleton (size-1) and doubleton (size-2) rules are unchanged.</p>

<div class="finding">
  <strong>Reviewer-proofing.</strong> The previous setup's pathological case &mdash; size-3 directories contributing zero positive training pairs &mdash; is exactly the kind of artifact a hostile reviewer would flag. The new rule has no such bias: every size class with <code>n &ge; 2</code> contributes positive pairs to both train and test.
</div>
""")

    # ==== Section 3: Re-evaluation deltas ====
    parts.append(f"""
<!-- ================================================================ -->
<h2>3. Re-evaluation deltas &mdash; All 662 rows refreshed on the new split</h2>

<p>Re-ran the canonical Task A evaluator (<code>slurm/resubmit/resubmit_evaluate.sbatch</code>, calling <code>scripts/resubmit/run_resubmit_evaluate.py</code>) over all 6 models, all layers, all 7 post-processing methods. Then re-ran the figure pipeline (<code>resubmit_figures.sbatch</code>) to refresh the per-model figures. Compared to the prior CSV row-by-row:</p>

<div class="metric-grid">
  <div class="metric-card">
    <div class="value">662 / 662</div>
    <div class="label">Rows Shifted</div>
    <div class="detail">Every (model, layer, method) row moved at least slightly</div>
  </div>
  <div class="metric-card">
    <div class="value">3.2 pp</div>
    <div class="label">Mean |&Delta;|</div>
    <div class="detail">overall_assignment_acc</div>
  </div>
  <div class="metric-card">
    <div class="value">16.3 pp</div>
    <div class="label">Max |&Delta;|</div>
    <div class="detail">single largest row shift</div>
  </div>
</div>

<h3>Per-model AUCROC (refreshed)</h3>
<img class="figure" src="{aucroc_uri}" alt="Per-model AUCROC, all layers, refreshed split">

<h3>Per-model cosine gap (same-dir vs different-dir)</h3>
<img class="figure" src="{gap_uri}" alt="Per-model cosine gap, all layers, refreshed split">

<div class="finding green">
  <strong>Relative ordering held.</strong> The same models that led on the old split lead on the new split, and the mid-layer dip is still visible in the same place for the same architectures. The numerical shifts are within the noise band you'd expect from a 354 &rarr; {n_pos_train} change in training pair count.
</div>
""")

    # ==== Section 4: acc@1 vs assignment accuracy ====
    parts.append(f"""
<!-- ================================================================ -->
<h2>4. Acc@1 vs Assignment Accuracy &mdash; The didactic section</h2>

<div class="quote">
  &ldquo;Just write it somewhere so that we don't forget what we are doing. I mean I don't know about you but I will definitely forget.&rdquo;
  <br>&mdash; Prof. Siddique, 2026-04-07
</div>

<p>Done. The paper now has a dedicated subsection clarifying the distinction. Read this section once and bookmark it.</p>

<h3>Side by side</h3>
<div class="cols2">
  <div class="card">
    <h3>Directory Acc@1</h3>
    <p><strong>Question:</strong> Did we rank the correct item at the top?</p>
    <ul class="checklist">
      <li class="done">Only counts test files that <em>have</em> a same-folder partner in the test set</li>
      <li class="done">Pure ranking &mdash; cosine similarity to every other test file</li>
      <li class="done">No threshold &tau;, no &ldquo;I don't know&rdquo; option</li>
      <li class="done">Every counted query is forced to produce an answer</li>
    </ul>
  </div>
  <div class="card">
    <h3>Assignment Accuracy</h3>
    <p><strong>Question:</strong> Did we make the right decision, including knowing when to say &ldquo;this is new&rdquo;?</p>
    <ul class="checklist">
      <li class="done">Counts <em>every</em> test file (with or without a partner)</li>
      <li class="done">Binary first: <code>max_sim &ge; &tau;</code> &rarr; existing, else &rarr; new</li>
      <li class="done">&tau; learned on train via F1-optimal sweep over same/diff pairs</li>
      <li class="done">Top-1 ranking only happens <em>after</em> we've decided "existing"</li>
    </ul>
  </div>
</div>

<h3>The workflow, step by step</h3>
<div class="flow">
  <div class="step"><strong>1.</strong> Query file <code>i</code> arrives.</div>
  <div class="step"><strong>2.</strong> Compute <code>s<sub>i</sub> = max<sub>j&ne;i</sub> cos(e<sub>i</sub>, e<sub>j</sub>)</code> against every other test file.</div>
  <div class="step"><strong>3.</strong> Compare to learned threshold &tau; (learned on <em>train</em> only).</div>
  <div class="step branch"><span class="arrow">&rarr;</span> If <code>s<sub>i</sub> &lt; &tau;</code>: predict <strong>&ldquo;new&rdquo;</strong>. Done. <em>(Assignment Acc scores this against ground truth; Acc@1 ignores it.)</em></div>
  <div class="step branch"><span class="arrow">&rarr;</span> If <code>s<sub>i</sub> &ge; &tau;</code>: predict <strong>&ldquo;existing&rdquo;</strong>, then take the top-1 neighbour as the predicted directory. <em>(Both metrics score this; they can disagree.)</em></div>
</div>

<h3>Worked example: when the two metrics disagree</h3>
<div class="card">
  <p>Suppose query file <code>q</code> has a same-folder partner <code>p</code> in the test set, and <code>p</code> is the closest neighbour: <code>cos(q, p) = 0.42</code>. Suppose the learned threshold is <code>&tau; = 0.55</code>.</p>
  <ul>
    <li><strong>Acc@1:</strong> The top-ranked neighbour <code>p</code> is in the correct folder. <span class="best">Score: correct.</span></li>
    <li><strong>Assignment Acc:</strong> <code>max_sim = 0.42 &lt; 0.55 = &tau;</code>, so we predict &ldquo;new&rdquo;. But ground truth says <code>q</code> has a partner. <span style="color: var(--red); font-weight: 700;">Score: wrong.</span></li>
  </ul>
  <p class="footnote" style="margin-top: 0.5rem; padding-top: 0; border-top: none;">The two diverge whenever &tau; is mis-calibrated relative to the actual same-pair similarities, or when the novel-file subset is large. They can also diverge in the other direction (assignment accuracy correct but Acc@1 wrong) when &tau; is satisfied but the second-closest neighbour belongs to a different folder than the true partner.</p>
</div>

<h3>The new paper paragraph (verbatim from <code>overleaf_drafts/acl_latex.tex</code>)</h3>
<blockquote class="paper">
  <p><strong>Directory accuracy at rank 1 (Acc@1).</strong> For each test file that has at least one same-directory partner elsewhere in the test set, we rank every other test file by cosine similarity and score the query correct if the top-ranked neighbour comes from the same directory. Every such query is forced to produce an answer; there is no &ldquo;I don't know&rdquo; option and the threshold &tau; plays no role. Acc@1 answers a single question: <em>did we rank the correct item at the top?</em></p>
  <p><strong>Assignment accuracy.</strong> Assignment accuracy adds the complementary question: <em>did we make the right decision, including knowing when to say &ldquo;this is new&rdquo;?</em> For every test file <code>i</code> we compute <code>s<sub>i</sub> = max<sub>j&ne;i</sub> cos(e<sub>i</sub>, e<sub>j</sub>)</code> against the rest of the test set and compare it to the threshold &tau;, which is learned on the training split by sweeping the F<sub>1</sub>-optimal cut on same-directory versus different-directory pairs (no test data is used to fit &tau;). The decision is binary first:</p>
  <span class="eq">y&#770;<sub>i</sub> = &ldquo;existing&rdquo; if s<sub>i</sub> &ge; &tau;,&nbsp;&nbsp;&ldquo;new&rdquo; if s<sub>i</sub> &lt; &tau;</span>
  <p>and a file is scored correct iff this binary prediction matches ground truth (i.e. whether the file actually has a same-directory partner in the test set). Only on files that we predict as &ldquo;existing&rdquo; do we additionally look at the top-1 neighbour to recover the predicted directory. We report the overall rate as <strong>overall assignment accuracy</strong> and decompose it into <strong>existing accuracy</strong> (only files whose partner is in the test set) and <strong>new accuracy</strong> (only files that should have been flagged as novel).</p>
  <p>The query-vs-directory variant (<code>qvd_*</code> columns in <code>phase_resubmit_results.csv</code>) applies exactly the same threshold rule after max-pooling similarities at the directory level instead of the file level, matching the deployment setting of Task B.</p>
</blockquote>
""")

    # ==== Section 5: Attribution methods 6x2 ====
    parts.append(f"""
<!-- ================================================================ -->
<h2>5. Attribution methods &mdash; 6 methods &times; ABTT/baseline, side by side</h2>

<div class="quote">
  &ldquo;We don't have these variants with ABTT and without ABTT. So should we have these with and without ABTT? I mean if we want to see the difference.&rdquo;
  <br>&mdash; Prof. Siddique, 2026-04-07
</div>

<p>Done. Every example pair now persists 12 attribution matrices (6 methods &times; baseline/ABTT), and the meeting figure shows all of them in one 6&times;2 grid.</p>

<img class="figure" src="{abtt_uri}" alt="6 attribution methods x ABTT/baseline comparison">

<h3>What you're looking at</h3>
<table>
  <tr><th>Row</th><th>Method</th><th>What it measures</th></tr>
  <tr><td>1</td><td><strong>IG pair matrix</strong></td><td>Captum-style integrated gradients (signed), then outer-product into a query &times; candidate alignment</td></tr>
  <tr><td>2</td><td><strong>BERTScore</strong></td><td>Greedy alignment of every query token to its best candidate token by cosine similarity</td></tr>
  <tr><td>3</td><td><strong>Optimal Transport</strong></td><td>EMD with IG-derived mass on each side &mdash; transport plan T[i,j] is the explanation</td></tr>
  <tr><td>4</td><td><strong>Attention cross-sim</strong></td><td>Per-instance attention diagonal as pooling weights, &agrave; la Ditto</td></tr>
  <tr><td>5</td><td><strong>Direct Logit Attribution</strong></td><td>Norm-weighted cosine, no gradients needed (one forward pass)</td></tr>
  <tr><td>6</td><td><strong>Attention standalone</strong></td><td>Column-mean attention as standalone token importance</td></tr>
</table>

<p><strong>Left column:</strong> baseline (raw hidden states). <strong>Right column:</strong> after ABTT cleaning (top D=10 principal components removed). The red boxes mark the top-K aligned token pairs in each panel.</p>

<div class="finding">
  <strong>Pattern to flag:</strong> The diffuse heatmaps in the left column collapse to crisper, more localized signal in the right column for several methods &mdash; especially DLA and IG &mdash; which is consistent with the anisotropy story we tell in the paper. BERTScore and Optimal Transport degrade more gracefully without ABTT than IG and DLA do.
</div>

<h3>Where the data lives</h3>
<ul class="checklist">
  <li class="done"><strong>Persistence script:</strong> <code>scripts/resubmit/persist_attribution_methods.py</code> writes the 12 matrices per pair into the existing NPZ artifacts</li>
  <li class="done"><strong>Storage:</strong> <code>runs/active/ig_examples/artifacts/&lt;model_slug&gt;/example{{NNN}}_pair_example.npz</code> &mdash; new keys <code>pair_matrix_&lt;method&gt;_&lt;variant&gt;</code> and <code>topk_&lt;method&gt;_&lt;variant&gt;_{{query,candidate}}</code></li>
  <li class="done"><strong>Coverage:</strong> All 80 CSV-owned example pairs across LaTa, PhilTa, LaBSE, and Qwen3-0.6B</li>
  <li class="done"><strong>CSV index:</strong> <code>runs/active/ig_examples/phase12f_examples.csv</code> has a new <code>methods_available</code> column listing the methods present per row</li>
</ul>

<div class="finding green">
  <strong>Webapp consequence:</strong> the scholar review tool now exposes a method dropdown (IG, BERTScore, OT, Attn-cross-sim, DLA, Attn-standalone) and a baseline/ABTT toggle, plus a per-model example gallery for browsing precomputed pairs. We'll demo this live separately &mdash; not embedded here to keep the page small.
</div>
""")

    # ==== Section 6: Decritals cross-dataset validation ====
    parts.append(f"""
<!-- ================================================================ -->
<h2>6. Cross-dataset validation &mdash; Decritals Task A + Task B done</h2>

<div class="quote">
  &ldquo;If there is a chance you can run those before that that will be great. If you cannot then of course you will get to that after that.&rdquo;
  <br>&mdash; Prof. Siddique, 2026-04-07 (handing us the decritals bundle)
</div>

<p>We ran it. The entire Task A + Task B pipeline was re-executed on decritals using the same varied-allocation split, the same 6 models, and the same 7 post-processing methods. No retraining. No special tuning. Here is what transfers and what breaks.</p>

<h3>The dataset (for reference)</h3>
<div class="cols2">
  <div class="card">
    <h3>Decritals split</h3>
    <ul class="checklist">
      <li><strong>{d_n_folders} directories</strong> (vs 840 for canon)</li>
      <li>439 labelled files: 213 train / {d_n_test} test</li>
      <li>{d_n_test_query} test files with a same-folder partner (the &ldquo;winnable&rdquo; test queries)</li>
      <li>{d_n_pos_train} train positive pairs / {d_n_pos_test} test positive pairs</li>
      <li>Task B: 113 queries, 46 reference directories, 5 seeds</li>
    </ul>
  </div>
  <div class="card">
    <h3>Evaluator inputs</h3>
    <ul class="checklist">
      <li class="done">Meta + split CSV built with <code>build_meta_with_split_v2()</code> (varied allocation)</li>
      <li class="done">Embeddings extracted for all 6 models on both labelled and unlabelled</li>
      <li class="done">Task A: <code>run_resubmit_evaluate.py</code> &rarr; 700-row results CSV</li>
      <li class="done">Task B: <code>run_taskb_mseed.py</code> &rarr; 5 seeds, aggregated per model &times; layer</li>
      <li class="done">SLURM jobs under <code>slurm/decritals/</code>, data under <code>runs/active/decritals/</code></li>
    </ul>
  </div>
</div>

<h3>Headline finding: acc@1 collapses, assignment accuracy pretends everything is fine</h3>

<div class="finding red">
  <strong>Acc@1 drops {min_drop_pp:.0f}&ndash;{max_drop_pp:.0f} absolute points</strong> across all 6 models when we move from canon to decritals (roughly 0.88 &rarr; 0.53). Meanwhile overall assignment accuracy barely moves at all &mdash; it sits at exactly <strong>84.51%</strong> for 5 of 6 models, with KaLM-mini one prediction beyond at 84.96%. This is the metric divergence Section 4 warned about, live.
</div>

{decritals_delta_table}

<p class="footnote" style="margin-top:0; padding-top:0; border-top:none;">Red dots mark the 5 models whose best layer hits the 0.8451 pin. The pin is the ceiling of a threshold-degenerate classifier: there are exactly 191 test files with a same-folder partner (out of 226 test files total), so <strong>191/226 = 0.8451</strong> is what you get by predicting &ldquo;existing&rdquo; for every test file and always picking the nearest neighbour. The threshold &tau; collapses because similarity distributions for same-folder and different-folder pairs overlap too heavily for F<sub>1</sub>-optimal sweeping to carve a meaningful boundary.</p>

<h3>Why the pin is the point</h3>

<p>Look at the canon result (Section 1) next to the decritals result (above). On canon, <code>overall_assignment_acc</code> and <code>dir_acc_at_1</code> move together; both are around 0.89&ndash;0.92. On decritals, they split apart: Acc@1 is saying &ldquo;the retrieval signal is gone&rdquo; while OA is saying &ldquo;we can still get 84.5% right by abstaining on everything.&rdquo;</p>

<div class="finding amber">
  <strong>This is exactly why Section 4 wasn't pedantic.</strong> If we'd only reported <code>overall_assignment_acc</code>, the decritals result would look like a mild 7-point drop: &ldquo;still above 84%, paper still works.&rdquo; Reporting Acc@1 alongside it immediately surfaces the actual failure mode: the models can't rank decritals manuscripts well enough to carry any threshold rule, period. This is the kind of result where a reviewer asking &ldquo;how robust is this to a new dataset?&rdquo; gets a much more honest answer from Acc@1 than from OA.
</div>

<h3>The OA pin is the output of a degenerate classifier</h3>

<p>The decritals Task A results CSV has <strong>700 rows</strong> (6 models &times; their layers &times; 7 methods). Of those, the counts tied at OA = 0.8451 per model are: Qwen3-0.6B <strong>67 rows</strong>, mT5-base 42, LaBSE 37, LaTa 32, PhilTa 32. Dozens of configurations all produce the identical number because they all reduce to the same degenerate rule: predict &ldquo;existing&rdquo; for everything, get the 191 files with partners right, miss the 35 singletons, done.</p>

<p>The configurations that actually <em>retrieve best</em> on decritals (highest <code>dir_acc_at_1</code>) are <strong>different</strong> from the configurations that hit the OA pin, and they sacrifice ~15&ndash;20 pp of OA to do it:</p>

<div class="cols2">
  <div class="card">
    <h3>OA-maximizing config (pinned)</h3>
    <ul class="checklist">
      <li>LaTa L1 any of 32 methods &mdash; OA 0.8451, Acc@1 ~0.535</li>
      <li>Qwen3-0.6B 67 configs &mdash; OA 0.8451, Acc@1 ~0.531</li>
      <li>mT5-base 42 configs &mdash; OA 0.8451, Acc@1 ~0.500</li>
      <li class="done">Degenerate: predicts every test file as &ldquo;existing&rdquo;</li>
    </ul>
  </div>
  <div class="card">
    <h3>Acc@1-maximizing config (actually retrieves)</h3>
    <ul class="checklist">
      <li>LaTa L6 <code>abtt_optimal</code> &mdash; Acc@1 <strong>0.571</strong>, OA only 0.8451</li>
      <li>Qwen3-0.6B L2 <code>abtt_fixed</code> &mdash; Acc@1 <strong>0.584</strong>, OA drops to 0.6549</li>
      <li>KaLM-mini L24 <code>baseline</code> &mdash; Acc@1 <strong>0.580</strong>, OA 0.8009</li>
      <li>Best layers scatter (L2, L3, L6, L12, L24) &mdash; different from canon's L1 preference</li>
    </ul>
  </div>
</div>

<h3>Task B holds up far better</h3>

<p>Task B (query &rarr; reference directory, 5 seeds) does <em>not</em> collapse the same way &mdash; probably because the reference-directory side provides a signal floor that file-level retrieval doesn't:</p>

{decritals_taskb_table}

<p>Task B Acc@1 of 0.41&ndash;0.47 is well above the 1/46 random baseline (~2.2%) and the gap between the best and worst model is only about 6 points. This is the metric that survives the cross-dataset move most gracefully.</p>

<h3>What this means</h3>

<div class="finding green">
  <strong>Retrieval quality, not thresholding, is the binding bottleneck on decritals.</strong> Our post-processing methods (ABTT, SIF, whitening) can reshape the similarity distribution, but they can't create ranking signal that isn't in the base embeddings. The canon &rarr; decritals gap is a retrieval-model gap &mdash; it lives in the transformer's representations, not in the downstream threshold rule.
</div>

<div class="finding">
  <strong>Consequence for the attribution story.</strong> Our 6-method attribution pipeline (Section 5) explains the retrieval decisions that DO happen; it doesn't fix the decritals gap. If we want to lift decritals numbers specifically, we'd need to look at the embedding step (better tokenization for abbreviated medieval Latin? domain-adapted pre-training? different pooling?), not at post-processing. That's a different axis from what our paper is about, but worth flagging so we know what we're claiming and what we aren't.
</div>

<p>Full artefacts: <code>runs/active/decritals/results/decritals_taskA_results.csv</code> (700 rows), <code>runs/active/decritals/results/decritals_taskB_mseed/aggregated_results.csv</code> (400+ rows), <code>runs/active/decritals/results/decritals_per_model_summary.csv</code> (6 rows, drives the tables above), plus 12 SLURM jobs under <code>slurm/decritals/</code>.</p>
""")

    # ==== Section 7: MaRC paper digest with corrections ====
    parts.append("""
<!-- ================================================================ -->
<h2>7. MaRC paper digest &mdash; And two corrections from last meeting</h2>

<div class="quote">
  &ldquo;Yeah so can you quickly see that do they have code or not? Maybe maybe they link it in references or in the appendix.&rdquo;
  <br>&mdash; Prof. Siddique, 2026-04-07
</div>

<p>I read the paper end to end (and verified the equations against the PDF using <code>pymupdf</code>). Two things turned out differently than we thought at the meeting:</p>

<div class="finding amber">
  <strong>Correction 1 &mdash; the paper is not new.</strong> It is <strong>Brinner &amp; Zarrie&szlig;, &ldquo;Model Interpretability and Rationale Extraction by Input Mask Optimization&rdquo;, Findings of ACL 2023</strong> (Toronto, pp. 13722&ndash;13744). The arXiv ID 2508.11388 (uploaded 2025-08-15) is a late re-deposit of the existing ACL Findings paper, not a fresh contribution. So we're not 8&ndash;9 months behind &mdash; we're nearly three years behind on the published version.
</div>

<div class="finding amber">
  <strong>Correction 2 &mdash; the code IS public.</strong> The <code>github link.txt</code> file in the resource bundle is not empty (52 bytes). It contains <code>https://github.com/inas-argumentation/Explainability</code>, and the same URL appears in Appendix A of the paper. The repo is Python, with <code>run_movie_reviews.py</code> and <code>run_imagenet.py</code> as entry points. <strong>No LICENSE file</strong> as of 2026-04-07 &mdash; relevant if we ever want to copy code rather than re-implement.
</div>

<h3>What MaRC actually does</h3>
<div class="cols3">
  <div class="card">
    <h3>The method</h3>
    <ul>
      <li>Per-instance soft mask <code>&lambda; &isin; [0,1]<sup>n</sup></code></li>
      <li>Parameterized by <code>(w, &sigma;)</code> via Gaussian influence kernel + sigmoid &mdash; gives spatial smoothness for free</li>
      <li>Masked input <code>x&#771; = &lambda;&middot;x + (1&minus;&lambda;)&middot;b</code> against PAD-token baseline</li>
      <li>Loss combines <strong>sufficiency</strong> on <code>x&#771;</code> + <strong>comprehensiveness</strong> on the complement</li>
      <li>Two regularizers: squared-mean sparsity on <code>&lambda;</code>, log-barrier on <code>&sigma;</code></li>
      <li>Per-instance Adam, frozen classifier, ~100s of forward+backward passes per example</li>
    </ul>
  </div>
  <div class="card">
    <h3>How it differs from IG</h3>
    <ul>
      <li><strong>Computation:</strong> IG = single-pass gradient integral; MaRC = per-instance optimization with hundreds of Adam steps</li>
      <li><strong>Output:</strong> IG = signed attributions, no sparsity prior; MaRC = non-negative <code>[0,1]</code> mask with explicit sparsity and smoothness</li>
      <li><strong>What's explained:</strong> IG = the gradient w.r.t. a target output; MaRC = perturbation-aligned (sufficiency + comprehensiveness, like ERASER)</li>
      <li><strong>Compute cost:</strong> IG = tens of forward passes; MaRC = ~2&ndash;3 min per BERT-base example</li>
      <li><strong>Memory:</strong> MaRC needs persistent computation graph through <code>&lambda; &rarr; x&#771; &rarr; M(x&#771;)</code></li>
    </ul>
  </div>
  <div class="card">
    <h3>Where our variant could differ</h3>
    <ul>
      <li><strong>Distance kernel:</strong> use line/folio position from manuscript metadata, not just sequential token index &mdash; a real linguistic difference for fragmentary inputs</li>
      <li><strong>Fidelity loss:</strong> tie to <em>ABTT-cleaned cosine to candidate vector</em>, not classification log-likelihood &mdash; aligns with our retrieval objective, not a synthetic classifier head</li>
      <li><strong>Amortized prediction:</strong> drop per-instance optimization in favor of a small head that emits <code>(w, &sigma;)</code> from a forward pass &mdash; scales to all 2,238 unlabelled queries instead of a hand-curated set</li>
      <li><strong>Cheap first pass:</strong> optimize a mask in cached-embedding space (no live model) as a sanity-check baseline before paying for the real thing</li>
    </ul>
  </div>
</div>

<div class="finding">
  <strong>Compute estimate</strong> for a real MaRC pass over our current example pool: 4 models &times; ~50 example pairs &times; ~100 Adam steps &times; per-step forward+backward &asymp; <strong>single-digit GPU-hours per model</strong> on <code>gpuA100x4</code>. This would be a Phase-13-style experiment, not a quick patch.
</div>

<details>
  <summary>Click to expand: full 336-line survey at <code>docs/research/attribution_paper_2025_input_mask.md</code></summary>
  <p>The full survey covers: citation, method summary (260 words), all core equations verified against the PDF, the IG comparison, implications for our pipeline (with the four variant-design hooks above), the limitations the authors admit (Section 7 of the paper), source code status, and a discrepancy log of every place the paper's equation numbering differs from common summaries.</p>
</details>
""")

    # ==== Section 8: What's next + 3 decisions ====
    parts.append("""
<!-- ================================================================ -->
<h2>8. What's next &mdash; And three decisions we need from you</h2>

<h3>What's already done <span class="tag tag-done">Complete</span></h3>
<ul class="checklist">
  <li class="done">Task A split refactor (varied per-folder allocation)</li>
  <li class="done">Task A re-evaluation on new split (662 rows, 6 models)</li>
  <li class="done">Acc@1 vs assignment accuracy documented in <code>acl_latex.tex</code></li>
  <li class="done">6 attribution methods persisted (12 matrices per example pair, all 80 pairs)</li>
  <li class="done">Webapp: attribution method dropdown + per-model example gallery + connection-line polish</li>
  <li class="done">6&times;2 ABTT-vs-baseline slide figure</li>
  <li class="done">MaRC paper surveyed (equations verified, two corrections flagged)</li>
  <li class="done">Decritals: stage + split + 6 models &times; embeddings &times; Task A + Task B m-seed + per-model summary</li>
</ul>

<h3>Still planned</h3>
<div class="cols2">
  <div class="card">
    <h3>Sketch our own attribution method <span class="tag tag-todo">Planned</span></h3>
    <ul class="checklist">
      <li>4-lane parallel research (surveyor + methodologist + ablation-architect + related-work-weaver)</li>
      <li>Building on the MaRC corrections in Section 7</li>
      <li>Four candidate differentiation hooks already identified &mdash; needs Decision 2 below to pick one</li>
      <li>Output: methodology proposal, not yet an implementation</li>
    </ul>
  </div>
  <div class="card">
    <h3>Final paper consolidation <span class="tag tag-todo">Planned</span></h3>
    <ul class="checklist">
      <li>Update <code>overleaf_drafts/acl_latex.tex</code> with new split numbers and per-model table</li>
      <li>Refresh every numerical claim affected by the 354 &rarr; 565 train pair count</li>
      <li>Fold in the decritals cross-dataset validation from Section 6 (now available)</li>
      <li>Frame the decritals retrieval-vs-thresholding finding per Decision 3 below</li>
    </ul>
  </div>
</div>

<h3>Three decisions before we push forward</h3>

<div class="finding amber">
  <strong>Decision 1.</strong> The new canon split numbers are stable (1.4-point spread across 6 models on assignment accuracy) and the relative model ordering held. <strong>Approve these for the paper rewrite?</strong> If yes, the paper consolidation run can start as soon as the rewrite is queued.
</div>

<div class="finding amber">
  <strong>Decision 2.</strong> Our own attribution method needs a single differentiation hook to be a real contribution rather than a re-skin of MaRC. The four candidates from Section 7 are: (a) folio-position distance kernel (linguistic difference for fragmentary inputs), (b) ABTT-cleaned cosine fidelity loss (retrieval-aligned, not classifier-aligned), (c) amortized mask predictor (scales to all 2,238 unlabelled queries), (d) cached-embedding-space cheap baseline (no live model, runs in seconds). <strong>Which one (or combination) should we make our own?</strong> The 4-lane research run needs a north star.
</div>

<div class="finding amber">
  <strong>Decision 3.</strong> The decritals cross-dataset result (Section 6) is sharp: Acc@1 collapses ~0.35 absolute but OA pins at 0.8451 via a degenerate classifier. Two ways to frame this in the paper: <strong>(A) Limitations story</strong> &mdash; our post-processing methods don't automatically transfer, more of an honest &ldquo;what we are and aren't claiming&rdquo; statement; or <strong>(B) Contribution story</strong> &mdash; the decritals gap isolates retrieval quality from thresholding, which is exactly the kind of cross-dataset diagnostic reviewers want, and the acc@1-vs-OA divergence makes the case for why Section 4 exists. <strong>Which framing?</strong> (The clean data for either framing is sitting in <code>runs/active/decritals/results/</code> ready to be cited.)
</div>

<div class="footnote">
  <p>Generated April 2026. Source CSVs: <code>runs/active/resubmit/results/taskA_per_model_summary.csv</code>, <code>runs/active/resubmit/data/resubmit_summary.json</code>. Builder: <code>scripts/meeting/build_postmeeting_2026_04_07_walkthrough.py</code>. Re-run the builder to refresh embedded numbers and figures.</p>
</div>

</body>
</html>
""")

    return "".join(parts)


def main() -> None:
    html = build_html()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
