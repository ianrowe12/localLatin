#!/bin/bash
set -euo pipefail

DEFAULT_WORK="${WORK:-$HOME}"
REPO_ROOT="${REPO_ROOT:-$DEFAULT_WORK/localLatin}"
RUNS_ROOT="${RUNS_ROOT:-$REPO_ROOT/runs}"
PHASE7_ROOT="${PHASE7_ROOT:-$RUNS_ROOT/phase7_results}"
META_CSV="${META_CSV:-$RUNS_ROOT/ff1_lata_postact/meta.csv}"

MODELS="${MODELS:-bowphs/LaTa,bowphs/PhilTa}"
REPRS="${REPRS:-hidden,ff1}"
LAYERS_HIDDEN="${LAYERS_HIDDEN:-0-12}"
LAYERS_FF1="${LAYERS_FF1:-1-12}"

MAX_LENGTH="${MAX_LENGTH:-512}"
BATCH_SIZE="${BATCH_SIZE:-12}"
SIF_A="${SIF_A:-0.001}"
ABTT_D="${ABTT_D:-10}"

MUSTS_LANGS="${MUSTS_LANGS:-sinhala,tamil,english,french}"
MUSTS_MODEL="${MUSTS_MODEL:-xlm-roberta-base}"
MUSTS_BATCH="${MUSTS_BATCH:-32}"
MUSTS_MAXLEN="${MUSTS_MAXLEN:-128}"
PHASE7_USE_SIF="${PHASE7_USE_SIF:-1}"

if [ ! -d "$REPO_ROOT" ]; then
  echo "ERROR: REPO_ROOT not found: $REPO_ROOT"
  exit 1
fi

cd "$REPO_ROOT"

if [ ! -f "$META_CSV" ]; then
  echo "ERROR: META_CSV not found: $META_CSV"
  echo "Run src/index_canon_cli.py first or set META_CSV."
  exit 1
fi

echo "Phase 7 sweep starting..."
python scripts/run_layer_sweep.py \
  --meta_csv "$META_CSV" \
  --runs_root "$RUNS_ROOT" \
  --out_dir "$PHASE7_ROOT" \
  --models "$MODELS" \
  --reprs "$REPRS" \
  --layers_hidden "$LAYERS_HIDDEN" \
  --layers_ff1 "$LAYERS_FF1" \
  --max_length "$MAX_LENGTH" \
  --batch_size "$BATCH_SIZE" \
  --sif_a "$SIF_A" \
  --D "$ABTT_D"

SWEEP_RUN_DIR="$(python - <<'PY'
import os
from pathlib import Path

root = Path(os.environ.get("PHASE7_ROOT", "") or "")
if not root.exists():
    raise SystemExit(1)
dirs = [p for p in root.iterdir() if p.is_dir()]
if not dirs:
    raise SystemExit(1)
latest = max(dirs, key=lambda p: p.stat().st_mtime)
print(str(latest))
PY
)"

echo "Phase 7 sweep outputs: $SWEEP_RUN_DIR"
export SWEEP_RUN_DIR

python scripts/run_musts_eval.py \
  --languages "$MUSTS_LANGS" \
  --model_name "$MUSTS_MODEL" \
  --batch_size "$MUSTS_BATCH" \
  --max_length "$MUSTS_MAXLEN" \
  --out_dir "$SWEEP_RUN_DIR" \
  --sif_a "$SIF_A" \
  --phase7_D "$ABTT_D" \
  --phase7_use_sif "$PHASE7_USE_SIF"

echo ""
echo "Professor Check:"
if [ "$PHASE7_USE_SIF" = "1" ]; then
  echo "  Integration: SIF pooling enabled (a=${SIF_A})"
else
  echo "  Integration: SIF pooling disabled (a=${SIF_A})"
fi

python - <<'PY'
import os
import pandas as pd
from pathlib import Path

run_dir = Path(os.environ.get("SWEEP_RUN_DIR", ""))
layer_csv = run_dir / "phase7_layer_sweep.csv"
musts_csv = run_dir / "phase7_musts_results.csv"

df = pd.read_csv(layer_csv)
baseline = df[(df["method"] == "baseline") & (df["pooling"] == "mean")]
worst = baseline.sort_values("gap").iloc[0]
match = df[
    (df["method"] == "abtt")
    & (df["pooling"] == "mean")
    & (df["model"] == worst["model"])
    & (df["repr"] == worst["repr"])
    & (df["layer"] == worst["layer"])
]
abtt_gap = float(match.iloc[0]["gap"]) if not match.empty else float("nan")
print(
    f"  Gap analysis: worst layer {worst['model']} {worst['repr']} layer {int(worst['layer'])} "
    f"baseline gap={worst['gap']:.4f} vs abtt gap={abtt_gap:.4f}"
)

musts = pd.read_csv(musts_csv)
targets = ["sinhala", "tamil"]
print("  MUSTS (Spearman):")
for lang in targets:
    base = musts[(musts["language"] == lang) & (musts["method"] == "base_mean")]
    p7 = musts[(musts["language"] == lang) & (musts["method"].str.startswith("phase7"))]
    base_val = float(base.iloc[0]["spearman"]) if not base.empty else float("nan")
    p7_val = float(p7.iloc[0]["spearman"]) if not p7.empty else float("nan")
    print(f"    {lang}: base={base_val:.4f} phase7={p7_val:.4f}")
PY

echo "Phase 7 pipeline complete."
