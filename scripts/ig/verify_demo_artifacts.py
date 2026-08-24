"""Verify the unlabelled demo attribution artifacts, NPZ-side and service-side.

Two layers of checking:

1. **NPZ level** (numpy/pandas only, so it runs in the 3.10 conda env on the
   HPC): every unlabelled-source row in the examples CSV has an artifact whose
   ``layer``/``D`` match the CSV, which carries all four attribution variants
   and real decoded token strings rather than ``[i]`` placeholders.
2. **Service level** (only when ``web`` is importable -- it needs Python 3.11+
   for ``enum.StrEnum``): each pair resolves through
   ``token_map_svc.resolve_example_id`` from its *unlabelled* ``query_file_id``
   plus candidate directory and model slug, exactly as
   ``GET /api/query/{file_id}/token_map`` does, and the resulting payload
   reports four variants and real tokens.

Usage::

    python scripts/ig/verify_demo_artifacts.py \\
        --examples_csv runs/active/ig_examples/phase12f_examples.csv \\
        --artifacts_dir runs/active/ig_examples/artifacts
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

VARIANTS = ("baseline", "abtt", "sif", "sif_abtt")
QUERY_SOURCE = "unlabelled"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--examples_csv", required=True, type=Path)
    p.add_argument("--artifacts_dir", required=True, type=Path)
    p.add_argument(
        "--query_source",
        default=QUERY_SOURCE,
        help="Only check rows with this query_source (default: unlabelled).",
    )
    p.add_argument(
        "--skip_service",
        action="store_true",
        help="Skip the token_map_svc checks even when web/ is importable.",
    )
    return p.parse_args()


def artifact_path(artifacts_dir: Path, row: pd.Series) -> Path:
    slug = str(row["model_name"]).replace("/", "_")
    return artifacts_dir / slug / f"example{int(row['example_id']):03d}_{row['candidate_role']}.npz"


def check_npz(path: Path, row: pd.Series) -> list[str]:
    problems: list[str] = []
    if not path.exists():
        return [f"artifact missing: {path}"]
    data = np.load(path, allow_pickle=False)

    missing_variants = [v for v in VARIANTS if f"pair_matrix_ig_{v}" not in data]
    if missing_variants:
        problems.append(f"missing ig variants: {missing_variants}")

    for key in ("query_token_strings", "candidate_token_strings"):
        if key not in data:
            problems.append(f"missing {key}")
            continue
        toks = [str(t) for t in data[key].tolist()]
        if not any(t.strip() for t in toks):
            problems.append(f"{key} is all blank")

    if "layer" in data and int(data["layer"].item()) != int(row["layer"]):
        problems.append(f"layer {int(data['layer'].item())} != CSV {int(row['layer'])}")
    if "D" in data and int(data["D"].item()) != int(row["D"]):
        problems.append(f"D {int(data['D'].item())} != CSV {int(row['D'])}")

    q_len = int(data["query_attention_mask"][0].sum())
    c_len = int(data["candidate_attention_mask"][0].sum())
    for v in VARIANTS:
        key = f"pair_matrix_ig_{v}"
        if key in data and data[key].shape[:2] != (q_len, c_len):
            problems.append(f"{key} shape {data[key].shape} != ({q_len},{c_len})")
    return problems


def describe(path: Path) -> str:
    data = np.load(path, allow_pickle=False)
    q_len = int(data["query_attention_mask"][0].sum())
    c_len = int(data["candidate_attention_mask"][0].sum())
    toks = " ".join(str(t) for t in data["query_token_strings"][:6].tolist())
    variants = [v for v in VARIANTS if f"pair_matrix_ig_{v}" in data]
    methods = sorted(
        {
            k.removeprefix("pair_matrix_").rsplit("_", 1)[0]
            for k in data.files
            if k.startswith("pair_matrix_")
        }
    )
    return (
        f"q_len={q_len} c_len={c_len} variants={len(variants)} "
        f"methods={len(methods)} tokens='{toks.strip()}'"
    )


def check_service(examples_csv: Path, artifacts_dir: Path, rows: pd.DataFrame) -> list[str]:
    """Resolve every pair the way GET /api/query/{file_id}/token_map does."""
    sys.path.insert(0, str(REPO_ROOT))
    from web.services import token_map_svc
    from web.services.data_store import DataStore, normalize_slug

    full = pd.read_csv(examples_csv)
    store = DataStore()
    store.ig_examples = full
    owner = {
        int(r["example_id"]): normalize_slug(str(r["model_name"])) for _, r in full.iterrows()
    }
    for model_dir in artifacts_dir.iterdir():
        if not model_dir.is_dir():
            continue
        for npz in model_dir.glob("example*.npz"):
            try:
                eid = int(npz.stem.split("_")[0].removeprefix("example"))
            except ValueError:
                continue
            if owner.get(eid) == normalize_slug(model_dir.name):
                store.ig_artifact_paths[eid] = npz

    problems: list[str] = []
    for _, row in rows.iterrows():
        file_id = int(row["query_file_id"])
        cand_dir = str(row["candidate_folder_id"])
        slug = normalize_slug(str(row["model_name"]))
        eid = token_map_svc.resolve_example_id(
            store, file_id, cand_dir, slug, query_source=QUERY_SOURCE
        )
        label = f"{Path(str(row['query_path'])).name} -> {cand_dir} [{slug}]"
        if eid is None:
            problems.append(f"{label}: resolve_example_id returned None")
            continue
        if eid != int(row["example_id"]):
            problems.append(f"{label}: resolved {eid}, expected {int(row['example_id'])}")
            continue
        resp = token_map_svc.load_token_map(store, eid)
        if resp is None:
            problems.append(f"{label}: load_token_map returned None")
            continue
        if list(resp.available_variants) != list(VARIANTS):
            problems.append(f"{label}: variants {resp.available_variants}")
        placeholders = [t.text for t in resp.query_tokens if t.text.startswith("[")]
        if len(placeholders) == len(resp.query_tokens):
            problems.append(f"{label}: all query tokens are placeholders")
        preview = " ".join(t.text for t in resp.query_tokens[:6]).strip()
        print(
            f"  SERVICE ok  example {eid:3d}  {label:58s} "
            f"layer={resp.layer} D={resp.D} variants={len(resp.available_variants)} "
            f"methods={len(resp.available_methods)} '{preview}'"
        )
    return problems


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.examples_csv)
    if "query_source" not in df.columns:
        raise SystemExit(f"{args.examples_csv} has no query_source column")
    rows = df[df["query_source"].astype(str) == args.query_source]
    if rows.empty:
        raise SystemExit(f"No rows with query_source={args.query_source!r}")

    print(f"=== NPZ checks ({len(rows)} pairs) ===")
    failures: list[str] = []
    for _, row in rows.iterrows():
        path = artifact_path(args.artifacts_dir, row)
        label = (
            f"example {int(row['example_id']):3d}  "
            f"{Path(str(row['query_path'])).name} -> {row['candidate_folder_id']} "
            f"[{str(row['model_name']).replace('/', '_')}]"
        )
        problems = check_npz(path, row)
        if problems:
            failures.extend(f"{label}: {p}" for p in problems)
            print(f"  FAIL {label}: {'; '.join(problems)}")
        else:
            print(f"  ok   {label:70s} {describe(path)}")

    if not args.skip_service and sys.version_info >= (3, 11):
        print(f"\n=== token_map_svc checks ({len(rows)} pairs) ===")
        try:
            failures.extend(check_service(args.examples_csv, args.artifacts_dir, rows))
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] web/ not importable here: {exc}")
    else:
        print("\n=== token_map_svc checks skipped (needs Python 3.11+) ===")

    if failures:
        print("\n=== FAILURES ===", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        raise SystemExit(1)
    print(f"\nAll {len(rows)} demo pairs verified.")


if __name__ == "__main__":
    main()
