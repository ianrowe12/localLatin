# Benchmark v1 (frozen 2026-09-06)

The dataset the ARR October 2026 submission reports on. Frozen by issue #112, epic #107.

## What is in it

| | Count |
|---|---|
| Labelled files (`data/canon_labelled/`) | 1,705 |
| Labelled directories | 840 |
| Singleton directories | 545 |
| Winnable files (directory has >= 2 members) | 1,160 |
| Unlabelled query files (`data/canon_unlabelled/`) | 2,238 |
| Files in both pools | 0 |

Directory size histogram (size: directories): 1: 545, 2: 108, 3: 32, 4: 25, 5: 79,
6: 22, 7: 16, 8: 9, 9: 3, 10: 1.

`BN2123` contributes 140 labelled and 148 unlabelled files (288 units). The two pools
are disjoint by filename and by content sha256; no file appears in both.

## Corrections applied

| File | Was | Now | Provenance |
|---|---|---|---|
| `BN2123.89r.5.txt` | `Can.apost.48` | `Can.apost.49` | Prof. Firey, email 2026-09; confirmed independently by her TEI export, whose `<div type="scholSource">` for this unit reads `Can.apost.49` |
| `BN2123.89r.6.txt` | `Can.apost.49` | `Can.apost.50` | same |

Both are pure `git mv`: the texts are byte-identical to what they were, so no
re-embedding is needed (see the file_id note below).

Directory sizes after the moves: `Can.apost.48` 6 -> 5, `Can.apost.49` 6 -> 6 (one out,
one in), `Can.apost.50` 5 -> 6. Total files, directories and winnable counts are
unchanged, so the CLAUDE.md figures (1,705 files / 840 directories / 2,238 unlabelled)
still hold.

### Known, deliberately not corrected

`BN2123.104r.2.txt` sits in `CMAC.585.5` in our corpus. The 2026-09 export contains
**two** unit divs with `xml:id="BN2123.104r.2"`: `n="206"` with key `CMAC.585.5` (the
one we hold) and `n="207"` with key `CLYO.518.4` (a canon we do not hold). Our file
matches the first, so the label is right and the collision is an upstream id bug to
report, not a correction to apply. Nothing in the benchmark changes.

## Freeze manifest

`docs/research/benchmark_v1_manifest.txt` lists `sha256  path` for all 1,705 labelled
files, sorted by path, and ends with a digest over that listing:

```
485e4f222838ed9d8a9977f65ec7401c0badeda5eeca0f22ce4b8f0669b5a91d
```

```bash
python scripts/data/benchmark_manifest.py --check   # exits non-zero if the corpus moved
```

## Split

`runs/active/resubmit/data/phase_resubmit_split.csv` (gitignored, so it is not part of
this PR; regenerate it with the commands below).

The v2 split generator (`src/canon_split_v2.py`, seed 42) draws from one RNG stream in
folder-size-class order, so a directory changing size reshuffles unrelated files. Rerunning
it verbatim after the corrections moves **43 rows across 35 directories**: 6 files flip
train/test (in `Can.apost.48`, `Can.apost.5`, `Can.apost.49/50`) and 42 files change Task B
role, most of them singletons in unrelated directories whose coin flips shifted. That would
make every previously computed number incomparable for no scientific reason.

So the corrected split is produced by **carrying the previous assignment over**: every
file keeps the `split` and `taskb_role` it had (matched on filename, unique across the
corpus), and only directory-derived columns are recomputed
(`build_meta_with_carried_over_split` in `src/canon_split_v2.py`).

```bash
# Back up first: the CSV is gitignored, so this file is the only copy.
mkdir -p runs/active/resubmit/data/benchmark_v1
cp runs/active/resubmit/data/phase_resubmit_split.csv \
   runs/active/resubmit/data/benchmark_v1/phase_resubmit_split.pre_correction_backup.csv

python scripts/resubmit/run_resubmit_data_prep.py \
    --carry_over_from runs/active/resubmit/data/benchmark_v1/phase_resubmit_split.pre_correction_backup.csv \
    --out_dir runs/active/resubmit/data
```

### What the corrected split changes

| | Pre-correction | Corrected |
|---|---|---|
| Train files / test files | 847 / 858 | 847 / 858 |
| Positive pairs, train | 565 | 565 |
| Positive pairs, test | 595 | **596** |
| Test queries (`is_test_query`) | 535 | 535 |
| Task B queries / references | 429 / 429 | 429 / 429 |
| Task B queries with a reference directory | 243 | 243 |
| Winnable files | 1,160 | 1,160 |
| Directories | 840 | 840 |

Row-level diff: **12 rows change something other than `file_id`, all of them in
`Can.apost.48`, `Can.apost.49` and `Can.apost.50`** — the two moved files change
`folder_id`, the ten files that stayed behind change `folder_size`. No file changes
`split`, `taskb_role`, `is_test_query`, `has_test_partner` or `has_reference_dir`. The
single extra positive test pair is `BN2123.89r.6` joining the test files of
`Can.apost.50`.

Roles of the two corrected files, unchanged by the correction:

* `BN2123.89r.5.txt`: train, Task B role `train`. It is now a training partner for
  `Can.apost.49` instead of `Can.apost.48`.
* `BN2123.89r.6.txt`: test, Task B role `query`, `has_reference_dir` True. Its gold
  answer is now `Can.apost.50`, which has references in test, so it stays winnable.

### file_id renumbering (matters for embeddings)

`file_id` is the row index into the cached embedding matrices, and it is assigned by
sorting on `(folder_id, filename)`. The moves shift 17 rows inside the window
`file_id` 1554-1570 (`Can.apost.48`, `Can.apost.49`, `Can.apost.5`, `Can.apost.50`, in
that lexicographic order). The texts did not change, so **no re-embedding is needed**,
but any cached matrix keyed by row index must either be re-extracted or permuted:

| old | new | file | new directory |
|---|---|---|---|
| 1554 | 1559 | `BN2123.89r.5.txt` | `Can.apost.49` |
| 1555 | 1554 | `C1525.7v.5.txt` | `Can.apost.48` |
| 1556 | 1555 | `Hat42.149r.2.txt` | `Can.apost.48` |
| 1557 | 1556 | `KoeD213.10r.1.txt` | `Can.apost.48` |
| 1558 | 1557 | `Vat5845.10r.7.txt` | `Can.apost.48` |
| 1559 | 1558 | `BAV1341.7v.19.txt` | `Can.apost.49` |
| 1560 | 1570 | `BN2123.89r.6.txt` | `Can.apost.50` |
| 1561 | 1560 | `C1525.7v.6.txt` | `Can.apost.49` |
| 1562 | 1561 | `Hat42.149r.3.txt` | `Can.apost.49` |
| 1563 | 1562 | `KoeD213.10r.2.txt` | `Can.apost.49` |
| 1564 | 1563 | `Vat5845.10v.1.txt` | `Can.apost.49` |
| 1565 | 1564 | `BAV1341.6v.55.txt` | `Can.apost.5` |
| 1566 | 1565 | `C1525.4v.3.txt` | `Can.apost.5` |
| 1567 | 1566 | `Hat42.144r.5.txt` | `Can.apost.5` |
| 1568 | 1567 | `KoeD213.5r.1.txt` | `Can.apost.5` |
| 1569 | 1568 | `Vat5845.7r.8.txt` | `Can.apost.5` |
| 1570 | 1569 | `BAV1341.7v.20.txt` | `Can.apost.50` |

Issue #113 (re-run and regenerate) is where this landed, and it took the second
route: nothing was re-extracted or permuted on disk. Every consumer now aligns
cached rows to split rows by filename, using the `meta.csv` the extractor writes
into each run directory as the row-order manifest
(`src/embedding_alignment.py`). The embeddings stay byte-identical, which
`scripts/resubmit/verify_embedding_alignment.py` checks for named files across
all 19 caches. See `benchmark_v1_rerun_diff.md` for the resulting number diff.

One consequence reaches outside the paper. The reviewer pilot's predictions pick
their layer by argmax over near-tied layers, so the correction flipped
Qwen3-0.6B's deployed `sif_abtt` layer from 7 to 1 and would have rewritten 1,276
of its 2,238 reviewer-facing top-1 answers. Those CSVs now hold their deployed
layer unless a re-run beats it by more than 0.005 on the selection metric
(`scripts/resubmit/deployed_unlabelled_layers.json`); the paper's tables keep the
plain argmax. Section 5 of `benchmark_v1_rerun_diff.md` has the numbers.

## Freeze rule

1. Benchmark v1 is the 1,705 labelled files and 840 directories whose digest is above,
   plus the 2,238 unlabelled queries. Every number in the ARR October 2026 submission
   comes from it.
2. New CCL material — including the 228 extra units in the 2026-09 `BN2123` export —
   goes to the webapp / v2 corpus, never into `data/canon_labelled` for this paper.
3. Label corrections from Prof. Firey are the one admissible change to v1, because they
   fix ground truth rather than grow the dataset. Each one must be logged in the table
   above with its provenance, applied with `git mv`, followed by a manifest rebuild and a
   carried-over split regeneration, and reported in the PR that makes it.
4. Any change to the file set (additions or deletions) ends v1 and starts v2. The
   carry-over path deliberately refuses to run when files appear or disappear.
5. Re-derivation from a TEI export is documented in `data_derivation.md` and scripted in
   `scripts/data/tei_to_canon.py`; use `--dry-run` to review an export before anyone
   proposes changing the corpus.
