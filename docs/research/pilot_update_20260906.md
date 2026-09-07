# Reviewer pilot update, 6 September 2026

For Abigail, and for Siddique's information. Everything below is live on
<https://ai.csr.uky.edu>. Nothing about how the interface works has changed
since the [26 August note](pilot_update_20260826.md); this is a data refresh.

## For Abigail, in two lines

Your two key corrections are live: `BN2123.89r.5` now sits under `Can.apost.49`
and `BN2123.89r.6` under `Can.apost.50`, so the shortlists you see are scored
against the corrected labels.

Fourteen of the 13,428 top-1 answers the pilot serves changed under the
deployed SIF+ABTT pipeline, and all fourteen fall inside the two corrected
directories, so nothing you have already reviewed elsewhere has moved.

## What that means in practice

The pilot serves 2,238 unlabelled fragments against six models, which is 13,428
first-place answers in total. The corrected labels move fourteen of them. The
other three post-processing variants, which the interface no longer exposes but
which are still shipped, move eight (`raw`), six (`sif`) and eleven (`abtt`).
The full accounting, per model and per variant, is in
[`benchmark_v1_rerun_diff.md`](benchmark_v1_rerun_diff.md) section 5.

Two things are deliberately unchanged:

- **The layer each model serves.** The corrected labels nudged one model's
  layer-selection metric by 0.23 of a point, which was enough to flip
  Qwen3-0.6B from layer 7 to layer 1 and would have rewritten 1,276 of its
  2,238 answers, more than half, from a correction that touches two files. The
  pipeline now keeps the layer already deployed unless a new one beats it by
  more than half a point of assignment accuracy, so all 24 (variant, model)
  cells kept their layer and the change stayed at fourteen answers. The paper
  reports the unmodified argmax and is unaffected by this rule.
- **The token-highlight artifacts.** All 39,647 of them are byte-identical. The
  fragments themselves were never edited, only their directory labels, so
  nothing needed recomputing.

Your reviewer notes, decisions and any directories you created are untouched.
They are keyed by fragment and directory name, not by rank or by layer, and the
feedback database sits outside everything a data deploy can write to.

## New CCL material

This refresh carries no new fragments. The corpus is frozen at benchmark v1 so
that the paper's numbers stay reproducible against a fixed set of texts.

New material from CCL will arrive by a different route. The TEI converter
(#111) now turns a CCL manuscript export into the plain-text unit files the
pilot reads, with the derivation rule written down in
[`data_derivation.md`](data_derivation.md) and verified against the existing
`BN2123` files. Your `BN2123` export alone carries roughly 220 units that are
not in the current corpus. Those, and any further exports you send, will go into
a benchmark v2 ingest rather than being dropped into the live instance one at a
time, so that the pilot and the paper keep describing the same corpus.

## Provenance

- Data release: `data-20260906-v1`, three parts, 39,660 files, all confined to
  `runs/active/`.
- Predictions and query-query matrices rebuilt on the corrected split under the
  sticky-layer rule (issue #113, epic #107).
- Deployed by the controlled path in issue #127: deploy flag enabled for the
  single dispatch, authenticated smoke run, flag disabled again.
