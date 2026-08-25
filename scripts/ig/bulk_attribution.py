"""Shared logic for the full-corpus IG attribution run (issue #84).

What the bulk run covers
------------------------
Every unlabelled review-queue query (2,238) against the *union of top-1
directories across the four deployed variants*, for each of the six served
models. The demo run (#86) did four queries by hand; this generalises it.

The unit of an artifact is ``(query, candidate_dir, model, layer)``, not
``(query, candidate_dir, model)``. That is the one place this deviates from the
issue text, and it is forced by the data: the four variants do **not** share a
layer on four of the six models.

    model        raw    abtt   sif    sif_abtt
    LaTa         L1     L1     L1     L1
    PhilTa       L1     L1     L1     L1
    mT5-base     L6     L1     L3     L1
    LaBSE        L11    L11    L12    L11
    Qwen3-0.6B   L28    L5     L28    L7
    KaLM-mini    L22    L1     L23    L1

Each variant's layer is the one its own deployed ranking was computed at, so a
single artifact cannot honestly carry all four panels: three of KaLM's four
variants would be explained at a layer their ranking never used. That is the
same class of mistake #87 fixed for the ABTT subspace, one axis over. So an
artifact is built at one layer and carries exactly the variants deployed at that
layer, and the resolver picks the artifact by the variant the reviewer is
looking at.

That also makes the run cheaper than a naive reading suggests: an artifact whose
variant set needs only ``baseline`` IG (or only ``abtt`` IG) runs two integrated
gradients passes instead of four.

What each artifact stores
-------------------------
Only what ``web/services/token_map_svc.py`` serves for the ``ig`` method:

    similarity_matrix            cos(query token, candidate token), float16
    pair_matrix_ig_<variant>     one per variant present, float16
    topk_ig_<variant>_{query,candidate}
    query/candidate_ig_<variant> per-token IG scores, float32
    query/candidate_sif_weights  mean-1 normalised SIF weights, float32
    query/candidate_token_strings, query/candidate_input_ids
    layer, D, D_sif, and the per-variant cleaner provenance

Not stored, and why:

* **the other five methods** (bertscore, ot, attention_weighted, dla,
  attention_standalone). 6 methods x 4 variants of dense QxC grids is what makes
  the demo artifacts 4.5 MB each; at 40k artifacts that is 180 GB.
* **raw token hidden states and attention matrices**. The hidden states exist in
  the demo artifacts so the pair matrices can be rebuilt CPU-only against a
  different cleaner, which is exactly how #88 fixed 128 artifacts without a GPU.
  At this scale they are 768 KB of the 4.5 MB, and the serving path only ever
  used them to derive ``similarity_matrix`` -- which is stored directly instead,
  at a sixth of the size. The tradeoff is real and deliberate: **a future
  cleaner change means re-running the GPU job, not a CPU rewrite.**
* **the PC matrices** (``pcs`` / ``pcs_sif``). Identical for every artifact of a
  given ``(model, layer)``, so 45 KB x 40k of duplication. The artifact records
  ``D``, ``D_sif``, the per-variant cleaner pooling and a SHA-1 of each cleaner,
  and the cleaners themselves live once per ``(model, layer)`` in the bulk PC
  file.

Fidelity notes
--------------
* **SIF weights** are estimated the way the *deployed* SIF pooling estimated
  them (``src/extract_hidden_cli.py``): the whole labelled train split, at
  ``max_length`` 512, through the ``tokenizer_empty`` keep-lookup. The paper-set
  artifacts drop each evaluated pair's own text from that corpus, which is right
  for a test-split metric and wrong here -- the bulk panels explain a live
  ranking whose own weights came from the undropped corpus, and the candidates
  are labelled train files, so dropping them would empty the corpus.
* **ABTT cleaners** are fit per pooling with the deployed recipe
  (``pooling_cleaners.fit_cleaner``), into a bulk-private PC root. Nothing reads
  or writes ``runs/phase12_release/pcs``, which sidesteps the two stale mean
  fits at ``bowphs_LaTa/layer4`` and ``bowphs_PhilTa/layer6``; neither layer is
  deployed anyway. ``scripts/ig/fit_pooling_cleaners.py --reseed_stale_mean``
  repairs those two files separately.
* **IG scores** for ``sif`` / ``sif_abtt`` are the ``baseline`` / ``abtt`` IG
  scaled by the SIF weights, the same convention
  ``scripts/resubmit/persist_sif_attribution.py`` established: SIF-ness enters
  through the token weights, not through a re-run of the gradient.
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts" / "resubmit", Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from canon_retrieval import l2_normalize  # noqa: E402

from run_resubmit_unlabelled_retrieval import VARIANTS as CSV_VARIANTS  # noqa: E402
from run_resubmit_unlabelled_retrieval import model_slug  # noqa: E402

from pooling_cleaners import (  # noqa: E402
    Cleaner,
    fit_cleaner,
    load_pooled_embeddings,
    pc_file_path,
    read_cleaner,
    write_cleaner,
)

# The prediction CSVs call the uncorrected variant "raw"; the artifacts call it
# "baseline". This is the only place the bulk pipeline bridges the two, mirroring
# `toAttributionVariant` on the frontend.
CSV_TO_ARTIFACT_VARIANT: dict[str, str] = {
    "raw": "baseline",
    "abtt": "abtt",
    "sif": "sif",
    "sif_abtt": "sif_abtt",
}
ARTIFACT_TO_CSV_VARIANT: dict[str, str] = {v: k for k, v in CSV_TO_ARTIFACT_VARIANT.items()}

# Render order, matching token_map_svc.ATTRIBUTION_VARIANTS.
ARTIFACT_VARIANT_ORDER: tuple[str, ...] = ("baseline", "abtt", "sif", "sif_abtt")

# Which artifact variants need which IG target. An artifact only runs the passes
# its own variants use.
BASELINE_IG_VARIANTS = frozenset({"baseline", "sif"})
ABTT_IG_VARIANTS = frozenset({"abtt", "sif_abtt"})

# Priority when a directory is top-1 under several variants: the candidate file
# inside the directory is chosen by the highest-priority one, because that is the
# variant whose score the recomputation is checked against. sif_abtt first, since
# it is the webapp default.
VARIANT_PRIORITY: tuple[str, ...] = ("sif_abtt", "sif", "abtt", "raw")

# Chunk order from issue #84. Also fixes the example_id ranges, so an id always
# decodes to one model and reruns are stable.
MODEL_PRIORITY: tuple[str, ...] = (
    "bowphs/LaTa",
    "bowphs/PhilTa",
    "google/mt5-base",
    "sentence-transformers/LaBSE",
    "Qwen/Qwen3-Embedding-0.6B",
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
)

MODEL_TYPES: dict[str, str] = {
    "bowphs/LaTa": "t5",
    "bowphs/PhilTa": "t5",
    "google/mt5-base": "t5",
    "sentence-transformers/LaBSE": "bert",
    "Qwen/Qwen3-Embedding-0.6B": "decoder",
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5": "decoder",
}

MODEL_SHORT: dict[str, str] = {
    "bowphs/LaTa": "LaTa",
    "bowphs/PhilTa": "PhilTa",
    "google/mt5-base": "mT5-base",
    "sentence-transformers/LaBSE": "LaBSE",
    "Qwen/Qwen3-Embedding-0.6B": "Qwen3-0.6B",
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5": "KaLM-mini",
}

# example_id space. The paper artifacts occupy 1..128, so the bulk run starts far
# above them and gives each model its own 100k block. 40k artifacts total, so the
# blocks are ~10x larger than they need to be and can absorb a re-enumeration.
EXAMPLE_ID_BASE = 1_000_000
EXAMPLE_ID_STRIDE = 100_000

# Marks bulk rows in phase12f_examples.csv. The webapp gallery excludes this
# bucket -- 40k cards is not a gallery, and building it would open every NPZ.
BULK_BUCKET = "unlabelled_bulk"
QUERY_SOURCE = "unlabelled"
CANDIDATE_ROLE = "pair_example"


def example_id_block(model_name: str) -> int:
    if model_name not in MODEL_PRIORITY:
        raise KeyError(f"Unknown model {model_name!r}; known: {list(MODEL_PRIORITY)}")
    return EXAMPLE_ID_BASE + MODEL_PRIORITY.index(model_name) * EXAMPLE_ID_STRIDE


# ---------------------------------------------------------------------------
# Pair enumeration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BulkPair:
    """One artifact to build: a query, a candidate directory, and a layer."""

    example_id: int
    filename: str
    file_id: int
    query_row: int  # row index into meta_unlabelled.csv
    candidate_dir: str
    layer: int
    # Variants deployed at this layer -- the panels the artifact carries.
    variants: tuple[str, ...]
    # Variants (artifact vocabulary) under which this directory is top-1 at this
    # layer. Always a subset of `variants`, never empty.
    top1_variants: tuple[str, ...]
    # Highest-priority CSV variant in `top1_variants`, and the rank1_score it
    # reported. The candidate file is chosen and checked against these.
    check_variant: str
    check_score: float

    def artifact_path(self, artifacts_dir: Path, slug: str) -> Path:
        return Path(artifacts_dir) / slug / f"example{self.example_id}_{CANDIDATE_ROLE}.npz"


def layer_variant_map(frames: dict[str, pd.DataFrame], model_name: str) -> dict[int, tuple[str, ...]]:
    """``{layer: artifact variants deployed at that layer}`` for one model."""
    by_layer: dict[int, list[str]] = {}
    for csv_variant, df in frames.items():
        sub = df[df["model"] == model_name]
        if sub.empty:
            continue
        layers = sorted({int(x) for x in sub["layer"].unique()})
        if len(layers) != 1:
            raise SystemExit(
                f"{model_name}/{csv_variant}: predictions span layers {layers}; "
                "a variant must have exactly one deployed layer."
            )
        by_layer.setdefault(layers[0], []).append(CSV_TO_ARTIFACT_VARIANT[csv_variant])
    return {
        layer: tuple(v for v in ARTIFACT_VARIANT_ORDER if v in set(names))
        for layer, names in sorted(by_layer.items())
    }


def enumerate_pairs(
    frames: dict[str, pd.DataFrame],
    model_name: str,
    filename_to_row: dict[str, int],
    file_ids: dict[str, int],
) -> list[BulkPair]:
    """Every ``(query, top-1 directory, layer)`` this model needs an artifact for.

    Sorted by ``(filename, candidate_dir, layer)`` and numbered from the model's
    example_id block, so the id of a pair does not depend on how much of the run
    has already happened. A resumed chunk lands on the same ids.
    """
    lv = layer_variant_map(frames, model_name)
    layer_of: dict[str, int] = {}
    for layer, names in lv.items():
        for name in names:
            layer_of[ARTIFACT_TO_CSV_VARIANT[name]] = layer

    # (filename, dir, layer) -> {csv_variant: rank1_score}
    hits: dict[tuple[str, str, int], dict[str, float]] = {}
    for csv_variant, df in frames.items():
        sub = df[df["model"] == model_name]
        layer = layer_of[csv_variant]
        for fname, cand_dir, score in zip(
            sub["filename"].astype(str), sub["rank1_dir"], sub["rank1_score"]
        ):
            if not isinstance(cand_dir, str) or not cand_dir or pd.isna(score):
                continue  # a query the degenerate-file guard left unpredicted
            hits.setdefault((fname, cand_dir, layer), {})[csv_variant] = float(score)

    block = example_id_block(model_name)
    pairs: list[BulkPair] = []
    for idx, key in enumerate(sorted(hits)):
        fname, cand_dir, layer = key
        scored = hits[key]
        check_variant = next(v for v in VARIANT_PRIORITY if v in scored)
        pairs.append(
            BulkPair(
                example_id=block + idx,
                filename=fname,
                file_id=file_ids[fname],
                query_row=filename_to_row[fname],
                candidate_dir=cand_dir,
                layer=layer,
                variants=lv[layer],
                top1_variants=tuple(
                    v for v in ARTIFACT_VARIANT_ORDER
                    if ARTIFACT_TO_CSV_VARIANT[v] in scored
                ),
                check_variant=check_variant,
                check_score=scored[check_variant],
            )
        )
    return pairs


def load_variant_frames(unlabelled_root: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for csv_variant in CSV_VARIANTS:
        path = Path(unlabelled_root) / f"unlabelled_predictions_{csv_variant}.csv"
        if not path.exists():
            raise SystemExit(f"Missing variant predictions CSV: {path}")
        frames[csv_variant] = pd.read_csv(path)
    return frames


# ---------------------------------------------------------------------------
# Per-(model, layer) embedding + cleaner context
# ---------------------------------------------------------------------------


def cleaner_sha1(cleaner: Cleaner) -> str:
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(cleaner.pcs, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(cleaner.mean_vec, dtype=np.float32).tobytes())
    return h.hexdigest()


@dataclass
class LayerContext:
    """Embeddings and both ABTT cleaners for one ``(model, layer)``.

    ``vectors(variant)`` returns the L2-normalised labelled and unlabelled
    matrices the deployed ranking scored with, so the candidate file inside a
    predicted directory can be re-derived and checked against ``rank1_score``.
    """

    slug: str
    layer: int
    cleaners: dict[str, Cleaner]
    _raw: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    _cleaned: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def vectors(self, csv_variant: str) -> tuple[np.ndarray, np.ndarray]:
        pooling, apply_abtt, _ = CSV_VARIANTS[csv_variant]
        source = self._cleaned if apply_abtt else self._raw
        return source[pooling]


def build_layer_context(
    slug: str,
    layer: int,
    labelled_bases: Path,
    unlabelled_bases: Path,
    split: pd.DataFrame,
    pc_root: Path,
    write_pcs: bool = True,
) -> LayerContext:
    """Fit (or read) both poolings' cleaners and pre-normalise every variant space.

    The cleaners are the deployed ones: labelled TRAIN split only, D swept by
    ``find_optimal_D``, one fit per pooling -- the recipe
    ``run_resubmit_unlabelled_retrieval.main()`` uses and
    ``pooling_cleaners.fit_cleaner`` encodes. ``pc_root`` is the bulk run's own,
    so a cached fit is reused across chunks without touching the shared paper
    PC files.
    """
    ctx = LayerContext(slug=slug, layer=layer, cleaners={})
    pc_path = pc_file_path(pc_root, slug, layer)
    cached: dict[str, Cleaner] = {}
    if pc_path.exists():
        with np.load(pc_path, allow_pickle=False) as data:
            for pooling in ("mean", "sif"):
                found = read_cleaner(data, pooling)
                if found is not None:
                    cached[pooling] = found

    for pooling in ("mean", "sif"):
        lab = load_pooled_embeddings(
            labelled_bases, slug, pooling, layer, n_expected=len(split)
        )
        unlab = load_pooled_embeddings(unlabelled_bases, slug, pooling, layer)
        cleaner = cached.get(pooling)
        if cleaner is None:
            cleaner = fit_cleaner(labelled_bases, slug, pooling, layer, split)
        else:
            print(f"[{slug} L{layer}] {pooling} cleaner from cache (D={cleaner.D})")
        ctx.cleaners[pooling] = cleaner
        ctx._raw[pooling] = (l2_normalize(lab), l2_normalize(unlab))
        ctx._cleaned[pooling] = (
            l2_normalize(cleaner.clean_tokens(lab)),
            l2_normalize(cleaner.clean_tokens(unlab)),
        )

    if write_pcs and set(cached) != {"mean", "sif"}:
        arrays: dict[str, np.ndarray] = {}
        if pc_path.exists():
            with np.load(pc_path, allow_pickle=False) as data:
                arrays = {k: data[k] for k in data.files}
        for pooling in ("mean", "sif"):
            write_cleaner(arrays, ctx.cleaners[pooling])
        pc_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = pc_path.with_name(pc_path.stem + ".tmp.npz")
        np.savez(tmp, **arrays)
        tmp.replace(pc_path)
        print(
            f"[{slug} L{layer}] wrote {pc_path} "
            f"(D={ctx.cleaners['mean'].D}, D_sif={ctx.cleaners['sif'].D})"
        )
    return ctx


# ---------------------------------------------------------------------------
# Registry rows
# ---------------------------------------------------------------------------

# Column order of the bulk registry, a superset-compatible subset of
# phase12f_examples.csv. register_bulk_examples.py reindexes onto whatever the
# canonical CSV actually has.
REGISTRY_COLUMNS: tuple[str, ...] = (
    "example_id",
    "model_name",
    "model_short",
    "model_type",
    "layer",
    "method",
    "repr",
    "pooling",
    "D",
    "query_index",
    "candidate_index",
    "query_file_id",
    "candidate_file_id",
    "query_path",
    "candidate_path",
    "query_folder_id",
    "candidate_folder_id",
    "candidate_label",
    "candidate_role",
    "gold_similar",
    "baseline_pred",
    "abtt_pred",
    "bucket",
    "query_source",
    "top1_variants",
    "variants_available",
    "methods_available",
)


def registry_row(
    pair: BulkPair,
    model_name: str,
    candidate_index: int,
    candidate_file_id: int,
    query_path: Path,
    candidate_path: Path,
    d_mean: int,
) -> dict:
    """One phase12f_examples.csv row for a built artifact.

    ``variants_available`` is the load-bearing new column: it is what lets the
    resolver pick the artifact built at the layer the requested variant is
    deployed at, instead of whichever ``(file_id, dir, model)`` row it happens to
    hit first. ``gold_similar`` is 0 because an unlabelled query has no gold
    directory; ``baseline_pred`` / ``abtt_pred`` are truthful about whether this
    directory was that variant's top-1.
    """
    return {
        "example_id": pair.example_id,
        "model_name": model_name,
        "model_short": MODEL_SHORT[model_name],
        "model_type": MODEL_TYPES[model_name],
        "layer": pair.layer,
        "method": "abtt_optimal",
        "repr": "hidden",
        "pooling": "mean",
        "D": d_mean,
        "query_index": -1,
        "candidate_index": candidate_index,
        "query_file_id": pair.file_id,
        "candidate_file_id": candidate_file_id,
        "query_path": str(query_path),
        "candidate_path": str(candidate_path),
        "query_folder_id": "",
        "candidate_folder_id": pair.candidate_dir,
        "candidate_label": pair.candidate_dir,
        "candidate_role": CANDIDATE_ROLE,
        "gold_similar": 0,
        "baseline_pred": int("baseline" in pair.top1_variants),
        "abtt_pred": int("abtt" in pair.top1_variants),
        "bucket": BULK_BUCKET,
        "query_source": QUERY_SOURCE,
        "top1_variants": ",".join(pair.top1_variants),
        "variants_available": ",".join(pair.variants),
        "methods_available": "ig",
    }


def slug_for(model_name: str) -> str:
    return model_slug(model_name)
