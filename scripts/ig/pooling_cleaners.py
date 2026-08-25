"""Per-pooling ABTT cleaners for the attribution artifact pipeline (issue #87).

Why this module exists
----------------------
The deployed retrieval serves four variants, and two of them apply ABTT::

    variant     pooling   ABTT   cleaner fitted on
    ----------  --------  -----  --------------------------------------
    raw         mean      no     -
    abtt        mean      yes    mean-pooled labelled TRAIN embeddings
    sif         SIF       no     -
    sif_abtt    SIF       yes    SIF-pooled  labelled TRAIN embeddings

``run_resubmit_unlabelled_retrieval.main()`` reloads the embedding cache with
the *variant's own* pooling before it sweeps D and fits ``EmbeddingCleaner``, so
``abtt`` and ``sif_abtt`` remove **different subspaces with independently chosen
D**. For LaTa layer 1 that is mean D=10 against SIF D=3; for PhilTa layer 1 both
sweeps land on D=10 yet the two subspaces still differ, with the last two
principal-angle cosines at 0.37 and 0.09.

Until this module existed the attribution artifacts stored a single
``(pcs, mean_vec)`` pair -- the mean-pooled fit -- and the ``sif_abtt`` panel was
a SIF reweighting of tokens cleaned with it. So the default panel in the webapp
explained a configuration the ranking never used.

The format
----------
Both the shared PC files (``<pc_root>/<slug>/layer{N}_pcs.npz``) and the
per-pair artifacts (``example*_pair_example.npz``) now key one cleaner per
pooling space:

    pooling   pcs key      mean key        D key
    --------  -----------  --------------  -------
    mean      ``pcs``      ``mean_vec``    ``D``
    sif       ``pcs_sif``  ``mean_vec_sif``  ``D_sif``

The mean keys are exactly the ones the old single-cleaner format used, so every
reader that predates this change keeps working unchanged and every file written
before it still loads: :func:`read_cleaner` returns ``None`` for a pooling whose
keys are absent, and ``D`` is inferred from ``pcs.shape[0]`` when the D key is
missing (the legacy PC files carry no D).

Issue #84 generates attribution artifacts for the full review queue. It should
use :func:`fit_cleaner` for both poolings and :func:`write_cleaner` to stamp
both into every NPZ it writes, so those 27k artifacts carry the right subspace
from the start.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts" / "resubmit") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "resubmit"))

from sif_abtt import EmbeddingCleaner  # noqa: E402

from run_resubmit_unlabelled_retrieval import D_VALUES, find_optimal_D  # noqa: E402

POOLINGS: tuple[str, ...] = ("mean", "sif")

# Layout of the canonical embedding cache
# (``runs/active/resubmit_bases/phase9_bases/<slug>/<repr>_<pooling>_tokempty/<file>``).
# ``repr`` is "hidden" for every deployed configuration, but it is a real axis of
# the cache (``ff1`` bases exist for the T5 models), so it is a parameter rather
# than a constant baked into the path helpers.
DEFAULT_REPR = "hidden"
POOLING_TAG = {"mean": "mean", "sif": "sif"}
POOLING_SUFFIX = {"mean": "", "sif": "_sif"}


def pooling_subdir(pooling: str, repr_name: str = DEFAULT_REPR) -> str:
    return f"{repr_name}_{POOLING_TAG[validate_pooling(pooling)]}_tokempty"


# Back-compat alias for callers that only ever meant the hidden-state cache.
POOLING_SUBDIR = {p: f"{DEFAULT_REPR}_{POOLING_TAG[p]}_tokempty" for p in POOLINGS}


@dataclass(frozen=True)
class CleanerKeys:
    pcs: str
    mean_vec: str
    d: str


# The mean row is the legacy single-cleaner format, kept byte-for-byte.
CLEANER_KEYS: dict[str, CleanerKeys] = {
    "mean": CleanerKeys(pcs="pcs", mean_vec="mean_vec", d="D"),
    "sif": CleanerKeys(pcs="pcs_sif", mean_vec="mean_vec_sif", d="D_sif"),
}

# Stamped into an artifact to record which pooling space the persisted
# ``sif_abtt`` matrices were cleaned in. Absent on pre-#87 artifacts, which
# were all cleaned in the mean-pooled space.
SIF_ABTT_CLEANER_KEY = "sif_abtt_cleaner_pooling"

# Unified provenance (issue #84). ``SIF_ABTT_CLEANER_KEY`` records the cleaner
# for exactly one variant, so an artifact could not say which space the *other*
# ABTT panel came from -- a reader had to know that ``abtt`` is always mean.
# These two parallel arrays say it for every variant the artifact carries, and
# ``SIF_ABTT_CLEANER_KEY`` is still written alongside them so pre-#84 readers
# keep working.
ARTIFACT_VARIANTS_KEY = "artifact_variants"
VARIANT_CLEANER_POOLING_KEY = "variant_cleaner_pooling"

# Which pooling space each deployed variant's ABTT cleaner is fit in. "" means
# the variant applies no cleaner at all.
VARIANT_CLEANER_POOLING: dict[str, str] = {
    "baseline": "",
    "abtt": "mean",
    "sif": "",
    "sif_abtt": "sif",
}


def write_variant_provenance(
    target: MutableMapping[str, np.ndarray], variants: Sequence[str]
) -> None:
    """Stamp per-variant cleaner provenance into a dict destined for ``np.savez``.

    ``variants`` is the list of variants the artifact actually carries, in the
    order the webapp renders them. Unknown variant names are rejected rather
    than recorded with an empty cleaner, since a silent "" would read as "this
    panel applies no ABTT" -- the exact class of wrong claim issue #87 fixed.
    """
    names = [str(v) for v in variants]
    unknown = [v for v in names if v not in VARIANT_CLEANER_POOLING]
    if unknown:
        raise ValueError(f"Unknown attribution variant(s): {unknown}")
    target[ARTIFACT_VARIANTS_KEY] = np.array(names, dtype="<U16")
    target[VARIANT_CLEANER_POOLING_KEY] = np.array(
        [VARIANT_CLEANER_POOLING[v] for v in names], dtype="<U8"
    )
    if "sif_abtt" in names:
        target[SIF_ABTT_CLEANER_KEY] = np.array(["sif"], dtype="<U8")


@dataclass(frozen=True)
class Cleaner:
    """A fitted ABTT cleaner, tagged with the pooling space it was fit in."""

    pooling: str
    pcs: np.ndarray  # (D, dim)
    mean_vec: np.ndarray  # (dim,)
    D: int

    def clean_tokens(self, hidden: np.ndarray) -> np.ndarray:
        """Per-token ABTT: center, then project out the top-D directions.

        Identical arithmetic to ``run_resubmit_ig_comparison.clean_tokens``;
        kept here so callers that hold a :class:`Cleaner` do not have to unpack
        it back into two arrays.
        """
        centered = np.asarray(hidden, dtype=np.float32) - self.mean_vec
        return (centered - centered @ self.pcs.T @ self.pcs).astype(np.float32)


def validate_pooling(pooling: str) -> str:
    if pooling not in CLEANER_KEYS:
        raise ValueError(f"Unknown pooling {pooling!r}; known: {sorted(CLEANER_KEYS)}")
    return pooling


def read_cleaner(data: Mapping[str, np.ndarray], pooling: str) -> Cleaner | None:
    """Read one pooling's cleaner out of an NPZ mapping, or ``None`` if absent.

    Works on ``np.load(...)`` handles and on plain dicts. ``D`` falls back to
    ``pcs.shape[0]`` so the legacy PC files -- which store ``pcs``/``mean_vec``
    and nothing else -- read cleanly.
    """
    keys = CLEANER_KEYS[validate_pooling(pooling)]
    if keys.pcs not in data or keys.mean_vec not in data:
        return None
    pcs = np.asarray(data[keys.pcs], dtype=np.float32)
    mean_vec = np.asarray(data[keys.mean_vec], dtype=np.float32)
    if keys.d in data:
        d = int(np.asarray(data[keys.d]).reshape(-1)[0])
    else:
        d = int(pcs.shape[0])
    # A file may hold more PCs than the D it was used at (the IG generator
    # slices ``pcs[:D]``), so trim rather than trusting the row count.
    return Cleaner(pooling=pooling, pcs=pcs[:d], mean_vec=mean_vec, D=d)


def write_cleaner(target: MutableMapping[str, np.ndarray], cleaner: Cleaner) -> None:
    """Stamp one pooling's cleaner into a dict destined for ``np.savez``."""
    keys = CLEANER_KEYS[validate_pooling(cleaner.pooling)]
    target[keys.pcs] = np.asarray(cleaner.pcs, dtype=np.float32)
    target[keys.mean_vec] = np.asarray(cleaner.mean_vec, dtype=np.float32)
    target[keys.d] = np.array([int(cleaner.D)], dtype=np.int32)


def embeddings_path(
    bases_root: Path, slug: str, pooling: str, layer: int, repr_name: str = DEFAULT_REPR
) -> Path:
    fname = f"{repr_name}_layer{layer}_embeddings{POOLING_SUFFIX[validate_pooling(pooling)]}.npy"
    return Path(bases_root) / slug / pooling_subdir(pooling, repr_name) / fname


def load_pooled_embeddings(
    bases_root: Path,
    slug: str,
    pooling: str,
    layer: int,
    n_expected: int | None = None,
    repr_name: str = DEFAULT_REPR,
) -> np.ndarray:
    path = embeddings_path(bases_root, slug, pooling, layer, repr_name)
    if not path.exists():
        raise SystemExit(f"Embedding cache missing: {path}")
    emb = np.load(path)
    if n_expected is not None and emb.shape[0] != n_expected:
        raise SystemExit(
            f"[{slug}] {pooling}-pooled cache at layer {layer} has {emb.shape[0]} rows "
            f"but the split has {n_expected}. Wrong --labelled_bases / --split_csv pair."
        )
    return emb


def fit_cleaner(
    bases_root: Path,
    slug: str,
    pooling: str,
    layer: int,
    split: pd.DataFrame,
    fixed_d: int | None = None,
    d_values: Sequence[int] = D_VALUES,
    verbose: bool = True,
    repr_name: str = DEFAULT_REPR,
) -> Cleaner:
    """Fit the deployed cleaner for one (model, layer, pooling).

    This is deliberately the *same* recipe as
    ``run_resubmit_unlabelled_retrieval.main()``: the labelled TRAIN split only,
    D chosen by :func:`find_optimal_D` (assignment accuracy on train), then
    ``EmbeddingCleaner(num_components=D, center=True).fit(train_emb)``.
    Degenerate (blank-source) rows are left in the fit on purpose, exactly as
    that script leaves them -- the guard there acts at scoring time, not at fit
    time, and dropping them here would perturb every direction.

    Passing ``fixed_d`` skips the sweep, for callers that already know D.
    """
    validate_pooling(pooling)
    emb = load_pooled_embeddings(
        bases_root, slug, pooling, layer, n_expected=len(split), repr_name=repr_name
    )
    train_mask = split["split"].to_numpy() == "train"
    train_emb = emb[train_mask]
    train_folders = split.loc[train_mask, "folder_id"].to_numpy()

    if fixed_d is None:
        if verbose:
            print(
                f"[{slug}] layer {layer} {pooling}-pooled: sweeping D over "
                f"{list(d_values)} on {train_emb.shape[0]} train docs"
            )
        d = int(find_optimal_D(train_emb, train_folders, list(d_values)))
    else:
        d = int(fixed_d)
    if verbose:
        print(f"[{slug}] layer {layer} {pooling}-pooled: D = {d}")

    cleaner = EmbeddingCleaner(num_components=d, center=True)
    cleaner.fit(train_emb)
    return Cleaner(
        pooling=pooling,
        pcs=np.asarray(cleaner.pcs, dtype=np.float32),
        mean_vec=np.asarray(cleaner.mean_vec, dtype=np.float32),
        D=d,
    )


def principal_angle_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosines of the principal angles between the row spaces of ``a`` and ``b``.

    Both inputs are ``(k, dim)`` bases. Returns ``min(k_a, k_b)`` singular
    values in descending order; all-ones means the two subspaces coincide.
    """
    qa = np.linalg.qr(np.asarray(a, dtype=np.float64).T)[0]
    qb = np.linalg.qr(np.asarray(b, dtype=np.float64).T)[0]
    return np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), 0.0, 1.0)


def cleaners_match(left: Cleaner | None, right: Cleaner | None, atol: float = 1e-6) -> bool:
    if left is None or right is None:
        return False
    return (
        left.D == right.D
        and left.pcs.shape == right.pcs.shape
        and left.mean_vec.shape == right.mean_vec.shape
        and np.allclose(left.pcs, right.pcs, atol=atol)
        and np.allclose(left.mean_vec, right.mean_vec, atol=atol)
    )


def pc_file_path(pc_root: Path, slug: str, layer: int) -> Path:
    return Path(pc_root) / slug / f"layer{layer}_pcs.npz"
