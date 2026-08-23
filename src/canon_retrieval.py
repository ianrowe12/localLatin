import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CanonMetaStats:
    total_files: int
    total_folders: int
    singleton_folders: int
    winnable_files: int


def list_txt_files(canon_root: str) -> List[Tuple[str, str, str]]:
    """Return sorted list of (folder_id, filename, path)."""
    entries: List[Tuple[str, str, str]] = []
    for dirpath, _, filenames in os.walk(canon_root):
        txt_files = [f for f in filenames if f.lower().endswith(".txt")]
        if not txt_files:
            continue
        folder_id = os.path.basename(dirpath)
        for fname in txt_files:
            path = os.path.join(dirpath, fname)
            entries.append((folder_id, fname, path))
    entries.sort(key=lambda x: (x[0], x[1]))
    return entries


def build_meta(canon_root: str, output_csv: str) -> pd.DataFrame:
    entries = list_txt_files(canon_root)
    df = pd.DataFrame(entries, columns=["folder_id", "filename", "path"])
    folder_sizes = df.groupby("folder_id")["filename"].transform("count")
    df["folder_size"] = folder_sizes
    df["is_singleton"] = df["folder_size"] == 1
    df["is_winnable"] = df["folder_size"] >= 2
    df = df.reset_index(drop=True)
    df["file_id"] = np.arange(len(df), dtype=np.int32)
    df.to_csv(output_csv, index=False)
    return df


def meta_stats(meta: pd.DataFrame) -> CanonMetaStats:
    total_files = int(len(meta))
    total_folders = int(meta["folder_id"].nunique())
    singleton_folders = int((meta["folder_size"] == 1).sum())
    winnable_files = int(meta["is_winnable"].sum())
    return CanonMetaStats(
        total_files=total_files,
        total_folders=total_folders,
        singleton_folders=singleton_folders,
        winnable_files=winnable_files,
    )


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_texts(paths: Sequence[str]) -> List[str]:
    return [load_text(p) for p in paths]


def token_lengths(tokenizer, texts: Sequence[str], max_length: int = 512) -> np.ndarray:
    enc = tokenizer(
        list(texts),
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
    )
    lengths = np.array([int(np.sum(mask)) for mask in enc["attention_mask"]], dtype=np.int32)
    return lengths


def mean_pool(hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask.astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)
    masked = hidden * mask
    denom = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
    return masked.sum(axis=1) / denom


def last_token_pool(hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    lengths = attention_mask.sum(axis=1).astype(np.int64)
    last_idx = np.maximum(lengths - 1, 0)
    batch_idx = np.arange(hidden.shape[0])
    return hidden[batch_idx, last_idx]


def l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize along ``axis``.

    Note on degenerate rows: the ``eps`` floor means a zero vector is mapped to
    a zero vector (``0 / eps == 0``), *not* to NaN. Cosine scores against such a
    row are therefore 0.0 rather than undefined, which silently hides the fact
    that the row carries no signal. Worse, if principal-component removal with
    ``center=True`` runs first (see :class:`sif_abtt.EmbeddingCleaner`), every
    zero vector becomes the same non-zero vector ``-mean_vec``, so two unrelated
    empty documents normalize to *identical* directions and score a spurious
    cosine of exactly 1.0. Callers must therefore detect degenerate rows on the
    **pre-ABTT** embeddings with :func:`zero_norm_mask` and exclude them; see
    :func:`build_directory_index` and :func:`top_k_directories`.
    """
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


# --- Degenerate-embedding guards (issue #66) --------------------------------

ZERO_NORM_EPS = 1e-8


def zero_norm_mask(embeddings: np.ndarray, eps: float = ZERO_NORM_EPS) -> np.ndarray:
    """Boolean mask of rows that carry no usable direction.

    ``True`` where a row's L2 norm is <= ``eps`` or is not finite (NaN / inf).
    Such rows must never take part in cosine retrieval: normalizing them yields
    either the zero vector or NaN, and after mean-centering they all collapse
    onto a single shared direction (spurious cosine 1.0).

    Works on 1-D (single vector) and 2-D (n_rows, dim) input; a 1-D input
    returns a 1-element mask.
    """
    arr = np.asarray(embeddings, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"zero_norm_mask expects a 1-D or 2-D array, got ndim={arr.ndim}.")
    nonfinite = ~np.isfinite(arr).all(axis=1)
    norms = np.linalg.norm(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), axis=1)
    return nonfinite | (norms <= eps)


def blank_text_mask(texts: Sequence) -> np.ndarray:
    """Boolean mask of entries that are empty, whitespace-only, or missing.

    Source-level companion to :func:`zero_norm_mask`: an empty ``.txt`` file has
    no content to embed, so whatever vector the model emits for it is an artefact
    of the tokenizer's special tokens rather than of the document. Two such files
    can be byte-identical after tokenization and score a genuine-looking cosine of
    1.0 even under mean pooling, where the norm is not zero.
    """
    out = np.zeros(len(texts), dtype=bool)
    for i, t in enumerate(texts):
        if t is None:
            out[i] = True
            continue
        if isinstance(t, float) and np.isnan(t):  # pandas NaN for a missing cell
            out[i] = True
            continue
        out[i] = not str(t).strip()
    return out


def build_directory_index(
    folder_ids: Sequence,
    exclude_refs: np.ndarray = None,
) -> Tuple[dict, List[str], List[str]]:
    """Group reference indices by directory, dropping excluded reference files.

    Parameters
    ----------
    folder_ids:
        Directory label per reference file, positionally aligned with the
        reference embedding matrix.
    exclude_refs:
        Optional boolean mask (same length) marking reference files to exclude.

    Returns
    -------
    (dir_to_indices, dropped_dirs, excluded_files_by_dir_order)
        ``dir_to_indices`` maps directory name -> usable reference row indices,
        containing only directories with at least one usable file.
        ``dropped_dirs`` lists, in first-seen order, directories that lost every
        one of their files and are therefore not retrievable at all.
        The third element lists the directory of each excluded reference file,
        in reference order, for logging.
    """
    n = len(folder_ids)
    if exclude_refs is None:
        exclude_refs = np.zeros(n, dtype=bool)
    else:
        exclude_refs = np.asarray(exclude_refs, dtype=bool)
        if exclude_refs.shape != (n,):
            raise ValueError(
                f"exclude_refs has shape {exclude_refs.shape}, expected ({n},)."
            )

    all_dirs: List[str] = []
    dir_to_indices: dict = {}
    excluded_dirs: List[str] = []
    for i, fid in enumerate(folder_ids):
        name = str(fid)
        if name not in dir_to_indices:
            dir_to_indices[name] = []
            all_dirs.append(name)
        if exclude_refs[i]:
            excluded_dirs.append(name)
            continue
        dir_to_indices[name].append(i)

    dropped_dirs = [d for d in all_dirs if not dir_to_indices[d]]
    for d in dropped_dirs:
        del dir_to_indices[d]
    return dir_to_indices, dropped_dirs, excluded_dirs


def top_k_directories(
    query_sims: np.ndarray,
    dir_to_indices: dict,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    """Top-``top_k`` (directory, max-cosine) pairs for one query.

    ``dir_to_indices`` must already have excluded reference rows removed (see
    :func:`build_directory_index`), so every listed directory has >= 1 usable
    file and ``np.max`` is well defined.

    Ties are broken by ``dir_to_indices`` insertion order (Python's sort is
    stable). This is deliberate: it reproduces the pre-guard ordering byte for
    byte on queries the guard does not touch.
    """
    dir_scores: List[Tuple[str, float]] = []
    for dir_name, file_indices in dir_to_indices.items():
        if not file_indices:
            continue
        dir_scores.append((dir_name, float(np.max(query_sims[file_indices]))))
    dir_scores.sort(key=lambda x: x[1], reverse=True)
    return dir_scores[:top_k]


def similarity_matrix(embeddings_norm: np.ndarray) -> np.ndarray:
    return embeddings_norm @ embeddings_norm.T


def sanity_checks(sim: np.ndarray) -> dict:
    diag_mean = float(np.mean(np.diag(sim)))
    symmetric = bool(np.allclose(sim, sim.T, atol=1e-5))
    off_diag = sim[~np.eye(sim.shape[0], dtype=bool)]
    off_diag_mean = float(np.mean(off_diag))
    return {
        "symmetric": symmetric,
        "diag_mean": diag_mean,
        "off_diag_mean": off_diag_mean,
    }


def accuracy_at_k(
    sim: np.ndarray,
    folder_ids: Sequence[str],
    query_mask: Sequence[bool],
    k: int,
) -> float:
    folder_ids = np.array(folder_ids)
    query_mask = np.array(query_mask, dtype=bool)
    n = sim.shape[0]
    correct = 0
    total = int(query_mask.sum())
    for i in range(n):
        if not query_mask[i]:
            continue
        scores = sim[i].copy()
        scores[i] = -np.inf
        topk_idx = np.argpartition(-scores, k)[:k]
        if np.any(folder_ids[topk_idx] == folder_ids[i]):
            correct += 1
    return correct / total if total > 0 else 0.0


def accuracy_at_k_all(sim: np.ndarray, folder_ids: Sequence[str], k: int) -> float:
    query_mask = np.ones(sim.shape[0], dtype=bool)
    return accuracy_at_k(sim, folder_ids, query_mask, k)


def accuracy_at_k_winnable(
    sim: np.ndarray, folder_ids: Sequence[str], is_winnable: Sequence[bool], k: int
) -> float:
    return accuracy_at_k(sim, folder_ids, is_winnable, k)


def upper_triangle(sim: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(sim.shape[0], k=1)
    return sim[idx]


def upper_triangle_labels(folder_ids: Sequence[str]) -> np.ndarray:
    folder_ids = np.array(folder_ids)
    idx = np.triu_indices(len(folder_ids), k=1)
    return (folder_ids[idx[0]] == folder_ids[idx[1]]).astype(bool)


def sweep_thresholds(sim_upper: np.ndarray, labels: np.ndarray, thresholds: Iterable[float]) -> pd.DataFrame:
    labels = labels.astype(bool)
    total_pos = int(labels.sum())
    total_neg = int(len(labels) - total_pos)
    rows = []
    for t in thresholds:
        preds = sim_upper >= t
        tp = int(np.sum(preds & labels))
        fp = int(np.sum(preds & ~labels))
        fn = int(np.sum(~preds & labels))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        rows.append(
            {
                "threshold": float(t),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "total_pos": total_pos,
                "total_neg": total_neg,
            }
        )
    return pd.DataFrame(rows)


def save_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
