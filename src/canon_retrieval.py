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


def l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


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
    is_winnable: Sequence[bool],
    k: int,
) -> float:
    folder_ids = np.array(folder_ids)
    is_winnable = np.array(is_winnable, dtype=bool)
    n = sim.shape[0]
    correct = 0
    total = int(is_winnable.sum())
    for i in range(n):
        if not is_winnable[i]:
            continue
        scores = sim[i].copy()
        scores[i] = -np.inf
        topk_idx = np.argpartition(-scores, k)[:k]
        if np.any(folder_ids[topk_idx] == folder_ids[i]):
            correct += 1
    return correct / total if total > 0 else 0.0


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
