"""Service for loading IG artifacts and computing token-level similarity."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from web.models import AutoHighlight, TokenEntry, TokenMapExampleSummary, TokenMapResponse, TopMatch
from web.services.data_store import DataStore, normalize_slug

logger = logging.getLogger(__name__)


def _try_decode_tokens(input_ids: np.ndarray, model_slug: str) -> list[str] | None:
    """Try to decode token IDs using HuggingFace tokenizer. Returns None if unavailable."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None

    slug_to_hf = {
        "bowphs_LaTa": "bowphs/LaTa",
        "bowphs_PhilTa": "bowphs/PhilTa",
    }
    hf_id = slug_to_hf.get(model_slug)
    if hf_id is None:
        return None

    try:
        tokenizer = _get_tokenizer(hf_id)
        ids = input_ids.flatten().tolist()
        return [tokenizer.decode([tid]) for tid in ids]
    except Exception as e:
        logger.warning("Failed to decode tokens for %s: %s", model_slug, e)
        return None


@lru_cache(maxsize=4)
def _get_tokenizer(hf_id: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(hf_id)


@lru_cache(maxsize=64)
def _load_npz(path: str) -> dict[str, np.ndarray]:
    return dict(np.load(path, allow_pickle=False))


def list_examples(store: DataStore, model: str | None = None, bucket: str | None = None) -> list[TokenMapExampleSummary]:
    if store.ig_examples is None:
        return []

    df = store.ig_examples
    if model:
        slug = normalize_slug(model)
        # Filter by model short name or slug
        mask = df["model_name"].apply(lambda x: normalize_slug(str(x)) == slug)
        df = df[mask]
    if bucket:
        df = df[df["bucket"] == bucket]

    results = []
    for _, row in df.iterrows():
        eid = int(row["example_id"])
        if eid not in store.ig_artifact_paths:
            continue
        results.append(TokenMapExampleSummary(
            example_id=eid,
            model=normalize_slug(str(row["model_name"])),
            bucket=str(row["bucket"]),
            query_path=str(row.get("query_path", "")),
            candidate_path=str(row.get("candidate_path", "")),
        ))
    return results


def resolve_example_id(
    store: DataStore,
    file_id: int,
    candidate_dir: str,
    model: str | None = None,
) -> int | None:
    """Find the example_id for a (query file_id, candidate_dir) pair."""
    if store.ig_examples is None:
        return None

    df = store.ig_examples
    mask = (df["query_file_id"] == file_id) & (df["candidate_folder_id"] == candidate_dir)
    if model:
        slug = normalize_slug(model)
        mask = mask & df["model_name"].apply(lambda x: normalize_slug(str(x)) == slug)

    matches = df[mask]
    if matches.empty:
        return None

    eid = int(matches.iloc[0]["example_id"])
    return eid if eid in store.ig_artifact_paths else None


def load_token_map(store: DataStore, example_id: int) -> TokenMapResponse | None:
    if example_id not in store.ig_artifact_paths:
        return None

    npz_path = store.ig_artifact_paths[example_id]
    data = _load_npz(str(npz_path))

    # Extract metadata
    layer = int(data["layer"].item()) if "layer" in data else 0
    D = int(data["D"].item()) if "D" in data else 0

    # Get model slug from the artifact path (parent dir name)
    model_slug = normalize_slug(npz_path.parent.name)

    # Get example metadata from the CSV
    bucket = ""
    query_path = ""
    candidate_path = ""
    if store.ig_examples is not None:
        match = store.ig_examples[store.ig_examples["example_id"] == example_id]
        if len(match) > 0:
            row = match.iloc[0]
            bucket = str(row.get("bucket", ""))
            query_path = str(row.get("query_path", ""))
            candidate_path = str(row.get("candidate_path", ""))

    # Token embeddings
    query_hidden = data["query_hidden"]    # (Q, dim)
    cand_hidden = data["candidate_hidden"]  # (C, dim)
    q_len = query_hidden.shape[0]
    c_len = cand_hidden.shape[0]

    # Cosine similarity matrix
    q_norm = query_hidden / (np.linalg.norm(query_hidden, axis=1, keepdims=True) + 1e-8)
    c_norm = cand_hidden / (np.linalg.norm(cand_hidden, axis=1, keepdims=True) + 1e-8)
    sim_matrix = (q_norm @ c_norm.T).tolist()

    # IG weights
    q_ig_base = data.get("query_ig_baseline", np.zeros(q_len))
    q_ig_abtt = data.get("query_ig_abtt", np.zeros(q_len))
    c_ig_base = data.get("candidate_ig_baseline", np.zeros(c_len))
    c_ig_abtt = data.get("candidate_ig_abtt", np.zeros(c_len))

    # IG-weighted matrix (from visualization pipeline logic)
    ig_weighted = None
    if "query_ig_abtt" in data and "candidate_ig_abtt" in data:
        weight = np.sqrt(np.abs(q_ig_abtt[:q_len, None]) * np.abs(c_ig_abtt[None, :c_len]))
        sign = np.sign(q_ig_abtt[:q_len, None]) * np.sign(c_ig_abtt[None, :c_len])
        cos = np.array(sim_matrix)
        ig_weighted = (cos * weight * sign).tolist()

    # Top matches (sparse format for frontend connection lines)
    top_matches: dict[str, list[TopMatch]] = {}
    sim_arr = np.array(sim_matrix)
    for qi in range(q_len):
        row = sim_arr[qi]
        top_k = min(3, c_len)
        top_idx = np.argsort(row)[-top_k:][::-1]
        top_matches[str(qi)] = [
            TopMatch(candidate_idx=int(ci), score=float(row[ci]))
            for ci in top_idx
        ]

    # Auto-highlights: top K=5 query tokens by |IG score|
    auto_highlights = None
    if "query_ig_abtt" in data:
        abs_ig = np.abs(q_ig_abtt[:q_len])
        if abs_ig.max() > 1e-6:
            K = 5
            top_k_idx = np.argsort(abs_ig)[-K:][::-1]
            auto_highlights = []
            for qi in top_k_idx:
                qi = int(qi)
                row = sim_arr[qi]
                n_matches = min(2, c_len)
                top_ci = np.argsort(row)[-n_matches:][::-1]
                auto_highlights.append(AutoHighlight(
                    query_idx=qi,
                    ig_score=float(abs_ig[qi]),
                    matches=[TopMatch(candidate_idx=int(ci), score=float(row[ci])) for ci in top_ci],
                ))

    # Decode tokens
    query_input_ids = data.get("query_input_ids")
    cand_input_ids = data.get("candidate_input_ids")

    q_token_strs = _try_decode_tokens(query_input_ids, model_slug) if query_input_ids is not None else None
    c_token_strs = _try_decode_tokens(cand_input_ids, model_slug) if cand_input_ids is not None else None

    def _make_token_entries(count: int, decoded: list[str] | None) -> list[TokenEntry]:
        entries = []
        for i in range(count):
            text = decoded[i] if decoded and i < len(decoded) else f"[{i}]"
            is_content = len(text.strip().lstrip("▁##Ġ")) > 2
            entries.append(TokenEntry(idx=i, text=text, is_content=is_content))
        return entries

    return TokenMapResponse(
        example_id=example_id,
        model=model_slug,
        layer=layer,
        D=D,
        bucket=bucket,
        query_path=query_path,
        candidate_path=candidate_path,
        query_tokens=_make_token_entries(q_len, q_token_strs),
        candidate_tokens=_make_token_entries(c_len, c_token_strs),
        similarity_matrix=sim_matrix,
        ig_weighted_matrix=ig_weighted,
        top_matches=top_matches,
        query_ig_baseline=q_ig_base[:q_len].tolist(),
        query_ig_abtt=q_ig_abtt[:q_len].tolist(),
        candidate_ig_baseline=c_ig_base[:c_len].tolist(),
        candidate_ig_abtt=c_ig_abtt[:c_len].tolist(),
        auto_highlights=auto_highlights,
    )
