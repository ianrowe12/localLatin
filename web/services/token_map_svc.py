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


ATTRIBUTION_METHODS = (
    "ig", "bertscore", "ot",
    "attention_weighted", "dla", "attention_standalone",
    "retrieval_mark",
)
# Order matters: the frontend renders the variant control in this order and
# falls back to the first available entry. "baseline" and "abtt" predate the
# SIF variants, so they stay first and keep their existing semantics.
ATTRIBUTION_VARIANTS = ("baseline", "abtt", "sif", "sif_abtt")
BUCKET_ORDER = ["correct_similar", "correct_not_similar", "wrong_similar", "wrong_not_similar"]

# Buckets the Attribution gallery does not list. The full-corpus run (issue #84)
# adds tens of thousands of rows that exist to be resolved from a live query, not
# browsed: listing them would mean tens of thousands of cards, and the grouped
# listing opens every artifact it lists.
GALLERY_EXCLUDED_BUCKETS = frozenset({"unlabelled_bulk"})

# Fallback only. Artifacts built after issue #47 carry
# query_token_strings / candidate_token_strings, so no tokenizer is needed at
# serve time. Kept in sync with scripts/ig/persist_decoded_tokens.py.
SLUG_TO_HF = {
    "bowphs_LaTa": "bowphs/LaTa",
    "bowphs_PhilTa": "bowphs/PhilTa",
    "google_mt5-base": "google/mt5-base",
    "sentence-transformers_LaBSE": "sentence-transformers/LaBSE",
    "Qwen_Qwen3-Embedding-0.6B": "Qwen/Qwen3-Embedding-0.6B",
    "Qwen_Qwen3-Embedding-8B": "Qwen/Qwen3-Embedding-8B",
    "KaLM-Embedding_KaLM-embedding-multilingual-mini-instruct-v2.5": (
        "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5"
    ),
}


def _stored_tokens(data: dict[str, np.ndarray], key: str, count: int) -> list[str] | None:
    """Read decoded token strings persisted in the artifact, if present."""
    arr = data.get(key)
    if arr is None:
        return None
    try:
        tokens = [str(t) for t in np.asarray(arr).ravel().tolist()]
    except Exception as e:  # noqa: BLE001
        logger.warning("Malformed %s in artifact: %s", key, e)
        return None
    if len(tokens) < count:
        logger.warning("%s has %d entries for %d tokens", key, len(tokens), count)
        return None
    return tokens[:count]


def _cell(row, key: str) -> str:
    """Read a CSV cell as a display string, mapping missing/NaN to "".

    Unlabelled-query rows leave columns like ``query_folder_id`` empty, which
    pandas reads back as NaN. ``str(nan)`` would surface a literal "nan" in the
    gallery.
    """
    value = row.get(key, "")
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def _csv_list(row, key: str) -> list[str] | None:
    """Parse a comma-separated CSV cell, or ``None`` when it is absent/blank."""
    raw = _cell(row, key).strip()
    if not raw or raw.lower() == "nan":
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def _gallery_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "bucket" not in df.columns:
        return df
    return df[~df["bucket"].astype(str).isin(GALLERY_EXCLUDED_BUCKETS)]


def _optional_vector(data: dict[str, np.ndarray], key: str, count: int) -> list[float] | None:
    arr = data.get(key)
    if arr is None:
        return None
    return np.asarray(arr, dtype=np.float32).ravel()[:count].tolist()


def _try_decode_tokens(input_ids: np.ndarray, model_slug: str) -> list[str] | None:
    """Try to decode token IDs using HuggingFace tokenizer. Returns None if unavailable."""
    try:
        from transformers import AutoTokenizer  # noqa: F401
    except ImportError:
        return None

    hf_id = SLUG_TO_HF.get(model_slug)
    if hf_id is None:
        return None

    try:
        tokenizer = _get_tokenizer(hf_id)
        ids = input_ids.flatten().tolist()
        return [tokenizer.decode([tid]) for tid in ids]
    except Exception as e:
        logger.warning("Failed to decode tokens for %s: %s", model_slug, e)
        return None


@lru_cache(maxsize=8)
def _get_tokenizer(hf_id: str):
    # No trust_remote_code: every model in SLUG_TO_HF resolves to a built-in
    # fast tokenizer (Qwen2TokenizerFast / T5TokenizerFast / BertTokenizerFast),
    # so the serving path never needs to execute unpinned code from the Hub.
    # This is only a fallback anyway -- artifacts carry decoded token strings.
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(hf_id)


@lru_cache(maxsize=64)
def _load_npz(path: str) -> dict[str, np.ndarray]:
    return dict(np.load(path, allow_pickle=False))


def list_examples(store: DataStore, model: str | None = None, bucket: str | None = None) -> list[TokenMapExampleSummary]:
    if store.ig_examples is None:
        return []

    df = _gallery_rows(store.ig_examples)
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
            query_path=_cell(row, "query_path"),
            candidate_path=_cell(row, "candidate_path"),
        ))
    return results


def list_examples_grouped(store: DataStore) -> dict:
    if store.ig_examples is None:
        return {"by_model": {}, "bucket_order": BUCKET_ORDER}

    by_model: dict[str, list[dict]] = {}
    for _, row in _gallery_rows(store.ig_examples).iterrows():
        eid = int(row["example_id"])
        if eid not in store.ig_artifact_paths:
            continue

        npz_path = store.ig_artifact_paths[eid]

        # The CSV columns are authoritative when present -- opening every listed
        # NPZ is what made this endpoint scale with the artifact count rather
        # than the gallery size. Fall back to inspecting the file for rows
        # written before those columns existed.
        methods = _csv_list(row, "methods_available")
        variants = _csv_list(row, "variants_available")
        if methods is None or variants is None:
            try:
                data = _load_npz(str(npz_path))
            except Exception as e:
                logger.warning("Failed to load NPZ for example %d: %s", eid, e)
                continue
            if methods is None:
                methods = [
                    m for m in ATTRIBUTION_METHODS
                    if any(f"pair_matrix_{m}_{v}" in data for v in ATTRIBUTION_VARIANTS)
                ]
            if variants is None:
                variants = [
                    v for v in ATTRIBUTION_VARIANTS
                    if any(f"pair_matrix_{m}_{v}" in data for m in ATTRIBUTION_METHODS)
                ]
        methods = [m for m in ATTRIBUTION_METHODS if m in set(methods)]
        variants = [v for v in ATTRIBUTION_VARIANTS if v in set(variants)]

        slug = normalize_slug(npz_path.parent.name)
        query_path = _cell(row, "query_path")
        by_model.setdefault(slug, []).append({
            "example_id": eid,
            "model_slug": slug,
            "bucket": _cell(row, "bucket"),
            "query_file_id": int(row.get("query_file_id", -1)),
            "query_folder_id": _cell(row, "query_folder_id"),
            "query_filename": Path(query_path).name if query_path else "",
            "candidate_folder_id": _cell(row, "candidate_folder_id"),
            "candidate_label": _cell(row, "candidate_label"),
            "methods_available": methods,
            "variants_available": variants,
            "gold_similar": int(row.get("gold_similar", 0) or 0),
            "baseline_pred": int(row.get("baseline_pred", 0) or 0),
            "abtt_pred": int(row.get("abtt_pred", 0) or 0),
        })

    # Sort cards within each model by (bucket order, example_id)
    bucket_idx = {b: i for i, b in enumerate(BUCKET_ORDER)}
    for cards in by_model.values():
        cards.sort(key=lambda c: (bucket_idx.get(c["bucket"], 999), c["example_id"]))

    return {"by_model": by_model, "bucket_order": BUCKET_ORDER}


def resolve_example_id(
    store: DataStore,
    file_id: int,
    candidate_dir: str,
    model: str | None = None,
    query_source: str | None = None,
    variant: str | None = None,
) -> int | None:
    """Find the example_id for a (query file_id, candidate_dir) pair.

    ``query_file_id`` is only unique within a corpus: the labelled examples
    number the canon split from 0 and the unlabelled review queue numbers its
    own files from 0, so the two ranges overlap. ``query_source`` disambiguates
    them, and the filter is strict -- a request for one corpus never falls back
    to a row from the other. Both the argument and the column are optional:
    artifact CSVs written before unlabelled examples existed have neither, and
    those resolve exactly as before.

    ``variant`` narrows further, and for the same reason. On four of the six
    models the four post-processing variants are deployed at *different layers*
    (KaLM-mini ranks raw at L22, abtt and sif_abtt at L1, sif at L23), so one
    ``(file_id, candidate_dir, model)`` has several artifacts, each built at one
    layer and carrying only the variants deployed there. Without this filter the
    first matching row wins and the reviewer gets an explanation computed at a
    layer their ranking never used. Rows whose ``variants_available`` is absent
    or blank are treated as carrying every variant -- that is what the paper-set
    artifacts do.

    Rows that have a matching artifact on disk win over rows that do not, so a
    stale CSV entry cannot mask a usable one.
    """
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

    if query_source is not None and "query_source" in matches.columns:
        # Strict: no fallback to the unnarrowed set. Falling back would resolve
        # an unlabelled query to a labelled artifact describing a different
        # manuscript, which is the collision this argument exists to prevent
        # (13 such combinations sit inside the deployed top-10). Returning None
        # renders as "no artifact for this pair", i.e. the Attribution toggle is
        # absent, which is what the demo script promises off the demo set.
        matches = matches[matches["query_source"].astype(str) == query_source]
        if matches.empty:
            return None

    if variant is not None and "variants_available" in matches.columns:
        # Strict, like the query_source filter above: an artifact that does not
        # carry this variant was built at another layer, and serving it would
        # explain a configuration the ranking never used.
        def _carries(cell) -> bool:
            listed = _csv_list({"variants_available": cell}, "variants_available")
            return listed is None or variant in listed

        matches = matches[matches["variants_available"].apply(_carries)]
        if matches.empty:
            return None

    ids = [int(eid) for eid in matches["example_id"]]
    for eid in ids:
        if eid in store.ig_artifact_paths:
            return eid
    return None


def load_token_map(
    store: DataStore,
    example_id: int,
    method: str | None = None,
    variant: str | None = None,
) -> TokenMapResponse | None:
    """Build the token-map payload for one example.

    ``method`` / ``variant`` narrow which ``pair_matrices`` and ``top_highlights``
    entries are serialised. An unfiltered response carries every persisted
    method x variant matrix (7 x 4 dense QxC float grids, tens of MB on long
    pairs), which the UI never renders all at once -- it shows exactly one cell
    of that grid. ``available_methods`` / ``available_variants`` always report
    the artifact's full contents regardless of the filters, so a client can
    still discover what else it may request (issue #72).
    """
    if example_id not in store.ig_artifact_paths:
        return None

    npz_path = store.ig_artifact_paths[example_id]
    data = _load_npz(str(npz_path))

    # Extract metadata
    layer = int(data["layer"].item()) if "layer" in data else 0
    D = int(data["D"].item()) if "D" in data else 0
    # The sif_abtt variant is cleaned in the SIF-pooled ABTT subspace, whose D is
    # swept independently of the mean-pooled one (LaTa layer 1: mean 10, SIF 3).
    # Absent on pre-per-pooling artifacts, where sif_abtt reused the mean fit.
    D_sif = int(np.asarray(data["D_sif"]).reshape(-1)[0]) if "D_sif" in data else None

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
            bucket = _cell(row, "bucket")
            query_path = _cell(row, "query_path")
            candidate_path = _cell(row, "candidate_path")

    # Token-token cosine similarity. Artifacts from the full-corpus run (issue
    # #84) store it directly and omit the raw hidden states it was derived from:
    # a (Q, dim) float32 block is six times the size of the (Q, C) grid, and this
    # is the only thing the serving path ever used it for. Older artifacts carry
    # the hidden states and no stored matrix, so both shapes are read.
    stored_sim = data.get("similarity_matrix")
    if stored_sim is not None:
        sim_arr = np.asarray(stored_sim, dtype=np.float32)
        q_len, c_len = int(sim_arr.shape[0]), int(sim_arr.shape[1])
    elif "query_hidden" in data and "candidate_hidden" in data:
        query_hidden = data["query_hidden"]    # (Q, dim)
        cand_hidden = data["candidate_hidden"]  # (C, dim)
        q_len = query_hidden.shape[0]
        c_len = cand_hidden.shape[0]
        q_norm = query_hidden / (np.linalg.norm(query_hidden, axis=1, keepdims=True) + 1e-8)
        c_norm = cand_hidden / (np.linalg.norm(cand_hidden, axis=1, keepdims=True) + 1e-8)
        sim_arr = np.asarray(q_norm @ c_norm.T, dtype=np.float32)
    else:
        logger.warning(
            "Artifact %s has neither similarity_matrix nor hidden states", npz_path
        )
        return None
    sim_matrix = sim_arr.tolist()

    # IG weights
    q_ig_base = data.get("query_ig_baseline", np.zeros(q_len))
    q_ig_abtt = data.get("query_ig_abtt", np.zeros(q_len))
    c_ig_base = data.get("candidate_ig_baseline", np.zeros(c_len))
    c_ig_abtt = data.get("candidate_ig_abtt", np.zeros(c_len))

    # IG-weighted matrix. Prefer the persisted pair_matrix_ig_abtt (already
    # computed with ABTT-cleaned hidden states and L1-normalized IG weights
    # by persist_attribution_methods.py) so this view stays consistent with
    # the slide and paper figures. Fall back to an inline, L1-normalized
    # computation if the persisted key is missing.
    ig_weighted = None
    if "pair_matrix_ig_abtt" in data:
        mat = np.asarray(data["pair_matrix_ig_abtt"], dtype=np.float32)[:q_len, :c_len]
        ig_weighted = mat.tolist()
    elif "query_ig_abtt" in data and "candidate_ig_abtt" in data:
        q_abs = np.abs(q_ig_abtt[:q_len])
        c_abs = np.abs(c_ig_abtt[:c_len])
        q_norm = q_abs / (q_abs.sum() + 1e-12)
        c_norm = c_abs / (c_abs.sum() + 1e-12)
        weight = np.sqrt(q_norm[:, None] * c_norm[None, :])
        sign = np.sign(q_ig_abtt[:q_len, None]) * np.sign(c_ig_abtt[None, :c_len])
        cos = np.array(sim_matrix)
        ig_weighted = (cos * weight * sign).tolist()

    # What this artifact actually holds. Reported in full even when the request
    # narrows the payload, so the client can discover the other combinations.
    available_methods = [
        m for m in ATTRIBUTION_METHODS
        if any(f"pair_matrix_{m}_{v}" in data for v in ATTRIBUTION_VARIANTS)
    ]
    available_variants = [
        v for v in ATTRIBUTION_VARIANTS
        if any(f"pair_matrix_{m}_{v}" in data for m in ATTRIBUTION_METHODS)
    ]

    wanted_methods = [m for m in available_methods if method is None or m == method]
    wanted_variants = [v for v in available_variants if variant is None or v == variant]

    # Load the requested method x variant matrices defensively (skip missing keys)
    pair_matrices: dict[str, dict[str, list[list[float]]]] = {}
    top_highlights: dict[str, dict[str, dict[str, list[int]]]] = {}

    for m in wanted_methods:
        for v in wanted_variants:
            mkey = f"pair_matrix_{m}_{v}"
            qk = f"topk_{m}_{v}_query"
            ck = f"topk_{m}_{v}_candidate"
            if mkey not in data:
                continue
            mat = np.asarray(data[mkey], dtype=np.float32)
            # Trim to actual sequence lengths so the response never reports padding cells
            mat = mat[:q_len, :c_len]
            pair_matrices.setdefault(m, {})[v] = mat.tolist()
            if qk in data and ck in data:
                top_highlights.setdefault(m, {})[v] = {
                    "query": np.asarray(data[qk]).astype(int).tolist(),
                    "candidate": np.asarray(data[ck]).astype(int).tolist(),
                }

    # Top matches (sparse format for frontend connection lines)
    top_matches: dict[str, list[TopMatch]] = {}
    for qi in range(q_len):
        row = sim_arr[qi]
        top_k = min(3, c_len)
        top_idx = np.argsort(row)[-top_k:][::-1]
        top_matches[str(qi)] = [
            TopMatch(candidate_idx=int(ci), score=float(row[ci]))
            for ci in top_idx
        ]

    # Auto-highlights: top K=5 query tokens by |IG score|. Prefer the ABTT
    # vector, but a bulk artifact only carries the variants deployed at its
    # layer, so an artifact built for `sif` alone has no `query_ig_abtt`. Take
    # whichever IG vector it does have rather than dropping the highlights.
    auto_highlights = None
    ig_source = next(
        (
            f"query_ig_{v}"
            for v in ("abtt", "sif_abtt", "baseline", "sif")
            if f"query_ig_{v}" in data
        ),
        None,
    )
    if ig_source is not None:
        abs_ig = np.abs(np.asarray(data[ig_source], dtype=np.float32)[:q_len])
        if abs_ig.size and abs_ig.max() > 1e-6:
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

    # Tokens: prefer the strings persisted into the artifact (issue #47), fall
    # back to decoding the ids with a tokenizer, and finally to "[i]" markers.
    query_input_ids = data.get("query_input_ids")
    cand_input_ids = data.get("candidate_input_ids")

    q_token_strs = _stored_tokens(data, "query_token_strings", q_len)
    c_token_strs = _stored_tokens(data, "candidate_token_strings", c_len)
    if q_token_strs is None and query_input_ids is not None:
        q_token_strs = _try_decode_tokens(query_input_ids, model_slug)
    if c_token_strs is None and cand_input_ids is not None:
        c_token_strs = _try_decode_tokens(cand_input_ids, model_slug)

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
        D_sif=D_sif,
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
        available_methods=available_methods,
        available_variants=available_variants,
        pair_matrices=pair_matrices,
        top_highlights=top_highlights,
        query_sif_weights=_optional_vector(data, "query_sif_weights", q_len),
        candidate_sif_weights=_optional_vector(data, "candidate_sif_weights", c_len),
    )
