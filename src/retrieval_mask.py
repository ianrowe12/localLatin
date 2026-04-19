"""MaRC-style mask optimization adapted to retrieval.

This module implements a retrieval-fidelity variant of Brinner & Zarrieß (2023)
MaRC masks. Rather than preserving a classifier's output logit under a learned
token mask, we preserve the cosine similarity between a pooled hidden-state
representation of the query and a FROZEN partner vector (pooled candidate):

    L(m) = (cos(pool(forward(m ⊙ E(x_q))), partner) - cos_orig) ** 2
         + lambda_sparsity * mean(m[content_positions])
         + gamma_tv * mean(|m[i+1] - m[i]|)   (over content positions)

Two variants are supported:

    baseline: raw cosine of mean-pooled hidden state
    abtt:     cosine after ABTT cleaning
              cleaned = (pooled - mean_vec) - (pooled - mean_vec) @ pcs.T @ pcs

The mask `m` is parameterized via m = sigmoid(zeta), with `zeta` frozen at a
large negative value outside content positions (so those tokens are always
masked out). Content positions are determined by
``attention_mask[i] * token_keep_lookup[input_ids[i]] > 0``.

Public API:

    ``MaskOptimConfig``      — dataclass of optimization hyperparameters.
    ``RetrievalMaskOptimizer`` — per-(model, layer) optimizer.
    ``compute_pair_masks``    — runs the 4 optimizations (q/c × baseline/abtt)
                                for a single query-candidate pair and assembles
                                the pair matrices + top-k indices.
    ``load_scores_from_npz``  — reads pair-level artifact files and returns
                                per-method (query_scores, candidate_scores)
                                marginals suitable for downstream comparison.

Model dispatch (T5 / BERT / decoder) reuses ``get_embedding_layer`` and
``get_model_for_forward`` from ``src/attribution_targets.py`` so that this
module works with all six models listed in ``CLAUDE.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from attribution_targets import get_embedding_layer, get_model_for_forward


Variant = Literal["baseline", "abtt"]


@dataclass
class MaskOptimConfig:
    """Hyperparameters for MaRC-style mask optimization."""

    lr: float = 0.1
    steps: int = 200
    lambda_sparsity: float = 0.01
    gamma_tv: float = 0.001
    init_mask_logit: float = 2.197  # log(9) -> sigmoid(.) ≈ 0.9
    early_stop_thresh: float = 0.01  # |cos_m - cos_orig|
    early_stop_min_steps: int = 50


class RetrievalMaskOptimizer:
    """Optimize a soft token mask to preserve retrieval cosine similarity."""

    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        layer_idx: int,
        token_keep_lookup: torch.Tensor,
        config: MaskOptimConfig = MaskOptimConfig(),
        device: str | torch.device = "cuda",
    ):
        self.model = model
        self.model_type = model_type
        self.layer_idx = layer_idx
        self.config = config
        self.device = torch.device(device)

        # Model-type-aware dispatch: embedding layer and the module to call
        # forward on (T5 encoder vs. whole model otherwise).
        self.forward_module = get_model_for_forward(model, model_type)
        self.embedding_layer = get_embedding_layer(model, model_type)

        # token_keep_lookup is a 1D float tensor indexed by vocab id.
        self.token_keep_lookup = token_keep_lookup.to(self.device).float()

        # Freeze all model parameters — we only optimize the mask.
        for p in self.model.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _content_positions(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return a (seq,) boolean tensor of positions that are content."""
        # input_ids / attention_mask have shape (1, seq).
        keep = self.token_keep_lookup[input_ids[0]]  # (seq,)
        am = attention_mask[0].float()
        return (am * keep) > 0.0

    def _pool_hidden(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean-pool hidden states using the attention * token-keep mask.

        Mirrors ``src/retrieval_targets.py`` lines 43-56.
        """
        mask = attention_mask.float() * self.token_keep_lookup[input_ids]
        pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(
            dim=1, keepdim=True
        ).clamp(min=1.0)
        return pooled  # (batch, dim)

    def _masked_cosine_through_forward(
        self,
        m: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        partner: torch.Tensor,
        variant: Variant,
        pcs: Optional[torch.Tensor],
        mean_vec: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run m⊙E(x) through the model and compute cosine to partner.

        Uses ``inputs_embeds=`` directly on the appropriate forward module so
        that the mask multiplies the token embeddings before contextualization.
        """
        # Look up embeddings, then multiply by the soft mask broadcast over dim.
        embeds = self.embedding_layer(input_ids)  # (1, seq, dim)
        masked_embeds = embeds * m.view(1, -1, 1)

        outputs = self.forward_module(
            inputs_embeds=masked_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[self.layer_idx].float()  # (1, seq, dim)
        pooled = self._pool_hidden(hidden, input_ids, attention_mask)  # (1, dim)

        if variant == "abtt":
            if pcs is None or mean_vec is None:
                raise ValueError("abtt variant requires pcs and mean_vec.")
            centered = pooled - mean_vec
            proj = centered @ pcs.T @ pcs
            pooled = centered - proj

        return F.cosine_similarity(pooled, partner.unsqueeze(0), dim=-1).squeeze(0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def optimize(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        partner_fixed: torch.Tensor,
        variant: Variant,
        pcs: Optional[torch.Tensor] = None,
        mean_vec: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit a soft mask so cos(pool(forward(m⊙E(x))), partner) ≈ cos_orig."""
        assert input_ids.dim() == 2 and input_ids.size(0) == 1, (
            "RetrievalMaskOptimizer expects a single-example batch (1, seq)."
        )

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        partner_fixed = partner_fixed.to(self.device).float()
        if pcs is not None:
            pcs = pcs.to(self.device).float()
        if mean_vec is not None:
            mean_vec = mean_vec.to(self.device).float()

        seq_len = input_ids.size(1)
        is_content = self._content_positions(input_ids, attention_mask)  # (seq,)

        # Compute the ORIGINAL cosine (un-masked — m = 1 everywhere on content,
        # 0 on non-content). Use m=1 on all positions so the encoder sees its
        # usual contextual signal; pooling already zeroes non-content.
        with torch.no_grad():
            m_ones = torch.ones(seq_len, device=self.device)
            cos_orig = self._masked_cosine_through_forward(
                m_ones,
                input_ids,
                attention_mask,
                partner_fixed,
                variant,
                pcs,
                mean_vec,
            ).detach()

        # Parameterize zeta; freeze non-content positions to a very negative
        # value so sigmoid ~ 0 and the gradient is effectively zero there.
        if seed is not None:
            torch.manual_seed(int(seed))

        init_val = float(self.config.init_mask_logit)
        zeta_init = torch.full((seq_len,), init_val, device=self.device)
        zeta_init[~is_content] = -1e4
        zeta = zeta_init.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([zeta], lr=self.config.lr)

        convergence = []
        content_idx = is_content.nonzero(as_tuple=False).squeeze(-1)  # indices

        # Precompute contiguity order: apply TV on content positions, preserving
        # their ORIGINAL order in the sequence (consecutive by position index).
        # content_idx is monotonically increasing as produced by nonzero().

        for step in range(self.config.steps):
            optimizer.zero_grad(set_to_none=True)

            m = torch.sigmoid(zeta)

            cos_m = self._masked_cosine_through_forward(
                m,
                input_ids,
                attention_mask,
                partner_fixed,
                variant,
                pcs,
                mean_vec,
            )

            fidelity = (cos_m - cos_orig).pow(2)

            if content_idx.numel() > 0:
                m_content = m[content_idx]
                sparsity = m_content.mean()
            else:
                sparsity = torch.tensor(0.0, device=self.device)

            if content_idx.numel() >= 2:
                m_c = m[content_idx]
                tv = (m_c[1:] - m_c[:-1]).abs().mean()
            else:
                tv = torch.tensor(0.0, device=self.device)

            loss = (
                fidelity
                + self.config.lambda_sparsity * sparsity
                + self.config.gamma_tv * tv
            )

            loss.backward()

            # Null the gradient on non-content positions so they stay frozen.
            if zeta.grad is not None:
                zeta.grad[~is_content] = 0.0

            optimizer.step()

            # Re-pin the non-content zeta values (Adam momentum could nudge
            # them otherwise, although the grad is zero).
            with torch.no_grad():
                zeta[~is_content] = -1e4

            cos_m_val = float(cos_m.detach().item())
            cos_orig_val = float(cos_orig.item())
            convergence.append(cos_m_val)

            # Early stop
            if step + 1 >= self.config.early_stop_min_steps:
                if abs(cos_m_val - cos_orig_val) < self.config.early_stop_thresh:
                    break

        with torch.no_grad():
            final_mask = torch.sigmoid(zeta).detach().cpu().numpy().astype(np.float32)
            # Zero out non-content positions exactly (rather than ~0).
            final_mask[(~is_content).detach().cpu().numpy()] = 0.0

        convergence_arr = np.asarray(convergence, dtype=np.float32)
        return final_mask, convergence_arr


# ----------------------------------------------------------------------
# Pair-level orchestration
# ----------------------------------------------------------------------
def _per_token_abtt_clean(
    hidden: np.ndarray, pcs: np.ndarray, mean_vec: np.ndarray
) -> np.ndarray:
    """Per-token ABTT cleaning; mirrors scripts/ig/run_phase12f_visualize.py:53-56."""
    centered = hidden - mean_vec
    proj = centered @ pcs.T @ pcs
    return centered - proj


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine between rows of a (n, d) and rows of b (m, d) -> (n, m)."""
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-12)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-12)
    return a_norm @ b_norm.T


def compute_pair_masks(
    optimizer: RetrievalMaskOptimizer,
    query_input_ids: torch.Tensor,
    query_attention_mask: torch.Tensor,
    candidate_input_ids: torch.Tensor,
    candidate_attention_mask: torch.Tensor,
    candidate_raw_partner: torch.Tensor,
    candidate_abtt_partner: torch.Tensor,
    query_raw_partner: torch.Tensor,
    query_abtt_partner: torch.Tensor,
    pcs: torch.Tensor,
    mean_vec: torch.Tensor,
    query_hidden: np.ndarray,
    candidate_hidden: np.ndarray,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Run q/c × baseline/abtt mask optimizations for one pair and package outputs.

    Returns a dict of numpy arrays per the API contract. Pair matrices are
    truncated to the attention-masked lengths of the query and candidate.
    """
    out: dict[str, np.ndarray] = {}

    # --- 4 mask optimizations ------------------------------------------------
    q_mask_base, conv_q_base = optimizer.optimize(
        input_ids=query_input_ids,
        attention_mask=query_attention_mask,
        partner_fixed=candidate_raw_partner,
        variant="baseline",
        seed=seed,
    )
    q_mask_abtt, conv_q_abtt = optimizer.optimize(
        input_ids=query_input_ids,
        attention_mask=query_attention_mask,
        partner_fixed=candidate_abtt_partner,
        variant="abtt",
        pcs=pcs,
        mean_vec=mean_vec,
        seed=seed,
    )
    c_mask_base, conv_c_base = optimizer.optimize(
        input_ids=candidate_input_ids,
        attention_mask=candidate_attention_mask,
        partner_fixed=query_raw_partner,
        variant="baseline",
        seed=seed,
    )
    c_mask_abtt, conv_c_abtt = optimizer.optimize(
        input_ids=candidate_input_ids,
        attention_mask=candidate_attention_mask,
        partner_fixed=query_abtt_partner,
        variant="abtt",
        pcs=pcs,
        mean_vec=mean_vec,
        seed=seed,
    )

    # --- Truncate masks to attention-masked lengths -------------------------
    q_len = int(query_attention_mask[0].sum().item())
    c_len = int(candidate_attention_mask[0].sum().item())

    q_mask_base_t = q_mask_base[:q_len].astype(np.float32)
    q_mask_abtt_t = q_mask_abtt[:q_len].astype(np.float32)
    c_mask_base_t = c_mask_base[:c_len].astype(np.float32)
    c_mask_abtt_t = c_mask_abtt[:c_len].astype(np.float32)

    out["q_mask_retrieval_mark_baseline"] = q_mask_base_t
    out["q_mask_retrieval_mark_abtt"] = q_mask_abtt_t
    out["c_mask_retrieval_mark_baseline"] = c_mask_base_t
    out["c_mask_retrieval_mark_abtt"] = c_mask_abtt_t

    # --- Pair matrices ------------------------------------------------------
    q_hidden_t = query_hidden[:q_len].astype(np.float32)
    c_hidden_t = candidate_hidden[:c_len].astype(np.float32)

    pcs_np = pcs.detach().cpu().numpy().astype(np.float32)
    mean_np = mean_vec.detach().cpu().numpy().astype(np.float32)
    q_hidden_abtt = _per_token_abtt_clean(q_hidden_t, pcs_np, mean_np)
    c_hidden_abtt = _per_token_abtt_clean(c_hidden_t, pcs_np, mean_np)

    cos_qc_base = _cosine_rows(q_hidden_t, c_hidden_t)       # (q_len, c_len)
    cos_qc_abtt = _cosine_rows(q_hidden_abtt, c_hidden_abtt)

    pair_base = (
        q_mask_base_t[:, None] * c_mask_base_t[None, :] * cos_qc_base
    ).astype(np.float32)
    pair_abtt = (
        q_mask_abtt_t[:, None] * c_mask_abtt_t[None, :] * cos_qc_abtt
    ).astype(np.float32)

    out["pair_matrix_retrieval_mark_baseline"] = pair_base
    out["pair_matrix_retrieval_mark_abtt"] = pair_abtt

    # --- Top-k indices (5 fixed) --------------------------------------------
    def _topk(vec: np.ndarray, k: int = 5) -> np.ndarray:
        k_eff = min(k, vec.shape[0])
        order = np.argsort(vec)[::-1][:k_eff]
        if k_eff < k:
            pad = np.full((k - k_eff,), -1, dtype=np.int32)
            return np.concatenate([order.astype(np.int32), pad])
        return order.astype(np.int32)

    out["topk_retrieval_mark_baseline_query"] = _topk(
        np.abs(pair_base).sum(axis=1)
    )
    out["topk_retrieval_mark_baseline_candidate"] = _topk(
        np.abs(pair_base).sum(axis=0)
    )
    out["topk_retrieval_mark_abtt_query"] = _topk(
        np.abs(pair_abtt).sum(axis=1)
    )
    out["topk_retrieval_mark_abtt_candidate"] = _topk(
        np.abs(pair_abtt).sum(axis=0)
    )

    # --- Convergence traces --------------------------------------------------
    out["convergence_q_baseline"] = conv_q_base.astype(np.float32)
    out["convergence_q_abtt"] = conv_q_abtt.astype(np.float32)
    out["convergence_c_baseline"] = conv_c_base.astype(np.float32)
    out["convergence_c_abtt"] = conv_c_abtt.astype(np.float32)

    # --- Original cosines (scalar) ------------------------------------------
    # The last cos_m of a converged run approximates cos_orig well, but to
    # avoid conflating with the optimized value we expose the fixed partner
    # dot against the unmasked query.
    q_pool_base = q_hidden_t.mean(axis=0)
    c_partner_base = candidate_raw_partner.detach().cpu().numpy().astype(np.float32)
    q_norm = q_pool_base / (np.linalg.norm(q_pool_base) + 1e-12)
    p_norm = c_partner_base / (np.linalg.norm(c_partner_base) + 1e-12)
    out["cos_orig_baseline"] = np.asarray(
        float(q_norm @ p_norm), dtype=np.float32
    )

    q_pool_abtt_mean = q_hidden_abtt.mean(axis=0)
    c_partner_abtt = candidate_abtt_partner.detach().cpu().numpy().astype(np.float32)
    qa_norm = q_pool_abtt_mean / (np.linalg.norm(q_pool_abtt_mean) + 1e-12)
    pa_norm = c_partner_abtt / (np.linalg.norm(c_partner_abtt) + 1e-12)
    out["cos_orig_abtt"] = np.asarray(float(qa_norm @ pa_norm), dtype=np.float32)

    return out


# ----------------------------------------------------------------------
# NPZ score loader
# ----------------------------------------------------------------------
_OTHER_METHODS: tuple[str, ...] = (
    "ig",
    "bertscore",
    "ot",
    "attention_weighted",
    "dla",
    "attention_standalone",
)


def load_scores_from_npz(npz_path: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load per-method (query_scores, candidate_scores) from a pair-level NPZ.

    retrieval_mark scores come directly from the learned masks; other methods'
    scores come from row/col marginals of ``|pair_matrix|``. All arrays are
    float32 1D, higher = more important.
    """
    data = np.load(npz_path)
    keys = set(data.files)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # --- retrieval_mark: learned masks ---------------------------------------
    for variant in ("baseline", "abtt"):
        qk = f"q_mask_retrieval_mark_{variant}"
        ck = f"c_mask_retrieval_mark_{variant}"
        if qk in keys and ck in keys:
            out[f"retrieval_mark_{variant}"] = (
                np.asarray(data[qk], dtype=np.float32),
                np.asarray(data[ck], dtype=np.float32),
            )

    # --- Other methods: row / col marginals of |pair_matrix| ----------------
    for method in _OTHER_METHODS:
        # Conventional naming: pair_matrix_<method> (e.g. pair_matrix_ig_baseline)
        # OR pair_matrix_<method>_<variant>. Handle both.
        for key in data.files:
            if not key.startswith("pair_matrix_"):
                continue
            body = key[len("pair_matrix_"):]
            if body == method or body.startswith(method + "_"):
                matrix = np.asarray(data[key], dtype=np.float32)
                if matrix.ndim != 2:
                    continue
                abs_m = np.abs(matrix)
                q_scores = abs_m.sum(axis=1).astype(np.float32)
                c_scores = abs_m.sum(axis=0).astype(np.float32)
                out[body] = (q_scores, c_scores)

    return out


# ----------------------------------------------------------------------
# Inline smoke test
# ----------------------------------------------------------------------
def _smoke_test() -> None:
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    from token_filtering import build_token_keep_lookup

    model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device={device} model={model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    # Token-keep lookup — use "all" so every position is content.
    token_keep_np = build_token_keep_lookup(tokenizer, "all")
    token_keep = torch.as_tensor(token_keep_np, dtype=torch.float32)

    q_text = "The cat sits quietly on the soft pillow."
    c_text = "A feline is resting calmly on a cushion."

    q_enc = tokenizer(q_text, return_tensors="pt", padding=True, truncation=True).to(device)
    c_enc = tokenizer(c_text, return_tensors="pt", padding=True, truncation=True).to(device)

    layer_idx = 2  # MiniLM-L3 has 3 blocks => hidden_states has length 4 (0..3).

    # Compute partner vector by running c through the model (no_grad).
    with torch.no_grad():
        outs = model(
            input_ids=c_enc["input_ids"],
            attention_mask=c_enc["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        )
        c_hidden = outs.hidden_states[layer_idx].float()  # (1, seq, dim)
        c_mask = c_enc["attention_mask"].float()
        partner = (c_hidden * c_mask.unsqueeze(-1)).sum(dim=1) / c_mask.sum(
            dim=1, keepdim=True
        ).clamp(min=1.0)
        partner = partner[0]  # (dim,)

    cfg = MaskOptimConfig(steps=10, early_stop_min_steps=100)  # don't early-stop
    optim = RetrievalMaskOptimizer(
        model=model,
        model_type="bert",
        layer_idx=layer_idx,
        token_keep_lookup=token_keep,
        config=cfg,
        device=device,
    )

    mask1, conv1 = optim.optimize(
        input_ids=q_enc["input_ids"],
        attention_mask=q_enc["attention_mask"],
        partner_fixed=partner,
        variant="baseline",
        seed=42,
    )
    mask2, _ = optim.optimize(
        input_ids=q_enc["input_ids"],
        attention_mask=q_enc["attention_mask"],
        partner_fixed=partner,
        variant="baseline",
        seed=42,
    )

    q_seq = int(q_enc["input_ids"].size(1))
    q_am = q_enc["attention_mask"][0].cpu().numpy().astype(bool)

    print(f"[smoke] q_seq={q_seq} mask.shape={mask1.shape}")
    print(f"[smoke] mean(mask) = {mask1.mean():.4f}")
    print(f"[smoke] cos_orig ≈ {conv1[0]:.4f}  cos_m_final = {conv1[-1]:.4f}")
    print(f"[smoke] mask values: {np.round(mask1, 3)}")

    assert mask1.shape == (q_seq,), f"Expected shape ({q_seq},) got {mask1.shape}"
    assert not np.allclose(mask1, 1.0), "Mask did not move from init"
    assert not np.allclose(mask1, 0.0), "Mask collapsed to zero"
    assert np.allclose(mask1, mask2), "Determinism failed (same seed -> different masks)"

    content_vals = mask1[q_am]
    non_content_vals = mask1[~q_am] if (~q_am).any() else np.array([])
    assert content_vals.min() > 0.0, "content mask has zero values"
    assert content_vals.max() < 1.0, "content mask values not < 1 (sigmoid bounded)"
    if non_content_vals.size > 0:
        assert np.all(non_content_vals <= 1e-6), "non-content mask not ~0"

    # Test load_scores_from_npz round-trip with a tiny synthetic NPZ.
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        np.savez(
            tmp_path,
            q_mask_retrieval_mark_baseline=np.array([0.1, 0.9, 0.5], dtype=np.float32),
            c_mask_retrieval_mark_baseline=np.array([0.2, 0.8], dtype=np.float32),
            pair_matrix_ig_baseline=np.array([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]], dtype=np.float32),
        )
        scores = load_scores_from_npz(tmp_path)
        assert "retrieval_mark_baseline" in scores
        assert "ig_baseline" in scores
        q_s, c_s = scores["ig_baseline"]
        assert q_s.shape == (3,) and c_s.shape == (2,)
        print(f"[smoke] load_scores_from_npz round-trip OK: keys={sorted(scores.keys())}")
    finally:
        os.unlink(tmp_path)

    print("[smoke] all assertions passed.")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Ensure `src/` is on sys.path so intra-package imports resolve when this
    # file is run directly as ``python src/retrieval_mask.py`` from the repo
    # root.
    src_dir = Path(__file__).resolve().parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    _smoke_test()
