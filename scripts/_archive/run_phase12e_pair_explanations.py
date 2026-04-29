"""Compute Phase 12e pair explanations and companion attention artifacts.

For each selected query/candidate pair, this script saves:

- original token IDs and attention masks for both passages
- per-sequence IG toward pair cosine similarity (baseline and ABTT)
- raw token hidden states at the selected layer
- mean self-attention matrices at the selected layer
- the train-fit PCs and mean vector used for ABTT
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from attribution_targets import get_embedding_layer
from retrieval_targets import ABTTCosSimTarget, BaselineCosSimTarget
from token_filtering import TOKEN_FILTER_CHOICES, build_token_keep_lookup

try:
    from captum.attr import LayerIntegratedGradients
except ImportError:
    LayerIntegratedGradients = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 12e pair explanations.")
    parser.add_argument("--examples_csv", required=True, help="CSV from run_phase12e_select_examples.py")
    parser.add_argument("--pc_root", required=True, help="Root containing per-model layer PCs")
    parser.add_argument("--out_dir", required=True, help="Output root for .npz artifacts")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--n_steps", type=int, default=40)
    parser.add_argument(
        "--token_filter",
        choices=list(TOKEN_FILTER_CHOICES),
        default="tokenizer_empty",
        help="Pooling-time token filter for the pair-cosine targets.",
    )
    parser.add_argument("--half_precision", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Restrict to these model_name strings.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=0,
        help="0 = all; N = first N rows after filtering.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate CSV rows, text paths, PC files, and output paths without loading models.",
    )
    parser.add_argument(
        "--dry_run_require_pcs",
        action="store_true",
        help="With --dry_run, fail if planned PC files are missing.",
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip examples whose output NPZ already exists.",
    )
    return parser.parse_args()


def model_slug(name: str) -> str:
    return name.replace("/", "_")


def load_model(model_name: str, model_type: str, half_precision: bool, trust_remote_code: bool, device: str):
    kwargs = {"trust_remote_code": trust_remote_code}
    if half_precision:
        kwargs["torch_dtype"] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if model_type == "t5":
        full_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)
        model = full_model.get_encoder()
        resolved_type = "t5"
    else:
        model = AutoModel.from_pretrained(model_name, **kwargs)
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            resolved_type = "decoder_wrapped"
        elif hasattr(model, "layers"):
            resolved_type = "decoder"
        elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            resolved_type = "bert"
        else:
            resolved_type = model_type
    model.to(device).eval()
    model.requires_grad_(False)
    return model, tokenizer, resolved_type


def tokenize_text(tokenizer, text: str, max_length: int, device: str) -> dict[str, torch.Tensor]:
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt",
    )
    return {
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
    }


@torch.no_grad()
def encode_states(
    model,
    enc: dict[str, torch.Tensor],
    layer: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        output_hidden_states=True,
        output_attentions=True,
        return_dict=True,
    )
    hidden = outputs.hidden_states[layer].float().squeeze(0)
    attentions = getattr(outputs, "attentions", None)
    if attentions is None or layer == 0:
        attn_mean = torch.zeros(
            hidden.shape[0],
            hidden.shape[0],
            dtype=torch.float32,
            device=hidden.device,
        )
    else:
        attn_idx = min(layer - 1, len(attentions) - 1)
        attn_mean = attentions[attn_idx].float().mean(dim=1).squeeze(0)
    return hidden, attn_mean


def pool_hidden(
    hidden: torch.Tensor,
    enc: dict[str, torch.Tensor],
    token_keep_lookup: torch.Tensor,
) -> torch.Tensor:
    mask = enc["attention_mask"].float() * token_keep_lookup[enc["input_ids"]]
    pooled = (hidden.unsqueeze(0) * mask.unsqueeze(-1)).sum(dim=1)
    return (pooled / mask.sum(dim=1, keepdim=True).clamp(min=1.0)).squeeze(0)


def abtt_clean(vec: torch.Tensor, pcs: torch.Tensor, mean_vec: torch.Tensor) -> torch.Tensor:
    centered = vec - mean_vec
    proj = centered @ pcs.T @ pcs
    return centered - proj


def run_ig(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_fn,
    emb_layer: torch.nn.Module,
    n_steps: int,
) -> np.ndarray:
    if LayerIntegratedGradients is None:
        raise ImportError("captum is required for IG generation: pip install captum")
    lig = LayerIntegratedGradients(target_fn, emb_layer)
    attr = lig.attribute(
        input_ids,
        additional_forward_args=(attention_mask,),
        n_steps=n_steps,
        internal_batch_size=1,
    )
    return attr.sum(dim=-1).squeeze(0).detach().cpu().float().numpy()


def main() -> None:
    args = parse_args()
    examples = pd.read_csv(args.examples_csv)
    if args.models:
        examples = examples[examples["model_name"].isin(args.models)].reset_index(drop=True)
    if args.max_examples and args.max_examples > 0:
        examples = examples.head(args.max_examples).reset_index(drop=True)
    out_root = Path(args.out_dir)
    artifacts_root = out_root / "artifacts"
    if not args.dry_run:
        artifacts_root.mkdir(parents=True, exist_ok=True)
    pc_root = Path(args.pc_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dry_run:
        failures: list[str] = []
        for row in examples.itertuples(index=False):
            example_id = int(row.example_id)
            slug = model_slug(row.model_name)
            out_path = (
                artifacts_root / slug
                / f"example{example_id:03d}_{row.candidate_role}.npz"
            )
            if pd.isna(row.query_path) or not str(row.query_path).strip():
                failures.append(f"example {example_id}: missing query_path")
            elif not Path(row.query_path).exists():
                failures.append(f"example {example_id}: query_path missing on disk: {row.query_path}")
            if pd.isna(row.candidate_path) or not str(row.candidate_path).strip():
                failures.append(f"example {example_id}: missing candidate_path")
            elif not Path(row.candidate_path).exists():
                failures.append(
                    f"example {example_id}: candidate_path missing on disk: {row.candidate_path}"
                )
            if str(row.candidate_label) == "__NEW__":
                failures.append(f"example {example_id}: candidate is __NEW__")
            pc_path = pc_root / slug / f"layer{int(row.layer)}_pcs.npz"
            pc_status = "exists" if pc_path.exists() else "missing"
            if args.dry_run_require_pcs and not pc_path.exists():
                failures.append(f"example {example_id}: PC file missing: {pc_path}")
            print(
                f"[DRY] example {example_id:03d} {slug} layer={int(row.layer)} "
                f"D={int(row.D)} pc={pc_status} -> {out_path}"
            )
        if failures:
            raise SystemExit("Dry run failed:\n" + "\n".join(failures))
        print(f"Dry run OK for {len(examples)} examples")
        return

    current_model_name = None
    model = None
    tokenizer = None
    resolved_type = None
    emb_layer = None
    token_keep_lookup_t = None
    skipped = 0

    for row in examples.itertuples(index=False):
        if pd.isna(row.query_path) or not str(row.query_path).strip():
            print(f"Skipping example {int(row.example_id)}: missing query_path")
            skipped += 1
            continue
        if pd.isna(row.candidate_path) or not str(row.candidate_path).strip():
            print(
                f"Skipping example {int(row.example_id)} ({row.candidate_role}): "
                "missing candidate_path"
            )
            skipped += 1
            continue
        if str(row.candidate_label) == "__NEW__":
            print(
                f"Skipping example {int(row.example_id)} ({row.candidate_role}): "
                "candidate is __NEW__"
            )
            skipped += 1
            continue

        slug = model_slug(row.model_name)
        out_path = (
            artifacts_root / slug
            / f"example{int(row.example_id):03d}_{row.candidate_role}.npz"
        )
        if args.skip_existing and out_path.exists():
            print(f"[SKIP] Already exists: {out_path}")
            skipped += 1
            continue

        if row.model_name != current_model_name:
            current_model_name = row.model_name
            model, tokenizer, resolved_type = load_model(
                row.model_name,
                row.model_type,
                args.half_precision,
                args.trust_remote_code,
                device,
            )
            emb_layer = get_embedding_layer(model, resolved_type)
            token_keep_lookup = build_token_keep_lookup(tokenizer, args.token_filter)
            token_keep_lookup_t = torch.as_tensor(
                token_keep_lookup,
                device=device,
                dtype=torch.float32,
            )

        pc_path = pc_root / slug / f"layer{int(row.layer)}_pcs.npz"
        if not pc_path.exists():
            raise FileNotFoundError(f"PC file missing: {pc_path}")
        pc_data = np.load(pc_path)
        d_value = int(row.D)
        pcs = torch.from_numpy(pc_data["pcs"][:d_value]).to(device).float()
        mean_vec = torch.from_numpy(pc_data["mean_vec"]).to(device).float()

        query_text = Path(row.query_path).read_text(encoding="utf-8", errors="ignore")
        candidate_text = Path(row.candidate_path).read_text(encoding="utf-8", errors="ignore")

        query_enc = tokenize_text(tokenizer, query_text, args.max_length, device)
        candidate_enc = tokenize_text(tokenizer, candidate_text, args.max_length, device)
        query_hidden, query_attention = encode_states(model, query_enc, int(row.layer))
        candidate_hidden, candidate_attention = encode_states(model, candidate_enc, int(row.layer))

        query_raw = pool_hidden(query_hidden, query_enc, token_keep_lookup_t)
        candidate_raw = pool_hidden(candidate_hidden, candidate_enc, token_keep_lookup_t)
        query_abtt = abtt_clean(query_raw, pcs, mean_vec)
        candidate_abtt = abtt_clean(candidate_raw, pcs, mean_vec)

        query_base_target = BaselineCosSimTarget(
            model, int(row.layer), candidate_raw, token_keep_lookup=token_keep_lookup_t
        ).to(device)
        query_abtt_target = ABTTCosSimTarget(
            model,
            int(row.layer),
            candidate_abtt,
            pcs,
            mean_vec,
            token_keep_lookup=token_keep_lookup_t,
        ).to(device)
        candidate_base_target = BaselineCosSimTarget(
            model, int(row.layer), query_raw, token_keep_lookup=token_keep_lookup_t
        ).to(device)
        candidate_abtt_target = ABTTCosSimTarget(
            model,
            int(row.layer),
            query_abtt,
            pcs,
            mean_vec,
            token_keep_lookup=token_keep_lookup_t,
        ).to(device)

        query_ig_base = run_ig(
            query_enc["input_ids"],
            query_enc["attention_mask"],
            query_base_target,
            emb_layer,
            args.n_steps,
        )
        query_ig_abtt = run_ig(
            query_enc["input_ids"],
            query_enc["attention_mask"],
            query_abtt_target,
            emb_layer,
            args.n_steps,
        )
        candidate_ig_base = run_ig(
            candidate_enc["input_ids"],
            candidate_enc["attention_mask"],
            candidate_base_target,
            emb_layer,
            args.n_steps,
        )
        candidate_ig_abtt = run_ig(
            candidate_enc["input_ids"],
            candidate_enc["attention_mask"],
            candidate_abtt_target,
            emb_layer,
            args.n_steps,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            example_id=np.array([int(row.example_id)], dtype=np.int32),
            layer=np.array([int(row.layer)], dtype=np.int32),
            D=np.array([d_value], dtype=np.int32),
            query_input_ids=query_enc["input_ids"].detach().cpu().numpy(),
            query_attention_mask=query_enc["attention_mask"].detach().cpu().numpy(),
            candidate_input_ids=candidate_enc["input_ids"].detach().cpu().numpy(),
            candidate_attention_mask=candidate_enc["attention_mask"].detach().cpu().numpy(),
            query_hidden=query_hidden.detach().cpu().numpy().astype(np.float32),
            candidate_hidden=candidate_hidden.detach().cpu().numpy().astype(np.float32),
            query_attention=query_attention.detach().cpu().numpy().astype(np.float32),
            candidate_attention=candidate_attention.detach().cpu().numpy().astype(np.float32),
            query_ig_baseline=query_ig_base.astype(np.float32),
            query_ig_abtt=query_ig_abtt.astype(np.float32),
            candidate_ig_baseline=candidate_ig_base.astype(np.float32),
            candidate_ig_abtt=candidate_ig_abtt.astype(np.float32),
            pcs=pcs.detach().cpu().numpy().astype(np.float32),
            mean_vec=mean_vec.detach().cpu().numpy().astype(np.float32),
        )
        print(f"Saved {out_path}")

    if skipped:
        print(f"Skipped {skipped} malformed example rows")


if __name__ == "__main__":
    main()
