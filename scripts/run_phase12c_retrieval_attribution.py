"""Phase 12c: Run Captum IG attribution against retrieval cosine-similarity targets.

For each (query, positive_partner, negative_partner) triple at each selected layer:
  - BaselineCosSimTarget:  cos(mean_pool(query_hidden), partner_raw)
  - ABTTCosSimTarget:      cos(ABTT(mean_pool(query_hidden)), partner_abtt)

Produces 4 IG vectors per query per layer:
  ig_baseline_pos, ig_abtt_pos, ig_baseline_neg, ig_abtt_neg

Output: runs/phase12c/attributions/{slug}/layer{L}_retrieval_attr.npz
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from attribution_targets import get_embedding_layer
from retrieval_targets import ABTTCosSimTarget, BaselineCosSimTarget
from canon_retrieval import load_texts
from cli_utils import parse_layers

try:
    from captum.attr import LayerIntegratedGradients
except ImportError:
    raise ImportError("captum is required: pip install captum")


def parse_args():
    p = argparse.ArgumentParser(description="Phase 12c retrieval attribution.")
    p.add_argument("--model_name", required=True)
    p.add_argument("--model_type", required=True, choices=["t5", "bert", "decoder"])
    p.add_argument("--pc_dir", required=True, help="Dir with layer*_pcs.npz")
    p.add_argument("--pairs_csv", required=True, help="retrieval_pairs.csv")
    p.add_argument("--layers", required=True, help="Layer list, e.g. '1,4,8,12'")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--D", type=int, default=10, help="Number of PCs for ABTT")
    p.add_argument("--n_steps", type=int, default=50, help="IG integration steps")
    p.add_argument("--half_precision", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")
    return p.parse_args()


def load_model(model_name, model_type, half_precision, trust_remote_code, device):
    """Load model, return (forward_model, tokenizer, resolved_type)."""
    kwargs = {"trust_remote_code": trust_remote_code}
    if half_precision:
        kwargs["torch_dtype"] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )

    if model_type == "t5":
        full_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)
        fwd = full_model.get_encoder()
        fwd.to(device).eval()
        fwd.requires_grad_(False)
        resolved = "t5"
    else:
        model = AutoModel.from_pretrained(model_name, **kwargs)
        model.to(device).eval()
        model.requires_grad_(False)
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            resolved = "decoder_wrapped"
        elif hasattr(model, "layers"):
            resolved = "decoder"
        elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            resolved = "bert"
        else:
            resolved = model_type
        fwd = model

    return fwd, tokenizer, resolved


def embed_text(model, tokenizer, text, layer_idx, device, max_length=512):
    """Get mean-pooled hidden state at `layer_idx` for a single text. Returns (dim,) tensor."""
    enc = tokenizer(text, truncation=True, max_length=max_length,
                    padding=False, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
    hidden = outputs.hidden_states[layer_idx].float()  # (1, seq, dim)
    mask = attention_mask.unsqueeze(-1).float()
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return pooled.squeeze(0)  # (dim,)


def abtt_clean(vec, pcs, mean_vec):
    """Apply ABTT cleaning to a single vector. All tensors on same device."""
    centered = vec - mean_vec
    proj = centered @ pcs.T @ pcs
    return centered - proj


def run_ig_for_query(input_ids, attention_mask, target_fn, emb_layer, n_steps):
    """Run LayerIntegratedGradients, return per-token scores as numpy array."""
    lig = LayerIntegratedGradients(target_fn, emb_layer)
    attr = lig.attribute(
        input_ids,
        additional_forward_args=(attention_mask,),
        n_steps=n_steps,
        internal_batch_size=1,
    )
    return attr.sum(dim=-1).squeeze(0).detach().cpu().float().numpy()  # (seq_len,)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layers = parse_layers(args.layers)

    pairs = pd.read_csv(args.pairs_csv)
    n_queries = len(pairs)
    print(f"Loaded {n_queries} query-partner triples")

    # Load model
    print(f"Loading model: {args.model_name} ({args.model_type})")
    fwd_model, tokenizer, resolved_type = load_model(
        args.model_name, args.model_type, args.half_precision,
        args.trust_remote_code, device,
    )
    emb_layer = get_embedding_layer(fwd_model, resolved_type)

    # Pre-tokenize all query texts
    print("Pre-tokenizing queries...")
    query_texts = []
    query_encodings = []
    for _, row in pairs.iterrows():
        text = load_texts([row["query_path"]])[0]
        query_texts.append(text)
        enc = tokenizer(text, truncation=True, max_length=args.max_length,
                        padding=False, return_tensors="pt")
        query_encodings.append(enc)

    max_seq_len = max(e["input_ids"].shape[1] for e in query_encodings)
    print(f"  max_seq_len = {max_seq_len}")

    # Load partner texts
    print("Loading partner texts...")
    pos_texts = [load_texts([p])[0] for p in pairs["pos_partner_path"]]
    neg_texts = [load_texts([p])[0] for p in pairs["neg_partner_path"]]

    pc_dir = Path(args.pc_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for layer in layers:
        print(f"\n=== Layer {layer} ===")
        out_path = out_dir / f"layer{layer}_retrieval_attr.npz"
        if out_path.exists():
            print(f"  Already exists, skipping: {out_path}")
            continue

        layer_start = time.time()

        # Load PCs
        pc_path = pc_dir / f"layer{layer}_pcs.npz"
        if not pc_path.exists():
            print(f"  WARNING: {pc_path} not found, skipping")
            continue
        pc_data = np.load(pc_path)
        pcs = torch.from_numpy(pc_data["pcs"][:args.D]).to(device).float()
        mean_vec = torch.from_numpy(pc_data["mean_vec"]).to(device).float()

        # Pre-compute partner embeddings for all pairs at this layer
        print("  Pre-computing partner embeddings...")
        pos_raw_embs = []
        pos_abtt_embs = []
        neg_raw_embs = []
        neg_abtt_embs = []

        for qi in range(n_queries):
            pos_emb = embed_text(fwd_model, tokenizer, pos_texts[qi],
                                 layer, device, args.max_length)
            neg_emb = embed_text(fwd_model, tokenizer, neg_texts[qi],
                                 layer, device, args.max_length)
            pos_raw_embs.append(pos_emb)
            pos_abtt_embs.append(abtt_clean(pos_emb, pcs, mean_vec))
            neg_raw_embs.append(neg_emb)
            neg_abtt_embs.append(abtt_clean(neg_emb, pcs, mean_vec))

        print(f"  Partner embeddings computed: {n_queries} positive + {n_queries} negative")

        # Allocate output arrays
        ig_baseline_pos = np.zeros((n_queries, max_seq_len), dtype=np.float32)
        ig_abtt_pos = np.zeros((n_queries, max_seq_len), dtype=np.float32)
        ig_baseline_neg = np.zeros((n_queries, max_seq_len), dtype=np.float32)
        ig_abtt_neg = np.zeros((n_queries, max_seq_len), dtype=np.float32)
        seq_lengths = np.zeros(n_queries, dtype=np.int32)
        all_input_ids = np.zeros((n_queries, max_seq_len), dtype=np.int64)

        # Run IG for each query
        for qi in range(n_queries):
            enc = query_encodings[qi]
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            seq_len = input_ids.shape[1]
            seq_lengths[qi] = seq_len
            all_input_ids[qi, :seq_len] = enc["input_ids"][0].numpy()

            # Build 4 target functions for this query
            t_base_pos = BaselineCosSimTarget(
                fwd_model, layer, pos_raw_embs[qi]).to(device)
            t_abtt_pos = ABTTCosSimTarget(
                fwd_model, layer, pos_abtt_embs[qi], pcs, mean_vec).to(device)
            t_base_neg = BaselineCosSimTarget(
                fwd_model, layer, neg_raw_embs[qi]).to(device)
            t_abtt_neg = ABTTCosSimTarget(
                fwd_model, layer, neg_abtt_embs[qi], pcs, mean_vec).to(device)

            try:
                scores_bp = run_ig_for_query(input_ids, attention_mask,
                                             t_base_pos, emb_layer, args.n_steps)
                scores_ap = run_ig_for_query(input_ids, attention_mask,
                                             t_abtt_pos, emb_layer, args.n_steps)
                scores_bn = run_ig_for_query(input_ids, attention_mask,
                                             t_base_neg, emb_layer, args.n_steps)
                scores_an = run_ig_for_query(input_ids, attention_mask,
                                             t_abtt_neg, emb_layer, args.n_steps)

                ig_baseline_pos[qi, :seq_len] = scores_bp[:seq_len]
                ig_abtt_pos[qi, :seq_len] = scores_ap[:seq_len]
                ig_baseline_neg[qi, :seq_len] = scores_bn[:seq_len]
                ig_abtt_neg[qi, :seq_len] = scores_an[:seq_len]
            except Exception as e:
                print(f"  ERROR query {qi}: {e}")

            if (qi + 1) % 10 == 0:
                elapsed = time.time() - layer_start
                rate = (qi + 1) / elapsed
                remaining = (n_queries - qi - 1) / max(rate, 1e-6)
                print(f"  [{qi+1}/{n_queries}] {rate:.2f} q/s, ~{remaining/60:.0f} min left")

            # Memory cleanup every 10 queries
            if (qi + 1) % 10 == 0:
                del t_base_pos, t_abtt_pos, t_base_neg, t_abtt_neg
                torch.cuda.empty_cache()
                gc.collect()

        np.savez(
            out_path,
            ig_baseline_pos=ig_baseline_pos,
            ig_abtt_pos=ig_abtt_pos,
            ig_baseline_neg=ig_baseline_neg,
            ig_abtt_neg=ig_abtt_neg,
            seq_lengths=seq_lengths,
            input_ids=all_input_ids,
        )

        elapsed = time.time() - layer_start
        print(f"  Layer {layer} done in {elapsed/60:.1f} min → {out_path}")

    print("\nAll layers complete.")


if __name__ == "__main__":
    main()
