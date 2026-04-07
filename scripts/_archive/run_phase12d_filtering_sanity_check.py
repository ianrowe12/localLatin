"""Phase 12d: Filtering Sanity Check — Does simple token filtering match ABTT?

Compares 4 conditions on the exact same documents and layers:
  (a) Baseline:        mean-pool ALL tokens → cosine sim → AUCROC
  (b) Filter (no empty): mean-pool only non-empty tokens → AUCROC
  (c) Filter (content-only): mean-pool only 3+ char content tokens → AUCROC
  (d) ABTT (all tokens): ABTT-clean all tokens, mean-pool → AUCROC

For each (model, layer), produces all 4 metrics.
Output: runs/phase12d/filtering_results.csv + bar chart figure

Usage:
  python scripts/run_phase12d_filtering_sanity_check.py \
    --model_name bowphs/LaTa --model_type t5 \
    --layers 1,4,8,12 --max_length 512 --D 10
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from canon_retrieval import (
    l2_normalize,
    similarity_matrix,
    upper_triangle,
    upper_triangle_labels,
)
from pair_evaluation import safe_auc_roc
from sif_abtt import compute_pcs
from token_filtering import classify_token_for_filtering


def parse_args():
    p = argparse.ArgumentParser(description="Phase 12d filtering sanity check.")
    p.add_argument("--model_name", required=True)
    p.add_argument("--model_type", required=True, choices=["t5", "bert", "decoder"])
    p.add_argument("--layers", required=True, help="e.g. '1,4,8,12'")
    p.add_argument("--split_csv", default="runs/phase9/phase9_split.csv")
    p.add_argument("--out_dir", default="runs/phase12d")
    p.add_argument("--D", type=int, default=10, help="Number of PCs for ABTT")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--half_precision", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")
    return p.parse_args()


def load_model(model_name, model_type, half_precision, trust_remote_code, device):
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
    else:
        fwd = AutoModel.from_pretrained(model_name, **kwargs)
        fwd.to(device).eval()
        fwd.requires_grad_(False)

    return fwd, tokenizer


def get_hidden_states(model, tokenizer, text, layer_idx, device, max_length=512):
    """Get per-token hidden states at layer_idx. Returns (hidden, attention_mask, input_ids)."""
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
    hidden = outputs.hidden_states[layer_idx].float().cpu().numpy()  # (1, seq, dim)
    mask = attention_mask.cpu().numpy()  # (1, seq)
    ids = input_ids.cpu().numpy()  # (1, seq)
    return hidden[0], mask[0], ids[0]  # squeeze batch dim


def mean_pool_with_mask(hidden, mask):
    """Mean pool with a given mask. Returns (dim,) vector."""
    mask_f = mask.astype(np.float32)
    if mask_f.sum() == 0:
        return np.zeros(hidden.shape[1], dtype=np.float32)
    return (hidden * mask_f[:, None]).sum(axis=0) / mask_f.sum()


def build_token_masks(input_ids, attention_mask, tokenizer):
    """Build 3 masks: all, no-empty, content-only."""
    seq_len = int(attention_mask.sum())
    tokens = [tokenizer.decode([tid]) for tid in input_ids[:seq_len].tolist()]
    cats = [classify_token_for_filtering(t) for t in tokens]

    # Pad to full length
    full_len = len(attention_mask)

    mask_all = attention_mask.copy()

    mask_no_empty = np.zeros(full_len, dtype=np.float32)
    for i in range(seq_len):
        if cats[i] != "empty":
            mask_no_empty[i] = 1.0

    mask_content_only = np.zeros(full_len, dtype=np.float32)
    for i in range(seq_len):
        if cats[i] == "content":
            mask_content_only[i] = 1.0

    return mask_all.astype(np.float32), mask_no_empty, mask_content_only


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layers = [int(l) for l in args.layers.split(",")]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load split
    split = pd.read_csv(args.split_csv)
    print(f"Loaded {len(split)} files from split CSV")

    # Separate train/test
    train_mask = split["split"] == "train"
    test_mask = split["split"] == "test"
    train_paths = split.loc[train_mask, "path"].tolist()
    test_paths = split.loc[test_mask, "path"].tolist()
    test_folder_ids = split.loc[test_mask, "folder_id"].values
    test_is_query = split.loc[test_mask, "is_test_query"].values.astype(bool)

    n_train = len(train_paths)
    n_test = len(test_paths)
    print(f"  Train: {n_train}, Test: {n_test}")

    # Load model
    print(f"Loading model: {args.model_name} ({args.model_type})")
    model, tokenizer = load_model(
        args.model_name, args.model_type, args.half_precision,
        args.trust_remote_code, device,
    )

    # Model slug for output naming
    slug = args.model_name.replace("/", "_")

    # Read all texts
    print("Loading texts...")
    train_texts = []
    for p in train_paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            train_texts.append(f.read())
    test_texts = []
    for p in test_paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            test_texts.append(f.read())

    results = []

    for layer in layers:
        print(f"\n=== Layer {layer} ===")
        t0 = time.time()

        # ── Extract token-level hidden states for all files ──
        # Train: only need document-level embeddings for PCA fitting
        print("  Extracting train embeddings (all tokens)...")
        train_embs = []
        for i, text in enumerate(train_texts):
            hidden, mask, ids = get_hidden_states(model, tokenizer, text, layer, device, args.max_length)
            emb = mean_pool_with_mask(hidden, mask.astype(np.float32))
            train_embs.append(emb)
            if (i + 1) % 200 == 0:
                print(f"    [{i+1}/{n_train}]")
        train_embs = np.array(train_embs)

        # Fit ABTT on train embeddings
        print("  Fitting ABTT PCs on train set...")
        pcs, mean_vec = compute_pcs(train_embs, args.D, center=True)

        # Test: extract hidden states and build 3 embedding conditions
        print("  Extracting test embeddings with 4 conditions...")
        emb_baseline = []       # (a) all tokens, raw pool
        emb_no_empty = []       # (b) filter out empty
        emb_content_only = []   # (c) content only
        emb_abtt = []           # (d) all tokens, ABTT cleaned

        for i, text in enumerate(test_texts):
            hidden, mask, ids = get_hidden_states(model, tokenizer, text, layer, device, args.max_length)

            # Build masks
            m_all, m_no_empty, m_content = build_token_masks(ids, mask, tokenizer)

            # (a) Baseline: pool all tokens
            e_base = mean_pool_with_mask(hidden, m_all)
            emb_baseline.append(e_base)

            # (b) No-empty: pool non-empty tokens
            e_noempty = mean_pool_with_mask(hidden, m_no_empty)
            emb_no_empty.append(e_noempty)

            # (c) Content only: pool only 3+ char tokens
            e_content = mean_pool_with_mask(hidden, m_content)
            emb_content_only.append(e_content)

            # (d) ABTT: clean all-token pooled embedding
            centered = e_base - mean_vec
            proj = centered @ pcs.T @ pcs
            e_abtt = centered - proj
            emb_abtt.append(e_abtt)

            if (i + 1) % 100 == 0:
                print(f"    [{i+1}/{n_test}]")

        # Convert to arrays
        emb_baseline = np.array(emb_baseline)
        emb_no_empty = np.array(emb_no_empty)
        emb_content_only = np.array(emb_content_only)
        emb_abtt = np.array(emb_abtt)

        # ── Evaluate all 4 conditions ──
        conditions = {
            "baseline": emb_baseline,
            "filter_no_empty": emb_no_empty,
            "filter_content_only": emb_content_only,
            "abtt_D10": emb_abtt,
        }

        for cond_name, embs in conditions.items():
            embs_norm = l2_normalize(embs)
            sim = similarity_matrix(embs_norm)

            sims_ut = upper_triangle(sim)
            labels_ut = upper_triangle_labels(test_folder_ids)

            aucroc = safe_auc_roc(sims_ut, labels_ut)
            from canon_retrieval import accuracy_at_k
            acc1 = accuracy_at_k(sim, test_folder_ids, test_is_query, k=1)
            acc3 = accuracy_at_k(sim, test_folder_ids, test_is_query, k=3)

            # Cosine gap
            pos_mask = labels_ut.astype(bool)
            same_avg = float(np.mean(sims_ut[pos_mask])) if pos_mask.any() else 0
            diff_avg = float(np.mean(sims_ut[~pos_mask])) if (~pos_mask).any() else 0

            results.append({
                "model": args.model_name,
                "layer": layer,
                "condition": cond_name,
                "aucroc": aucroc,
                "acc_at_1": acc1,
                "acc_at_3": acc3,
                "same_avg_cos": same_avg,
                "diff_avg_cos": diff_avg,
                "cos_gap": same_avg - diff_avg,
            })
            print(f"    {cond_name:25s}  AUCROC={aucroc:.4f}  Acc@1={acc1:.4f}  Gap={same_avg-diff_avg:.4f}")

        elapsed = time.time() - t0
        print(f"  Layer {layer} done in {elapsed/60:.1f} min")

    # Save results
    df = pd.DataFrame(results)
    out_csv = out_dir / f"filtering_results_{slug}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Filtering vs ABTT")
    print("=" * 80)
    for layer in layers:
        ldf = df[df["layer"] == layer]
        print(f"\nLayer {layer}:")
        print(f"  {'Condition':25s}  {'AUCROC':>8s}  {'Acc@1':>8s}  {'Cos Gap':>8s}")
        print(f"  {'-'*25}  {'-'*8}  {'-'*8}  {'-'*8}")
        for _, row in ldf.iterrows():
            print(f"  {row['condition']:25s}  {row['aucroc']:8.4f}  {row['acc_at_1']:8.4f}  {row['cos_gap']:8.4f}")


if __name__ == "__main__":
    main()
