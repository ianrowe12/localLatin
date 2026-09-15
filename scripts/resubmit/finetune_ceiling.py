"""Supervised fine-tuning REFERENCE CEILING (issues #123, #138, #194, #210).

This is not a proposed method. It is an upper reference point: how far does the
retrieval task move when the encoder is allowed to see supervision that the
zero-shot + post-processing pipeline never gets? Everything is trained on the
TRAIN split only, with a DEV slice carved out of train by DIRECTORY for model
selection, and the TEST split is untouched until the final evaluation.

**The model is a parameter.** #123 ran one ceiling, on LaTa, and "it is the best
model" is not a reason a reviewer accepts for fine-tuning exactly one of six.
#194 therefore runs the identical recipe on a second, non-Latin-pretrained
encoder (Qwen3-Embedding-0.6B), and #210 on a third (KaLM-mini), which owns the
paper's best zero-shot ABTT cell. Same objective, optimiser, schedule, batch
size, seed, dev carve and early stopping; only ``--model_name`` and the labels
change. Every ceiling that is run is reported, whichever side of its own
zero-shot row it lands on. The model families the paper uses are shaped
differently -- a T5 seq2seq whose *encoder stack* is extracted, and a
decoder-only stack that IS the model -- so :class:`Encoder` dispatches on the
config rather than assuming T5.

Pipeline
--------
1. ``pairs``    Build positive pairs from train directories with >= 2 files.
                A fixed fraction of those directories is held out as DEV. The
                carve depends only on the split and the seed, so every model
                trains on exactly the same pairs.
2. ``train``    Contrastive fine-tuning of the encoder with mean pooling and
                in-batch negatives (symmetric InfoNCE, i.e. the objective
                behind sentence-transformers MultipleNegativesRankingLoss).
                Model selection on DEV directory accuracy@1 each epoch.
3. ``extract``  Mean-pooled embeddings for all 1,705 labelled files at every
                encoder layer, written in the canonical
                ``phase9_bases/<slug>/hidden_mean_tokempty/`` layout so the
                paper's evaluators can read them unchanged.
4. ``evaluate`` The SAME evaluator as the paper (``run_resubmit_evaluate``):
                Task A AUROC + cosine gap, Task B assignment accuracy and
                dir_acc@1 with tau learned on train, for baseline / ABTT(D=10) /
                ABTT(D swept on train).
5. ``mseed``    The paper's 5-seed Task B protocol (``run_taskb_mseed``) at the
                selected layers, for both the fine-tuned and the pre-trained
                encoder.

Pooling, token filtering (``tokenizer_empty``), max_length and row order all
match the paper's extraction exactly; ``--parity_check`` proves it by re-running
extraction with the *pre-trained* weights and diffing against that model's
cached baseline embeddings.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "resubmit"))

from canon_retrieval import (  # noqa: E402
    l2_normalize,
    load_texts,
    similarity_matrix,
    upper_triangle,
    upper_triangle_labels,
)
from finetune_pairs import (  # noqa: E402
    DevPoint,
    PairData,
    batch_pairs_by_round,
    build_pairs,
    is_better_checkpoint,
)
from embedding_alignment import AlignmentResolver  # noqa: E402
from pair_evaluation import safe_auc_roc  # noqa: E402
from token_filtering import build_token_keep_lookup, torch_token_keep_mask  # noqa: E402

import run_resubmit_evaluate as paper_eval  # noqa: E402
import run_taskb_mseed as paper_mseed  # noqa: E402
from canon_split_v2 import canon_taskb_query_reference_split  # noqa: E402


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

STAGES = ("pairs", "train", "extract", "evaluate", "mseed", "report")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Supervised fine-tuning reference ceiling.")
    p.add_argument("--split_csv", required=True, help="phase_resubmit_split.csv (row order = embedding row order).")
    p.add_argument("--data_root", default=str(REPO_ROOT), help="Root the split CSV 'path' column resolves against.")
    p.add_argument("--model_name", default="bowphs/LaTa",
                   help="HuggingFace id of the encoder to fine-tune; also names the "
                        "pre-trained comparison rows and the parity-check cache.")
    p.add_argument("--ft_model_label", default="bowphs/LaTa-ft",
                   help="Label used for the fine-tuned model in result rows and the bases slug.")
    p.add_argument("--display_name", default="LaTa",
                   help="Short name used in the comparison CSV's 'system' column and the "
                        "generated table, e.g. 'LaTa' or 'Qwen3-0.6B'.")
    p.add_argument("--results_prefix", default="finetune_lata",
                   help="Filename stem for this model's result CSVs, so two ceilings "
                        "can share one results directory without overwriting each other.")
    p.add_argument("--trust_remote_code", action="store_true",
                   help="Pass trust_remote_code to the tokenizer and model loaders.")
    p.add_argument("--grad_checkpointing", action="store_true",
                   help="Trade compute for activation memory during training. The "
                        "gradients are identical, so this changes the memory footprint "
                        "and nothing about the recipe; it is recorded in run_info.json.")

    p.add_argument("--out_dir", required=True, help="Run outputs (checkpoint, dev curve, configs).")
    p.add_argument("--bases_root", required=True,
                   help="Embedding cache root; files land in <bases_root>/phase9_bases/<slug>/hidden_mean_tokempty/.")
    p.add_argument("--baseline_bases_root", required=True,
                   help="Root of the paper's pre-trained LaTa embeddings (for parity check and mseed comparison).")
    p.add_argument("--baseline_results_csv", required=True,
                   help="phase_resubmit_results.csv, for the pre-trained LaTa comparison rows.")
    p.add_argument("--results_dir", required=True, help="Where the result CSVs are written.")
    p.add_argument("--tex_out", default="", help="Optional path for the generated LaTeX table rows.")

    p.add_argument("--stages", default="all",
                   help=f"Comma-separated subset of {STAGES}, or 'all'.")

    # data / dev carve
    p.add_argument("--dev_dir_frac", type=float, default=0.15,
                   help="Fraction of train directories with >=2 files held out as DEV.")

    # optimisation
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--batch_pairs", type=int, default=16, help="Positive pairs per batch (2x sequences).")
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--warmup_frac", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=3, help="Epochs without dev improvement before stopping.")
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # encoding
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--token_filter", default="tokenizer_empty")
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--layers", default="", help="Layers to extract, e.g. '1-12'. Empty = all encoder blocks.")

    # evaluation
    p.add_argument("--D_values", default="1,2,3,5,7,10")
    p.add_argument("--mseed_M", type=int, default=5)
    p.add_argument("--mseed_base_seed", type=int, default=42)

    p.add_argument("--tex_extra_run", action="append", default=[], metavar="SPEC",
                   help="Another finished ceiling to print ABOVE this one in --tex_out, "
                        "as '<display_name>:<results_prefix>:<out_dir>'. Its comparison "
                        "and 5-seed CSVs are read from --results_dir and its caption "
                        "facts from <out_dir>/run_info.json. Repeatable.")
    p.add_argument("--parity_check", action="store_true",
                   help="Re-extract with pre-trained weights and diff against the cached baseline embeddings.")
    p.add_argument("--cpu", action="store_true", help="Force CPU (for the evaluate/mseed stages).")
    return p.parse_args()


def parse_layer_spec(spec: str, n_blocks: int) -> List[int]:
    if not spec.strip():
        return list(range(1, n_blocks + 1))
    from cli_utils import parse_layers
    layers = parse_layers(spec)
    bad = [l for l in layers if l < 0 or l > n_blocks]
    if bad:
        raise ValueError(f"Layers out of range 0..{n_blocks}: {bad}")
    return layers


def model_slug(name: str) -> str:
    return name.replace("/", "_")


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #

def count_blocks(module) -> int:
    """How many transformer blocks a loaded stack has.

    The paper's models are three shapes and each hides its block list somewhere
    different: a T5 encoder stack in ``.block``, a decoder-only stack in
    ``.layers``, a BERT encoder in ``.encoder.layer``. Counting them is what
    turns ``--layers "1-N"`` and the "train on the last layer" rule into
    something the caller does not have to restate per model.
    """
    blocks = getattr(module, "block", None)          # T5Stack
    if blocks is None:
        blocks = getattr(module, "layers", None)     # Llama/Qwen-style stack
    if blocks is None:
        inner = getattr(module, "encoder", None)     # BertModel
        blocks = getattr(inner, "layer", None) if inner is not None else None
    if blocks is None:
        inner = getattr(module, "model", None)       # a wrapped decoder
        blocks = getattr(inner, "layers", None) if inner is not None else None
    if blocks is None:
        raise ValueError(
            f"Cannot find the transformer block list on {type(module).__name__}; "
            "add its attribute path to count_blocks()."
        )
    return len(blocks)


class Encoder:
    """One model's encoder stack with the paper's mean pooling and token filter.

    Dispatch is on the config, not on the model name. A seq2seq checkpoint
    (LaTa, PhilTa, mT5) contributes only its encoder, which is what the paper
    extracts from; every other checkpoint IS the stack, and ``AutoModel``
    already drops the LM head, which is how ``extract_encoder_cli`` loads the
    decoder-only models whose zero-shot rows these numbers are compared against.
    """

    def __init__(self, model_name: str, token_filter: str, max_length: int, device: str,
                 trust_remote_code: bool = False, grad_checkpointing: bool = False):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if getattr(config, "is_encoder_decoder", False):
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, trust_remote_code=trust_remote_code
            )
            self.encoder = model.get_encoder() if hasattr(model, "get_encoder") else model.encoder
            self.family = "seq2seq_encoder"
        else:
            self.encoder = AutoModel.from_pretrained(
                model_name, trust_remote_code=trust_remote_code
            )
            self.family = "single_stack"
            # A decoder stack would otherwise build a KV cache it never reads,
            # and refuse to checkpoint its activations.
            if hasattr(self.encoder, "config"):
                self.encoder.config.use_cache = False
        self.grad_checkpointing = bool(grad_checkpointing)
        if self.grad_checkpointing:
            # Recompute activations in the backward pass instead of storing
            # them. The gradients are identical; only the memory bill changes.
            self.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.encoder.to(device)
        self.device = device
        self.max_length = max_length
        keep = build_token_keep_lookup(self.tokenizer, token_filter)
        self.token_keep_lookup = keep
        self.n_blocks = count_blocks(self.encoder)

    def tokenize(self, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            list(texts),
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    def mean_pool(self, hidden: torch.Tensor, input_ids: torch.Tensor,
                  attention_mask: torch.Tensor) -> torch.Tensor:
        keep = torch_token_keep_mask(input_ids, attention_mask, self.token_keep_lookup)
        denom = keep.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (hidden * keep.unsqueeze(-1)).sum(dim=1) / denom

    def forward_pooled(self, texts: Sequence[str], layer: int = -1) -> torch.Tensor:
        """Pooled hidden state at one layer, with gradients (training path)."""
        enc = self.tokenize(texts)
        out = self.encoder(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = out.hidden_states[layer]
        return self.mean_pool(hidden, enc["input_ids"], enc["attention_mask"])

    @torch.no_grad()
    def encode_layers(
        self,
        texts: Sequence[str],
        layers: Sequence[int],
        batch_size: int,
        log_every: int = 0,
    ) -> Dict[int, np.ndarray]:
        """Mean-pooled embeddings for every requested layer in one pass per batch."""
        self.encoder.eval()
        buckets: Dict[int, List[np.ndarray]] = {l: [] for l in layers}
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = self.tokenize(batch)
            out = self.encoder(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
            for layer in layers:
                pooled = self.mean_pool(
                    out.hidden_states[layer], enc["input_ids"], enc["attention_mask"]
                )
                buckets[layer].append(pooled.float().cpu().numpy().astype(np.float32))
            if log_every and (start // batch_size) % log_every == 0:
                print(f"    encoded {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
        return {l: np.concatenate(v, axis=0) for l, v in buckets.items()}


# --------------------------------------------------------------------------- #
# Dev metric
# --------------------------------------------------------------------------- #

def dev_metrics(emb: np.ndarray, folder_ids: np.ndarray) -> Dict[str, float]:
    """Retrieval quality inside the held-out dev pool.

    dir_acc_at_1 is the fraction of dev files whose nearest other dev file sits
    in the same directory; every dev directory has >= 2 files, so the metric is
    well defined for every row. AUROC is over all dev pairs.
    """
    emb_norm = l2_normalize(emb)
    sim = similarity_matrix(emb_norm)
    sims = upper_triangle(sim)          # strict upper triangle: diagonal excluded
    labels = upper_triangle_labels(folder_ids)

    np.fill_diagonal(sim, -np.inf)
    nearest = np.argmax(sim, axis=1)
    acc1 = float(np.mean(folder_ids[nearest] == folder_ids))

    auc = float(safe_auc_roc(sims, labels))
    return {"dev_dir_acc_at_1": acc1, "dev_aucroc": auc, "n_dev_files": int(len(folder_ids))}


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train_contrastive(
    enc: Encoder,
    texts: Sequence[str],
    pair_data: PairData,
    folder_ids: np.ndarray,
    args: argparse.Namespace,
    out_dir: Path,
) -> Tuple[Dict[str, torch.Tensor], pd.DataFrame, Dict[str, float]]:
    """Symmetric InfoNCE over positive pairs with in-batch negatives."""
    device = enc.device
    dev_texts = [texts[r] for r in pair_data.dev_rows]
    dev_fids = folder_ids[pair_data.dev_rows]

    py_rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Epoch 0 = the pre-trained encoder, so the curve shows what training buys.
    base_emb = enc.encode_layers(dev_texts, [enc.n_blocks], args.eval_batch_size)[enc.n_blocks]
    best = dev_metrics(base_emb, dev_fids)
    history = [{"epoch": 0, "train_loss": float("nan"), **best}]
    print(f"  epoch 0 (pre-trained): dev_acc@1={best['dev_dir_acc_at_1']:.4f} "
          f"dev_auroc={best['dev_aucroc']:.4f}", flush=True)

    best_state = {k: v.detach().cpu().clone() for k, v in enc.encoder.state_dict().items()}
    best_epoch = 0
    best_score = best["dev_dir_acc_at_1"]
    best_tiebreak = best["dev_aucroc"]

    n_batches = len(batch_pairs_by_round(
        pair_data.train_pairs, pair_data.train_pair_dirs, args.batch_pairs, random.Random(0)
    ))
    total_steps = max(1, n_batches * args.epochs)
    warmup_steps = max(1, int(args.warmup_frac * total_steps))

    optim = torch.optim.AdamW(
        enc.encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    use_amp = device.startswith("cuda")
    amp_dtype = torch.bfloat16

    step = 0
    epochs_without_gain = 0
    # Patience is measured against the best *trained* epoch, not against epoch 0.
    # Epoch 0 stays selectable (if fine-tuning never beats the pre-trained
    # encoder on dev, that is the honest answer), but it must not be able to
    # end the run before training has had a chance.
    patience_ref = -float("inf")
    for epoch in range(1, args.epochs + 1):
        enc.encoder.train()
        batches = batch_pairs_by_round(
            pair_data.train_pairs, pair_data.train_pair_dirs, args.batch_pairs, py_rng
        )
        epoch_loss, n_seen = 0.0, 0
        for batch in batches:
            anchors = [texts[pair_data.train_pairs[i][0]] for i in batch]
            positives = [texts[pair_data.train_pairs[i][1]] for i in batch]
            optim.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                pooled = enc.forward_pooled(list(anchors) + list(positives))
                pooled = F.normalize(pooled.float(), dim=-1)
                a, b = pooled[: len(batch)], pooled[len(batch) :]
                logits = (a @ b.T) / args.temperature
                target = torch.arange(len(batch), device=device)
                loss = 0.5 * (F.cross_entropy(logits, target)
                              + F.cross_entropy(logits.T, target))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.encoder.parameters(), args.max_grad_norm)
            optim.step()
            sched.step()
            step += 1
            epoch_loss += float(loss.detach()) * len(batch)
            n_seen += len(batch)

        mean_loss = epoch_loss / max(1, n_seen)
        emb = enc.encode_layers(dev_texts, [enc.n_blocks], args.eval_batch_size)[enc.n_blocks]
        m = dev_metrics(emb, dev_fids)
        history.append({"epoch": epoch, "train_loss": mean_loss, **m})
        improved = is_better_checkpoint(
            DevPoint(epoch, m["dev_dir_acc_at_1"], m["dev_aucroc"]),
            DevPoint(best_epoch, best_score, best_tiebreak),
            len(pair_data.dev_rows),
        )
        print(f"  epoch {epoch}: loss={mean_loss:.4f} dev_acc@1={m['dev_dir_acc_at_1']:.4f} "
              f"dev_auroc={m['dev_aucroc']:.4f}{'  *' if improved else ''}", flush=True)
        if improved:
            best_score = m["dev_dir_acc_at_1"]
            best_tiebreak = m["dev_aucroc"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in enc.encoder.state_dict().items()}

        if m["dev_dir_acc_at_1"] > patience_ref:
            patience_ref = m["dev_dir_acc_at_1"]
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= args.patience:
                print(f"  early stop after epoch {epoch} "
                      f"({args.patience} epochs without dev gain)", flush=True)
                break

    hist_df = pd.DataFrame(history)
    selection = {
        "selected_epoch": best_epoch,
        "dev_dir_acc_at_1": best_score,
        "dev_aucroc": best_tiebreak,
        "n_dev_files": len(pair_data.dev_rows),
        "n_dev_dirs": len(pair_data.dev_dirs),
        "epochs_run": int(hist_df["epoch"].max()),
        "n_batches_per_epoch": n_batches,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    hist_df.to_csv(out_dir / "dev_curve.csv", index=False)
    torch.save(best_state, out_dir / "encoder_best.pt")
    with open(out_dir / "selection.json", "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)
    return best_state, hist_df, selection


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

def bases_dir(bases_root: Path, label: str) -> Path:
    return bases_root / "phase9_bases" / model_slug(label) / "hidden_mean_tokempty"


MANIFEST_COLUMNS = ("file_id", "folder_id", "filename", "path")


def write_row_manifest(out_dir: Path, split_meta: pd.DataFrame) -> Path:
    """Record, beside the matrices, the row order they were written in.

    Extraction here follows the split CSV, so the split's own order *is* the
    cache order. Saying so on disk is what keeps the cache self-describing:
    without a ``meta.csv`` an ``AlignmentResolver`` falls back to a
    ``row_order.csv`` at an ancestor bases root, which records the *paper's*
    extraction order and would permute these rows into the wrong labels.
    """
    cols = [c for c in MANIFEST_COLUMNS if c in split_meta.columns]
    if not ({"path", "filename"} & set(cols)):
        raise ValueError(
            "split_meta needs a 'path' or 'filename' column to write a row manifest"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "meta.csv"
    split_meta[cols].to_csv(path, index=False)
    return path


def extract_and_save(
    enc: Encoder, texts: Sequence[str], layers: Sequence[int],
    out_dir: Path, batch_size: int, split_meta: pd.DataFrame,
) -> Dict[int, np.ndarray]:
    if len(split_meta) != len(texts):
        raise ValueError(
            f"{len(texts)} texts but {len(split_meta)} split rows; the manifest "
            "would not describe the matrices."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    embs = enc.encode_layers(texts, layers, batch_size, log_every=20)
    for layer, emb in embs.items():
        np.save(out_dir / f"hidden_layer{layer}_embeddings.npy", emb)
        np.save(out_dir / f"hidden_layer{layer}_embeddings_norm.npy", l2_normalize(emb))
    manifest = write_row_manifest(out_dir, split_meta)
    print(f"  wrote {len(embs)} layers to {out_dir} (row order in {manifest.name})", flush=True)
    return embs


def parity_check(
    enc_pre: Encoder, texts: Sequence[str], layers: Sequence[int],
    baseline_bases_root: Path, batch_size: int, aligner: AlignmentResolver,
    model_name: str = "bowphs/LaTa",
) -> Dict[str, float]:
    """Confirm this script's extraction reproduces the paper's cached embeddings.

    ``texts`` are read in split-CSV order, while the cached matrix is frozen in
    the order the paper's extractor walked the corpus. A relabelling permutes
    one and not the other, so the reference is loaded through ``aligner``
    (built on the same split as ``texts``) rather than with ``np.load``:
    otherwise a pure re-ordering would show up as a parity failure.

    ``model_name`` picks which cached matrix to diff against, so a second
    ceiling is checked against its own zero-shot rows rather than LaTa's.
    """
    ref_dir = bases_dir(baseline_bases_root, model_name)
    probe = [l for l in layers if (ref_dir / f"hidden_layer{l}_embeddings.npy").exists()]
    probe = probe[-1:] + probe[:1]  # last and first available layer
    if not probe:
        return {"parity_layers": 0}
    embs = enc_pre.encode_layers(texts, sorted(set(probe)), batch_size)
    report: Dict[str, float] = {"parity_layers": len(embs)}
    for layer, emb in embs.items():
        ref = aligner.load(ref_dir / f"hidden_layer{layer}_embeddings.npy")
        diff = float(np.max(np.abs(emb - ref)))
        cos = float(np.mean(np.sum(l2_normalize(emb) * l2_normalize(ref), axis=1)))
        report[f"parity_layer{layer}_max_abs_diff"] = diff
        report[f"parity_layer{layer}_mean_cosine"] = cos
        print(f"  parity layer {layer}: max|diff|={diff:.3e} mean cosine={cos:.6f}", flush=True)
    return report


# --------------------------------------------------------------------------- #
# Evaluation with the paper's evaluator
# --------------------------------------------------------------------------- #

EVAL_METHODS = ("baseline", "abtt_fixed", "abtt_optimal")


def evaluate_layers(
    split_meta: pd.DataFrame,
    emb_dir: Path,
    layers: Sequence[int],
    model_label: str,
    D_values: List[int],
    aligner: AlignmentResolver,
) -> pd.DataFrame:
    rows = []
    for layer in layers:
        path = emb_dir / f"hidden_layer{layer}_embeddings.npy"
        if not path.exists():
            print(f"    layer {layer}: missing {path}, skipping", flush=True)
            continue
        emb_all = aligner.load(path)
        if emb_all.shape[0] != len(split_meta):
            raise ValueError(f"{path}: {emb_all.shape[0]} rows vs {len(split_meta)} split rows")
        for method in EVAL_METHODS:
            row = paper_eval.evaluate_single(
                emb_all=emb_all,
                split_meta=split_meta,
                method=method,
                D=10,
                sif_a=0.001,
                model_name=model_label,
                repr_name="hidden",
                pooling_src="mean",
                layer=layer,
                D_values=D_values,
            )
            if row is not None:
                rows.append(row)
                print(f"    layer {layer:>2} {method:<13} AUROC={row['aucroc']:.4f} "
                      f"dir_acc@1={row['dir_acc_at_1']:.4f} assign={row['overall_assignment_acc']:.4f}",
                      flush=True)
    return pd.DataFrame(rows)


def select_layer(df: pd.DataFrame, method: str, by: str) -> Optional[pd.Series]:
    """Layer selection exactly as the paper's headline tables: on TRAIN metrics."""
    sub = df[df["method"] == method]
    if sub.empty:
        return None
    return sub.loc[sub[by].idxmax()]


# --------------------------------------------------------------------------- #
# Multi-seed Task B (paper protocol)
# --------------------------------------------------------------------------- #

def run_mseed(
    split_meta: pd.DataFrame,
    configs: Sequence[Tuple[str, Path, int, str]],
    M: int,
    base_seed: int,
    D_values: List[int],
    aligner: AlignmentResolver,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Paper's 5-seed Task B protocol, restricted to the selected configurations.

    configs: (model_label, embedding_dir, layer, method)
    """
    per_seed: List[Dict] = []
    cache: Dict[Tuple[str, int], np.ndarray] = {}
    for seed in range(base_seed, base_seed + M):
        meta = canon_taskb_query_reference_split(split_meta.copy(), random_seed=seed)
        for label, emb_dir, layer, method in configs:
            key = (str(emb_dir), layer)
            if key not in cache:
                cache[key] = aligner.load(
                    emb_dir / f"hidden_layer{layer}_embeddings.npy"
                )
            row = paper_mseed.evaluate_model_for_seed(
                emb_all=cache[key],
                meta=meta,
                model_name=label,
                repr_name="hidden",
                pooling="mean",
                layer=layer,
                method=method,
                D_values=D_values,
                sif_a=0.001,
                top_k=5,
            )
            if row is not None:
                row["seed"] = seed
                per_seed.append(row)
                print(f"    seed {seed} {label} L{layer} {method}: "
                      f"dir_acc@1={row['dir_acc_at_1']:.4f}", flush=True)

    all_df = pd.DataFrame(per_seed)
    if all_df.empty:
        return all_df, all_df
    metrics = ["dir_acc_at_1", "dir_acc_at_3", "existing_acc", "new_acc",
               "overall_assignment_acc", "tau"]
    agg = (
        all_df.groupby(["model", "method", "layer"])[metrics]
        .agg(["mean", "std"])
    )
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg["n_seeds"] = M
    return all_df, agg


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def system_order(display_name: str) -> List[Tuple[str, str, bool]]:
    """The four comparison rows, named after the model under test.

    The names are what ``build_headline_tables`` and the generated table match
    on, so they are derived from one ``--display_name`` rather than repeated as
    literals per model.
    """
    return [
        (f"{display_name} (pre-trained)", "baseline", False),
        (f"{display_name} (pre-trained) + ABTT", "abtt_optimal", False),
        (f"{display_name} (fine-tuned)", "baseline", True),
        (f"{display_name} (fine-tuned) + ABTT", "abtt_optimal", True),
    ]


def build_comparison(
    ft_df: pd.DataFrame, base_df: pd.DataFrame, display_name: str = "LaTa"
) -> pd.DataFrame:
    """One row per system, with Task A and Task B each at their own selected layer.

    Layer selection follows the paper's headline tables: Task A by train AUROC,
    Task B by train directory accuracy at rank 1.
    """
    rows = []
    for system, method, is_ft in system_order(display_name):
        df = ft_df if is_ft else base_df
        a = select_layer(df, method, "train_aucroc")
        b = select_layer(df, method, "train_dir_acc_at_1")
        if a is None or b is None:
            continue
        rows.append({
            "system": system,
            "finetuned": is_ft,
            "method": method,
            "taskA_layer": int(a["layer"]),
            "taskA_aucroc": float(a["aucroc"]),
            "taskA_cosine_gap": float(a["gap"]),
            "taskA_D": int(a["D"]),
            "taskB_layer": int(b["layer"]),
            "taskB_assignment_acc": float(b["overall_assignment_acc"]),
            "taskB_dir_acc_at_1": float(b["dir_acc_at_1"]),
            "taskB_dir_acc_at_3": float(b["dir_acc_at_3"]),
            "taskB_existing_acc": float(b["existing_acc"]),
            "taskB_new_acc": float(b["new_acc"]),
            "taskB_tau": float(b["tau"]),
            "taskB_D": int(b["D"]),
        })
    return pd.DataFrame(rows)


class CeilingFacts:
    """The run-specific numbers the generated caption and notes quote.

    They were literals in the caption until benchmark v1 changed the dev carve
    and the pair count under them: a re-run then silently shipped a table whose
    prose described the previous run. Every one of them is now read off the run
    that produced the rows above it.
    """

    def __init__(
        self,
        pair_data: "PairData",
        split_meta: pd.DataFrame,
        D_values: Sequence[int],
        selection: Optional[Dict[str, object]],
        epoch_budget: Optional[int],
        ft_results: pd.DataFrame,
        display_name: str = "",
    ):
        # Empty when one model owns the table: the sentences then read as they
        # always did. Set it when the table carries more than one ceiling, so
        # every claim says which model it is about.
        self.display_name = display_name
        self.n_all_train_pairs = pair_data.n_all_train_pairs
        self.n_fit_pairs = len(pair_data.train_pairs)
        self.n_fit_dirs = len(pair_data.fit_dirs)
        self.n_dev_dirs = len(pair_data.dev_dirs)
        self.n_dev_files = len(pair_data.dev_rows)
        self.D_values = sorted(int(d) for d in D_values)

        self.selected_epoch = None if selection is None else int(selection["selected_epoch"])
        self.epochs_run = None if selection is None else int(selection["epochs_run"])
        self.epoch_budget = epoch_budget

        # How much of the test query pool sits in a directory that supplied a
        # training pair. Witnesses inside a directory are near-duplicates, so
        # this is the number that says how far the ceiling is flattered.
        fit = set(pair_data.fit_dirs)
        queries = split_meta[split_meta["is_test_query"].astype(bool)]
        self.n_test_queries = int(len(queries))
        self.n_test_queries_touched = int(queries["folder_id"].isin(fit).sum())
        self.pct_test_queries_touched = (
            100.0 * self.n_test_queries_touched / self.n_test_queries
            if self.n_test_queries else 0.0
        )

        # Is abtt_optimal a tuned optimum or a boundary hit? Answered from the
        # rows rather than asserted.
        self.n_optimal_rows = 0
        self.n_optimal_at_max_D = 0
        self.optimal_matches_fixed = None
        if not ft_results.empty and "D" in ft_results.columns:
            opt = ft_results[ft_results["method"] == "abtt_optimal"]
            fixed = ft_results[ft_results["method"] == "abtt_fixed"]
            self.n_optimal_rows = int(len(opt))
            self.n_optimal_at_max_D = int((opt["D"] == max(self.D_values)).sum())
            if len(opt) and len(opt) == len(fixed):
                merged = opt.merge(fixed, on="layer", suffixes=("_o", "_f"))
                self.optimal_matches_fixed = bool(
                    len(merged) == len(opt)
                    and np.allclose(merged["aucroc_o"], merged["aucroc_f"])
                    and np.allclose(merged["dir_acc_at_1_o"], merged["dir_acc_at_1_f"])
                )

    @classmethod
    def from_dict(cls, facts: Dict[str, object], display_name: str = "") -> "CeilingFacts":
        """Rebuild the facts of a run that finished in another job.

        A two-model table is rendered by whichever run reports last, and the
        other model's pair counts, epoch and $D$ sweep are already on disk in
        its ``run_info.json``. Reading them back beats recomputing them from a
        checkpoint this process does not have.
        """
        obj = cls.__new__(cls)
        obj.__dict__.update(facts)
        obj.display_name = display_name
        return obj

    def _who(self, lead: str = " for ") -> str:
        return f"{lead}{self.display_name}" if self.display_name else ""

    def d_grid_tex(self) -> str:
        return r"\{" + ",".join(str(d) for d in self.D_values) + r"\}"

    def abtt_caption_sentence(self) -> str:
        if not self.n_optimal_rows:
            return ""
        if self.n_optimal_at_max_D == self.n_optimal_rows:
            tail = (
                " so the tuned and fixed variants coincide."
                if self.optimal_matches_fixed
                else " so $D$ was never free to go higher."
            )
            return (
                f" ABTT rows{self._who()} sweep $D$ per layer on train and select "
                f"$D={max(self.D_values)}$ everywhere, which is the top of the grid "
                f"${self.d_grid_tex()}$," + tail
            )
        return (
            f" ABTT rows{self._who()} sweep $D$ per layer on train over the grid "
            f"${self.d_grid_tex()}$; {self.n_optimal_at_max_D} of {self.n_optimal_rows} "
            f"layer rows select the top of the grid."
        )

    def epoch_caption_sentence(self) -> str:
        if self.selected_epoch is None:
            return ""
        if self.selected_epoch == 0:
            return (
                f" Model selection{self._who()} kept the pre-trained encoder (epoch 0): "
                "no training epoch improved dev directory accuracy."
            )
        budget = f"{self.epoch_budget}-epoch budget" if self.epoch_budget else "epoch budget"
        if self.epochs_run is not None and self.selected_epoch == self.epochs_run:
            return (
                f" The selected checkpoint{self._who()} is epoch {self.selected_epoch}, "
                f"the terminal epoch of the sweep under the {budget}, so this is a ceiling "
                f"at this training budget rather than an asymptote."
            )
        return (
            f" The selected checkpoint{self._who()} is epoch {self.selected_epoch} of "
            f"{self.epochs_run} run under the {budget}, so this is a ceiling at this "
            f"training budget rather than an asymptote."
        )

    def notes(self) -> List[str]:
        """This run's bullet notes, wrapped into LaTeX comment lines.

        Bullets only: the section header is written once by ``write_tex``, so a
        two-model table does not repeat it.
        """
        bullets: List[str] = []
        if self.selected_epoch == 0:
            bullets.append(
                "Model selection kept epoch 0, the PRE-TRAINED encoder: no training epoch "
                "beat it on dev directory accuracy."
            )
        elif self.selected_epoch is not None:
            if self.epochs_run is not None and self.selected_epoch == self.epochs_run:
                bullets.append(
                    f"Epoch {self.selected_epoch} is the LAST epoch run under the "
                    f"{self.epoch_budget}-epoch budget, so the ceiling is 'at this training "
                    f"budget', not an asymptote."
                )
            else:
                bullets.append(
                    f"Epoch {self.selected_epoch} of {self.epochs_run} run was selected on "
                    f"dev directory accuracy."
                )
        if self.n_optimal_rows:
            grid = ",".join(str(d) for d in self.D_values)
            if self.n_optimal_at_max_D == self.n_optimal_rows:
                same = (
                    ", and equals abtt_fixed in every row"
                    if self.optimal_matches_fixed else ""
                )
                bullets.append(
                    f"abtt_optimal selects D={max(self.D_values)} at every layer, the maximum "
                    f"of the D grid {{{grid}}}{same}. Boundary hit, not a tuned optimum."
                )
            else:
                bullets.append(
                    f"abtt_optimal sweeps the D grid {{{grid}}} and selects its top value in "
                    f"{self.n_optimal_at_max_D} of {self.n_optimal_rows} layer rows."
                )
        bullets.append(
            "The pre-trained rows are COPIED from the paper's results CSV, not rescored "
            "here; only the fine-tuned bases pass through evaluate_layers."
        )
        # Which way this cuts depends on the model: for LaTa the ceiling sits
        # below the zero-shot ABTT cells, so overstating it makes that reading
        # conservative; for Qwen3-0.6B it sits above them, so overstating it
        # weakens that reading instead. The note therefore states the fact and
        # leaves the direction to the row it annotates.
        bullets.append(
            f"Witnesses in one directory are near-duplicates, and "
            f"{self.n_test_queries_touched} of the {self.n_test_queries} test query files "
            f"({self.pct_test_queries_touched:.1f}%) sit in a directory that supplied "
            f"training pairs. No test file was trained on, but the ceiling is if anything "
            f"overstated, so read any margin it holds over a zero-shot row as an upper "
            f"bound on that margin."
        )

        prefix = f"{self.display_name}: " if self.display_name else ""
        out: List[str] = []
        for bullet in bullets:
            out += textwrap.wrap(
                prefix + bullet,
                width=79,
                initial_indent="%   - ",
                subsequent_indent="%     ",
            )
        return out


TEX_HEADER = "% generated table"


# Keys that describe the job that ran rather than the artifacts on disk. A job
# that did not load the encoder has these namespaced under "report_<key>" so it
# cannot overwrite the training job's account of how the weights were made.
JOB_SCOPED_KEYS = ("config", "total_seconds", "device")


def merge_run_info(
    existing: Dict[str, object], new: Dict[str, object], touched_model: bool
) -> Dict[str, object]:
    """Fold one job's record into ``run_info.json`` without dropping another's.

    The stages run in separate jobs: training, extraction and the parity check
    need a GPU, scoring does not. Both used to dump this file wholesale, so the
    scoring job silently deleted the GPU job's ``parity`` report, its
    ``selection``, its ``train_seconds`` and its ``grad_checkpointing`` flag.
    That is the provenance the paper's appendix quotes, and losing it left the
    appendix asserting a record that no longer existed on disk. Every key the
    current job did not produce is now carried through untouched.

    Three keys describe the *job* rather than the artifacts, so they are
    namespaced rather than merged: a scoring job's ``config``, ``total_seconds``
    and ``device`` land under ``report_config``, ``report_total_seconds`` and
    ``report_device``, leaving the training job's as the record of how the
    weights were made. ``device`` joined them in #210: scoring runs on the CPU
    partition by budget rule, so without namespacing it a scoring job rewrote
    the record to say the weights were trained on ``cpu``.
    ``touched_model`` is true exactly when this job loaded the encoder.
    """
    merged = dict(existing)
    for key, value in new.items():
        if key in JOB_SCOPED_KEYS and not touched_model and key in merged:
            merged[f"report_{key}"] = value
        else:
            merged[key] = value
    return merged


def write_run_info(path: Path, new: Dict[str, object], touched_model: bool) -> None:
    """Merge ``new`` into the record at ``path`` and write it back."""
    existing: Dict[str, object] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  {path} is not valid JSON; replacing it", file=sys.stderr)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(merge_run_info(existing, new, touched_model), f, indent=2, default=str)


def load_selection(out_dir: Path) -> Optional[Dict[str, object]]:
    """Read the training stage's selection.json, or None if it has not run."""
    path = Path(out_dir) / "selection.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@dataclass
class CeilingSection:
    """One model's block of the generated table.

    The table carries a ceiling per fine-tuned model (issue #194), and every
    caption claim -- pair count, selected epoch, whether the $D$ sweep hit its
    boundary -- is a property of one run, not of the table. Keeping them
    together in a section is what stops the second model inheriting the first
    model's prose.
    """

    display_name: str
    comparison: pd.DataFrame
    facts: Optional[CeilingFacts] = None
    mseed_agg: Optional[pd.DataFrame] = None


def _row_tex(row: pd.Series) -> str:
    def fmt(x: float, nd: int = 3) -> str:
        return f"{x:.{nd}f}"

    name = str(row["system"]).replace("_", r"\_")
    return (
        f"{name} & {fmt(row['taskA_aucroc'])}\\,\\textsubscript{{{row['taskA_layer']}}} "
        f"& {fmt(row['taskA_cosine_gap'])}\\,\\textsubscript{{{row['taskA_layer']}}} "
        f"& {fmt(100 * row['taskB_assignment_acc'], 1)}\\,\\textsubscript{{{row['taskB_layer']}}} "
        f"& {fmt(100 * row['taskB_dir_acc_at_1'], 1)}\\,\\textsubscript{{{row['taskB_layer']}}} \\\\"
    )


def _pairs_clause(sections: Sequence[CeilingSection]) -> str:
    """The shared pair count, phrased for the caption, or nothing.

    Both ceilings are carved from the same split with the same seed, so they
    see the same pairs; if a future run breaks that, the caption drops the
    number rather than quoting one model's count for both.
    """
    counts = {
        (sec.facts.n_fit_pairs, sec.facts.n_all_train_pairs)
        for sec in sections if sec.facts is not None
    }
    if len(counts) != 1:
        return "positive pairs available in the train split "
    n_fit, n_all = counts.pop()
    # The headline captions say "499 of the 565 positive train pairs"; this
    # table says the same thing in the same words (review of 2026-09-15).
    return f"{n_fit} of the {n_all} positive train pairs "


def load_extra_section(spec: str, results_dir: Path) -> CeilingSection:
    """Build a section from a ceiling that ran in an earlier job.

    ``spec`` is ``<display_name>:<results_prefix>:<out_dir>``. Everything the
    section needs is already on disk, so a two-model table does not require
    re-scoring the first model.
    """
    parts = spec.split(":")
    if len(parts) != 3:
        raise SystemExit(
            f"--tex_extra_run expects '<display_name>:<results_prefix>:<out_dir>', got {spec!r}"
        )
    display_name, prefix, out_dir = (part.strip() for part in parts)
    comp_path = results_dir / f"{prefix}_ceiling_comparison.csv"
    if not comp_path.exists():
        raise SystemExit(f"--tex_extra_run {spec!r}: no comparison CSV at {comp_path}")
    comparison = pd.read_csv(comp_path)

    facts = None
    info_path = Path(out_dir) / "run_info.json"
    if info_path.exists():
        saved = json.loads(info_path.read_text(encoding="utf-8")).get("caption_facts")
        if saved:
            facts = CeilingFacts.from_dict(saved, display_name)
    if facts is None:
        print(f"  {spec}: no caption_facts in {info_path}; its caption sentences are omitted",
              file=sys.stderr)

    mseed_path = results_dir / f"{prefix}_mseed_aggregated.csv"
    mseed = pd.read_csv(mseed_path) if mseed_path.exists() else None
    return CeilingSection(display_name, comparison, facts, mseed)


def write_tex(sections: Sequence[CeilingSection], path: Path) -> None:
    """Emit the table body. Generated file: edit the generator, not this.

    The header names no repository path: this directory ships to Overleaf
    (issue #117).
    """
    sections = [sec for sec in sections if not sec.comparison.empty]
    if not sections:
        raise ValueError("write_tex needs at least one non-empty comparison")
    names = [sec.display_name for sec in sections]
    subject = names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" and {names[-1]}"

    lines = [
        TEX_HEADER,
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{\textbf{Task A}} & \multicolumn{2}{c}{\textbf{Task B}} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"\textbf{System} & AUROC & \makecell{Cosine\\gap} & \makecell{Assign.\\acc.} "
        r"& \makecell{Dir.\\acc.@1} \\",
        r"\midrule",
    ]
    for i, sec in enumerate(sections):
        if i:
            lines.append(r"\midrule")
        for _, row in sec.comparison.iterrows():
            lines.append(_row_tex(row))

    encoder = "encoder is" if len(sections) == 1 else "encoders are"
    caption = (
        r"\caption{Supervised fine-tuning reference ceiling on " + subject +
        r". The fine-tuned " + encoder + r" trained contrastively on "
        + _pairs_clause(sections) +
        r"(in-batch negatives, symmetric InfoNCE), with a dev slice carved "
        r"out of train by directory for model selection; the test split is untouched "
        r"until this evaluation. Task A columns are test AUROC and cosine gap at the "
        r"layer chosen by train AUROC; Task B columns are test assignment accuracy and "
        r"directory accuracy at rank 1, in percent, at the layer chosen by train "
        r"directory accuracy, with $\tau$ learned on train. Layer indices are "
        r"subscripts."
    )
    for sec in sections:
        if sec.facts is not None:
            caption += sec.facts.abtt_caption_sentence()
            caption += sec.facts.epoch_caption_sentence()
    caption += (
        r" This is a reference ceiling, not a proposed method: it consumes supervision "
        r"the zero-shot pipeline never sees.}"
    )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        caption,
        r"\label{tab:finetune_ceiling}",
        r"\end{table}",
    ]
    notes: List[str] = []
    for sec in sections:
        if sec.facts is not None:
            notes += sec.facts.notes()
    if notes:
        lines += ["", "% Notes for whoever moves these rows into the paper:"] + notes
    mseed_lines = []
    for sec in sections:
        if sec.mseed_agg is not None and not sec.mseed_agg.empty:
            for _, r in sec.mseed_agg.iterrows():
                mseed_lines.append(
                    f"%   {r['model']} L{int(r['layer'])} {r['method']}: "
                    f"dir_acc@1 = {r['dir_acc_at_1_mean']:.3f} +/- {r['dir_acc_at_1_std']:.3f}"
                )
    if mseed_lines:
        lines.append("")
        lines.append(r"% 5-seed Task B (mean +/- std over seeds 42-46), same protocol as the")
        lines.append(r"% multi-seed appendix table:")
        lines += mseed_lines
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {path}", flush=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    stages = set(STAGES) if args.stages.strip() == "all" else {
        s.strip() for s in args.stages.split(",") if s.strip()
    }
    unknown = stages - set(STAGES)
    if unknown:
        raise SystemExit(f"Unknown stage(s): {sorted(unknown)}; valid: {STAGES}")

    t0 = time.time()
    out_dir = Path(args.out_dir)
    results_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    split_meta = pd.read_csv(args.split_csv)
    # Cached rows sit in extraction order; the split is sorted by directory, so a
    # label correction permutes one and not the other. Align on filename.
    aligner = AlignmentResolver(split_meta)
    data_root = Path(args.data_root)
    paths = [str(data_root / p) for p in split_meta["path"].tolist()]
    folder_ids = split_meta["folder_id"].values
    D_values = [int(d) for d in args.D_values.split(",")]

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    ft_dir = bases_dir(Path(args.bases_root), args.ft_model_label)
    base_emb_dir = bases_dir(Path(args.baseline_bases_root), args.model_name)

    # ---- pairs ------------------------------------------------------------ #
    pair_data = build_pairs(split_meta, args.dev_dir_frac, args.seed)
    print(f"Train directories with >=2 files: {pair_data.n_train_multi_dirs} "
          f"({pair_data.n_all_train_pairs} positive pairs total)")
    print(f"DEV carve: {len(pair_data.dev_dirs)} directories / {len(pair_data.dev_rows)} files "
          f"held out; {len(pair_data.fit_dirs)} directories / {len(pair_data.train_pairs)} pairs "
          f"used for training")
    pd.DataFrame({"folder_id": pair_data.dev_dirs}).to_csv(out_dir / "dev_directories.csv", index=False)
    pd.DataFrame(
        {"a_row": [a for a, _ in pair_data.train_pairs],
         "b_row": [b for _, b in pair_data.train_pairs],
         "folder_id": pair_data.train_pair_dirs}
    ).to_csv(out_dir / "train_pairs.csv", index=False)

    texts: Optional[List[str]] = None
    enc: Optional[Encoder] = None
    layers: List[int] = []
    run_info: Dict[str, object] = {"config": vars(args), "device": device}

    needs_model = bool({"train", "extract"} & stages) or args.parity_check
    if needs_model:
        texts = load_texts(paths)
        enc = Encoder(
            args.model_name, args.token_filter, args.max_length, device,
            trust_remote_code=args.trust_remote_code,
            grad_checkpointing=args.grad_checkpointing,
        )
        layers = parse_layer_spec(args.layers, enc.n_blocks)
        run_info["model_family"] = enc.family
        run_info["n_blocks"] = enc.n_blocks
        run_info["grad_checkpointing"] = enc.grad_checkpointing
        print(f"Model {args.model_name} ({enc.family}): {enc.n_blocks} blocks; "
              f"extracting layers {layers}"
              + ("; gradient checkpointing ON" if enc.grad_checkpointing else ""))

        if args.parity_check:
            print(f"Parity check against the paper's cached {args.model_name} embeddings...")
            run_info["parity"] = parity_check(
                enc, texts, layers, Path(args.baseline_bases_root),
                args.eval_batch_size, aligner, model_name=args.model_name,
            )

    # ---- train ------------------------------------------------------------ #
    if "train" in stages:
        print("Contrastive fine-tuning...")
        t_train = time.time()
        _, _, selection = train_contrastive(
            enc, texts, pair_data, folder_ids, args, out_dir
        )
        run_info["selection"] = selection
        run_info["train_seconds"] = time.time() - t_train
        print(f"Selected epoch {selection['selected_epoch']} "
              f"(dev dir_acc@1 {selection['dev_dir_acc_at_1']:.4f}, "
              f"dev AUROC {selection['dev_aucroc']:.4f})")

    # ---- extract ---------------------------------------------------------- #
    if "extract" in stages:
        ckpt = out_dir / "encoder_best.pt"
        if "train" not in stages:
            if not ckpt.exists():
                raise SystemExit(f"No checkpoint at {ckpt}; run the train stage first.")
        state = torch.load(ckpt, map_location="cpu")
        enc.encoder.load_state_dict(state)
        enc.encoder.to(device)
        print("Extracting fine-tuned embeddings for all labelled files...")
        extract_and_save(enc, texts, layers, ft_dir, args.eval_batch_size, split_meta)

    if enc is not None:
        del enc
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---- evaluate --------------------------------------------------------- #
    ft_results = pd.DataFrame()
    ft_results_path = results_dir / f"{args.results_prefix}_layer_results.csv"
    if "evaluate" in stages:
        eval_layers = layers or sorted(
            int(p.stem.split("_layer")[1].split("_")[0])
            for p in ft_dir.glob("hidden_layer*_embeddings.npy")
        )
        print("Evaluating fine-tuned embeddings with the paper's evaluator...")
        ft_results = evaluate_layers(
            split_meta, ft_dir, eval_layers, args.ft_model_label, D_values, aligner
        )
        ft_results.to_csv(ft_results_path, index=False)
        print(f"  wrote {ft_results_path}")
    elif ft_results_path.exists():
        ft_results = pd.read_csv(ft_results_path)

    # ---- report ----------------------------------------------------------- #
    comparison = pd.DataFrame()
    if "report" in stages and ft_results.empty:
        # Reporting used to skip silently here, which looks exactly like a
        # successful no-op run: it writes no comparison, no table, and a
        # run_info.json with no caption_facts, overwriting the one the real run
        # left behind.
        raise SystemExit(
            f"report stage: no layer results at {ft_results_path}. Run the evaluate "
            "stage first, or point --results_dir at the run that produced them."
        )
    if "report" in stages:
        base_all = pd.read_csv(args.baseline_results_csv)
        base_df = base_all[
            (base_all["model"] == args.model_name)
            & (base_all["pooling"] == "mean")
            & (base_all["repr"] == "hidden")
        ].copy()
        comparison = build_comparison(ft_results, base_df, args.display_name)
        comp_path = results_dir / f"{args.results_prefix}_ceiling_comparison.csv"
        comparison.to_csv(comp_path, index=False)
        print(f"  wrote {comp_path}")
        print(comparison.to_string(index=False))

    # ---- mseed ------------------------------------------------------------ #
    mseed_agg = pd.DataFrame()
    mseed_agg_path = results_dir / f"{args.results_prefix}_mseed_aggregated.csv"
    if "mseed" in stages and not comparison.empty:
        configs = []
        for _, r in comparison.iterrows():
            emb_dir = ft_dir if r["finetuned"] else base_emb_dir
            if not (emb_dir / f"hidden_layer{int(r['taskB_layer'])}_embeddings.npy").exists():
                print(f"    mseed: no embeddings for {r['system']}, skipping")
                continue
            label = args.ft_model_label if r["finetuned"] else args.model_name
            cfg = (label, emb_dir, int(r["taskB_layer"]), r["method"])
            if cfg not in configs:  # two systems can select the same layer
                configs.append(cfg)
        print(f"Task B multi-seed ({args.mseed_M} seeds from {args.mseed_base_seed})...")
        mseed_all, mseed_agg = run_mseed(
            split_meta, configs, args.mseed_M, args.mseed_base_seed, D_values, aligner
        )
        if not mseed_all.empty:
            mseed_all.to_csv(
                results_dir / f"{args.results_prefix}_mseed_all_seeds.csv", index=False
            )
            mseed_agg.to_csv(mseed_agg_path, index=False)
            print(mseed_agg.to_string(index=False))
    elif mseed_agg_path.exists():
        # Same fallback the layer results get: regenerating the table alone must
        # not silently drop the 5-seed block, and must not re-run five seeds to
        # keep it either.
        mseed_agg = pd.read_csv(mseed_agg_path)
        print(f"  reusing {mseed_agg_path}")

    # ---- LaTeX ------------------------------------------------------------ #
    if args.tex_out and not comparison.empty:
        # The caption quotes the run's own pair count, dev carve, selected epoch
        # and D sweep, so a re-run on a changed split cannot ship prose that
        # describes the previous one. The scoring stages run in a separate job
        # from training, so the selection comes off disk when it is not in hand.
        selection = load_selection(out_dir)
        if selection is None:
            print("  no selection.json; caption will omit the checkpoint sentence",
                  file=sys.stderr)
        extra = [load_extra_section(spec, results_dir) for spec in args.tex_extra_run]
        # Name every claim only when more than one ceiling shares the table;
        # a single-model table reads as it always did.
        named = args.display_name if extra else ""
        facts = CeilingFacts(
            pair_data=pair_data,
            split_meta=split_meta,
            D_values=D_values,
            selection=selection,
            epoch_budget=args.epochs,
            ft_results=ft_results,
            display_name=named,
        )
        run_info["caption_facts"] = {
            k: v for k, v in vars(facts).items() if k != "display_name"
        }
        mine = CeilingSection(
            args.display_name,
            comparison,
            facts,
            mseed_agg if not mseed_agg.empty else None,
        )
        write_tex(extra + [mine], Path(args.tex_out))

    run_info["total_seconds"] = time.time() - t0
    write_run_info(out_dir / "run_info.json", run_info, needs_model)
    print(f"Done in {run_info['total_seconds'] / 60:.1f} min.")


if __name__ == "__main__":
    main()
