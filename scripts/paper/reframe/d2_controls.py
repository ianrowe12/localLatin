#!/usr/bin/env python3
"""D2 (paper spine, section 6; approved 2026-10-08): the two deferred control runs, on CPU.

(i)  google/t5-efficient-base, the original T5 layout (ReLU feed-forward, tied embeddings)
     pretrained on C4 only, through ``p2x2_panel.py --stage compute`` exactly as the other
     controls were run: same corpus, mean pooling with the ``tokenizer_empty`` filter, encoder
     hidden states 1-12, Task A test AUROC, train top-PC share, effective rank and mean
     pairwise cosine. T5-base, T5-v1.1-base, LaBERTa and PhilBERTa are re-extracted by the same
     run and compared with the committed P2x2 rows to prove the run is comparable.
(ii) The projection (ABTT fit on the training split) on the four controls, with the paper's
     own code path: ``run_resubmit_evaluate.evaluate_single`` with ``abtt_fixed`` (D=10) and
     ``abtt_optimal`` (D chosen on training DirAcc@1 from 1,2,3,5,7,10). A gate first
     recomputes those cells for LaTa, PhilTa, mT5-base and LaBSE and requires them to equal
     ``phase_resubmit_results.csv``.

Stages (each reads the previous stage's outputs; run from the repo root):
  config      architecture and provenance check of the three T5 checkpoints from their pinned
              config, weights, model card and Mesh TF gin; tokenizer identity on the corpus.
              Exit 2 (stop) if t5-efficient-base is not ReLU / tied / 12 layers / C4 only.
              -> <out_dir>/d2_config_check.csv
  extract     one CPU forward pass per batch for all 12 layers (the CLIs rerun the encoder per
              layer; the pooled vectors are the same). T5 checkpoints: the encoder of
              AutoModelForSeq2SeqLM and ``extract_hidden_cli.pool_hidden``, as gen_extract.py;
              encoder-only checkpoints: AutoModel and ``extract_encoder_cli.pool_embeddings``.
              -> <bases>/<slug>/hidden_mean_tokempty/{hidden_layerN_embeddings.npy,meta.csv,
                 config.json}
  panel       p2x2_panel.py --stage compute on LaTa (gates a, b, d) and the five D2 caches,
              then the comparability check of the four re-extracted controls against the
              committed runs/active/reframe/p2x2/p2x2_layers.csv.
              -> <out_dir>/p2x2_layers.csv, p2x2_repro.csv, d2_repro.csv
  projection  baseline, abtt_fixed and abtt_optimal at layers 1-12 of the gate models and the
              D2 caches; hard gate against the published cells.
              -> <out_dir>/d2_projection.csv (d2_projection.rejected.csv on a gate failure)
  render      <out_dir>/d2_facts.md and two appendix tables (not \\input anywhere):
              overleaf_drafts/tables/d2_t5_efficient.tex, d2_controls_abtt.tex

  python scripts/paper/reframe/d2_controls.py --stage config
  python scripts/paper/reframe/d2_controls.py --stage extract --threads 16
  python scripts/paper/reframe/d2_controls.py --stage panel --bases_root <root>/runs/active/resubmit_bases
  python scripts/paper/reframe/d2_controls.py --stage projection --bases_root <root>/runs/active/resubmit_bases
  python scripts/paper/reframe/d2_controls.py --stage render
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
for _p in (HERE, REPO / "src", REPO / "scripts" / "resubmit"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gen_ft_geometry import COLLAPSE_PC1, HEADER  # noqa: E402

SPLIT_CSV = Path("runs/active/resubmit/data/phase_resubmit_split.csv")
RES_CSV = Path("runs/active/resubmit/results/phase_resubmit_results.csv")
P2X2_REF = Path("runs/active/reframe/p2x2/p2x2_layers.csv")
BASES_ROOT = Path("runs/active/resubmit_bases")
OUT_DIR = Path("runs/active/reframe/d2")
BASES = OUT_DIR / "bases"
TAB_DIR = Path("overleaf_drafts/tables")
SUBDIR = "hidden_mean_tokempty"
LAYERS = tuple(range(1, 13))
MID = tuple(range(2, 12))      # mid-depth: layers 2-11, the paper's collapsed range
COLLAPSE_AUROC = 0.70          # the paper's collapsed layer: baseline AUROC below 0.70
BAND = (0.962, 0.987)          # claim 2: every panel layer after ABTT, three decimals
D_VALUES = (1, 2, 3, 5, 7, 10)  # the paper's selection grid (slurm/resubmit/*evaluate*.sbatch)
D_FIXED = 10
SIF_A = 0.001

# display, HF id, architecture class, pinned revision (the HF main commit on 2026-10-08;
# T5-v1.1-base is the GEN pin of gen_english_sample.MODEL_REVISIONS).
D2_MODELS = [
    ("T5-efficient-base", "google/t5-efficient-base", "seq2seq",
     "1f496af4864196a638fd96fed0b34f0e5603f738"),
    ("T5-base", "google-t5/t5-base", "seq2seq", "a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1"),
    ("T5-v1.1-base", "google/t5-v1_1-base", "seq2seq", "b5fc947a416ea3cb079532cb3c2bbadeb7f800fc"),
    ("LaBERTa", "bowphs/LaBerta", "encoder", "94fab85783dca8a16529cda2b58760d03bd5d9c1"),
    ("PhilBERTa", "bowphs/PhilBerta", "encoder", "acfe65d43b93aeba226528904be120a9d30ef8aa"),
]
NEW = "T5-efficient-base"
CONTROLS = ("LaBERTa", "PhilBERTa", "T5-base", "T5-v1.1-base")  # the paper's four controls
GATE_MODELS = [  # panel models whose ABTT cells the paper prints (phase_resubmit_results.csv)
    ("LaTa", "bowphs/LaTa"), ("PhilTa", "bowphs/PhilTa"), ("mT5-base", "google/mt5-base"),
    ("LaBSE", "sentence-transformers/LaBSE"),
]
PANEL = ("LaTa", "PhilTa", "mT5-base", "LaBSE", "Qwen3-0.6B", "KaLM-mini")
PANEL_IDS = {"LaTa": "bowphs/LaTa", "PhilTa": "bowphs/PhilTa", "mT5-base": "google/mt5-base",
             "LaBSE": "sentence-transformers/LaBSE", "Qwen3-0.6B": "Qwen/Qwen3-Embedding-0.6B",
             "KaLM-mini": "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5"}
METHODS = ("baseline", "abtt_fixed", "abtt_optimal")

# Comparability of the re-extraction against the committed P2x2 rows (GPU extraction, #248).
# 1e-4 is the tolerance p2x2_panel.py already uses across hardware (its gate c).
REPRO_TOL = {"aucroc": (1e-4, False), "pc1": (1e-4, False), "erank": (1e-4, True),
             "mean_cos": (1e-4, False)}
PUBLISHED_TOL = 1e-6
PUBLISHED_COLS = ("aucroc", "dir_acc_at_1", "train_dir_acc_at_1")


def _slug(model_id: str) -> str:
    return model_id.replace("/", "_")


def _model(name: str):
    for m in D2_MODELS:
        if m[0] == name:
            return m
    raise SystemExit(f"unknown D2 model {name}; choose from {[m[0] for m in D2_MODELS]}")


# --------------------------------------------------------------------------- config
def parse_gin(text: str) -> Dict[str, str]:
    """The few Mesh TF gin bindings the layout and pretraining claims rest on."""
    want = {"MIXTURE_NAME": r"^MIXTURE_NAME\s*=\s*(.+)$",
            "dropout_rate": r"^dropout_rate\s*=\s*(.+)$",
            "num_layers": r"^num_layers\s*=\s*(.+)$",
            "d_ff": r"^d_ff\s*=\s*(.+)$",
            "shared_embedding": r"^Bitransformer\.shared_embedding\s*=\s*(.+)$",
            "encoder_activation": r"^encoder/DenseReluDense\.activation\s*=\s*(.+)$",
            "train_steps": r"^run\.train_steps\s*=\s*(.+)$"}
    out = {}
    for key, pat in want.items():
        m = re.search(pat, text, flags=re.M)
        out[key] = m.group(1).strip().strip("'\"") if m else ""
    return out


def layout_problems(row: Dict) -> List[str]:
    """Why a checkpoint is not the original T5 layout pretrained on C4 only (empty: it is)."""
    bad = []
    if row.get("feed_forward_proj") != "relu":
        bad.append(f"feed_forward_proj is {row.get('feed_forward_proj')!r}, not 'relu'")
    if row.get("tie_word_embeddings") is not True:
        bad.append(f"tie_word_embeddings is {row.get('tie_word_embeddings')!r}, not True")
    if row.get("lm_head_separate") is True:
        bad.append("the checkpoint stores an lm_head that differs from the input embedding")
    if int(row.get("num_layers") or 0) != 12:
        bad.append(f"num_layers is {row.get('num_layers')}, not 12")
    if row.get("gin_mixture") != "c4_v220_unsupervised":
        bad.append(f"pretraining mixture is {row.get('gin_mixture')!r}, not C4 only "
                   "('c4_v220_unsupervised')")
    if row.get("gin_shared_embedding") not in ("True", True):
        bad.append(f"gin shared_embedding is {row.get('gin_shared_embedding')!r}")
    if row.get("gin_encoder_activation") != "relu":
        bad.append(f"gin encoder activation is {row.get('gin_encoder_activation')!r}")
    if row.get("card_datasets") != "c4":
        bad.append(f"model card datasets are {row.get('card_datasets')!r}, not c4")
    return bad


def config_stage(args) -> int:
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoTokenizer

    from token_filtering import build_token_keep_lookup

    split = pd.read_csv(args.split_csv)
    texts = [Path(args.data_root, p).read_text(encoding="utf-8") for p in split["path"]]
    rows, ids = [], {}
    for name, model_id, arch, rev in D2_MODELS:
        if arch != "seq2seq":
            continue
        cfg = AutoConfig.from_pretrained(model_id, revision=rev)
        row = {"model": name, "model_id": model_id, "revision": rev,
               "feed_forward_proj": cfg.feed_forward_proj,
               "dense_act_fn": getattr(cfg, "dense_act_fn", ""),
               "is_gated_act": getattr(cfg, "is_gated_act", ""),
               "tie_word_embeddings": bool(cfg.tie_word_embeddings),
               "num_layers": cfg.num_layers, "d_model": cfg.d_model, "d_ff": cfg.d_ff,
               "vocab_size": cfg.vocab_size, "dropout_rate_cfg": cfg.dropout_rate}
        raw = json.loads(Path(hf_hub_download(model_id, "config.json", revision=rev)).read_text())
        row["tie_word_embeddings_in_file"] = raw.get("tie_word_embeddings", "absent (default True)")
        # The stored weights: an untied checkpoint stores an lm_head distinct from `shared`.
        try:
            from safetensors.torch import load_file
            sd = load_file(hf_hub_download(model_id, "model.safetensors", revision=rev))
        except Exception:  # noqa: BLE001 (no safetensors file for this revision)
            sd = torch.load(hf_hub_download(model_id, "pytorch_model.bin", revision=rev),
                            map_location="cpu", weights_only=True)
        has_head = "lm_head.weight" in sd
        row["lm_head_in_weights"] = has_head
        row["lm_head_separate"] = bool(has_head and not torch.equal(sd["lm_head.weight"],
                                                                     sd["shared.weight"]))
        # Tied checkpoints may store copies of the input embedding under other keys.
        row["keys_equal_to_shared"] = ",".join(
            k for k, v in sd.items() if k != "shared.weight"
            and v.shape == sd["shared.weight"].shape and torch.equal(v, sd["shared.weight"]))
        del sd
        card = Path(hf_hub_download(model_id, "README.md", revision=rev)).read_text()
        m = re.search(r"^datasets:\s*\n((?:-\s*.+\n)+)", card, flags=re.M)
        row["card_datasets"] = ",".join(x.strip()[1:].strip() for x in m.group(1).splitlines()) if m else ""
        try:
            gin = parse_gin(Path(hf_hub_download(model_id, "operative_config.gin", revision=rev)).read_text())
        except Exception:  # noqa: BLE001 (only the t5-efficient checkpoints ship a gin)
            gin = {}
        for k in ("MIXTURE_NAME", "dropout_rate", "num_layers", "d_ff", "shared_embedding",
                  "encoder_activation", "train_steps"):
            row[f"gin_{k.lower() if k == 'MIXTURE_NAME' else k}"] = gin.get(k, "")
        row["gin_mixture"] = row.pop("gin_mixture_name")
        tok = AutoTokenizer.from_pretrained(model_id, revision=rev)
        keep = build_token_keep_lookup(tok, "tokenizer_empty")
        enc = [tok(t, truncation=True, max_length=512)["input_ids"] for t in texts]
        ids[name] = (enc, [[i for i in x if keep[i]] for x in enc])
        rows.append(row)
    ref, ref_kept = ids["T5-base"]
    for row in rows:
        mine, kept = ids[row["model"]]
        # All input ids, and the ids the tokenizer_empty filter keeps in the mean.
        row["passages_tokenized_unlike_t5_base"] = int(sum(a != b for a, b in zip(mine, ref)))
        row["passages_pooled_ids_unlike_t5_base"] = int(sum(a != b for a, b in zip(kept, ref_kept)))
        row["extra_input_tokens_vs_t5_base"] = int(sum(len(a) - len(b) for a, b in zip(mine, ref)))
        row["layout_problems"] = "; ".join(layout_problems(row)) if row["model"] == NEW else ""
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "d2_config_check.csv", index=False)
    print(df.T.to_string())
    new = df[df["model"] == NEW].iloc[0]
    if new["layout_problems"]:
        print(f"STOP: {NEW} is not the original T5 layout pretrained on C4 only: "
              f"{new['layout_problems']}")
        return 2
    print(f"{NEW}: original T5 layout (ReLU, tied embeddings, 12 layers), C4 only: confirmed")
    return 0


# --------------------------------------------------------------------------- extract
def extract_one(name: str, args) -> Dict:
    import torch
    from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer
    from transformers.utils import cached_file

    from extract_encoder_cli import pool_embeddings
    from extract_hidden_cli import pool_hidden
    from token_filtering import build_token_keep_lookup

    _, model_id, arch, rev = _model(name)
    loaded = Path(cached_file(model_id, "config.json", revision=rev)).parent.name
    if loaded != rev:
        raise SystemExit(f"{model_id}: loaded snapshot {loaded} != pinned revision {rev}")
    split = pd.read_csv(args.split_csv)
    texts = [Path(args.data_root, p).read_text(encoding="utf-8") for p in split["path"]]
    if args.limit:
        split, texts = split.iloc[: args.limit], texts[: args.limit]

    t0 = time.time()
    tok_name, tok_id, _, tok_rev = _model(args.tokenizer_from or name)
    tok = AutoTokenizer.from_pretrained(tok_id, revision=tok_rev)
    if arch == "seq2seq":
        net = AutoModelForSeq2SeqLM.from_pretrained(model_id, revision=rev,
                                                    dtype=torch.float32).get_encoder()
        n_layers, dim = len(net.block), int(net.config.d_model)

        def pool(h, enc, keep):
            return pool_hidden(h, enc["attention_mask"], "mean", input_ids=enc["input_ids"],
                               token_keep_lookup=keep)
    else:
        net = AutoModel.from_pretrained(model_id, revision=rev, dtype=torch.float32)
        n_layers, dim = int(net.config.num_hidden_layers), int(net.config.hidden_size)

        def pool(h, enc, keep):
            return pool_embeddings(h, enc["attention_mask"], "mean", input_ids=enc["input_ids"],
                                   token_keep_lookup=keep)
    net.eval()
    keep = build_token_keep_lookup(tok, args.token_filter)
    t_load = time.time() - t0

    raw_len = np.array([len(tok(t)["input_ids"]) for t in texts])
    order = np.argsort(-np.minimum(raw_len, args.max_length), kind="stable")
    pooled = np.zeros((n_layers, len(texts), dim), dtype=np.float32)
    t1 = time.time()
    with torch.no_grad():
        for b in range(0, len(order), args.batch_size):
            idx = order[b:b + args.batch_size]
            enc = tok([texts[i] for i in idx], truncation=True, max_length=args.max_length,
                      padding=True, return_tensors="pt")
            out = net(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                      output_hidden_states=True, return_dict=True)
            for layer in range(1, n_layers + 1):
                pooled[layer - 1, idx] = pool(out.hidden_states[layer], enc, keep).numpy()
            if (b // args.batch_size) % 50 == 0:
                print(f"  {name}: {b + len(idx)}/{len(texts)} passages, "
                      f"{time.time() - t1:.0f}s", flush=True)
    t_fwd = time.time() - t1

    run_dir = Path(args.bases) / _slug(model_id) / SUBDIR
    run_dir.mkdir(parents=True, exist_ok=True)
    for layer in range(1, n_layers + 1):
        np.save(run_dir / f"hidden_layer{layer}_embeddings.npy", pooled[layer - 1])
    split[["filename", "split"]].to_csv(run_dir / "meta.csv", index=False)
    cfg = dict(model=name, model_name=model_id, revision=rev, arch=arch,
               tokenizer=f"{tok_id}@{tok_rev}", n_rows=len(texts),
               n_layers=n_layers, d_model=dim, pooling="mean", token_filter=args.token_filter,
               max_length=args.max_length, batch_size=args.batch_size, order="split_csv",
               device="cpu", torch_threads=torch.get_num_threads(), torch=torch.__version__,
               load_seconds=round(t_load, 1), forward_seconds=round(t_fwd, 1),
               n_truncated=int((raw_len > args.max_length).sum()),
               n_zero_rows=int((~np.any(pooled[-1] != 0, axis=1)).sum()),
               layer_indexing="hidden_states[1..N]; 1 = first block, N = last block output")
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(json.dumps(cfg), flush=True)
    return cfg


def extract_stage(args) -> int:
    import torch
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    names = [m[0] for m in D2_MODELS] if args.models is None else args.models
    for name in names:
        run_dir = Path(args.bases) / _slug(_model(name)[1]) / SUBDIR
        done = all((run_dir / f"hidden_layer{layer}_embeddings.npy").exists() for layer in LAYERS)
        if done and (run_dir / "meta.csv").exists() and not args.force and not args.limit:
            print(f"{name}: complete in {run_dir}, skipped (--force re-extracts)")
            continue
        extract_one(name, args)
    return 0


# --------------------------------------------------------------------------- panel
def compare_to_reference(new: pd.DataFrame, ref: pd.DataFrame,
                         models: Sequence[str]) -> pd.DataFrame:
    """Per (model, layer, metric): the re-extracted value against the committed P2x2 row."""
    a = new[new["model"].isin(models)].set_index(["model", "layer"])
    b = ref[ref["model"].isin(models)].set_index(["model", "layer"])
    recs = []
    for key in b.index:
        for metric, (tol, relative) in REPRO_TOL.items():
            if key not in a.index:
                recs.append({"model": key[0], "layer": key[1], "metric": metric, "value": np.nan,
                             "reference": float(b.loc[key, metric]), "abs_diff": np.nan,
                             "rel_diff": np.nan, "tol": tol,
                             "tol_kind": "relative" if relative else "absolute", "ok": False})
                continue
            v, r = float(a.loc[key, metric]), float(b.loc[key, metric])
            d = abs(v - r)
            shown = d / abs(r) if relative else d
            recs.append({"model": key[0], "layer": key[1], "metric": metric, "value": v,
                         "reference": r, "abs_diff": d, "rel_diff": d / abs(r) if r else np.nan,
                         "tol": tol, "tol_kind": "relative" if relative else "absolute",
                         "ok": bool(shown <= tol)})
    return pd.DataFrame(recs)


def panel_stage(args) -> int:
    import p2x2_panel as p2
    out = Path(args.out_dir)
    names = ["LaTa"] + [m[0] for m in D2_MODELS]
    status = p2.main(["--stage", "compute", "--split_csv", str(args.split_csv),
                      "--bases_root", str(args.bases_root), "--p2x2_bases", str(args.bases),
                      "--out_dir", str(out), "--models", *names])
    if status:
        print("p2x2_panel.py reproduction gate failed; stopping")
        return status
    new = pd.read_csv(out / "p2x2_layers.csv")
    ref = pd.read_csv(args.p2x2_ref)
    rep = compare_to_reference(new, ref, CONTROLS)
    rep.to_csv(out / "d2_repro.csv", index=False, float_format="%.10g")
    for (model, metric), g in rep.groupby(["model", "metric"], sort=False):
        print(f"repro {model:13s} {metric:8s} max abs {g['abs_diff'].max():.2e} "
              f"max rel {g['rel_diff'].max():.2e}: {'PASS' if g['ok'].all() else 'FAIL'}")
    if not rep["ok"].all():
        print(f"comparability check FAILED on {int((~rep['ok']).sum())} cell(s)")
        return 1
    print("comparability check passed: the re-extracted controls match the committed P2x2 rows")
    return 0


# --------------------------------------------------------------------------- projection
def projection_rows(emb: np.ndarray, split: pd.DataFrame, name: str, model_id: str,
                    layer: int) -> List[Dict]:
    import run_resubmit_evaluate as paper_eval
    rows = []
    for method in METHODS:
        r = paper_eval.evaluate_single(emb_all=emb, split_meta=split, method=method, D=D_FIXED,
                                       sif_a=SIF_A, model_name=model_id, repr_name="hidden",
                                       pooling_src="mean", layer=layer, D_values=list(D_VALUES))
        rows.append({"model": name, "model_id": model_id, "layer": layer, "method": method,
                     "D": int(r["D"]) if method != "baseline" else 0, "aucroc": r["aucroc"],
                     "train_aucroc": r["train_aucroc"], "dir_acc_at_1": r["dir_acc_at_1"],
                     "train_dir_acc_at_1": r["train_dir_acc_at_1"], "tau": r["tau"],
                     "overall_assignment_acc": r["overall_assignment_acc"]})
    return rows


def published_gate(df: pd.DataFrame, res_csv: Path) -> List[str]:
    pub = pd.read_csv(res_csv)
    pub = pub[(pub["repr"] == "hidden") & (pub["pooling"] == "mean")
              & pub["method"].isin(METHODS)].set_index(["model", "layer", "method"])
    problems = []
    gate = df[df["model"].isin([g[0] for g in GATE_MODELS])]
    for r in gate.itertuples():
        key = (r.model_id, r.layer, r.method)
        if key not in pub.index:
            problems.append(f"{r.model} L{r.layer} {r.method}: no published cell")
            continue
        p = pub.loc[key]
        for col in PUBLISHED_COLS:
            d = abs(getattr(r, col) - float(p[col]))
            if not d <= PUBLISHED_TOL:
                problems.append(f"{r.model} L{r.layer} {r.method} {col}: {getattr(r, col):.8f} "
                                f"vs published {float(p[col]):.8f}")
        if r.method != "baseline" and int(r.D) != int(p["D"]):
            problems.append(f"{r.model} L{r.layer} {r.method} D {r.D} vs published {int(p['D'])}")
    return problems


def projection_stage(args) -> int:
    from embedding_alignment import STATUS_UNVERIFIED, AlignmentResolver
    split = pd.read_csv(args.split_csv)
    resolver = AlignmentResolver(split)
    layers = args.layers or list(LAYERS)
    jobs = []
    if not args.skip_gate:
        jobs += [(n, mid, Path(args.bases_root) / "phase9_bases" / _slug(mid) / SUBDIR)
                 for n, mid in GATE_MODELS]
    names = [m[0] for m in D2_MODELS] if args.models is None else args.models
    jobs += [(n, _model(n)[1], Path(args.bases) / _slug(_model(n)[1]) / SUBDIR) for n in names]
    rows: List[Dict] = []
    t0 = time.time()
    for name, model_id, run_dir in jobs:
        for layer in layers:
            path = run_dir / f"hidden_layer{layer}_embeddings.npy"
            if resolver.aligner_for(path).status == STATUS_UNVERIFIED:
                raise SystemExit(f"{path}: no meta.csv beside the cache; refusing to score it by "
                                 "row position")
            new = projection_rows(resolver.load(path), split, name, model_id, layer)
            rows += new
            b, f, o = (x["aucroc"] for x in new)
            print(f"{name:17s} L{layer:<2d} AUROC base {b:.4f}  ABTT D=10 {f:.4f}  "
                  f"ABTT D={new[2]['D']:<2d} {o:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    print(resolver.summary())
    df = pd.DataFrame(rows)
    problems = [] if args.skip_gate else published_gate(df, args.res_csv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / ("d2_projection.rejected.csv" if problems or args.skip_gate or args.layers
                    else "d2_projection.csv")
    df.to_csv(target, index=False, float_format="%.10g")
    print(f"wrote {target} ({len(df)} rows)")
    n_gate = int(df["model"].isin([g[0] for g in GATE_MODELS]).sum())
    for p in problems:
        print("PUBLISHED MISMATCH:", p)
    if problems:
        print(f"projection gate FAILED with {len(problems)} problem(s)")
        return 1
    if not args.skip_gate:
        print(f"projection gate passed: {n_gate} recomputed cells equal the published ones "
              f"(aucroc, dir_acc_at_1, train_dir_acc_at_1 within {PUBLISHED_TOL:.0e}; D exact)")
    return 0


# --------------------------------------------------------------------------- read-out
def collapse_readout(layers_df: pd.DataFrame, name: str) -> Dict:
    """Spine D2 test: baseline AUROC below 0.70 with top-PC share at least 0.76 at mid-depth."""
    s = layers_df[layers_df["model"] == name].set_index("layer").sort_index()
    mid = s.loc[[l for l in MID if l in s.index]]
    low = mid[mid["aucroc"] < COLLAPSE_AUROC]
    both = low[low["pc1"] >= COLLAPSE_PC1]
    return {"model": name, "n_mid": len(mid), "n_low_auroc": len(low),
            "n_collapsed_high_pc1": len(both), "collapsed_layers": list(both.index),
            "low_auroc_layers": list(low.index), "mid_auroc_min": float(mid["aucroc"].min()),
            "mid_auroc_min_layer": int(mid["aucroc"].idxmin()),
            "pc1_max": float(s["pc1"].max()), "pc1_max_layer": int(s["pc1"].idxmax()),
            "collapses": bool(len(both) > 0)}


def in_band(x: float, band=BAND) -> bool:
    """Inside the printed band at the paper's three decimals."""
    return band[0] <= round(float(x), 3) <= band[1]


def band_readout(proj: pd.DataFrame, method: str) -> pd.DataFrame:
    out = []
    for name, g in proj[proj["method"] == method].groupby("model", sort=False):
        g = g.sort_values("layer")
        out.append({"model": name, "method": method, "n_layers": len(g),
                    "auroc_min": float(g["aucroc"].min()),
                    "auroc_min_layer": int(g.loc[g["aucroc"].idxmin(), "layer"]),
                    "auroc_max": float(g["aucroc"].max()),
                    "auroc_max_layer": int(g.loc[g["aucroc"].idxmax(), "layer"]),
                    "n_in_band": int(sum(in_band(x) for x in g["aucroc"])),
                    "all_in_band": bool(all(in_band(x) for x in g["aucroc"])),
                    "D_values": ",".join(str(int(d)) for d in g["D"])})
    return pd.DataFrame(out)


def panel_band(res_csv: Path, method: str) -> pd.DataFrame:
    """The six panel models' published ABTT cells (100 layers)."""
    pub = pd.read_csv(res_csv)
    pub = pub[(pub["repr"] == "hidden") & (pub["pooling"] == "mean") & (pub["method"] == method)]
    inv = {v: k for k, v in PANEL_IDS.items()}
    pub = pub[pub["model"].isin(inv)].assign(model=lambda d: d["model"].map(inv))
    return pub[["model", "layer", "method", "D", "aucroc"]]


# --------------------------------------------------------------------------- render
def write_t5_efficient_table(layers_df: pd.DataFrame, proj: pd.DataFrame, path: Path) -> None:
    by = layers_df.set_index(["model", "layer"])
    pj = proj[proj["method"] == "abtt_optimal"].set_index(["model", "layer"])
    n = f"{int(layers_df['n_train'].iloc[0]):,}".replace(",", "{,}")
    lines = [HEADER, "% python scripts/paper/reframe/d2_controls.py --stage render",
             r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{@{}rcccccc@{}}", r"\toprule",
             r"& \multicolumn{5}{c}{T5-efficient-base} & T5-base \\",
             r"\cmidrule(lr){2-6}\cmidrule(lr){7-7}",
             r"Layer & AUROC & PC1 & Rank & Cos & ABTT & AUROC \\", r"\midrule"]
    for layer in LAYERS:
        x, t = by.loc[(NEW, layer)], by.loc[("T5-base", layer)]
        lines.append(f"{layer} & {x['aucroc']:.3f} & {x['pc1']:.3f} & {x['erank']:.2f} & "
                     f"{x['mean_cos']:.3f} & {pj.loc[(NEW, layer), 'aucroc']:.3f} & "
                     f"{t['aucroc']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{T5-efficient-base, the original T5 layout (ReLU feed-forward, tied "
              r"embeddings) pretrained on C4 alone without dropout, on the Latin corpus. "
              r"AUROC: test ranking "
              r"AUROC of cosine on unmodified mean-pooled vectors. PC1: top-PC share, the share "
              r"of centered variance on the first principal component of the " + n +
              r" training passages. Rank: their entropy effective rank. Cos: their mean "
              r"pairwise cosine. ABTT: test AUROC after ABTT fit on the training passages, "
              r"with $D$ chosen on training DirAcc@1. Last column: T5-base, which has the same "
              r"layout and pools the same tokens of every passage and was pretrained on C4 "
              r"mixed with supervised tasks, with dropout; its vectors come from the same "
              r"extraction run.}",
              r"\label{tab:d2_t5_efficient}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def write_abtt_table(proj: pd.DataFrame, path: Path) -> None:
    models = list(CONTROLS)
    base = proj[proj["method"] == "baseline"].set_index(["model", "layer"])
    opt = proj[proj["method"] == "abtt_optimal"].set_index(["model", "layer"])
    head = " & ".join(r"\multicolumn{2}{c}{" + m + "}" for m in models)
    rules = "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(models)))
    lines = [HEADER, "% python scripts/paper/reframe/d2_controls.py --stage render",
             r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{2.5pt}",
             r"\begin{tabular}{@{}r" + "cc" * len(models) + r"@{}}", r"\toprule",
             "& " + head + r" \\", rules,
             "Layer & " + " & ".join(["Base & ABTT"] * len(models)) + r" \\", r"\midrule"]
    for layer in LAYERS:
        cells = [f"{base.loc[(m, layer), 'aucroc']:.3f} & {opt.loc[(m, layer), 'aucroc']:.3f}"
                 for m in models]
        lines.append(f"{layer} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Test ranking AUROC of the four control models on the Latin corpus, "
              r"before (Base: unmodified mean-pooled vectors) and after ABTT fit on the training "
              r"passages, with $D$ chosen per layer on training DirAcc@1 from 1, 2, 3, 5, 7 and "
              r"10, as for the panel.}",
              r"\label{tab:d2_controls_abtt}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def facts(layers_df, proj, rep, cfg, res_csv: Path, path: Path) -> None:
    L: List[str] = ["# D2 numbers (generated)", "",
                    "Generated by `scripts/paper/reframe/d2_controls.py --stage render`. Latin "
                    "corpus, unmodified mean-pooled encoder hidden states, layers 1-12. AUROC: "
                    "Task A test AUROC. PC1, effective rank, mean cosine: training passages.", ""]
    w = L.append
    if cfg is not None:
        w("## Checkpoints")
        w("| model | revision | feed_forward_proj | tied (config) | lm_head separate in weights "
          "| layers | d_ff | card datasets | gin mixture | gin dropout | passages tokenized "
          "unlike T5-base (all ids / pooled ids) |")
        w("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in cfg.itertuples():
            w(f"| {r.model} | `{r.revision[:12]}` | {r.feed_forward_proj} | "
              f"{r.tie_word_embeddings} | {r.lm_head_separate} | {r.num_layers} | {r.d_ff} | "
              f"{r.card_datasets} | {r.gin_mixture if isinstance(r.gin_mixture, str) else ''} | "
              f"{'' if pd.isna(r.gin_dropout_rate) else r.gin_dropout_rate} | "
              f"{r.passages_tokenized_unlike_t5_base} / {r.passages_pooled_ids_unlike_t5_base} |")
        w("")
    w("## (i) Collapse read-out: baseline AUROC < 0.70 with top-PC share >= "
      f"{COLLAPSE_PC1} at layers 2-11")
    w("| model | mid layers | AUROC < 0.70 | of those PC1 >= 0.76 | lowest mid AUROC (layer) | "
      "PC1 max (layer) | collapses |")
    w("|---|---|---|---|---|---|---|")
    for name in [NEW, "T5-base", "T5-v1.1-base", "LaBERTa", "PhilBERTa", "LaTa"]:
        if name not in set(layers_df["model"]):
            continue
        c = collapse_readout(layers_df, name)
        w(f"| {name} | {c['n_mid']} | {c['n_low_auroc']} | {c['n_collapsed_high_pc1']} | "
          f"{c['mid_auroc_min']:.3f} ({c['mid_auroc_min_layer']}) | {c['pc1_max']:.3f} "
          f"({c['pc1_max_layer']}) | {'yes' if c['collapses'] else 'no'} |")
    w("")
    w("## Per layer: AUROC / top-PC share / effective rank / mean pairwise cosine")
    for name in [NEW, "T5-base", "T5-v1.1-base", "LaBERTa", "PhilBERTa"]:
        s = layers_df[layers_df["model"] == name].sort_values("layer")
        if len(s):
            w(f"- {name}: " + "; ".join(f"{int(r.layer)}: {r.aucroc:.3f} / {r.pc1:.3f} / "
                                         f"{r.erank:.2f} / {r.mean_cos:.3f}" for r in s.itertuples()))
    w("")
    if rep is not None:
        w("## Comparability: this run against the committed P2x2 rows")
        w("| model | metric | layers | max abs diff | max rel diff | tolerance | all within |")
        w("|---|---|---|---|---|---|---|")
        for (m, metric), g in rep.groupby(["model", "metric"], sort=False):
            w(f"| {m} | {metric} | {len(g)} | {g['abs_diff'].max():.2e} | "
              f"{g['rel_diff'].max():.2e} | {g['tol'].iloc[0]:.0e} ({g['tol_kind'].iloc[0]}) | "
              f"{bool(g['ok'].all())} |")
        w("")
    if proj is not None:
        w(f"## (ii) The projection: test AUROC after ABTT, band {BAND[0]:.3f}-{BAND[1]:.3f} "
          "(three decimals)")
        w("| model | method | layers | min (layer) | max (layer) | layers in band | all in band | "
          "D per layer |")
        w("|---|---|---|---|---|---|---|---|")
        for method in ("abtt_optimal", "abtt_fixed"):
            for r in band_readout(proj, method).itertuples():
                w(f"| {r.model} | {method} | {r.n_layers} | {r.auroc_min:.3f} "
                  f"({r.auroc_min_layer}) | {r.auroc_max:.3f} ({r.auroc_max_layer}) | "
                  f"{r.n_in_band} | {r.all_in_band} | {r.D_values} |")
        w("")
        w("## Panel plus controls")
        w("| method | set | layers | AUROC min | AUROC max | layers below 0.962 | layers above 0.987 |")
        w("|---|---|---|---|---|---|---|")
        for method in ("abtt_optimal", "abtt_fixed"):
            pan = panel_band(res_csv, method)
            ctl = proj[(proj["method"] == method) & proj["model"].isin(CONTROLS)]
            new = proj[(proj["method"] == method) & (proj["model"] == NEW)]
            sets = [("six-model panel (published)", pan["aucroc"]),
                    ("four controls", ctl["aucroc"]),
                    ("ten models (panel + controls)", pd.concat([pan["aucroc"], ctl["aucroc"]])),
                    ("ten models + T5-efficient-base",
                     pd.concat([pan["aucroc"], ctl["aucroc"], new["aucroc"]]))]
            for label, x in sets:
                r3 = x.round(3)
                w(f"| {method} | {label} | {len(x)} | {x.min():.3f} | {x.max():.3f} | "
                  f"{int((r3 < BAND[0]).sum())} | {int((r3 > BAND[1]).sum())} |")
        w("")
        w("## Per layer after ABTT (D chosen on training DirAcc@1): AUROC (D)")
        opt = proj[proj["method"] == "abtt_optimal"]
        for name in [NEW, *CONTROLS]:
            s = opt[opt["model"] == name].sort_values("layer")
            if len(s):
                w(f"- {name}: " + "; ".join(f"{int(r.layer)}: {r.aucroc:.3f} ({int(r.D)})"
                                             for r in s.itertuples()))
        w("")
        w("Routing (DirAcc@1, test) after ABTT, D chosen on train, for the record: "
          + "; ".join(f"{n} {100 * opt[opt['model'] == n]['dir_acc_at_1'].min():.1f}-"
                      f"{100 * opt[opt['model'] == n]['dir_acc_at_1'].max():.1f}"
                      for n in [NEW, *CONTROLS] if n in set(opt['model'])))
        w("")
    path.write_text("\n".join(L) + "\n")


def render_stage(args) -> int:
    out = Path(args.out_dir)
    layers_df = pd.read_csv(out / "p2x2_layers.csv")
    proj_p = out / "d2_projection.csv"
    proj = pd.read_csv(proj_p) if proj_p.exists() else None
    rep = pd.read_csv(out / "d2_repro.csv") if (out / "d2_repro.csv").exists() else None
    cfg_p = out / "d2_config_check.csv"
    cfg = pd.read_csv(cfg_p, keep_default_na=False) if cfg_p.exists() else None
    if cfg is not None:
        cfg["gin_dropout_rate"] = cfg["gin_dropout_rate"].replace("", np.nan)
    facts(layers_df, proj, rep, cfg, Path(args.res_csv), out / "d2_facts.md")
    wrote = [str(out / "d2_facts.md")]
    if proj is not None:
        tab = Path(args.tab_dir)
        tab.mkdir(parents=True, exist_ok=True)
        write_t5_efficient_table(layers_df, proj, tab / "d2_t5_efficient.tex")
        write_abtt_table(proj, tab / "d2_controls_abtt.tex")
        wrote += [str(tab / "d2_t5_efficient.tex"), str(tab / "d2_controls_abtt.tex")]
    print("rendered " + ", ".join(wrote))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    choices=["config", "extract", "panel", "projection", "render"])
    ap.add_argument("--split_csv", type=Path, default=SPLIT_CSV)
    ap.add_argument("--res_csv", type=Path, default=RES_CSV)
    ap.add_argument("--p2x2_ref", type=Path, default=P2X2_REF)
    ap.add_argument("--data_root", type=Path, default=Path("."))
    ap.add_argument("--bases_root", type=Path, default=BASES_ROOT,
                    help="holds phase9_bases/<slug>/hidden_mean_tokempty (the paper's cache)")
    ap.add_argument("--bases", type=Path, default=BASES, help="the D2 caches (written by extract)")
    ap.add_argument("--out_dir", type=Path, default=OUT_DIR)
    ap.add_argument("--tab_dir", type=Path, default=TAB_DIR)
    ap.add_argument("--models", nargs="*", default=None, help="D2 display names (default: all five; bare --models: none, e.g. a projection pilot on the gate models only)")
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="projection pilot: these layers only (output goes to *.rejected.csv)")
    ap.add_argument("--skip_gate", action="store_true", help="projection pilot without the gate")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--token_filter", default="tokenizer_empty")
    ap.add_argument("--limit", type=int, default=0, help="extract pilot: first N rows only")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tokenizer_from", default="",
                    help="extract sensitivity: tokenize with this D2 model's tokenizer instead")
    args = ap.parse_args(argv)
    if args.limit and args.stage == "extract":
        args.bases = Path(args.out_dir) / "pilot_bases"
    return {"config": config_stage, "extract": extract_stage, "panel": panel_stage,
            "projection": projection_stage, "render": render_stage}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
