"""CPU-only dataset-2 comparisons and depth sweep, separate from legacy score."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import score_dataset2 as sd
from lexical_baselines import char_tfidf_vectors
from run_resubmit_unlabelled_retrieval import resolve_source_path

REFERENCE = "tfidf_char35"
REFERENCE_DISPLAY = "Char. n-grams"
METRICS = ("acc@1", "acc@5", "acc@10", "mrr")
MISSING = "missing_cache"


def context(args):
    split = pd.read_csv(args.split_csv)
    meta = pd.read_csv(args.unlabelled_meta)
    root = Path(args.data_root)
    return {
        "split": split, "qmeta": meta,
        "labelled_bases": Path(args.labelled_bases),
        "unlabelled_bases": Path(args.unlabelled_bases),
        "lab_resolver": sd.AlignmentResolver(split),
        "unlab_resolver": sd.AlignmentResolver(meta),
        "query_names": meta["filename"].astype(str).tolist(),
        "folder_ids": split["folder_id"].astype(str).values,
        "dir_names": list(dict.fromkeys(split["folder_id"].astype(str))),
        "train_mask": (split["split"] == "train").values,
        "train_fids": split.loc[split["split"] == "train", "folder_id"].values,
        "lab_blank": sd.read_blank_source_mask(split["path"].tolist(), root),
        "unlab_blank": sd.read_blank_source_mask(meta["path"].tolist(), root),
    }


def labels(args, ctx):
    q = sd.load_snapshot(Path(args.snapshot), ctx["query_names"])
    index = sd.key_to_dir_index(ctx["dir_names"])
    q["label_cols"], q["unmapped_keys"] = sd.label_dirs(q, index)
    reviewers = q[["reviewer_status", "reviewer_ccl_key"]].rename(
        columns={"reviewer_status": "status", "reviewer_ccl_key": "ccl_key"},
    ).assign(conflicting_keys="")
    q["reviewer_label_cols"], _ = sd.label_dirs(reviewers, index)
    return q


def groups(q):
    sets = {
        "agreed": q["status"] == "agreed",
        "agreed_plus_conflict": q["status"].isin(["agreed", "conflict"]),
        "conflict": q["status"] == "conflict",
        "reviewers_agreed": q["reviewer_status"] == "agreed",
    }
    for name, mask in sets.items():
        yield name, "all", mask.to_numpy()
        for group in sorted(q.loc[mask, "seen_group"].unique()):
            yield name, group, (mask & (q["seen_group"] == group)).to_numpy()


def rank_vectors(order, excluded, q):
    rows = q["query_row"].to_numpy()
    result = {}
    for name, column in (("final", "label_cols"), ("reviewers", "reviewer_label_cols")):
        ranks = sd.ranks_of(order, q[column].tolist(), rows)
        ranks[excluded[rows]] = np.nan
        result[name] = ranks
    return result


def set_ranks(vectors, name):
    return vectors["reviewers" if name == "reviewers_agreed" else "final"]


def draws(args, name, group, n, domain):
    """Named streams cannot consume legacy draws or another label set's draws."""
    digest = hashlib.sha256(f"{name}\0{group}".encode()).digest()
    key = (298, domain, *np.frombuffer(digest, dtype="<u4").tolist())
    rng = np.random.default_rng(np.random.SeedSequence(args.seed, spawn_key=key))
    return rng.integers(0, n, size=(args.n_boot, n))


def reference_scores(ctx, data_root):
    texts = [
        resolve_source_path(path, Path(data_root)).read_text(encoding="utf-8", errors="ignore")
        for frame in (ctx["split"], ctx["qmeta"]) for path in frame["path"]
    ]
    n_lab = len(ctx["split"])
    vectors = char_tfidf_vectors(texts, np.flatnonzero(ctx["train_mask"]))
    lab, query = vectors[:n_lab], vectors[n_lab:]
    excluded_lab = ctx["lab_blank"] | (lab.getnnz(axis=1) == 0)
    excluded_query = ctx["unlab_blank"] | (query.getnnz(axis=1) == 0)
    cross = (query @ lab.T).toarray()
    scores = np.full((len(query.indptr) - 1, len(ctx["dir_names"])), -np.inf)
    for j, directory in enumerate(ctx["dir_names"]):
        cols = np.flatnonzero((ctx["folder_ids"] == directory) & ~excluded_lab)
        if len(cols):
            scores[:, j] = cross[:, cols].max(axis=1)
    scores[excluded_query] = np.nan
    return {"scores": scores, "excluded_queries": excluded_query,
            "excluded_refs": excluded_lab, "D": None, "status": sd.SCORED, "reason": ""}


def load_rankings(out, selection, ctx, q):
    cells = {}
    for s in selection.itertuples():
        path = out / "rankings" / f"{sd.slug(s.model)}__{s.method}.npz"
        if s.status != sd.SCORED or not path.exists():
            continue
        with np.load(path, allow_pickle=True) as saved:
            if (saved["query_filenames"].tolist() != ctx["query_names"]
                    or saved["dir_names"].tolist() != ctx["dir_names"]):
                raise ValueError("Saved rankings do not match the current metadata order")
            cells[(s.model, s.method)] = rank_vectors(
                saved["order"], saved["excluded_queries"], q,
            )
    return cells


def replay_reference_draws(args, selection, cells, q, reference, accuracy):
    """Replay score's draw schedule, retaining draws on the reference's exact cohort.

    Validation prevents mixing a snapshot, seed or replicate count with a
    different legacy score run. No score output is opened for writing.
    """
    rng = np.random.default_rng(args.seed)
    reviewers_rng = np.random.default_rng(np.random.SeedSequence(args.seed, spawn_key=(1,)))
    slots, matched = {}, {}
    for s in selection.itertuples():
        vectors = cells.get((s.model, s.method))
        if vectors is None:
            continue
        for name, group, mask in groups(q):
            ranks = set_ranks(vectors, name)
            valid = mask & ~np.isnan(ranks)
            n = int(valid.sum())
            if not n:
                continue
            slot = (name, group)
            if slot not in slots or slots[slot].shape[1] != n:
                generator = reviewers_rng if name == "reviewers_agreed" else rng
                slots[slot] = generator.integers(0, n, size=(args.n_boot, n))
            old = accuracy[(accuracy["model"] == s.display) & (accuracy["method"] == s.method)
                           & (accuracy["label_set"] == name) & (accuracy["seen_group"] == group)]
            expected = sd.bootstrap(ranks[valid], slots[slot])
            point = sd.metric_block(ranks[valid])
            for metric in METRICS:
                row = old[old["metric"] == metric]
                if len(row) != 1 or not np.allclose(
                    row[["value", "ci_lo", "ci_hi"]].to_numpy(dtype=float),
                    [[point[metric], *expected[metric]]], rtol=0, atol=1e-14,
                ):
                    raise ValueError("Legacy scores differ; use score's snapshot, seed and n_boot")
            ref_valid = mask & ~np.isnan(set_ranks(reference, name))
            if slot not in matched and np.array_equal(valid, ref_valid):
                matched[slot] = slots[slot]
    return matched


def best_methods(selection):
    """Choose post-processing using train only; ties follow METHODS order."""
    order = {method: i for i, method in enumerate(sd.METHODS)}
    sub = selection[(selection["method"] != "baseline")
                    & (selection["status"] == sd.SCORED)
                    & selection[sd.TRAIN_METRIC].notna()].copy()
    sub["_order"] = sub["method"].map(order)
    sub["_model_order"] = sub["model"].map(
        {model: i for i, model in enumerate(selection["model"].unique())},
    )
    return (sub.sort_values(["_model_order", sd.TRAIN_METRIC, "_order"],
                            ascending=[True, False, True],
                            kind="stable").drop_duplicates("model")
            .drop(columns=["_order", "_model_order"]))


def paired_rows(args, q, a, b, metadata, status=sd.SCORED, reason=""):
    for name, group, mask in groups(q):
        if not mask.any():
            continue
        valid = mask.copy()
        if a is not None and b is not None:
            ra, rb = set_ranks(a, name), set_ranks(b, name)
            valid &= ~np.isnan(ra) & ~np.isnan(rb)
        else:
            valid[:] = False
        n = int(valid.sum())
        cell_status = status if status != sd.SCORED or n else "no_paired_queries"
        idx = draws(args, name, group, n, 2) if n else None
        for metric in METRICS:
            estimate = lo = hi = None
            if n:
                av = (ra[valid] <= int(metric[4:])).astype(float) if metric != "mrr" else 1 / ra[valid]
                bv = (rb[valid] <= int(metric[4:])).astype(float) if metric != "mrr" else 1 / rb[valid]
                delta = av - bv
                estimate = float(delta.mean())
                lo, hi = np.percentile(delta[idx].mean(axis=1), [2.5, 97.5])
            yield {**metadata, "label_set": name, "seen_group": group,
                   "metric": metric, "n": n, "n_excluded": int(mask.sum() - n),
                   "estimate": estimate, "ci_lo": lo, "ci_hi": hi,
                   "status": cell_status, "reason": reason}


def cmd_compare(args):
    ctx = context(args)
    q = labels(args, ctx)
    out = Path(args.out_dir)
    results = out / "results"
    selection = pd.read_csv(results / "selected_layers.csv", dtype={"reason": "string"})
    selection["reason"] = selection["reason"].fillna("")
    reference = reference_scores(ctx, args.data_root)
    order = sd.rank_order(np.nan_to_num(reference["scores"], nan=-np.inf))
    ref_ranks = rank_vectors(order, reference["excluded_queries"], q)
    cells = load_rankings(out, selection, ctx, q)
    matched = replay_reference_draws(
        args, selection, cells, q, ref_ranks, pd.read_csv(results / "accuracy_long.csv"),
    )
    rows = []
    for name, group, mask in groups(q):
        ranks = set_ranks(ref_ranks, name)
        valid = mask & ~np.isnan(ranks)
        n = int(valid.sum())
        if not mask.any():
            continue
        idx = matched.get((name, group))
        stream = "legacy_score" if idx is not None else "reference_named"
        if idx is None and n:
            idx = draws(args, name, group, n, 1)
        point = sd.metric_block(ranks[valid]) if n else {}
        ci = sd.bootstrap(ranks[valid], idx) if n else {}
        for metric in METRICS:
            rows.append({"label_set": name, "seen_group": group, "n": n,
                         "n_excluded": int(mask.sum() - n), "model": REFERENCE_DISPLAY,
                         "method": REFERENCE, "layer": None, "D": None, "metric": metric,
                         "value": point.get(metric), "ci_lo": ci.get(metric, (None, None))[0],
                         "ci_hi": ci.get(metric, (None, None))[1],
                         "status": sd.SCORED if n else "no_usable_queries", "reason": "",
                         "bootstrap_stream": stream})
    best = best_methods(selection)
    contrasts = []
    for s in selection.itertuples():
        vectors = cells.get((s.model, s.method))
        status = s.status if s.status != sd.SCORED else (sd.SCORED if vectors else MISSING)
        contrasts.extend(paired_rows(args, q, vectors, ref_ranks, {
            "contrast": "model_minus_reference", "model": s.display, "method": s.method,
            "reference_method": REFERENCE, "layer": s.layer, "D": s.D,
        }, status, s.reason if status != MISSING else "selected cache ranking absent"))
    for s in best.itertuples():
        a, b = cells.get((s.model, s.method)), cells.get((s.model, "baseline"))
        contrasts.extend(paired_rows(args, q, a, b, {
            "contrast": "best_minus_baseline", "model": s.display, "method": s.method,
            "reference_method": "baseline", "layer": s.layer, "D": s.D,
        }, sd.SCORED if a and b else MISSING, "" if a and b else "selected cache ranking absent"))
    pd.DataFrame(rows).to_csv(results / "reference_accuracy_long.csv", index=False)
    pd.DataFrame(contrasts).to_csv(results / "paired_differences.csv", index=False)
    best.to_csv(results / "best_methods_train.csv", index=False)
    sd.save_ranking(out / "rankings" / f"{REFERENCE}__reference.npz",
                    reference, REFERENCE, REFERENCE, -1, ctx)
    pd.DataFrame({"xml_id": q["xml_id"], "rank": ref_ranks["final"],
                  "reviewer_rank": ref_ranks["reviewers"]}).to_csv(
        results / "reference_per_query_ranks.csv", index=False,
    )
    info = {
        "seed": args.seed, "n_boot": args.n_boot, "interval": "percentile95",
        "bootstrap_unit": "reviewed query",
        "selection": "benchmark v1 train DirAcc@1 only; method ties in METHODS order",
        "reference_bootstrap": "legacy draws for identical cohorts; named stream otherwise",
        "paired_bootstrap": "shared named draws on each contrast's non-excluded intersection",
        "status_counts": q["status"].value_counts().to_dict(),
        "reviewer_status_counts": q["reviewer_status"].value_counts().to_dict(),
    }
    (results / "comparison_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"Wrote reference and paired comparisons -> {results}")


def layer_cells(ctx, training):
    known = {sd.slug(model): model for model, _ in sd.MODELS}
    known.update({sd.slug(model): model for model in training["model"].dropna().unique()})
    for directory in sorted(ctx["unlabelled_bases"].iterdir()):
        if not directory.is_dir():
            continue
        model = known.get(directory.name, directory.name)
        for method, (pooling, _) in sd.METHODS.items():
            pattern = re.compile(rf"hidden_layer(\d+)_embeddings{sd.SUFFIX[pooling]}\.npy$")
            cache = directory / f"hidden_{pooling}_tokempty"
            layers = [int(m[1]) for path in cache.glob("*.npy")
                      if (m := pattern.fullmatch(path.name))]
            for layer in sorted(layers):
                yield model, method, layer


def train_rows(args):
    frames = [pd.read_csv(args.results_csv)]
    if Path(args.qwen8b_selection).exists():
        frames.append(pd.read_csv(args.qwen8b_selection))
    return pd.concat(frames, ignore_index=True)


def layer_d(training, model, method, layer):
    pooling, post = sd.METHODS[method]
    if post != "abtt_optimal":
        return None, "fixed" if post == "abtt_fixed" else "not_required"
    sub = training[(training["model"] == model) & (training["repr"] == "hidden")
                   & (training["pooling"] == pooling) & (training["layer"] == layer)
                   & (training["method"] == sd.RESULTS_METHOD[method])]
    if len(sub) > 1:
        raise ValueError(f"Duplicate train selection rows for {model}/{method}/L{layer}")
    if sub.empty or pd.isna(sub["D"].iloc[0]):
        return None, "retuned_benchmark_v1_train"
    return int(sub["D"].iloc[0]), "recorded_benchmark_v1_train"


def cmd_layers(args):
    ctx = context(args)
    q = labels(args, ctx)
    agreed = (q["status"] == "agreed").to_numpy()
    training = train_rows(args)
    rows = []
    for model, method, layer in layer_cells(ctx, training):
        d, source = layer_d(training, model, method, layer)
        result = sd.score_one(model, method, layer, d, ctx)
        status = MISSING if result is None else result["status"]
        row = {"model": model, "display": sd.DISPLAY.get(model, model), "method": method,
               "layer": layer, "D": None if result is None else result["D"],
               "D_source": source, "label_set": "agreed", "status": status,
               "reason": "labelled cache absent" if result is None else result["reason"],
               "n": 0, "n_excluded": int(agreed.sum()), "acc@1": None, "mrr": None}
        if status == sd.SCORED:
            order = sd.rank_order(np.nan_to_num(result["scores"], nan=-np.inf))
            ranks = rank_vectors(order, result["excluded_queries"], q)["final"]
            valid = agreed & ~np.isnan(ranks)
            row.update(n=int(valid.sum()), n_excluded=int((agreed & ~valid).sum()))
            if valid.any():
                metrics = sd.metric_block(ranks[valid])
                row.update({name: metrics[name] for name in ("acc@1", "mrr")})
            else:
                row.update(status="no_usable_queries", reason="no non-excluded agreed queries")
        rows.append(row)
        print(f"  {model} {method} L{layer}: {row['status']}", flush=True)
    if not rows:
        raise ValueError("No unlabelled embedding layers found")
    out = Path(args.out_dir) / "results"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "per_layer.csv", index=False)
    print(f"Wrote {len(rows)} layer cells -> {out / 'per_layer.csv'}")


def add_subcommands(sub):
    commands = {"compare": cmd_compare, "layers": cmd_layers}
    for name in commands:
        parser = sub.add_parser(name)
        for key in ("split_csv", "unlabelled_meta", "labelled_bases", "unlabelled_bases",
                    "results_csv", "qwen8b_selection", "out_dir"):
            parser.add_argument(f"--{key}", default=sd.DEFAULTS[key])
        parser.add_argument("--snapshot", required=True)
        parser.add_argument("--data_root", default="data")
        if name == "compare":
            parser.add_argument("--n_boot", type=int, default=2000)
            parser.add_argument("--seed", type=int, default=289)
    parser = sub.add_parser("render")
    parser.add_argument("--out_dir", default=sd.DEFAULTS["out_dir"])

    def render(args):
        from dataset2_render import cmd_render

        cmd_render(args)

    commands["render"] = render
    return commands
