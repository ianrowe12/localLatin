"""One selection rule for the two five-seed SIF+ABTT tables (issue #175).

``tables/taskB_topk.tex`` (written by ``visualize_taskb_mseed.py``) and
``tables/taskB_ranking_appendix_mseed.tex`` (written by
``build_per_layer_tables.py``) both report the five-seed Task B run in
``runs/active/resubmit/taskb_mseed/aggregated_results.csv``. Before #175 each
picked its reported configuration differently: the top-K table took the argmax
of the five-seed *test* mean over every method and layer (which landed on
``sif_abtt_fixed`` for LaTa and KaLM-mini), while the per-layer table bolded
the test argmax restricted to ``sif_abtt_optimal``. Neither matched the layer
the headline tables report.

Both generators now import this module and apply the headline rule: for each
model, report ``sif_abtt_optimal`` at the layer with the highest *training-set*
directory accuracy at rank 1 in the single-seed run
(``runs/active/resubmit/results/phase_resubmit_results.csv``), the layer behind
the SIF+ABTT cell of ``tables/taskB_headline.tex`` and listed in
``tables/selected_layers.tex`` (issue #219). The five-seed
CSV carries no train metric of its own, which is why the layer comes from the
single-seed CSV. Ties go to the lowest layer.
"""
from __future__ import annotations

import pandas as pd

MSEED_METHOD = "sif_abtt_optimal"
TRAIN_SELECT_METRIC = "train_dir_acc_at_1"
DEFAULT_REPR = "hidden"

# Shared caption sentence so the two tables state the rule in the same words.
SELECTION_RULE_CAPTION = (
    r"For each model the reported layer is chosen on the train split: the layer with the "
    r"highest training-set directory accuracy at rank~1 for SIF+ABTT in the single-seed run, "
    r"the layer at which Table~\ref{tab:taskB_headline} scores its SIF+ABTT cell "
    r"(listed in Table~\ref{tab:selected_layers}). "
    r"$D$ is tuned per layer on the train split. "
    r"This is not always the layer with the highest five-seed mean."
)


def train_selected_layers(
    results_df: pd.DataFrame,
    models: list[str],
    method: str = MSEED_METHOD,
    metric: str = TRAIN_SELECT_METRIC,
    repr_name: str = DEFAULT_REPR,
) -> dict[str, int]:
    """Map each model to the layer maximising ``metric`` (train) under ``method``.

    Raises ``KeyError`` for a model with no rows, so a silent fallback to a
    test-set argmax can never creep back in.
    """
    layers: dict[str, int] = {}
    for model in models:
        sub = results_df[
            (results_df["model"] == model)
            & (results_df["repr"] == repr_name)
            & (results_df["method"] == method)
        ]
        if sub.empty:
            raise KeyError(
                f"no rows for model={model!r} method={method!r} repr={repr_name!r} "
                f"in the single-seed results CSV; cannot select a train layer"
            )
        sub = sub.sort_values("layer", kind="stable")
        layers[model] = int(sub.loc[sub[metric].idxmax(), "layer"])
    return layers


def select_mseed_rows(
    agg_df: pd.DataFrame,
    layers: dict[str, int],
    method: str = MSEED_METHOD,
    repr_name: str = DEFAULT_REPR,
) -> pd.DataFrame:
    """One aggregated five-seed row per model at its train-selected layer."""
    rows = []
    for model, layer in layers.items():
        sub = agg_df[
            (agg_df["model"] == model)
            & (agg_df["repr"] == repr_name)
            & (agg_df["method"] == method)
            & (agg_df["layer"] == layer)
        ]
        if len(sub) != 1:
            raise KeyError(
                f"expected exactly one five-seed row for model={model!r} method={method!r} "
                f"layer={layer}, found {len(sub)}"
            )
        rows.append(sub.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)
