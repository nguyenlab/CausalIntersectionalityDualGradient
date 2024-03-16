"""
# Summary
Evaluation metrics
## References
"""
# import os
from typing import Callable, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from modality_evaluation.utils import simple_avg


def micace(
    preds: np.ndarray,
    labels: np.ndarray,
    preds_overall: np.ndarray,
    pred_labels: np.ndarray,
    expector: Callable[[List[float]], List[float]] = simple_avg,
    preds_threshold: float = 0.1,
    pick_correct_labels: bool = True,
):
    e_pos = expector(preds[labels == 1])
    e_neg = expector(preds[labels == 0])
    result = e_pos - e_neg
    e_pos_overall = preds_overall[labels == 1]
    e_neg_overall = preds_overall[labels == 0]
    result_overall = e_pos_overall - e_neg_overall
    if (
        expector(preds_overall[labels == 1]) - expector(preds_overall[labels == 0])
        >= preds_threshold
    ) and len(set(labels)) > 1:
        if pick_correct_labels:
            if np.sum(pred_labels[labels == 1]) - np.sum(
                pred_labels[labels == 0]
            ) == len(pred_labels[labels == 1]):
                calc_flg = 1
            else:
                calc_flg = 0
        else:
            calc_flg = 1
    else:
        calc_flg = 0
    return [result, result_overall, e_pos, e_neg], calc_flg


def _df_evaluator_single_col(
    df_processed: pd.DataFrame,
    conf_modality: str = "img",
    col_suffix: str = "_org_id",
    label_col: str = "label",
    pred_col: str = "preds",
    remove_txts: List[str] = ["No Confounder", "Not hit"],
    evaluator: Callable[[List[float], List[float]], float] = micace,
    export_cols: List[str] = ["id", "mod", "score", "micace"],
    overall_pred_col: str = "proba",
    preds_threshold: float = 0.1,
    pick_correct_labels: bool = False,
):
    def _batch_evaluator(
        df,
        pred_col=pred_col,
        label_col=label_col,
        evaluator=evaluator,
        pick_correct_labels=pick_correct_labels,
    ):
        preds, labels, preds_o, pred_labels = (
            df[pred_col].values,
            df[label_col].values,
            df[overall_pred_col].values,
            df[f"pred_{label_col}"].values,
        )
        result_batch, calc_flg = evaluator(
            preds,
            labels,
            preds_o,
            pred_labels,
            preds_threshold=preds_threshold,
            pick_correct_labels=pick_correct_labels,
        )
        return result_batch, calc_flg

    results = {k: [] for k in export_cols}

    mod_ids = (
        df_processed.loc[
            ~df_processed[conf_modality + col_suffix].isin(remove_txts),
            conf_modality + col_suffix,
        ]
        .unique()
        .tolist()
    )
    for mod_id in mod_ids:
        df_group = df_processed[
            df_processed[conf_modality + col_suffix] == mod_id
        ].copy()
        result_batch, calc_flg = _batch_evaluator(df_group)
        if calc_flg:
            results["score"].append(result_batch[0])
            results["micace"].append(result_batch[1])
            results["p_pos"].append(result_batch[2])
            results["p_neg"].append(result_batch[3])
            results["id"].append(mod_id)
            results["mod"].append(conf_modality)
        else:
            pass
    return results


def df_evaluator(
    df_conf: pd.DataFrame,
    df_score: pd.DataFrame,
    conf_modality: str = "img",
    label_col: str = "label",
    split_col: str = "split",
    splits: List[str] = ["dev_seen"],  # todo: feed from args
    pred_cols: List[str] = ["all", "text", "image", "cross"],
    colors: List[str] = ["#f7fbff", "#9ecae1", "#6baed6", "#2171b5"],  # Blues
    export_path: str = "/content/export/test.html",
    preds_threshold: float = 0.1,
    pick_correct_labels: bool = False,
):
    def _preprocess_df(
        df_conf, df_score, label_col=label_col, split_col=split_col, splits=splits
    ):
        # score dataframe
        df_score = df_score.rename({label_col: f"pred_{label_col}"}, axis=1)
        df_score["id"] = df_score["id"].astype("str").str.zfill(5)
        # label dataframe
        df_conf = (
            df_conf[(df_conf[label_col] != -1) & (df_conf[split_col].isin(splits))]
            .reset_index(drop=True)
            .drop_duplicates()
        )
        df_merged = df_score.merge(
            df_conf.set_index("id"),
            left_on="id",
            right_index=True,
            suffixes=["", "_label"],
        )
        return df_merged

    intermediate_path = export_path.split(".html")[0]
    df_processed = _preprocess_df(df_conf, df_score)
    df_processed.to_csv(f"{intermediate_path}_processed.csv", index=False)

    fig = go.Figure()
    for pred_col, color in zip(pred_cols, colors):
        result = _df_evaluator_single_col(
            df_processed,
            conf_modality=conf_modality,
            pred_col=pred_col,
            preds_threshold=preds_threshold,
            pick_correct_labels=pick_correct_labels,
        )
        df_result = pd.DataFrame(result)
        df_result.to_csv(f"{intermediate_path}_result_{pred_col}.csv", index=False)
        df_result = df_result[~df_result["score"].isnull()]
        mods = df_result["mod"].unique()
        values, errors = [], []
        for mod in mods:
            scores = df_result.loc[df_result["mod"] == mod, "score"]
            values.append(scores.mean())
            errors.append(scores.std(ddof=1) / np.sqrt(len(scores)))
        fig.add_trace(
            go.Bar(
                name=pred_col,
                x=mods,
                y=values,
                error_y=dict(type="data", array=errors),
                marker_color=color,
            )
        )
    fig.update_layout(template="plotly_dark", barmode="group")
    fig.write_html(export_path)

    return fig
