"""
# Summary
Shared in evaluation codes
## References
"""
import copy
from itertools import permutations
from numbers import Number
import os
from typing import Callable, List, Union

import numpy as np
import pandas as pd
import torch


def simple_sum(score: Union[List[Number], np.ndarray, torch.Tensor, Number]):
    """
    tensor of size (n,m,k) -> size n
    list of size n -> 1 float
    """
    if type(score) == torch.Tensor:
        sample_num = score.dim()
        dim = [idx for idx in range(sample_num) if idx not in [0]]
        out = score.sum(dim=dim)
    elif isinstance(score, Number):
        out = score
    else:
        out = sum(score)
    return out


def simple_avg(score: Union[List[Number], np.ndarray, torch.Tensor, Number]):
    """
    tensor of size (n,m,k) -> size n
    list of size n -> 1 float
    """
    summed = simple_sum(score)
    if type(score) == torch.Tensor:
        num_dim = torch.numel(score) / score.shape[0]
        out = summed / num_dim
    elif isinstance(score, Number):
        out = summed
    else:
        out = summed / len(score)
    return out


def nonzero_avg(score: Union[List[Number], np.ndarray, torch.Tensor, Number]):
    """
    tensor of size (n,m,k) -> size n
    list of size n -> 1 float
    averaged by number of non-zero elements
    """
    summed = simple_sum(score)
    if type(score) == torch.Tensor:
        # todo: version issue
        # num_dim = int(torch.count_nonzero(score))
        num_dim = len(torch.nonzero(score))
        out = summed / num_dim
    elif isinstance(score, Number):
        out = summed
    else:
        out = summed / len(score)
    return out


def simple_summary(score: Union[List[float], torch.Tensor], mode: str = "avg"):
    """
    Main function for score summarization
    """
    if mode == "avg":
        out = nonzero_avg(score)
    elif mode == "sum":
        out = simple_sum(score)
    else:
        if type(score) == torch.Tensor:
            out = score.clone().detach()
        else:
            out = copy.deepcopy(score)
    return out


def permute_2lists(list1,list2):
    out = []
    ps = permutations(list1, len(list2))
    for p in ps:
        z1, z2 = zip(p, list2)
        out.append(z1+"_"+z2)
    return out

def scorer(
    preds: np.ndarray,
    labels: np.ndarray,
    preds_overall: np.ndarray,
    pred_labels: np.ndarray,
    expector: Callable[[List[float]], List[float]] = simple_avg,
    preds_threshold: float = 0.1,
    pick_correct_labels: bool = True,
):
    if len(set(labels)) == 1:
        e_pos,e_neg,e_pos_overall,e_neg_overall,result_overall,result = 0,0,0,0,0,0
        calc_flg = 0
    else:
        e_pos = expector(preds[labels == 1])
        e_neg = expector(preds[labels == 0])
        result = e_pos - e_neg
        e_pos_overall = expector(preds_overall[labels == 1])
        e_neg_overall = expector(preds_overall[labels == 0])
        result_overall = e_pos_overall - e_neg_overall
        if result_overall >= preds_threshold:
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
    return [
        result,
        result_overall,
        e_pos,
        e_neg,
        e_pos_overall,
        e_neg_overall,
    ], calc_flg

def _df_evaluator_single_col(
    df_processed: pd.DataFrame,
    conf_modality: str = "img",
    col_suffix: str = "_org_id",
    label_col: str = "label",
    pred_col: str = "preds",
    remove_txts: List[str] = ["No Confounder", "Not hit"],
    evaluator: Callable[[List[float], List[float]], float] = scorer,
    id_col: str = "id",
    generated_cols: List[str] = [
        "mod",
        "score",
        "miate",
        "score_pos",
        "score_neg",
        "p_pos",
        "p_neg",
    ],
    overall_pred_col: str = "pred",
    preds_threshold: float = 0.1,
    pick_correct_labels: bool = False,
):
    def _batch_evaluator(
        df,
        pred_col=pred_col,
        label_col=label_col,
        evaluator=evaluator,
        pick_correct_labels=pick_correct_labels,
        overall_pred_col=overall_pred_col,
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

    export_cols = [id_col] + generated_cols
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
            results["miate"].append(result_batch[1])
            results["score_pos"].append(result_batch[2])
            results["score_neg"].append(result_batch[3])
            results["p_pos"].append(result_batch[4])
            results["p_neg"].append(result_batch[5])
            results[id_col].append(mod_id)
            results["mod"].append(conf_modality)
        else:
            pass
    return results

def makedirs_recursive(path):
    if not os.path.exists(path):
        os.makedirs(path)
