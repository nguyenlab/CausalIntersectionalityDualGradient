"""
# Summary
Modality scoring
## References
"""
import os
import re
from typing import Dict, List, Union

import einops
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from torch import Tensor
import torch.nn.functional as F

from evaluation.utils import _df_evaluator_single_col, makedirs_recursive, simple_summary

def binary_filpper(x: Tensor):
    out = torch.logical_not(x, out=torch.empty(x.shape[0], dtype=torch.int16))
    return out


def segment_modality_converter(segment: Tensor):
    # segment_reshaped = segment.reshape(len(segment), 1)
    # out = torch.mm(segment_reshaped, segment_reshaped.T)
    out = torch.mm(segment.T, segment)
    return out


def cross_modality_generator(modality1: Tensor, modality2: Tensor):
    unimodality = torch.logical_or(modality1, modality2)
    out = torch.logical_not(
        unimodality, out=torch.empty(unimodality.shape, dtype=torch.int16)
    )
    return out


def modality_generator_single(head_num: int, input_length: int, text_length: int):
    segment_all = torch.zeros(input_length)
    segment_txt = torch.ones(text_length)
    segment_ids = F.pad(
        input=segment_txt,
        pad=(0, len(segment_all) - len(segment_txt)),
        mode="constant",
        value=0,
    )
    segment_ids = torch.tensor([segment_ids.tolist()], dtype=torch.long)
    segment_ids_flipped = binary_filpper(segment_ids)
    modality1 = segment_modality_converter(segment_ids)
    modality1 = einops.repeat(modality1, "d e -> n d e", n=head_num)
    modality2 = segment_modality_converter(segment_ids_flipped)
    modality2 = einops.repeat(modality2, "d e -> n d e", n=head_num)

    cross_modality = cross_modality_generator(modality1, modality2)
    return {"text": modality1, "image": modality2, "cross": cross_modality}


def modality_generator(head_num: int, input_length: int, text_lengthes: List[int]):
    segment_all = torch.zeros(input_length)
    modalities1, modalities2, cross_modalities = [], [], []
    for text_length in text_lengthes:
        segment_txt = torch.ones(text_length)
        segment_ids = F.pad(
            input=segment_txt,
            pad=(0, len(segment_all) - len(segment_txt)),
            mode="constant",
            value=0,
        )
        segment_ids = torch.tensor([segment_ids.tolist()], dtype=torch.long)
        segment_ids_flipped = binary_filpper(segment_ids)
        modality1 = segment_modality_converter(segment_ids)
        modality1 = einops.repeat(modality1, "d e -> n d e", n=head_num)
        modalities1.append(modality1)
        modality2 = segment_modality_converter(segment_ids_flipped)
        modality2 = einops.repeat(modality2, "d e -> n d e", n=head_num)
        modalities2.append(modality2)
        cross_modality = cross_modality_generator(modality1, modality2)
        cross_modalities.append(cross_modality)
    modalities1, modalities2, cross_modalities = (
        torch.stack(modalities1),
        torch.stack(modalities2),
        torch.stack(cross_modalities),
    )
    return {
        "text": modalities1,
        "image": modalities2,
        "cross": cross_modalities,
    }


class UndefinedOptionError(Exception):
    pass


def compression(score: Union[List[float], torch.Tensor], compress_mode: str = "avg"):
    if compress_mode in ["sum", "avg"] or compress_mode is None:
        out = simple_summary(score, compress_mode)
    else:
        raise UndefinedOptionError("compress_mode should be [avg, sum, None]")
    return out


def zero_exception_loc(score: torch.Tensor, loc: List[int]):
    """
    Zero out the score at the exception location
    0: [CLS], l: [SEP]
    """
    for l in loc:
        score[:, :, [0, l], :] = 0
        score[:, :, :, [0, l]] = 0
    return score


def modality_scorer(
    scores: List[Tensor],
    modalities: Dict[str, Tensor],
    device: str,
    cmp_mode: str = "avg",
):
    out = {}
    for mod in modalities.keys():
        mod_bin, grad = modalities[mod].to(device), scores.to(device)
        # grad = scores.to(device) if not scores.is_cuda else scores.to(device)
        mod_score = torch.mul(mod_bin, grad)
        if cmp_mode is None:
            out[mod] = mod_score
        else:
            out[mod] = compression(mod_score, cmp_mode)
    return out


def modality_summarizer(scores: Dict[str, List[Tensor]], cmp_mode: str = "avg"):
    out = {k: [] for k in scores.keys()}
    for mod in scores.keys():
        for score in scores[mod]:
            # to make it work as pd.df column
            out[mod] += score.tolist()

    return out


def generate_text_lengthes(input_ids, tokenizer, batch_size):
    out = []
    for iid in input_ids:
        tokens = np.array(tokenizer.convert_ids_to_tokens(iid))
        text_length = np.where(tokens == "[SEP]")[0][0] - 1
        out.append(text_length)
    out += [batch_size] * (batch_size - len(out))
    return out


def midas_scorer(
    input_ids: Tensor,
    scores: Dict[str, Tensor],
    tokenizer,
    device: str,
    cmp_mode: str = "avg",
    batch_size: int = 16,
):
    out = {k: None for k in scores.keys()}
    text_lengthes = generate_text_lengthes(input_ids, tokenizer, batch_size)
    for score_mod in scores.keys():
        score = scores[score_mod]
        head_num, input_length = score.shape[1], score.shape[2]
        modalities = modality_generator(
            head_num,
            input_length,
            text_lengthes,
        )
        out[score_mod] = modality_scorer(score, modalities, device)
        out[score_mod]["all"] = compression(score)
    return out


def df_evaluator(
    df_conf: pd.DataFrame,
    df_score: pd.DataFrame,
    logger,
    conf_modality: str = "img",
    label_col: str = "label",
    split_col: str = "split",
    splits: List[str] = ["dev_seen"],  # todo: feed from args
    pred_cols: List[str] = ["all", "text", "image", "cross"],
    id_col: str = "id",
    colors: List[str] = ["#C0C0C0", "#9ecae1", "#6baed6", "#2171b5"],  # Blues
    export_path: str = "/content/export/test.html",
    preds_threshold: float = 0.1,
    pick_correct_labels: bool = False,
    save_formats: List[str] = ["eps", "png"],
):
    def _preprocess_df(
        df_conf,
        df_score,
        label_col=label_col,
        split_col=split_col,
        splits=splits,
    ):
        # score dataframe
        df_score = df_score.rename({label_col: f"pred_{label_col}"}, axis=1)
        df_score[id_col] = df_score[id_col].astype("str").str.zfill(5)
        # label dataframe
        df_conf = (
            df_conf[(df_conf[label_col] != -1) & (df_conf[split_col].isin(splits))]
            .reset_index(drop=True)
            .drop_duplicates()
        )
        df_merged = df_score.merge(
            df_conf.set_index(id_col),
            left_on=id_col,
            right_index=True,
            suffixes=["", "_label"],
        )
        return df_merged

    intermediate_path = export_path.split(".html")[0]
    df_processed = _preprocess_df(df_conf, df_score)
    logger.info(f"#{len(df_processed)} confounders detected")
    df_processed.to_csv(f"{intermediate_path}_processed.csv", index=False)

    fig = go.Figure()
    for pred_col, color in zip(pred_cols, colors):
        logger.info(f"Eval started for {pred_col}")
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
        df_result["mod"] = df_result["mod"].replace(
            {
                "txt": "text",
                "img": "image",
            },
        )
        mods = df_result["mod"].unique()
        model = intermediate_path.split("/")[-1].split("_")[1]
        df_result["model"] = model
        df_result["model(mod)"] = df_result["model"] + "(" + df_result["mod"] + ")"
        model_mods = df_result["model(mod)"].unique()
        values, errors = [], []
        for mod in mods:
            scores = df_result.loc[df_result["mod"] == mod, "score"]
            values.append(scores.mean())
            errors.append(scores.std(ddof=1) / np.sqrt(len(scores)))
        fig.add_trace(
            go.Bar(
                name=pred_col,
                x=model_mods,
                y=values,
                error_y=dict(type="data", array=errors),
                marker_color=color,
            )
        )
        logger.info(f"Eval complete for {pred_col}")
    fig.update_layout(
        template="plotly_white",
        barmode="group",
        yaxis_exponentformat="e",
    )
    if "eps" in save_formats:
        fig.write_image(f"{intermediate_path}_midas.eps", engine="kaleido")
    if "html" in save_formats:
        fig.write_html(export_path)

    return fig


def midas_evaluator(
    args,
    logger,
    df_score: pd.DataFrame,
    conf_modality: str = "img",
    label_col: str = "label",
    split_col: str = "split",
    splits: List[str] = ["dev_seen"],  # todo: feed from args
    score_types: List[str] = ["attention", "attattr", "gradient"],
    pred_cols: List[str] = ["all", "text", "image", "cross"],
    id_col: str = "id",
    colors: List[str] = ["#C0C0C0", "#9ecae1", "#6baed6", "#2171b5"],  # Blues
    preds_threshold: float = 0.1,
    pick_correct_labels: bool = False,
    save_formats: List[str] = ["eps", "png"],
):
    df_conf = pd.read_parquet(args.conf_path, engine="pyarrow")
    figs = []
    for score_type in score_types:
        logger.info(f"Eval started for {conf_modality} modality {score_type}")
        pred_type_dict = {col: re.sub(f"{score_type}_", "", col) for col in df_score.columns if score_type in col}
        df_midas = df_score[[id_col, "pred", label_col]+list(pred_type_dict.keys())].rename(pred_type_dict, axis=1)
        if pick_correct_labels:
            export_path = f"{args.output_dir}/evaluation/{args.checkpoint_path}/correct_label/{conf_modality}/{score_type}/blip2_bert_correct.html"
        else:
            export_path = f"{args.output_dir}/evaluation/{args.checkpoint_path}/{conf_modality}/{score_type}/blip2_bert.html"
        makedirs_recursive(os.path.dirname(export_path))
        fig = df_evaluator(
            df_conf,
            df_midas,
            logger,
            conf_modality=conf_modality,
            label_col=label_col,
            split_col=split_col,
            splits=splits,
            pred_cols=pred_cols,
            colors=colors,
            export_path=export_path,
            preds_threshold=preds_threshold,
            pick_correct_labels=pick_correct_labels,
            save_formats=save_formats,
        )
        figs.append(fig)
        logger.info(f"Eval complete for {conf_modality} modality {score_type}")
    return figs