import argparse
import glob
import os
import re
from typing import List, Union

import dask.dataframe as dd
from dask import delayed
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils import makedirs_recursive


def miate_parser():
    parser = argparse.ArgumentParser(
        prog="miateEvaluator",
        description="Evaluate miate of models with same type",
    )
    parser.add_argument(
        "file_path", type=str, default="./data", help="Path where scores are saved"
    )
    parser.add_argument("model_type", type=str, default="O", help="vilio model type")
    parser.add_argument(
        "calc_type", type=str, default="attattr", help="Score calculation type"
    )
    parser.add_argument(
        "--pick_correct_labels", "-pc", action="store_true", help="Correct label only"
    )
    args = parser.parse_args()
    return args


@delayed
def read_csv_w_model(file_path: str, model: str = "blip2_bert"):
    df = pd.read_csv(file_path)
    conf_mod = re.sub(r".*/(img|txt)/.*", r"\1", file_path)
    df["conf_modality"], df["model"] = conf_mod, model
    # add any other standardization routines, e.g. dtype conversion
    return df


def miate_evaluator(
    score_path: str,
    model_type: str,
    calc_type: str = "attattr",
    mod: str = "all",
    colors: List[str] = ["#42b7b9", "#c75dab"],
    save_formats: List[str] = ["eps", "html"],
    axes_limit: List[Union[float, int]] = [0, 1],
):
    if args.pick_correct_labels:
        makedirs_recursive(f"{score_path}/correct_label/miate/")
        files = glob.glob(
            f"{score_path}/correct_label/img/{calc_type}/{model_type}*{mod}.csv"
        )
        files += glob.glob(
            f"{score_path}/correct_label/txt/{calc_type}/{model_type}*{mod}.csv"
        )
    else:
        os.makedirs(f"{score_path}/miate/", exist_ok=True)
        files = glob.glob(f"{score_path}/img/{calc_type}/{model_type}*{mod}.csv")
        files += glob.glob(f"{score_path}/txt/{calc_type}/{model_type}*{mod}.csv")
    fig = go.Figure()
    # todo: pass model_type as arg
    ddf = dd.from_delayed([read_csv_w_model(f) for f in files])
    df = ddf.compute()
    df["conf_modality"] = df["conf_modality"].replace(
        {
            "txt": "text conf.",
            "img": "image conf.",
        },
    )
    for conf_mod, color in zip(df["conf_modality"].unique(), colors):
        values, errors, models = [], [], []
        for model in df["model"].unique():
            miate = df.loc[
                (df["model"] == model) & (df["conf_modality"] == conf_mod), "miate"
            ].values.astype("float")
            values.append(miate.mean())
            errors.append(miate.std(ddof=1) / np.sqrt(len(miate)))
            models.append(model)
        fig.add_trace(
            go.Bar(
                name=conf_mod,
                x=models,
                y=values,
                error_y=dict(type="data", array=errors),
                marker_color=color,
            )
        )
    pick_correct_labels = "correct_label" in score_path
    axes_limit = axes_limit if pick_correct_labels else [0, 0.5]
    fig.update_layout(
        template="plotly_white",
        barmode="group",
        yaxis_range=axes_limit,
    )
    if args.pick_correct_labels:
        export_name = f"{score_path}/correct_label/miate/{model_type}_{calc_type}_miate"
    else:
        export_name = f"{score_path}/miate/{model_type}_{calc_type}_miate"
    if "eps" in save_formats:
        fig.write_image(f"{export_name}.eps", engine="kaleido")
    if "html" in save_formats:
        fig.write_html(f"{export_name}.html")
    return


if __name__ == "__main__":
    args = miate_parser()
    _ = miate_evaluator(args.file_path, args.model_type, args.calc_type)
