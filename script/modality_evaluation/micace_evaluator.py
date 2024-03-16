import argparse
import glob
from typing import List, Union

import dask.dataframe as dd
from dask import delayed
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def micace_parser():
    parser = argparse.ArgumentParser(
        prog="MicaceEvaluator",
        description="Evaluate micace of models with same type",
    )
    parser.add_argument(
        "file_path", type=str, default="./data", help="Path where scores are saved"
    )
    parser.add_argument("model_type", type=str, default="O", help="vilio model type")
    parser.add_argument(
        "calc_type", type=str, default="attattr", help="Score calculation type"
    )
    args = parser.parse_args()
    return args


@delayed
def read_csv_w_model(file_path: str):
    df = pd.read_csv(file_path)
    conf_mod, model = (
        file_path.split("/")[-1].split("_")[0],
        file_path.split("/")[-1].split("_")[1],
    )
    df["conf_modality"], df["model"] = conf_mod, model
    # add any other standardization routines, e.g. dtype conversion
    return df


def micace_evaluator(
    score_path: str,
    model_type: str,
    calc_type: str = "attattr",
    mod: str = "all",
    colors: List[str] = ["#42b7b9", "#c75dab"],
    save_formats: List[str] = ["eps", "html"],
    axes_limit: List[Union[float, int]] = [0, 1],
):
    fig = go.Figure()
    files = glob.glob(f"{score_path}/{calc_type}/*_{model_type}*{mod}.csv")
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
            micace = df.loc[
                (df["model"] == model) & (df["conf_modality"] == conf_mod), "micace"
            ].values.astype("float")
            values.append(micace.mean())
            errors.append(micace.std(ddof=1) / np.sqrt(len(micace)))
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
    if "eps" in save_formats:
        fig.write_image(
            f"{score_path}/{calc_type}/{model_type}_miate.eps", engine="kaleido"
        )
    if "html" in save_formats:
        fig.write_html(f"{score_path}/{calc_type}/{model_type}_miate.html")
    return


if __name__ == "__main__":
    args = micace_parser()
    _ = micace_evaluator(args.file_path, args.model_type)
