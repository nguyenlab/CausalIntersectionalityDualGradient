"""
# Summary
Preprocessing required for modality evaluation
## References
"""
from typing import List

import dask.dataframe as dd
from dask import delayed
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset


@delayed
def read_jsonl_w_source(file_path: str, split: str):
    df = pd.read_json(file_path, orient="records", lines=True).rename(
        {"image": "img"}, axis=1
    )
    df["split"] = split
    return df.drop_duplicates()


def generate_confounders(
    path_conf: str,
    json_path: str = "data/",
    modality: str = "img",
    col_suffix: str = "_org_id",
    remove_txts: List[str] = ["No Confounder", "Not hit"],
    split: str = "dev_seen",
    conf_num: int = 3,
    mod_splits: List[str] = ["dev_seen", "train"],
    rs: int = 19871221,
):
    """
    Generate confounders for dataset
    """
    # load jsonl files / confounder file
    ddf_memes = dd.from_delayed(
        [read_jsonl_w_source(f"{json_path}/{s}.jsonl", s) for s in mod_splits]
    )
    df_memes = ddf_memes.compute()
    df_conf = pd.read_parquet(path_conf, engine="pyarrow")

    # get positive samples
    df_mod = (
        df_conf[
            (~df_conf[modality + col_suffix].isin(remove_txts))
            & (df_conf["split"] == split)
            & (df_conf["label"] == 1)
        ]
        .reset_index(drop=True)
        .drop_duplicates()
    )
    positive_ids = df_mod[modality + col_suffix].drop_duplicates().to_numpy()
    # sample confounders
    conf_mod = "txt" if modality == "img" else "img"
    for i, positive_id in enumerate(positive_ids):
        neg_ids = (
            df_conf.loc[
                ~df_conf[conf_mod + col_suffix].isin(remove_txts)
                & (df_conf[modality + col_suffix] != positive_id),
                conf_mod + col_suffix,
            ]
            .drop_duplicates()
            .sample(n=conf_num, replace=False, random_state=rs + i)
            .to_numpy()
        )
        negative_df = df_memes[df_memes["id"].isin(neg_ids.astype(int))]
        if modality == "img":
            df_gen = pd.DataFrame(
                {
                    "id": np.repeat(positive_id, conf_num),
                    "img": np.repeat(f"img/{positive_id}.png", conf_num),
                    "text": negative_df["text"].values,
                    "label": np.repeat(0, conf_num),
                }
            )
        else:
            positive_df = df_memes[df_memes["id"] == int(positive_id)]
            df_gen = pd.DataFrame(
                {
                    "id": np.repeat(positive_id, conf_num),
                    "img": negative_df["img"].values,
                    "text": positive_df["text"].values[0],
                    "label": np.repeat(0, conf_num),
                }
            )
        df_gen["split"] = split
        df_memes = df_memes.append(df_gen[df_memes.columns], ignore_index=True)
    df_memes.to_json(
        f"{json_path}/{split}_negatives.jsonl",
        force_ascii=False,
        orient="records",
        lines=True,
    )
    return df_memes


def batch_confounders(
    dataset,
    path_conf: str,
    modality: str = "img",
    col_suffix: str = "_org_id",
    remove_txts: List[str] = ["No Confounder", "Not hit"],
    split: str = "dev_seen",
    batch_size: int = 8,
):
    """
    Sort dataset by confounder ids
    If batch size is larger than # of confounders, repeat samples
    todo: split/batch info from args
    """
    df_conf = pd.read_parquet(path_conf, engine="pyarrow")

    out = {}
    if "test" in split and modality == "img":
        out[modality] = dataset
    else:
        df_sorted = (
            df_conf[
                (~df_conf[modality + col_suffix].isin(remove_txts))
                & (df_conf["split"] == split)
            ]
            .sort_values(by=modality + col_suffix)
            .reset_index(drop=True)
            .drop_duplicates()
        )
        ids = df_sorted["id"].to_numpy()
        conf_ids = df_sorted[modality + col_suffix].to_numpy()
        ds = np.array([d[0] for d in dataset])
        # todo: ids with each conf_id to group of batch
        indices = []
        for conf_id in np.unique(conf_ids):
            id_group = ids[conf_ids == conf_id]
            if len(id_group) <= 1:
                continue
            else:
                num_repeat, num2add = divmod(batch_size, len(id_group))
                id_group = np.repeat(id_group, num_repeat).tolist()
                id_group += id_group[:(num2add)]
            for i in id_group:
                id_loc = np.where(ds == int(i))[0][0]
                indices.append(id_loc)

    return indices


class OrderedHMDataset(Dataset):
    def __init__(
        self,
        dataset,
        path_conf: str,
        modality: str = "img",
        col_suffix: str = "_org_id",
        remove_txts: List[str] = ["No Confounder", "Not hit"],
        split: str = "dev_seen",
        batch_size: int = 8,
    ):
        self.indices = batch_confounders(
            dataset,
            path_conf,
            modality=modality,
            split=split,
            batch_size=batch_size,
            col_suffix=col_suffix,
            remove_txts=remove_txts,
        )
        self.dataset = Subset(dataset, indices=self.indices)
        self.id2datum = {
            dataset.id2datum[dataset[idx][0]]["id"]: dataset.id2datum[dataset[idx][0]]
            for idx in self.indices
        }
        # self.id2datum = {
        #     dataset.id2datum[self.dataset[idx][0]]["id"]: dataset.id2datum[
        #         self.dataset[idx][0]
        #     ]
        #     for idx in self.indices
        # }

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[idx]


def min_max_scaler(
    vs: torch.Tensor, norm_size: int = 2, new_max: float = 1.0, new_min: float = 0.0
) -> torch.Tensor:
    """
    scale tensor
    [min, max] -> [new_min,new_max]
    """
    dvs, mds = divmod(vs.shape[0], norm_size)
    assert (
        mds == 0
    ), "# samples {vs.shape[0]} should be divided by norm_size {norm_size}"
    vs_p = []
    for dv in range(dvs):
        v = vs[dv * norm_size : (dv + 1) * norm_size]
        v_min, v_max = v.min(), v.max()
        v_p = (v - v_min) / (v_max - v_min) * (new_max - new_min) + new_min
        vs_p.append(v_p)
    vs_p = torch.stack(vs_p).reshape(vs.shape)
    return vs_p
