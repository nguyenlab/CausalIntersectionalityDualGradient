import argparse
from datetime import datetime
import time
from typing import Dict, List, Union

import numpy as np
from numpy.core.defchararray import add
from numpy.random import default_rng
import pandas as pd

from utils import get_module_logger, makedirs_recursive


def arg_parser():
    """
    Add arguments to main function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cp",
        "--caption_dir",
        help="Directory for captions",
    )
    parser.add_argument(
        "-m",
        "--meme_dir",
        help="Directory for hateful memes",
    )
    parser.add_argument(
        "-cf",
        "--conf_dir",
        help="Directory for hateful memes",
    )
    parser.add_argument(
        "-sf",
        "--save_dir",
        help="Directory for saving prompts",
    )
    args = parser.parse_args()
    return args


def load_inputs(
    args,
    id_org_col="id",
    id_col="image_id",
    split_col="split",
    split="dev_seen",
    text_org_col="text",
    modalities=["img", "txt"],
    no_conf=["No Confounder", "Not hit"],
    caption_col="text_captioned",
    export_cols=["text_org", "label"],
    group_col="org_id",
):
    """
    Load inputs
    """
    export_cols += [id_col, caption_col, group_col]
    rename_dict = {id_org_col: id_col}
    df_caption = pd.read_parquet(f"{args.caption_dir}/hm_captions.parquet")
    df_conf = pd.read_parquet(f"{args.conf_dir}/confounders.parquet").rename(
        rename_dict, axis=1
    )
    df_conf = df_conf[df_conf[split_col] == split].reset_index(drop=True)
    rename_dict[text_org_col] = "text_org"
    df_json = pd.read_json(
        f"{args.meme_dir}/{split}.jsonl",
        orient="records",
        lines=True,
    )[[id_org_col, text_org_col]].rename(rename_dict, axis=1)
    df_json[id_col] = df_json[id_col].astype(str).str.zfill(5)
    df = df_caption.merge(df_conf, on=id_col, how="inner")
    df = df.merge(df_json, on=id_col, how="left")
    df_out = []
    for mod in modalities:
        df_mod = (
            df[~df[f"{mod}_{group_col}"].isin(no_conf)]
            .sort_values(
                by=[f"{mod}_{group_col}", "label"],
                ascending=[True, False],
            )
            .reset_index(drop=True)
        )
        df_mod[group_col] = df_mod[f"{mod}_{group_col}"].astype(str).str.zfill(5)
        df_out.append(df_mod[export_cols])
    df_out = pd.concat(df_out)
    return df_out


def shuffle_df_group(df, group_col, rs=1987):
    """
    Shuffle dataframe with group
    """
    df_out = df.copy()
    df_out["rand"] = df_out.groupby(group_col).cumcount()
    df_out = df_out.sample(frac=1, random_state=rs).reset_index(drop=True)
    df_out = df_out.sort_values(by=["rand"]).reset_index(drop=True)
    df_out = df_out.drop(columns=["rand"])
    return df_out


def escape_instruction_chars(text):
    """
    Escape instruction characters
    """
    text = text.replace("caption", "text")
    text = text.replace("image", "picture")
    return text


def generate_prompt_df(
    df,
    rs=1987,
    source_cols={"image_desc": "text_captioned", "caption": "text_org"},
    label_col="label",
    label_map={1: "hateful", 0: "benign"},
    out_col="data",
    prefix="caption: '",
    connection="', image: '",
    suffix="'",
):
    """
    Generate dataset w/ caption and image description
    """
    # df_out = df.copy()
    df_out = shuffle_df_group(df, group_col="org_id", rs=rs)
    df_out[label_col] = df_out[label_col].map(label_map)
    df_out[out_col] = (
        prefix
        + df_out[source_cols["caption"]].apply(escape_instruction_chars)
        + connection
        + df_out[source_cols["image_desc"]].apply(escape_instruction_chars)
        + suffix
    )
    return df_out


def generate_few_shot_prompt(
    data_array: np.array,
    gid_num: Union[int, str],
    label_prefix: str = "System:",
    label_prefix_gt: str = "The most likely hateful or sarcastic pair is ",
    prefix: str = "#",
    suffix: str = ":",
    label_suffix: str = ", with the potential label 'hateful'.\n\nUser:",
    connector: str = "\n",
    label_map: Dict[int, str] = {1: "hateful", 0: "benign"},
):
    """
    Generate prompts for few shot classification
    """
    label_last = f"\n{label_prefix}"

    descs_last = []
    descs_org_last = data_array[data_array[:, 3] == gid_num, 0]
    for i_last, desc_org_last in enumerate(descs_org_last):
        prefix_sample = f"{prefix}{i_last}{suffix}"
        desc = f"{prefix_sample} {desc_org_last}"
        descs_last.append(desc)
    desc_last = "\n".join(descs_last)

    out_list = []
    out_list_record = []
    id_nums = np.unique(data_array[:, 3])
    for i, id_num in enumerate(id_nums):
        data_gid = data_array[data_array[:, 3] == id_num]
        descs = []
        descs_org, labels_org = data_gid[:, 0], data_gid[:, 1]
        for i2, desc_org in enumerate(descs_org):
            prefix_sample = f"{prefix}{i2}{suffix}"
            desc = f"{prefix_sample} {desc_org}"
            descs.append(desc)
        pos_indices = np.where(labels_org == label_map[1])[0]
        pos_indices = add(prefix, pos_indices.astype(str))
        label = f"{' and '.join(pos_indices)}"
        label = f"{label_prefix}\n\n{label_prefix_gt}{label}{label_suffix}"

        desc = "\n".join(descs)
        if i:
            out_prev = "\n".join(out_list_record[:i]) + "\n"
        else:
            out_prev = ""
        out_list.append(out_prev + desc_last + label_last)
        out_list_record.append(desc + connector + label)

    return out_list


def save_prompts(
    save_dir,
    prompts,
    _prefix="User: Out of image-caption pairs #0 to #",
    suffix=", select the most likely hateful or sarcastic pair, together with potential label (hateful or sarcastic). If all pairs are benign, say 'All pairs are benign'.\n",
):
    prefix = f"{_prefix}{prompts[0].count('#')-1}{suffix}"
    for i, prompt in enumerate(prompts):
        with open(
            f"{save_dir}/prompt{str(i).zfill(5)}.txt", "a", encoding="utf-8"
        ) as f:
            f.write(prefix + prompt)
    return prompts


def generate_prompts(
    df,
    save_dir,
    out_col="data",
    group_col="org_id",
    label_col="label",
    shuffle=True,
    rs=1987,
    sort_col="sort_num",
):
    df_id_labels = (
        df.groupby(
            [
                group_col,
                label_col,
            ]
        )
        .count()[out_col]
        .reset_index()
    )
    df_id_labels = df_id_labels.groupby(group_col).count()[label_col].reset_index()
    non_unique_labels = df_id_labels.loc[
        df_id_labels[label_col] >= 2, group_col
    ].values.tolist()
    df_selected = (
        df.loc[df[group_col].isin(non_unique_labels), [out_col, label_col, group_col]]
        .drop_duplicates(subset=out_col)
        .reset_index(drop=True)
    )
    ids = df_selected[group_col].unique().tolist()
    out = []
    for i, gid in enumerate(non_unique_labels):
        ids = df_selected[group_col].unique()
        df_scope = df_selected.copy()
        if shuffle:
            rng = default_rng(seed=rs + i)
            rands = rng.choice(len(ids) + 1, size=len(ids), replace=False)
            id_dict = {iid: n for iid, n in zip(ids, rands)}
            id_dict[gid] = 99999
            df_scope[sort_col] = df_scope[group_col].map(id_dict)
        else:
            id_dict = {iid: iid for iid in ids}
            id_dict[gid] = f"99999_{gid}"
            df_scope[sort_col] = df_scope[group_col].map(id_dict)

        df_sorted = df_scope.sort_values(by=sort_col).reset_index(drop=True)
        df_indices = df_sorted[(df_sorted[sort_col] == id_dict[gid])].reset_index(
            drop=True
        )
        pos_indices = "_".join(
            df_indices[(df_indices[label_col] == "hateful")].index.astype(str).tolist()
        )
        max_idx = str(df_indices.index.max())
        idxes_label = f"{pos_indices}_pos_max_{max_idx}"

        save_dir_gid = f"{save_dir}/{gid}_{idxes_label}"
        makedirs_recursive(save_dir_gid)

        arr_prompts = df_sorted.values
        out_scope = generate_few_shot_prompt(arr_prompts, id_dict[gid])
        out_scope = save_prompts(save_dir_gid, out_scope)
        out.append(out_scope)
    return out


if __name__ == "__main__":
    t0 = time.time()
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    args = arg_parser()
    logger = get_module_logger(__name__)

    df = load_inputs(args)
    num_samples = len(df)
    logger.info(f"Loaded inputs with length {num_samples}")

    df = generate_prompt_df(df)
    assert num_samples == len(
        df
    ), "Length should be same before/after prompt source generation"
    logger.info(f"Generated prompt source with length {len(df)}")

    prompts = generate_prompts(df, args.save_dir)
    logger.info(f"All process complete w/ {time.time()-t0} seconds")
