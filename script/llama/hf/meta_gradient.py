import argparse
from more_itertools import locate
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from calculation import (
    attentions2vectors,
    extract_modality_attentions,
    extract_modality_indices,
)
from utils import get_module_logger, load_specified_files, makedirs_recursive


def arg_parser():
    """
    Add arguments to main function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-sp",
        "--save_path",
        help="Directory for saving extracted info",
    )
    parser.add_argument(
        "-rp",
        "--result_path",
        help="Directory for few shot result",
    )
    parser.add_argument(
        "-pp",
        "--prompt_path",
        help="Directory for source prompts",
    )
    parser.add_argument(
        "-mn",
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-13b-chat-hf",
        help="Model name on HuggingFace format",
    )
    parser.add_argument(
        "-nah",
        "--num_hidden_layers",
        type=int,
        default=40,
        help="# of hidden layers",
    )
    args = parser.parse_args()
    return args


def extract_info(texts: list, prompts: list):
    out = []
    for text, prompt in zip(texts, prompts):
        out.append(text.replace(prompt, ""))
    return out


def extract_consecutive_chars(lst: list, char1: str, char2: str):
    indices = [
        i for i in range(len(lst) - 1) if lst[i][1:] == char1 and lst[i + 1] == char2
    ]
    return indices


def find_indices(list_to_check, item_to_find):
    indices = locate(list_to_check, lambda x: x[1:] == item_to_find)
    return list(indices)


def main():
    args = arg_parser()
    logger = get_module_logger(__name__)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # makedirs_recursive(args.save_path)
    atts, att_nums = load_specified_files(
        args.result_path, file_type="attention", extension="pt"
    )
    logger.info(f"Loaded {len(atts)} attention maps")
    # dict of image ids > list with #<few shots> elements > tensor with size [1, 40, #<tokens>, #<tokens>]

    output = {}
    os.makedirs(f"{args.save_path}/atts", exist_ok=True)

    for ky in atts.keys():
        att_nums_iid = att_nums[ky]
        num_prompts = len(att_nums_iid)
        atts_shots = {"caption": [], "image": [], "cross": []}
        # extract attention maps for each prompt
        for i, att_num in enumerate(att_nums_iid):
            atts_prompt = atts[ky][i]

            # load prompt and extract indices of caption and image
            prompt_file = f"{args.prompt_path}/{ky}/prompt{att_num}.txt"
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt = f.read()
            tokens = tokenizer.tokenize(prompt)
            caption_indices = extract_consecutive_chars(tokens, "ca", "ption")
            caption_indices.append(len(tokens) - 3)
            image_indices = find_indices(tokens, "image")
            if not i:
                num_pairs = len(caption_indices)
            else:
                caption_indices = caption_indices[-num_pairs:]
                image_indices = image_indices[-num_pairs:]
            diffs = []
            modality_indices = extract_modality_indices(caption_indices, image_indices)
            for k in modality_indices.keys():
                print(k)
                for idx in modality_indices[k]:
                    print(idx)
                    print("".join(tokens[idx[0] : idx[1]]))
                    print("==================")
            assert num_pairs == len(caption_indices)
            modality_attentions = extract_modality_attentions(
                atts_prompt, caption_indices, image_indices, args.num_hidden_layers
            )
            modality_vectors = attentions2vectors(modality_attentions, i)
            for ky_shots in atts_shots.keys():
                atts_shots[ky_shots].append(modality_vectors[ky_shots])
        # calculate zsl and delta attention maps for each prompt
        weights = {}
        os.makedirs(f"{args.save_path}/atts/{ky}", exist_ok=True)
        for ky_att in atts_shots.keys():
            atts_shots_mod = atts_shots[ky_att]
            for i, atts_shot_mode in enumerate(atts_shots_mod):
                if not i:
                    weights[f"{ky_att}_zsl"] = atts_shot_mode
                else:
                    zsl_weights = weights[f"{ky_att}_zsl"]
                    weights[f"{ky_att}_delta_{i}"] = (
                        atts_shot_mode[-len(zsl_weights) :] - zsl_weights
                    )
        # save zsl and delta attention maps
        for save_ky in weights.keys():
            torch.save(weights[save_ky], f"{args.save_path}/atts/{ky}/{save_ky}.pt")
        output[ky] = weights

    logger.info("Extracted meta gradients")

    return output


if __name__ == "__main__":
    _ = main()
