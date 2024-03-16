import argparse
import os
import time

import pandas as pd
from PIL import Image
from lavis.models import load_model_and_preprocess
import torch

from utils import get_module_logger, read_jsonls_w_source


def arg_parser():
    """
    Add arguments to main function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_dir",
        help="Directory for hateful memes",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Directory for output captions",
    )
    args = parser.parse_args()
    return args


def image2text(
    image_file,
    model,
    vis_processors,
    caption,
    caption_ps=[
        "I know caption '",
        "' is embedded on this image. Other than that, ",
    ],
    detail_p="when explained in details, ",
    end_p="this image is about ",
):
    """
    image2text with BLIP2 model
    """
    raw_image = Image.open(image_file).convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    text = model.generate({"image": image})[0]
    text_d = model.generate({"image": image, "prompt": detail_p + end_p})[0]
    caption_prompt_base = caption_ps[0] + caption + caption_ps[1]
    text_c = model.generate(
        {
            "image": image,
            "prompt": caption_prompt_base + end_p,
        }
    )[0]
    text_cd = model.generate(
        {
            "image": image,
            "prompt": caption_prompt_base + detail_p + end_p,
        }
    )[0]
    return {
        "image_id": image_file.split("/")[-1].split(".")[0],
        "text": text,
        "text_detailed": text_d,
        "text_captioned": text_c,
        "text_detailed_captioned": text_cd,
    }


def get_caption(df_memes, iid):
    """
    Get ground truth caption from memes dataframe
    """
    return df_memes[df_memes["id"] == int(iid)].text.values[0]


def images2texts(args, model, vis_processors):
    """
    Main function for batch captioning
    """
    out = {
        "image_id": [],
        "text": [],
        "text_detailed": [],
        "text_captioned": [],
        "text_detailed_captioned": [],
    }
    df_memes = read_jsonls_w_source(args.input_dir)
    for file in os.listdir(f"{args.input_dir}/img"):
        image_file = f"{args.input_dir}/img/{file}" if ".png" in file else None
        if image_file:
            caption = get_caption(df_memes, file.split(".")[0])
            text_dict = image2text(image_file, model, vis_processors, caption)
            for k in text_dict.keys():
                out[k].append(text_dict[k])
    return out


def save_text_df(args, texts_dict):
    """
    Save dataframe w/ captions to designated directory
    """
    text_df = pd.DataFrame(texts_dict)
    text_df.to_parquet(
        f"{args.output_dir}/hm_captions.parquet", index=False, engine="pyarrow"
    )
    return text_df


"""
todo:
prompt: image shows xx and yy
"""

if __name__ == "__main__":
    t0 = time.time()
    # logger & parser
    logger = get_module_logger(__name__)
    args = arg_parser()
    os.makedirs(args.output_dir, exist_ok=True)
    # load
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5",
        model_type="pretrain_flant5xxl",
        is_eval=True,
        device=device,
    )
    # process
    texts_dict = images2texts(args, model, vis_processors)
    text_df = save_text_df(args, texts_dict)
    logger.info(f"Extracted captions with # {len(text_df)}")
    logger.info(f"Processing complete in {round(time.time()-t0, 2)} seconds.")
