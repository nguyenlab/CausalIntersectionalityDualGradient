"""
# Summary
Highlight sample with high MICACE score
## References
"""
import copy
import html
import os
import random
import re
from typing import List, Dict

import imgkit
import numpy as np
from PIL import Image, ImageDraw
import torch

from modality_evaluation.utils import nan_to_num

seed = 19871221

# Python random
random.seed(seed)
# Numpy
np.random.seed(seed)
# Pytorch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True


def highlight_sample(
    dump: str,
    exp_name: str,
    scores: Dict[str, torch.Tensor],
    ids: torch.Tensor,
    boxes: torch.Tensor,
    sents: torch.Tensor,
    json_path: str = "data/",
    max_alpha: float = 0.8,
    max_highlight_text_num: int = 10,
    max_highlight_image_num: int = 3,
    tokens_all: List[str] = None,
    sample_num: int = 10,
    sample_type: str = "topk",
    save_formats: List[str] = ["png"],
    is_relative: bool = True,
    extract_type: str = "attattr",
    tr_name: str = "bert-base-uncased",
    special_char: str = "[^A-Za-z0-9\s]+",
):
    """
    Highlight sample with high MICACE score
    todo: confounder-wise normalization?
    todo: convert to class
    """

    def extract_topk(tokens, scores, k=10):
        """
        Extract top k tokens
        """
        k_bound = min(k, len(tokens))
        topk = torch.topk(scores, k_bound)
        topk_tokens = tokens[topk.indices]
        return topk_tokens, topk.indices

    def sort_values_by_scores(scores, values):
        """
        Sort value tensor by score, descending order
        """
        sorts = torch.sort(scores, descending=True)
        sorted_scores, score_indices = sorts.values, sorts.indices
        sorted_values = np.array(values)[score_indices].tolist()
        return sorted_scores, sorted_values, score_indices

    def highlight_token(
        scores,
        tokens,
        max_alpha=max_alpha,
        max_highlight_num=max_highlight_text_num,
        is_relative=True,
    ):
        """
        Highlight token with high MICACE score
        todo: now only support html, hopefully support eps
        """
        max_highlight_num = (
            max_highlight_num if len(tokens) > max_highlight_num else len(tokens)
        )
        token_topk = extract_topk(
            np.array(tokens),
            scores,
            max_highlight_num,
        )[0]
        scores_sorted, tokens_sorted, scores_indices = sort_values_by_scores(
            scores, tokens
        )
        highlighted_tokens = {i: "tmp" for i in range(len(scores))}
        for i, (score_s, token_s, score_i) in enumerate(
            zip(scores_sorted, tokens_sorted, scores_indices)
        ):
            if is_relative:
                score_highlight = max_alpha * (
                    (max_highlight_num - i) / max_highlight_num
                )
            else:
                score_highlight = score_s.item() / max_alpha
            if token_s in token_topk:
                highlighted_tokens[score_i.item()] = (
                    '<span style="background-color:rgba(135,206,250,'
                    + str(score_highlight)
                    + ');">'
                    + html.escape(token_s)
                    + "</span>"
                )
            else:
                highlighted_tokens[score_i.item()] = token_s

        return highlighted_tokens.values(), token_topk

    def dict_token_to_text(tokens, words):
        """Create dictionary of token key to text value"""
        out = {word: [] for word in words}
        for token in tokens:
            words2check = copy.deepcopy(words)
            for word in words2check:
                if token in word:
                    out[word].append(token)
                    break
                else:
                    pass
                    # words2check.remove(words2check[0])
        return out

    def strp_whitespace_between_subword(highlighted_tokens, token_dict):
        """Strip whitespace between subword"""
        out = []
        for token in highlighted_tokens:
            if ">" in token:
                token_strp = token.split(">")[1].split("<")[0]
            else:
                token_strp = token
            for word in token_dict:
                if token_strp in token_dict[word][:-2]:
                    out.append(token)
                    break
                else:
                    out.append(token)
                    out.append(" ")
                    break
        return out

    def highlight_text(
        words,
        scores,
        tokens,
        is_relative=True,
    ):
        """Highlight text with high MICACE score"""
        highlighted_tokens, token_topk = highlight_token(
            scores, tokens, is_relative=is_relative
        )
        token_dict = dict_token_to_text(token_topk, words)
        highlighted_tokens_strp = strp_whitespace_between_subword(
            highlighted_tokens, token_dict
        )
        out = "".join(highlighted_tokens_strp)
        return out

    def convert_box_to_actual(box, img):
        """
        Opposite to delta, convert box with relative size to that with actual size
        https://detectron2.readthedocs.io/en/latest/_modules/detectron2/modeling/box_regression.html
        """
        img_h, img_w = img.size
        # In case of UNITER with 7 elements in box, first 4 are used - see hm.py
        out = box[:4] * torch.tensor([img_w, img_h, img_w, img_h])
        return out

    def draw_box(box, img, label):
        """
        Draw box on image
        todo: image setting as input
        """
        top, left, bottom, right = box
        draw = ImageDraw.Draw(img, "RGBA")
        label_size = draw.textsize(label)
        draw.rectangle([left, top, bottom, right], fill=(231, 38, 235, 50))
        draw.rectangle(
            [left, top, bottom, right],
            outline=(231, 38, 235, 100),
            width=3,
        )
        draw.text((left, top - label_size[1]), label, fill=(231, 38, 235))
        return img

    def highlight_image(
        boxes: torch.Tensor,
        image_id: str,
        scores: torch.Tensor,
        json_path: str = json_path,
        max_highlight_num: int = max_highlight_image_num,
    ):
        """
        Highlight image with high MICACE score
        """
        max_highlight_num = (
            max_highlight_num if max_highlight_num < boxes.shape[0] else boxes.shape[0]
        )
        img = Image.open(f"{json_path}img/{image_id}.png")
        box_topk_idx = extract_topk(boxes, scores, max_highlight_num)[1]
        for i, (box, score) in enumerate(zip(boxes, scores)):
            if i in box_topk_idx:
                if torch.max(box) > 1:
                    box_actual = box
                else:
                    box_actual = convert_box_to_actual(box, img)
                img_label = f"score: {score.item()}"
                img = draw_box(box_actual, img, img_label)
            else:
                continue
        return img

    def save_sample(
        filedir: str,
        exp_name: str,
        image_id: str,
        highlighted_text: str,
        highlighted_image: Image,
        save_formats: List[str] = save_formats,
        modality: str = None,
        extract_type: str = extract_type,
    ):
        """
        Save sample with high MICACE score
        """
        os.makedirs(f"{filedir}/sample_analysis", exist_ok=True)
        for fm in save_formats:
            os.makedirs(f"{filedir}/sample_analysis/{exp_name}/{fm}", exist_ok=True)
            os.makedirs(
                f"{filedir}/sample_analysis/{exp_name}/{fm}/midas_{modality}",
                exist_ok=True,
            )
            os.makedirs(
                f"{filedir}/sample_analysis/{exp_name}/{fm}/midas_{modality}/{extract_type}",
                exist_ok=True,
            )
            os.makedirs(
                f"{filedir}/sample_analysis/{exp_name}/{fm}/midas_{modality}/{extract_type}/image",
                exist_ok=True,
            )
            os.makedirs(
                f"{filedir}/sample_analysis/{exp_name}/{fm}/midas_{modality}/{extract_type}/text",
                exist_ok=True,
            )
        if modality != "image" and highlighted_text is not None:
            if "html" in save_formats:
                with open(
                    f"{filedir}/sample_analysis/{exp_name}/html/midas_{modality}/{extract_type}/text/{image_id}_text.html",
                    "w",
                ) as file:
                    file.write(highlighted_text)
            else:
                pass
            if "eps" in save_formats:
                imgkit.from_string(highlighted_text, "tmp.png")
                highlighted_text_im = Image.open("tmp.png")
                highlighted_text_im.convert("RGB").save(
                    f"{filedir}/sample_analysis/{exp_name}/eps/midas_{modality}/{extract_type}/text/{image_id}_text.eps",
                    lossless=True,
                )
                os.remove("tmp.png")
            else:
                pass
            if "png" in save_formats:
                imgkit.from_string(
                    highlighted_text,
                    f"{filedir}/sample_analysis/{exp_name}/png/midas_{modality}/{extract_type}/text/{image_id}_text.png",
                )
            else:
                pass
        else:
            pass
        if modality != "text" and highlighted_image is not None:
            if "eps" in save_formats:
                highlighted_image.save(
                    f"{filedir}/sample_analysis/{exp_name}/eps/midas_{modality}/{extract_type}/image/{image_id}.eps",
                    lossless=True,
                )
            else:
                pass
            if "png" in save_formats:
                highlighted_image.save(
                    f"{filedir}/sample_analysis/{exp_name}/png/midas_{modality}/{extract_type}/image/{image_id}.png"
                )
            else:
                pass
        else:
            pass
        return (highlighted_text, highlighted_image)

    def min_max_norm(scores):
        """
        Min-max normalization
        todo: 0-1 normalization
        """
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        return scores

    def shift_norm(scores, new_min=0, new_max=255):
        """Shift normalization"""
        out = min_max_norm(scores)
        out = out * (new_max - new_min) + new_min
        return out

    filedir = "/".join(dump.split("/")[:-2])
    # filedir = "/content/export"

    # df_json = pd.read_json(f"{json_path}/{split}.jsonl", orient="records", lines=True)
    for modality in scores.keys():
        score_mod = scores[modality].detach().cpu()
        score_mod = nan_to_num(score_mod, nan=0.0)
        score_token = score_mod.sum(axis=[1, 3])
        score_token = shift_norm(score_token)
        out = {"text": [], "image": []}
        # todo: split score to text and image section upon num of tokens
        for i, sent, score, box, tokens in zip(
            ids, sents, score_token, boxes, tokens_all
        ):
            image_id = str(i.item()).zfill(5)
            sent = re.sub(special_char, "", sent)
            tokens = [
                re.sub(special_char, "", token)
                for token in tokens
                if len(re.sub(special_char, "", token)) >= 1
            ]
            words = sent.split(" ")
            tokens = tokens[1:-1]  # remove cls and sep
            num_tokens = len(tokens)
            score_txt, score_img = (
                score[1 : num_tokens + 1],
                score[num_tokens + 2 : (num_tokens + 2) + len(box)],
            )
            highlighted_text, highlighted_image = None, None
            if modality != "image":
                highlighted_text = highlight_text(
                    words, score_txt, tokens, is_relative=is_relative
                )
                out["text"].append(highlighted_text)
            else:
                pass
            if modality != "text":
                highlighted_image = highlight_image(box, image_id, score_img)
                out["image"].append(highlighted_image)
            save_sample(
                filedir,
                exp_name,
                image_id,
                highlighted_text,
                highlighted_image,
                modality=modality,
            )

    return out
