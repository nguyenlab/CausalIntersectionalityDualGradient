from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
from math import sqrt
from typing import List

import einops
import torch
from torch import Tensor
import torch.nn.functional as F


def segment_modality_converter(segment: Tensor):
    out = torch.mm(segment.T, segment)
    return out


def cross_modality_generator(
    modality_all: Tensor, modality1: Tensor, modality2: Tensor
):
    out = modality_all - modality1
    out = out - modality2
    return out


def segment_generator(modality_indices: List[int], input_length: int):
    segment = torch.zeros((1, input_length), dtype=torch.long)
    for mod_idx in modality_indices:
        segment[0, mod_idx[0] : mod_idx[1]] = 1
    return segment


def round_half_up(f: float):
    return Decimal(f).quantize(Decimal("0"), rounding=ROUND_HALF_UP)


def modality_generator(
    caption_indices: List[int],
    image_indices: List[int],
    num_hidden_layers: int,
    input_length: int,
):
    segment_all = segment_generator(caption_indices + image_indices, input_length)
    segment_caption = segment_generator(caption_indices, input_length)
    segment_image = segment_generator(image_indices, input_length)
    assert (
        segment_all.sum() == segment_caption.sum() + segment_image.sum()
    ), f"Segmentation: # all elements {segment_all.sum()} should match that of caption {segment_caption.sum()} + image {segment_image.sum()}"
    modality_all = segment_modality_converter(segment_all)
    modality_caption = segment_modality_converter(segment_caption)
    modality_image = segment_modality_converter(segment_image)
    assert round_half_up(sqrt(modality_all.sum())) == round_half_up(
        sqrt(modality_caption.sum()) + sqrt(modality_image.sum())
    ), f"Modality Matrix: # all elements {round_half_up(sqrt(modality_all.sum()))} should match that of caption {sqrt(modality_caption.sum())} + image {sqrt(modality_image.sum())}"
    modality_all = einops.repeat(modality_all, "d e -> 1 n d e", n=num_hidden_layers)
    modality_caption = einops.repeat(
        modality_caption, "d e -> 1 n d e", n=num_hidden_layers
    )
    modality_image = einops.repeat(
        modality_image, "d e -> 1 n d e", n=num_hidden_layers
    )
    assert round_half_up(sqrt(modality_all.sum())) == round_half_up(
        sqrt(modality_caption.sum()) + sqrt(modality_image.sum())
    ), f"Extended Modality Matrix: # all elements {round_half_up(sqrt(modality_all.sum()))} should match that of caption {round_half_up(sqrt(modality_caption.sum()))} + image {round_half_up(sqrt(modality_image.sum()))}"
    modality_cross = cross_modality_generator(
        modality_all, modality_caption, modality_image
    )
    assert (
        modality_all.sum()
        == modality_cross.sum() + modality_caption.sum() + modality_image.sum()
    ), f"Modality Matrix w/ Cross-Modal Attention: # all elements {modality_all.sum()} should match that of caption {modality_caption.sum()} + image {modality_image.sum()} + cross {modality_cross.sum()}"
    return {
        "caption": modality_caption,
        "image": modality_image,
        "cross": modality_cross,
    }
