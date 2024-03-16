"""
# Summary
Modality scoring
## References
"""
from typing import Dict, List, Union

import einops
import torch
from torch import Tensor
import torch.nn.functional as F

from modality_evaluation.utils import simple_summary


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
    return {"text": modalities1, "image": modalities2, "cross": cross_modalities}


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
