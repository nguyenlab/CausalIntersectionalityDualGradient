from typing import List, Tuple, Dict, Union
import torch
from torch import Tensor

from modality_generation import modality_generator


def extract_modality_indices(
    caption_start_indices: List[int],
    image_start_indices: List[str],
) -> Dict[str, int]:
    """
    Extracts the indices of the caption and image tokens from the tokenized input.
    """
    out = {"caption": [], "image": []}
    for p_idx in range(len(caption_start_indices) - 1):
        out["caption"].append(
            (caption_start_indices[p_idx], image_start_indices[p_idx + 1])
        )
        out["image"].append(
            (image_start_indices[p_idx + 1], caption_start_indices[p_idx + 1] - 2)
        )
    return out


def convert_modalities_to_attention_weights(
    modalities: Dict[str, Tensor], attentions: Tensor
) -> Dict[str, Tensor]:
    """
    Converts modalities to attention weights.
    """
    out = {"caption": [], "image": [], "cross": []}
    for modality in modalities.keys():
        out[modality] = torch.mul(modalities[modality], attentions)
    return out


def extract_modality_attentions(
    attentions: Tensor,
    caption_start_indices: List[int],
    image_start_indices: List[str],
    num_hidden_layers: int = 40,
) -> Dict[str, Tensor]:
    """
    Extracts the attention weights for the caption/image/cross-modal inputs from the tokenized input.
    """
    modality_indices = extract_modality_indices(
        caption_start_indices, image_start_indices
    )
    modalities = modality_generator(
        modality_indices["caption"],
        modality_indices["image"],
        num_hidden_layers,
        attentions.shape[-1],
    )
    modality_attentions = convert_modalities_to_attention_weights(
        modalities, attentions
    )
    return modality_attentions


def attentions2vectors(modality_attentions: Dict[str, Tensor], shot_num: int):
    out = {}
    for ky in modality_attentions.keys():
        vector = modality_attentions[ky].sum(axis=[0, 1, 3])
        out[ky] = vector[vector != 0]
    return out
