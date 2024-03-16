"""
# Summary
Shared in evaluation codes
## References
"""
import copy
from numbers import Number
from typing import List, Union

import numpy as np
import torch


def simple_sum(score: Union[List[Number], np.ndarray, torch.Tensor, Number]):
    """
    tensor of size (n,m,k) -> size n
    list of size n -> 1 float
    """
    if type(score) == torch.Tensor:
        sample_num = score.dim()
        dim = [idx for idx in range(sample_num) if idx not in [0]]
        out = score.sum(dim=dim)
    elif isinstance(score, Number):
        out = score
    else:
        out = sum(score)
    return out


def simple_avg(score: Union[List[Number], np.ndarray, torch.Tensor, Number]):
    """
    tensor of size (n,m,k) -> size n
    list of size n -> 1 float
    """
    summed = simple_sum(score)
    if type(score) == torch.Tensor:
        num_dim = torch.numel(score) / score.shape[0]
        out = summed / num_dim
    elif isinstance(score, Number):
        out = summed
    else:
        out = summed / len(score)
    return out


def nonzero_avg(score: Union[List[Number], np.ndarray, torch.Tensor, Number]):
    """
    tensor of size (n,m,k) -> size n
    list of size n -> 1 float
    averaged by number of non-zero elements
    """
    summed = simple_sum(score)
    if type(score) == torch.Tensor:
        # todo: version issue
        # num_dim = int(torch.count_nonzero(score))
        num_dim = len(torch.nonzero(score))
        out = summed / num_dim
    elif isinstance(score, Number):
        out = summed
    else:
        out = summed / len(score)
    return out


def simple_summary(score: Union[List[float], torch.Tensor], mode: str = "avg"):
    """
    Main function for score summarization
    """
    if mode == "avg":
        out = nonzero_avg(score)
    elif mode == "sum":
        out = simple_sum(score)
    else:
        if type(score) == torch.Tensor:
            out = score.clone().detach()
        else:
            out = copy.deepcopy(score)
    return out


def nan_to_num(a, nan=None):
    """
    to use with torch 1.6
    https://github.com/pytorch/pytorch/blob/31f311a816c026bbfca622d6121d6a7fab44260d/torch/_refs/__init__.py
    https://github.com/pytorch/pytorch/issues/9190
    """
    result = torch.where(torch.isnan(a), torch.tensor(nan), a)
    return result
