"""
# Summary
Functions for attention attribute score/ integrated gradients
## References
- Hao et al., 2021 (https://github.com/YRdddream/attattr)
- Sundararajan et al., 2017 (https://github.com/ankurtaly/Integrated-Gradients)
"""
import torch


def min_max_scaler(
    vs: torch.Tensor,
    norm_size: int = 2,
    new_max: float = 1.0,
    new_min: float = 0.0,
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


def scaled_input(emb: torch.Tensor, steps: int, baseline: bool = None):
    """
    Generate concatenated samples for step-wise gradient calculation
    ((1/m)*X, (2/m)*X, ..., (m/m)*X)
    """
    if not baseline:
        baseline = torch.zeros(emb.shape)
    else:
        pass
    emb_diff = emb - baseline
    scaled_inputs = [
        baseline + (float(i) / steps) * emb_diff for i in range(0, steps + 1)
    ]
    scaled_inputs = torch.stack(scaled_inputs)

    return scaled_inputs


def attention_attribute(
    A: torch.Tensor,
    F: torch.Tensor,
    m: int,
    normed: bool = True,
    norm_size: int = 2,
):
    """
    Calculate AttAttr (Hao et al.)
    If normed, attention matrix is scaled to [0,1]
    """
    if normed:
        A_out = min_max_scaler(A, norm_size)
    else:
        A_out = A
    return torch.mul(F, A_out) / m
