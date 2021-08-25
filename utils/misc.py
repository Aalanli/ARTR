# %%
import math
from typing import List

import torch
from torch.autograd import Function


def gen_test_data(high, low, samples, device='cpu'):
    import random
    dimensions = [(random.randint(low, high), random.randint(low, high)) for _ in range(samples)]
    return [torch.randn(3, y, x, device=device) for x, y in dimensions]


def calculate_param_size(model):
    params = 0
    for i in model.parameters():
        params += math.prod(list(i.shape))
    return params


@torch.no_grad()
def make_equal1D(tensors: List[torch.Tensor]):
    shapes       = [t.shape[-1] for t in tensors]
    max_len      = max(shapes)
    batch        = len(shapes)
    num_channels = tensors[0].shape[0]
    out = torch.zeros([batch, num_channels, max_len], device=tensors[0].device, dtype=tensors[0].dtype)
    for i in range(batch):
        out[i, :, :shapes[i]].copy_(tensors[i])
    return out


class MakeEqual(Function):
    """
    Efficient implementation of pad and concat, equalizes inputs and passes gradients.
    Receives lists of tensors of shape [num_channels, n1] and outputs [batch, num_channels, n]
    """
    @staticmethod
    def forward(ctx, *tensors: torch.Tensor):
        shapes       = [t.shape[-1] for t in tensors]
        max_len      = max(shapes)
        batch        = len(shapes)
        num_channels = tensors[0].shape[0]
        out = torch.zeros([batch, num_channels, max_len], device=tensors[0].device, dtype=tensors[0].dtype)
        for i in range(batch):
            out[i, :, :shapes[i]].copy_(tensors[i])        
        ctx.shapes = shapes
        return out
    
    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        shapes = ctx.shapes
        batch = len(shapes)
        tensors = [t[0, :, :shapes[i]] for i, t in enumerate(grad.chunk(batch, dim=0))]
        return tuple(tensors)

make_equal = MakeEqual.apply



