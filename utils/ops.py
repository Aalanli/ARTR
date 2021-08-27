# %%
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torchvision.ops.boxes import box_area
import numpy as np


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def xyxy_to_cwh(bbox):
    bbox = np.asarray(bbox)
    bbox = np.reshape(bbox, [-1, 4])
    for i, box in enumerate(bbox):
        bbox[i] = np.array([box[0], box[1], (box[2] - box[0]), (box[3] - box[1])])
    return bbox


def unnormalize_box(w, h, box):
    box = box * torch.tensor([w, h, w, h], dtype=torch.float32)
    return box_cxcywh_to_xyxy(box)


def unnormalize_im(tensor: torch.Tensor, mean: List[float] = [0.485, 0.456, 0.406], 
                std: List[float] = [0.229, 0.224, 0.225], inplace=False):
    """inverse of F.normalize"""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('Input tensor should be a torch tensor. Got {}.'.format(type(tensor)))

    if tensor.ndim < 3:
        raise ValueError('Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = '
                         '{}.'.format(tensor.size()))

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list, mask_list=None, exclude_mask_dim=-3) -> Tuple[torch.Tensor, torch.Tensor]:
    # TODO make this more general
    max_size = max_by_axis([list(img.shape) for img in tensor_list])
    # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
    batch_shape = [len(tensor_list)] + max_size
    mask_shape = batch_shape[:exclude_mask_dim] + batch_shape[exclude_mask_dim + 1:]
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device
    tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
    mask = torch.ones(mask_shape, dtype=torch.bool, device=device)
    
    i = 0
    if tensor_list[0].ndim == 3:
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
            if mask_list is None:
                m[:img.shape[1], :img.shape[2]] = False
            else:
                m[:img.shape[1], :img.shape[2]].copy_(mask_list[i])
            i += 1
    elif tensor_list[0].ndim == 4:
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2], :img.shape[3]].copy_(img)
            if mask_list is None:
                m[:img.shape[1], :img.shape[2], :img.shape[3]] = False
            else:
                m[:img.shape[1], :img.shape[2], img.shape[3]].copy_(mask_list[i])
            i += 1
    else:
        raise ValueError('not supported')
    return tensor, mask


def l2_query_distance_(image: torch.Tensor, query, bbox: torch.Tensor):
    dist = torch.tensor(0.0)
    h, w = image.shape[-2:]
    bbox = unnormalize_box(w, h, bbox).int()
    boxes = [image[:, box[1]:box[3], box[0]:box[2]] for box in bbox]
    for b in boxes:
        size = list(b.shape)[1:]  # [x, y]
        q_ = list(map(lambda x: F.interpolate(x.unsqueeze(0), size).squeeze(), query))
        losses = torch.tensor([F.mse_loss(b, y) for y in q_])
        dist.add_(losses.min())
    return dist


def l2_query_distance_vectorized(image: torch.Tensor, query: torch.Tensor, q_mask: torch.Tensor, bbox: torch.Tensor):
    dist = 0.0
    h, w = image.shape[-2:]
    bbox = unnormalize_box(w, h, bbox).int()
    boxes = [image[:, box[1]:box[3], box[0]:box[2]] for box in bbox]
    for b in boxes:
        size = list(b.shape)[1:]  # [x, y]
        b = b.unsqueeze(0)  # [1, 3, x, y]
        q = F.interpolate(query, size)  # [n, 3, x, y]
        m = ~F.interpolate(q_mask.unsqueeze(1), size)  # [n, 1, x, y]
        mse = ((b - q) ** 2 * m).mean(1).min()
        dist += float(mse)
        
    return dist


def give_grad_copy(model):
    grads = []
    for i in model.parameters():
        grads.append(i.grad.clone().detach())
    return grads


def split_batches(model, x, y, loss_func, splits, fp16=True):
    for x1, y1, in zip(x.chunk(splits), y.chunk(splits)):
        with torch.cuda.amp.autocast(enabled=fp16):
            pred = model(x1)
            loss = loss_func(y1, pred)
            loss = loss.div(splits)
        loss.backward()


def list_error(list1, list2):
    error = 0
    for g, g1 in zip(list1, list2):
        partial_error = ((g - g1)**2).mean()
        error += float(partial_error)
    return error

