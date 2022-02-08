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


class EasyDict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(f"{name} not in dictionary")
    def __setattr__(self, name: str, value) -> None:
        self[name] = value 
    def search_common_naming(self, name, seperator='_'):
        name = name + seperator
        return {k.replace(name, ''): v for k, v in self.items() if name in k}
    def get_copy(self):
        return EasyDict(self.copy())


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


import json
import os

import numpy as np

import torch
import torchvision.transforms.functional as F

from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm

import data.transforms as T
from utils.similarity import metric_functions


def format_json(json_file):
    """Formats coco json by grouping features by their image_ids"""
    with open(json_file, 'r') as f:
        annote = json.load(f)
    
    image_ids = {}  # key = image_ids, item = nested dict classes and their bounding boxes
    for i in annote['annotations']:
        if i['image_id'] in image_ids:
            if i['category_id'] in image_ids[i['image_id']]:
                image_ids[i['image_id']][i['category_id']].append(i['bbox'])
            else:
                image_ids[i['image_id']].update({i['category_id']: [i['bbox']]})
        else:
            image_ids[i['image_id']] = {i['category_id']: [i['bbox']]}
    return image_ids


def id_to_file(id: str, image_dir) -> str:
    """image_id to file name"""
    id = str(id)
    file = '0' * (12 - len(id)) + id + '.jpg'
    return image_dir + '/' + file


def get_im_size(image, format='channel first'):
    t = type(image)
    if t == Image.Image:
        w, h = image.size
    elif t == torch.Tensor or t == np.ndarray:
        if format == 'channel first':
            h, w = image.shape[-2:]
    return h, w


def fetch_query(image, bbox, mode='xywh'):
    """
    Fetches query images, alias to cropping
    Args:
        image: str or PIL.Image
        bbox: an array
        mode: the format of the bbox
            xyxy: top left corner x, y and bottom right corner x, y
            xywh: top left corner x, y and bottom right corner (x + w), (y + h)
    modes:
        xywh, xyxy
    """
    if type(image) == str:
        image = Image.open(image, mode='RGB')
    
    h, w = get_im_size(image)
    if mode == 'xywh':
        x, y, w1, h1 = bbox
        x1, y1 = x + w1, y + h1
    elif mode == 'xyxy':
        x, y, x1, y1 = bbox
    # box checking
    # zero width
    if x1 < x or y1 < y:
        return None
    # point is outside the image
    if x < 0 or x1 > w or y < 0 or y1 > h:
        return None
    return F.crop(image, y, x, h1, w1)


def make_query_pool(image_dir, json_file, name):
    """constructs the instance query pool, saves a counter dictionary at location image_dir"""
    if not os.path.exists(name):
        os.mkdir(name)
    image_ids = format_json(json_file)
    classes_counter = {}
    for id in tqdm(image_ids):
        file = id_to_file(id, image_dir)
        img = Image.open(file)
        for class_id in image_ids[id]:
            path = name + '/' + str(class_id)
            if not os.path.exists(path):
                os.mkdir(path)
                classes_counter[int(class_id)] = 0
            for bbox in image_ids[id][class_id]:
                query: Image.Image = fetch_query(img, bbox)
                if query is None:
                    print('broken box')
                    continue
                query.save(path + '/' + str(classes_counter[class_id]) + '.jpg')
                classes_counter[class_id] += 1
    with open(name + '/instances.json', 'w') as f:
        json.dump(classes_counter, f)
    print('done')


def remap_image_names(json_file, im_dir, save_name):
    """Renames the coco dataset images to contiguous integer names"""
    im_ids = format_json(json_file)
    new_ids = {}
    for i, name in tqdm(enumerate(im_ids)):
        file_name = id_to_file(name, im_dir)
        os.rename(file_name, os.path.join(im_dir, str(i) + '.jpg'))
        new_ids[i] = im_ids[name]
    with open(os.path.dirname(json_file) + f'/{save_name}', 'w') as f:
        json.dump(new_ids, f)


def visualize_output(image: torch.Tensor, queries: List[torch.Tensor], bbox: torch.Tensor) -> None:
    h, w = get_im_size(image)
    image = T.unnormalize_im(image)
    plt.imshow(image.permute(1, 2, 0))
    ax = plt.gca()
    for box in bbox:
        box = T.unnormalize_box(w, h, box)
        rect = Rectangle(box[:2], box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()
    for q in queries:
        q = T.unnormalize_im(q)
        plt.imshow(q.permute(1, 2, 0))
        plt.show()

def visualize_model_output(image, queries, true_box, pred_box, probs=None, prob_thres=0.7):
    h, w = get_im_size(image)
    image = T.unnormalize_im(image)
    plt.imshow(image.permute(1, 2, 0))
    ax = plt.gca()
    for box in true_box:
        box = T.unnormalize_box(w, h, box)
        rect = Rectangle(box[:2], box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
    for i, box in enumerate(pred_box):
        if probs is None or (probs is not None and probs[i] > prob_thres):
            box = T.unnormalize_box(w, h, box)
            rect = Rectangle(box[:2], box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    plt.show()
    for q in queries:
        q = T.unnormalize_im(q)
        plt.imshow(q.permute(1, 2, 0))
        plt.show()


def compute_image_similarity(x, y, alpha=0.5):
    """score between 0 and 1; 1 is most similar"""
    return metric_functions['fsim'](x, y) * alpha + metric_functions['ssim'](x, y) * (1 - alpha)


def compute_highest_similarity(queries: List[Image.Image], im: Image.Image, bboxes: torch.Tensor, alpha=0.5, mode='xyxy'):
    score = 0
    if bboxes.shape[0] == 0: return score
    im_bbox = [fetch_query(im, i, mode=mode) for i in bboxes]
    for q in queries:
        scores = []
        for b in im_bbox:
            b = b.resize(q.size)
            scores.append(compute_image_similarity(np.asarray(q), np.asarray(b), alpha=alpha))
        score += max(scores)
    return score


def alert_nan(x: torch.Tensor, message=""):
    if torch.isnan(x).any():
        print(message + " is nan.")
