# %%
import json
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import torchvision.transforms.functional as F


# %%
data_root = 'datasets/coco/val2017'
with open('datasets/coco/annotations/instances_val2017.json', 'r') as f:
    annote = json.load(f)
print(annote.keys())

# %%
def id_to_file(id: str, dir: str) -> str:
    id = str(id)
    file = '0' * (12 - len(id)) + id + '.jpg'
    return dir + '/' + file


def plot_box(image: np.ndarray, bbox: List) -> None:
    bbox = np.array(bbox)
    plt.imshow(image)
    ax = plt.gca()
    if len(bbox.shape) == 1:
        rect = Rectangle(bbox[:2], bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    else:
        for box in bbox:
            rect = Rectangle(box[:2], box[2], box[3], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    plt.show()

# %%
img = Image.fromarray(np.random.uniform(0.0, 1.0, [128, 128, 3]), mode='RGB')
img = F.crop(img, 64, 60, 10, 90)
plt.imshow(img)
plt.show()

# %%
image_ids = {}
for i in annote['annotations']:
    if i['image_id'] in image_ids:
        if i['category_id'] in image_ids[i['image_id']]:
            image_ids[i['image_id']][i['category_id']].append(i['bbox'])
        else:
            image_ids[i['image_id']].update({i['category_id']: [i['bbox']]})
    else:
        image_ids[i['image_id']] = {i['category_id']: [i['bbox']]}

# %%
for k in image_ids:
    for id, bbox in image_ids[k].items():
        img = Image.open(id_to_file(k, data_root))
        plot_box(img, bbox)
        bbox = bbox[0]
        print(bbox)
        img = F.crop(img, bbox[1], bbox[0], bbox[3], bbox[2])
        plt.imshow(img)
        plt.show()
        break
    break
# %%
def sample_dist(images_dict):
    for id in images_dict:
        file = id_to_file(id, data_root)
        arr = np.asarray(Image.open(file))
        plot_box(arr, images_dict[id])
        yield None

# %%
sampler = sample_dist(image_ids)

# %%
import torch

a = torch.tensor([[1, 2, 3], [2, -1, 2], [4, 2, 213]]).type(torch.float32)
print(torch.nn.functional.softmax(a, dim=-1))
