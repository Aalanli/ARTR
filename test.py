# %%
import json
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

# %%
data_root = 'dataset/val2017'
with open('dataset/annotations/instances_val2017.json', 'r') as f:
    annote = json.load(f)
print(annote.keys())

# %%
print(annote['annotations'][0].keys())
# %%

print(annote['annotations'][3].keys())

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
image_ids = {}
for i in annote['annotations']:
    if i['image_id'] in image_ids:
        image_ids[i['image_id']].append(i['bbox'])
    else:
        image_ids[i['image_id']] = [i['bbox']]

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
next(sampler)