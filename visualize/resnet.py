# %%
import typing as T

import torch
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models import resnet50
from matplotlib import pyplot as plt

import data.dataset as data

root = "datasets/coco/"
proc = 'train'
dataset = data.CocoDetection(root + proc + '2017', root + f'annotations/instances_{proc}2017.json', data.img_transforms('train'), None)

# %%
model = resnet50(pretrained=True)

def intercede(model):
    return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
    body = IntermediateLayerGetter(model, return_layers=return_layers)
    return body

model = intercede(model)

def visualize(ouputs: T.Dict[str, torch.Tensor], dim: int, layers=None):
    for n, t in ouputs.items():
        if layers is not None and n not in layers:
            continue
        print(n)
        # normalize
        t = t.detach()
        t = t.squeeze()[dim].squeeze()
        t = t.cpu()
        plt.imshow(t)
        plt.show()

# %%
im = dataset[0][0]
outputs = model(im.unsqueeze(0))
plt.imshow(im.permute(1, 2, 0))
plt.show()
# %%
for i in range(32, 128):
    visualize(outputs, i, ["2"])

# %%
