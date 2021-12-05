# %%
import typing as T

import torch
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models import resnet50, resnet34
from matplotlib import pyplot as plt

import data.dataset as data

root = "datasets/coco/"
proc = 'train'
query_process = data.GetQuery(data.query_transforms(), 3, 2, stretch_limit=0.4, min_size=32,
                            query_pool=root + proc + "2017_query_pool", prob_pool=0.1, max_queries=4)
dataset = data.CocoDetection(root + proc + '2017', root + f'annotations/instances_{proc}2017.json', data.img_transforms('train'), query_process)
# %%
model = resnet34(pretrained=True)

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
import torch.nn.functional as F
def feature_similarity(im, features):  # im.shape = [B, C, X1, X2], features.shape = [B, C, Y1, Y2]
    im = im.softmax(1)
    features = -features.softmax(1).log()
    features = F.adaptive_avg_pool2d(features, (1, 1))
    return F.conv2d(im, features)

im, qr, label = dataset[6]
plt.imshow(qr[0].permute(1, 2, 0))
plt.show()
plt.imshow(im.permute(1, 2, 0))
plt.show()

outputs = model(im.unsqueeze(0))
features = model(qr[0].unsqueeze(0))

feat_maps = [feature_similarity(i, f) for (k, i), (k1, f) in zip(outputs.items(), features.items())]
print([(i.max(), i.min()) for i in feat_maps])
for i in range(4):
    plt.imshow(feat_maps[i][0, 0].detach())
    plt.show()