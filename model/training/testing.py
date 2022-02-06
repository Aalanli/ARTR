# %%
import torch
from utils.misc import EasyDict
from utils.ops import unnormalize_im
from model.training.sweep import get_dataset
from model.resnet_parts import Backbone
import torchvision.transforms.functional as F

import torch.nn.functional as G
from matplotlib import pyplot as plt


def get_parameters():
    args = EasyDict()

    args.batch_size = 1
    args.mixed_precision = False
    args.fix_label = None

    args.transforms = False
    args.query_pool_prob = 0.4
    args.query_std = 2
    args.query_mu = 3
    args.query_stretch_limit = 0.7
    args.min_query_dim = 32
    args.max_queries = 10

    args.cost_class = 1
    args.cost_bbox = 5 * 4
    args.cost_giou = 2 * 4
    args.cost_eof = 0.05
    args.losses = ['boxes', 'labels', 'cardinality']

    args.lr = 1e-4
    args.weight_decay = 1e-4
    args.lr_drop = 200
    args.weight_dict = {'loss_giou': 2, 'loss_bbox': 5, 'loss_ce': 1}
    args.name = 'modelV3_test'
    return args

data = get_dataset(get_parameters())
# %%
im, _, _ = next(iter(data))
im = unnormalize_im(im[0], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
def show(im):
    plt.imshow(im.permute(1, 2, 0))
    plt.show()
print(im.shape)
show(im)
# %%
im1 = G.interpolate(im.unsqueeze(0), scale_factor=0.2, mode='nearest').squeeze()
im2 = torch.nn.AvgPool2d(5,5)(im.unsqueeze(0) ** 5).squeeze()
im2 = im2 ** (1/5)
show(im1)
show(im2)

# %%
model = Backbone("resnet50", True, True, False)
# %%
def get_activations(model: torch.nn.Module, names):
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output
        return hook
    for act in names:
        model.get_submodule(act).register_forward_hook(get_activation(act))
    return model, activations

act_names = ["layer1.1.in_proj." + str(i) for i in range(4, 8)]
model, activations = get_activations(model, act_names)
# %%
def transform(im, resize=None, rotate=None):
    if resize is not None:
        _, x, y = im.shape
        im = F.resize(im, [int(x * resize), int(y * resize)])
    if rotate is not None:
        im = F.rotate(im, rotate)
    return im

transform_im = lambda xs, im: [im] + list(map(lambda x: transform(im, x[0], x[1]), xs))
transforms = [(1.5, 20), (2.0, 0), (0.3, 0)]
im, _, _ = next(iter(data))
ims = transform_im(transforms, im[0])
features = list(map(lambda im: model(im.unsqueeze(0)), ims))

for i in map(str, range(0,3)):
    orig = features[0][i]
    print(orig.shape)
    for f in features[1:]:
        feat = f[i]
        print(feat.shape)
        if feat.shape[2:] > orig.shape[2:]:
            feat = G.interpolate(f[i], orig.shape[2:])
        else:
            orig = G.interpolate(orig, feat.shape[2:])
        diff = G.mse_loss(orig, feat)
        print(diff)

# %%
def visualize_attn_maps(im):
    im = [i.to('cuda') for i in im]
    y = model(im)
    for i in range(len(im)):
        plt.imshow(im[i].permute(1, 2, 0).cpu())
        plt.show()
        for j in range(3,4):
            for k in range(32):
                plt.imshow(y['attn_maps'][j][i][k].squeeze().detach().cpu())
                plt.show()

visualize_attn_maps(im)

# %%
print([n.shape for i, n in activations.items()])
# %%
for i, n in activations.items():
    n = n.detach().cpu().flatten(2).squeeze()
    plt.imshow(n)
    plt.show()

# %%
import torch

f = torch.nn.InstanceNorm1d(512)
#f = torch.nn.LayerNorm([512, 128])
print(f(torch.rand(4, 512, 128)).shape)