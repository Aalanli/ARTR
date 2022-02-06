# %%
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.ops import make_equal_3D, nested_tensor_from_tensor_list
from utils.misc import alert_nan
from model.resnet_parts import Backbone

from matplotlib import pyplot as plt

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def ConvMixer(dim, depth, out_size, kernel_size=9, patch_size=7, input_dim=1):
    return nn.Sequential(
        nn.Conv2d(input_dim, dim, kernel_size=1, stride=1),
        nn.Conv2d(dim, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((out_size, out_size))
    )

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )

def MLPMixer(in_features, out_features, dim, depth, expansion_factor = 4, dropout = 0.):
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        nn.Conv1d(in_features, out_features, 1),
        nn.Linear(dim, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(out_features, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim)
    )

class AttnMap(nn.Module):
    def __init__(self, channels, out_channels=1, kernel_size=1):
        super().__init__()
        self.feats = nn.Parameter(torch.empty(out_channels, channels, kernel_size, kernel_size))
        nn.init.normal_(self.feats)
    
    def norm(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = [B, C, X1, X2]
        flat = x.view(x.shape[0], x.shape[1], -1)
        min = flat.min(-1).values[:, :, None, None]
        max = flat.max(-1).values[:, :, None, None]
        x = (x - min) / (max - min + 1e-5)
        if torch.isnan(x).any():
            print(flat.min(), flat.max())
        return x
    
    def forward(self, x):
        x, mask = x
        # x.shape = [B, N, X1, X2]
        #alert_nan(x, "attnMap before")
        #x = torch.log_softmax(x, 1)
        #alert_nan(x, "attnMap after")
        #feats = torch.softmax(self.feats, 1)
        #alert_nan(feats, "feats")
        attn_map = F.conv2d(x, self.feats, padding='same')
        alert_nan(attn_map, "attn_map")
        return self.norm(attn_map).masked_fill(mask[:, None, :, :], 0), attn_map


class ExtractCoord(nn.Module):
    def __init__(self, dim, depth, out_size, kernel_size, patch_size, input_dim=1):
        super().__init__()
        self.in_proj = ConvMixer(dim, depth, out_size, kernel_size, patch_size, input_dim)
    
    def forward(self, x):
        x = self.in_proj(x).flatten(2).transpose(-1, -2)
        return x


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class Detector(nn.Module):
    def __init__(self, 
        dims,                 # dimensions of conv_mixer for various levels of feature maps
        final_dim,            # final aggregate dim for all levels
        depths,               # depths of conv mixer for different levels
        out_sizes,            # reduction sizes for different conv mixer levels
        kernel_sizes,         # kernel sizes for conv mixer levels
        patch_sizes,          # conv mixer levels
        mlp_depth,            # mlp mixer depth
        queries=50,           # final set prediction numbers
        attn_out_channels=1,  # attn map reduction
        attn_kernel_size=1,   # attn kernel sizes
        classes=1):  
        super().__init__()
        self.final_dim = final_dim
        self.backbone = Backbone("resnet50", True, True, True)

        dims = self.broadcast_arg(dims)
        depths = self.broadcast_arg(depths)
        out_sizes = self.broadcast_arg(out_sizes)
        patch_sizes = self.broadcast_arg(patch_sizes)
        k1 = self.broadcast_arg(attn_kernel_size)
        k2 = self.broadcast_arg(kernel_sizes)
        attn_ch = self.broadcast_arg(attn_out_channels)

        self.layer1 = self.make_layer(256, attn_ch[0], k1[0], k2[0], dims[0], depths[0], out_sizes[0], patch_sizes[0])
        self.layer2 = self.make_layer(512, attn_ch[1], k1[1], k2[1], dims[1], depths[1], out_sizes[1], patch_sizes[1])
        self.layer3 = self.make_layer(1024, attn_ch[2], k1[2], k2[2], dims[2], depths[2], out_sizes[2], patch_sizes[2])
        self.layer4 = self.make_layer(2048, attn_ch[3], k1[3], k2[3], dims[3], depths[3], out_sizes[3], patch_sizes[3])

        cat_dim = sum([i**2 for i in out_sizes])
        self.mixer = MLPMixer(cat_dim, queries, final_dim, mlp_depth)
        self.bbox_embed = MLP(final_dim, final_dim, 4, 3)
        self.class_embed = nn.Linear(final_dim, classes + 1)
    
    def make_layer(self, feat_in, feat_out, k1, k2, dim, depth, out_size, patch_size):
        return nn.ModuleList([
            AttnMap(feat_in, feat_out, k1),
            ExtractCoord(dim, depth, out_size, k2, patch_size, feat_out),
            nn.Linear(dim, self.final_dim)
        ])

    def broadcast_arg(self, x):
        if not isinstance(x, list):
            return [x for _ in range(4)]
        return x
    
    def execute_layer(self, layer, x):
        x, attn = layer[0](x)
        for l in layer[1:]:
            x = l(x)
        return x, attn
    
    def forward(self, im, *args, **kwargs):
        if isinstance(im, list):
            im, mask = nested_tensor_from_tensor_list(im)
        inter = self.backbone(im, mask)
        x0, attn0 = self.execute_layer(self.layer1, inter["0"])
        alert_nan(x0, "x0")
        x1, attn1 = self.execute_layer(self.layer2, inter["1"])
        alert_nan(x1, "x1")
        x2, attn2 = self.execute_layer(self.layer3, inter["2"])
        alert_nan(x2, "x2")
        x3, attn3 = self.execute_layer(self.layer4, inter["3"])
        alert_nan(x3, "x3")
        alert_nan(attn3, "attn3")
        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.mixer(x)
        classes = self.class_embed(x)
        bboxes = self.bbox_embed(x).sigmoid()
        assert not torch.isnan(classes).any(), "classes has nan"
        assert not torch.isnan(bboxes).any(), "boxes has nan"
        return {"pred_logits": classes, "pred_boxes": bboxes, "attn_maps": [attn0, attn1, attn2, attn3]}


