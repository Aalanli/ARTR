# %%
from typing import List, Tuple

import torch
from torch import nn
from torch import Tensor

from model.modelV2.feat_extractor import SpatialFeatureExtractor
from model.modelV2.feat_maps import QueryBackbone
from model.pos_emb import FixedPositionalEmbedding
from utils.ops import nested_tensor_from_tensor_list


class DetectorBackbone(nn.Module):
    """
    Resnet shape guarantees:
    input:
        imgs is a list of images of shape [3, x, y], len(imgs) = batch
        qrs is a list of a list of images of shape [3, x, y], len(qrs) = batch
    
    output:
        imgs is a tensor of shape [batch, channels, x1, y1]
        mask is a tensor of shape [batch, x1, y1] of type bool
    """
    def __init__(self, layer_maps, arch='resnet34', transformer_kwargs=None, dilation=None, pretrained=True):
        super().__init__()
        self.spatial_extractor = SpatialFeatureExtractor(layer_maps, transformer_kwargs, dilation)
        self.query_backbone = QueryBackbone(arch, pretrained)
        self.feature_dims = self.query_backbone.resnet.feature_dims
        self.pos_embs = nn.ModuleList([FixedPositionalEmbedding(dim) for dim in self.feature_dims])
    
    def forward(self, imgs: List[Tensor], qrs: List[List[Tensor]]) -> Tuple[Tensor, Tensor]:
        imgs, mask = nested_tensor_from_tensor_list(imgs)

        features = self.query_backbone(qrs)
        for i in range(len(features)): features[i] = features[i].transpose(-1, -2)
        pos_emb = [pos(x) for pos, x in zip(self.pos_embs, features)]
        imgs = self.spatial_extractor(imgs, feat_maps=features, pos=pos_emb)
        # mask.shape = [B, X1, Y1], imgs.shape = [B, C, X2, Y2]
        # interpolate mask to fit downsized image dim
        X2, Y2 = imgs.shape[-2:]
        mask = nn.functional.interpolate(mask.unsqueeze_(0).float(), size=(X2, Y2)).squeeze_().bool()
        return imgs, mask

