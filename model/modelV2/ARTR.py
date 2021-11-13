# %%
from typing import List, Dict

import torch
from torch import Tensor
from torch import nn
from model.modelV2.backbone import DetectorBackbone
from model.modelV2.detector import Detector


class ARTR(nn.Module):
    def __init__(
        self,
        layer_maps=[[0, 1], [1, 0], [1, 0], [1, 0]],
        backbone_transformer_kwargs={'nhead': 8, 'dim_feedforward': 1024, 'dropout': 0.1, 'activation': 'relu', 'normalize_before': False},
        feat_maps_arch='resnet34',
        pretrained=True,
        n_queries=50,
        d_model=256,
        nhead=8,
        n_enc_layers=3,
        n_dec_layers=3,
        dim_feedforward=2048,
        dropout=0.1,
        activation='relu',
        normalize_before=False
    ):
        super().__init__()
        self.backbone = DetectorBackbone(layer_maps, feat_maps_arch, backbone_transformer_kwargs, pretrained=pretrained)
        self.detector = Detector(n_queries, d_model, nhead, n_enc_layers, n_dec_layers, dim_feedforward, dropout, activation, normalize_before)
    
    def forward(self, img: List[Tensor], qrs: List[List[Tensor]]) -> Dict[str, Tensor]:
        features, mask = self.backbone(img, qrs)
        return self.detector(features, mask)
