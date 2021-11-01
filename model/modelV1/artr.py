from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ARTR(nn.Module):
    def __init__(self, backbone_process: nn.Module, transformer_process: nn.Module, n_queries: int):
        super().__init__()
        self.backbone = backbone_process
        self.transformer = transformer_process
        d_model = self.transformer.d_model

        self.class_embed = nn.Linear(d_model, 92)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)
        self.query_embed = nn.Parameter(torch.Tensor(n_queries, d_model))
        torch.nn.init.normal_(self.query_embed)        

    def forward(self, tar_im: List[torch.Tensor], query_im: List[List[torch.Tensor]]):
        """
        len(tar_im) = batch_size, len(query_im) = batch_size, len(query_im[i]) = # of query images for tar_im[i]
        dist.shape = [batch_size]; a scaler difference between the bounding boxes and query images 
        """
        # backbone portion
        (t_features, t_mask, t_pos), (q_features, q_mask, q_pos) = self.backbone(tar_im, query_im)
        

        # transformer portion
        batch_size = t_pos.shape[0]
        query_embed = self.query_embed.unsqueeze(0).repeat(batch_size, 1, 1)
        out = self.transformer(q_features, t_features, query_embed, q_mask, t_mask, q_pos, t_pos)
        output_class = self.class_embed(out)
        output_bbox = self.bbox_embed(out).sigmoid()
        return {'pred_logits': output_class, 'pred_boxes': output_bbox}
    
    @property
    def config(self):
        return None