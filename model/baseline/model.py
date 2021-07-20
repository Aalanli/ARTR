# %%
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from model.transformer import EncoderLayer, DecoderLayer, PositionEmbeddingSine
from model.resnet_parts import Backbone
from utils.ops import nested_tensor_from_tensor_list

class Transformer(nn.Module):
    def __init__(
        self, 
        d_model, 
        heads, 
        proj_forward, 
        enc_q_nlayers,
        enc_t_nlayers,
        dec_nlayers,
        activation=F.relu,
        dropout=0.1, 
        bias=None):
        super().__init__()
        self.d_model = d_model
        args = [d_model, heads, proj_forward, activation, dropout, bias]
        self.enc_q = nn.ModuleList([EncoderLayer(*args) for _ in range(enc_q_nlayers)])
        self.enc_t = nn.ModuleList([EncoderLayer(*args) for _ in range(enc_t_nlayers)])

        self.dec_kv = nn.ModuleList([DecoderLayer(*args) for _ in range(dec_nlayers)])
        self.dec_final = nn.ModuleList([DecoderLayer(*args) for _ in range(dec_nlayers)])
    
    def forward(self, im_query, im_target, query_embed, mask_q=None, mask_t=None, pos_q=None, pos_t=None):
        for layer in self.enc_q:
            im_query = layer(im_query, mask_q, pos_q)
        for layer in self.enc_t:
            im_target = layer(im_target, mask_t, pos_t)

        x = torch.zeros_like(query_embed)  # flow-through variable
        kv = torch.zeros_like(im_query)    # flow-through variable
        for kv_layer, out_layer in zip(self.dec_kv, self.dec_final):
            kv = kv_layer(kv, im_query, im_target, mask_t, pos_t)
            x = out_layer(x, query_embed, kv, mask_q, pos_q)
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
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ARTR(nn.Module):
    def __init__(self, backbone_args: List, pos_embed_args: List, transformer_args: List, num_queries: int):
        super().__init__()
        self.backbone = Backbone(*backbone_args)
        self.pos_embed = PositionEmbeddingSine(*pos_embed_args)
        self.transformer = Transformer(*transformer_args)
        d_model = self.transformer.d_model

        self.class_embed = nn.Linear(d_model, 2)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)
        self.query_embed = nn.Parameter(torch.Tensor(num_queries, d_model))
        torch.nn.init.normal_(self.query_embed)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, d_model, kernel_size=1)
    
    def forward(self, tar_im: List[torch.Tensor], query_im = List[List[torch.Tensor]]):
        """len(tar_im) = batch_size, len(query_im) = batch_size, len(query_im[i]) = # of query images for tar_im[i]"""
        tar_im, t_mask = nested_tensor_from_tensor_list(tar_im)
        t_features, t_mask = self.backbone(tar_im, t_mask)      # [batch, num_channels, x, y]
        t_features = self.input_proj(t_features)                # [batch, d_model, x, y]
        t_pos = self.pos_embed(t_mask)                          # [batch, d_model, x, y]
        t_features = t_features.flatten(2).transpose(-1, -2)    # [batch, x*y, d_model]
        t_pos = t_pos.flatten(2).transpose(-1, -2)              # [batch, x*y, d_model]
        t_mask = t_mask.flatten(1)                              # [batch, x*y]

        q_features = []
        q_mask = []
        for q in query_im:
            qf, qm = nested_tensor_from_tensor_list(q)
            q_features.append(qf)
            q_mask.append(qm)
        q_features, q_mask = nested_tensor_from_tensor_list(q_features, q_mask, exclude_mask_dim=-3)
        # [batch, n, 3, x1, y1], [batch, n, x1, y1]
        for b in range(q_features.shape[0]):
            q_features[b], q_mask[b] = self.backbone(q_features[b], q_mask[b])
            q_features[b] = self.input_proj(q_features[b])
            q_mask[b] = self.pos_embed(q_mask[b])
        # [batch, n, d_model, x1, x2], [batch, n, x1, x2]
        q_pos = torch.empty_like(q_features)
        for qp in q_pos:
            qp.copy_(self.pos_embed(qp))
        q_features = q_features.transpose(2, 1).flatten(2).transpose(-1, -2)  # [batch, n*x1*y1, d_model]
        q_mask = q_mask.flatten(1)                                            # [batch, n*x1*y1]
        q_pos = q_pos.transpose_(2, 1).flatten(2).transpose_(-1, -2)          # [batch, n*x1*y1, d_model]

        query_embed = self.query_embed.unsqueeze(1).repeat(1, q_features.shape[0], 1)

        out = self.transformer(q_features, t_features, query_embed, q_mask, t_mask, q_pos, t_pos)

        output_class = self.class_embed(out)
        output_bbox = self.bbox_embed(out).sigmoid()
        return {'pred_logits': output_class, 'pred_boxes': output_bbox[-1]}

