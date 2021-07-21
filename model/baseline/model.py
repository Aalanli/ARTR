# %%
from typing import List

import torch
from torch._C import dtype
import torch.nn.functional as F
from torch import nn

from model.transformer import EncoderLayer, DecoderLayer, PositionEmbeddingSine
from model.resnet_parts import Backbone
from utils.ops import nested_tensor_from_tensor_list, max_by_axis

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
        self.query_proj = nn.Parameter(torch.Tensor(num_queries, num_queries))
        torch.nn.init.normal_(self.query_proj)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, d_model, kernel_size=1)
    
    def forward(self, tar_im: List[torch.Tensor], query_im: List[List[torch.Tensor]], dist: torch.Tensor) -> torch.Tensor:
        """
        len(tar_im) = batch_size, len(query_im) = batch_size, len(query_im[i]) = # of query images for tar_im[i]
        dist.shape = [batch_size]; a scaler difference between the bounding boxes and query images 
        """
        tar_im, t_mask = nested_tensor_from_tensor_list(tar_im)
        t_features, t_mask = self.backbone(tar_im, t_mask)['0']   # [batch, num_channels, x, y]
        t_features = self.input_proj(t_features)                  # [batch, d_model, x, y]
        t_pos = self.pos_embed(t_mask)                            # [batch, x, y, d_model]
        t_features = t_features.flatten(2).transpose(-1, -2)      # [batch, x*y, d_model]
        t_pos = t_pos.reshape(list(t_features.shape))             # [batch, x*y, d_model]
        t_mask = t_mask.flatten(1)                                # [batch, x*y]
        
        q_features = []
        q_mask = []
        q_pos = []
        for q in query_im:
            # treat each query as a batch
            qf, qm = nested_tensor_from_tensor_list(q)
            qf, qm = self.backbone(qf, qm)['0']                   # [n, 3, x1, y1], [n, x1, y1]
            qf = self.input_proj(qf).transpose(1, 0).flatten(1)   # [d_model, n*x1*y1]
            qpos = self.pos_embed(qm).transpose(-1, 0).flatten(1) # [d_model, n*x1*y1]
            qm = qm.flatten(0)                                    # [n*x1*y1]
            q_features.append(qf)
            q_mask.append(qm)
            q_pos.append(qpos)
        max_len = max([i.shape[0] for i in q_mask])
        for b in range(len(q_mask)):
            # pad across aggregated samples
            # pad with zeros
            q_features[b] = F.pad(q_features[b], (0, max_len - q_features[b].shape[-1]))
            q_pos[b] = F.pad(q_pos[b], (0, max_len - q_pos[b].shape[-1]))
            # pad with ones
            q_mask[b] = torch.cat([q_mask[b], torch.ones(max_len - q_mask[b].shape[-1], dtype=torch.bool, device=q_mask[b].device)])
        
        q_features, q_mask, q_pos = map(lambda x: torch.stack(x), (q_features, q_mask, q_pos))

        q_features = q_features.transpose(-1, -2)                 # [batch, n*x1*y1, d_model]
        q_pos = q_pos.transpose(-1, -2)                           # [batch, n*x1*y1, d_model]

        # transform the queries by the difference scalar
        query_proj = (self.query_proj.unsqueeze(-1) * (dist + 1e-6)).transpose(-1, 0)  # [batch, num_queries, num_queries]
        query_embed = query_proj @ self.query_embed                                    # [batch, num_queries, d_model]

        print(q_features.shape, t_features.shape, q_pos.shape, t_pos.shape)
        out = self.transformer(q_features, t_features, query_embed, q_mask, t_mask, q_pos, t_pos)

        output_class = self.class_embed(out)
        output_bbox = self.bbox_embed(out).sigmoid()
        return {'pred_logits': output_class, 'pred_boxes': output_bbox[-1]}
