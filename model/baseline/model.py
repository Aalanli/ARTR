# %%
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from model.transformer import EncoderLayer, DecoderLayer, PositionEmbeddingSineMaskless
from model.resnet_parts import Backbone
from utils.misc import make_equal, make_equal1D


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
        key_padding_mask_q = mask_q[:, None, None, :].bool()
        key_padding_mask_t = mask_t[:, None, None, :].bool()
        for layer in self.enc_q:
            im_query = layer(im_query, key_padding_mask_q, pos_q)
        for layer in self.enc_t:
            im_target = layer(im_target, key_padding_mask_t, pos_t)

        attn_2d_mask = mask_q[:, :, None].logical_or(mask_t[:, None, :])
        attn_2d_mask.unsqueeze_(1)         # [batch, 1, L, D]
        x = torch.zeros_like(query_embed)  # flow-through variable
        kv = torch.zeros_like(im_query)    # flow-through variable
        for kv_layer, out_layer in zip(self.dec_kv, self.dec_final):
            kv = kv_layer(kv, im_query, im_target, attn_2d_mask, pos_t)
            x = out_layer(x, query_embed, kv, key_padding_mask_q, pos_q)
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
    def __init__(
        self, 
        backbone_args:    Dict = {'name': 'resnet50', 'train_backbone': True, 'return_interm_layers': False, 'dilation': False}, 
        pos_embed_args:   Dict = {'num_pos_feats': 128, 'temperature': 10000, 'normalize': False, 'scale': None}, 
        transformer_args: Dict = {'d_model': 256, 'heads': 8, 'proj_forward': 1024, 'enc_q_nlayers': 3,
                                  'enc_t_nlayers': 3, 'dec_nlayers': 3, 'activation': F.relu, 'dropout': 0.1, 'bias': None}, 
        num_queries:      int  = 50
        ):
        super().__init__()
        # default args
        backbone_args_ = {'name': 'resnet50', 'train_backbone': True, 'return_interm_layers': False, 'dilation': False}
        pos_embed_args_ = {'num_pos_feats': 128, 'temperature': 10000, 'normalize': False, 'scale': None}
        transformer_args_ = {'d_model': 256, 'heads': 8, 'proj_forward': 1024, 'enc_q_nlayers': 3,
                             'enc_t_nlayers': 3, 'dec_nlayers': 3, 'activation': F.relu, 'dropout': 0.1, 'bias': None}
        backbone_args_.update(backbone_args)
        pos_embed_args_.update(pos_embed_args)
        transformer_args_.update(transformer_args)
        self.config_ = [backbone_args_, pos_embed_args_, transformer_args_, num_queries]

        self.backbone = Backbone(**backbone_args_)
        self.pos_embed = PositionEmbeddingSineMaskless(**pos_embed_args_)
        self.transformer = Transformer(**transformer_args_)
        d_model = self.transformer.d_model

        self.class_embed = nn.Linear(d_model, 2)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)
        self.query_embed = nn.Parameter(torch.Tensor(num_queries, d_model))
        torch.nn.init.normal_(self.query_embed)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, self.transformer.d_model, kernel_size=1)
        
        self.input_proj = nn.Conv2d(self.backbone.num_channels, d_model, kernel_size=1)
    
    def make_equal_masks(self, ims: List[torch.Tensor]):
        """Accepts a list of flattened images and produces binary masks for attention"""
        shapes = [i.shape[-1] for i in ims]
        batch = len(ims)
        max_len = max(shapes)
        mask = torch.zeros(batch, max_len, dtype=ims[0].dtype, device=ims[0].device)
        for i, s in enumerate(shapes):
            mask[i, s:].fill_(1)
        return mask.bool()
    
    def make_pos_embed(self, x: List[torch.Tensor]):
        """Makes a 2D positional embedding from a 2D image"""
        # [embed_dim, x * y]
        return self.pos_embed(x).permute(2, 0, 1).flatten(1)

    def make_equal_ims(self, ims: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Unrolls resnet features and calculates masks and positional embed,
        vectorizes all.
        """
        pos_embed = []
        for i in range(len(ims)):
            ims[i] = self.backbone(ims[i].unsqueeze_(0))['0']
            ims[i] = self.input_proj(ims[i]).squeeze()
            pos_embed.append(self.make_pos_embed(ims[i]))
            ims[i] = ims[i].flatten(1)
        pos_embed = make_equal1D(pos_embed)  # no grad
        masks = self.make_equal_masks(ims)   # no grad
        ims = make_equal(*ims)               # with grad
        return ims, masks, pos_embed
    
    def make_equal_queries(self, qrs: List[List[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Same as make_equal_ims method, but unrolls multiple queries into
        one for each sample in a batch
        """
        pos_embed = []
        for j in range(len(qrs)):
            pos_embed_ = []
            for i in range(len(qrs[j])):
                qrs[j][i] = self.backbone(qrs[j][i].unsqueeze_(0))['0']
                qrs[j][i] = self.input_proj(qrs[j][i]).squeeze()
                pos_embed_.append(self.make_pos_embed(qrs[j][i]))
                qrs[j][i] = qrs[j][i].flatten(1)
            # [embed_dim, n * x * y]
            pos_embed.append(torch.cat(pos_embed_, dim=-1))
            qrs[j] = torch.cat(qrs[j], dim=-1)
        pos_embed = make_equal1D(pos_embed)
        masks = self.make_equal_masks(qrs)
        qrs = make_equal(*qrs)
        return qrs, masks, pos_embed
    
    def forward(self, tar_im: List[torch.Tensor], query_im: List[List[torch.Tensor]]):
        """
        len(tar_im) = batch_size, len(query_im) = batch_size, len(query_im[i]) = # of query images for tar_im[i]
        dist.shape = [batch_size]; a scaler difference between the bounding boxes and query images 
        """
        t_features, t_mask, t_pos = self.make_equal_ims(tar_im)
        q_features, q_mask, q_pos = self.make_equal_queries(query_im)
        t_features = t_features.transpose(-1, -2)
        q_features = q_features.transpose(-1, -2)
        t_pos.transpose_(-1, -2)
        q_pos.transpose_(-1, -2)

        batch_size = t_pos.shape[0]
        query_embed = self.query_embed.unsqueeze(0).repeat(batch_size, 1, 1)
        out = self.transformer(q_features, t_features, query_embed, q_mask, t_mask, q_pos, t_pos)
        output_class = self.class_embed(out)
        output_bbox = self.bbox_embed(out).sigmoid()
        return {'pred_logits': output_class, 'pred_boxes': output_bbox[-1]}
    
    @property
    def config(self):
        return self.config_


