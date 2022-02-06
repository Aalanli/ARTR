import math

import torch
import torch.nn as nn


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask):
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t  # [batch, y, x, num_pos_feats]
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3)
        return pos


class PositionEmbeddingSineMaskless(PositionEmbeddingSine):
    def forward(self, h, w, dtype, device):
        y_embed = torch.arange(0, h, 1, dtype=dtype, device=device).unsqueeze_(1).tile(1, w)
        x_embed = torch.arange(0, w, 1, dtype=dtype, device=device).unsqueeze_(0).tile(h, 1)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=dtype, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t  # [batch, y, x, num_pos_feats]
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2)
        return pos


class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.inv_freq = 1. / (10000 ** (torch.arange(0, self.dim, 2).float() / self.dim))
        # the amount to increment after exceeding current seq_len
        self.seg_div = 256

        self.cur_seq_len = 256
        self.emb = None
    
    def get_seg(self, new_len):
        return math.ceil(new_len / self.seg_div) * self.seg_div
    
    def calculate(self, length):
        position = torch.arange(0, length, dtype=torch.float)
        sinusoid_inp = torch.einsum("i,j->ij", position, self.inv_freq)
        return torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
    
    def forward(self, x):
        seq_len = x.shape[1]
        if self.emb is None or seq_len > self.cur_seq_len or x.device != self.emb.device:
            self.cur_seq_len = self.get_seg(seq_len)
            self.emb = self.calculate(self.cur_seq_len).to(x)
            self.emb.unsqueeze_(0)
        return self.emb[:, :seq_len, :]
