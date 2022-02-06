# %%
from functools import partial
from math import sqrt, ceil
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.pos_emb import PositionEmbeddingSineMaskless
from model.modelV4.backbone import ConvNeXt, ConvNeXtIsotropic, convnext_isotropic_small, convnext_small
from utils.misc import calculate_param_size
from utils.ops import make_equal_3D


def nan_guard(x, layer=''):
    if torch.isnan(x).any():
        print(layer, "has nan")
    
def split_im(im, splits):
    b, c, y, x = im.shape
    y_k, x_k = ceil(y / splits), ceil(x / splits)
    y_pad, x_pad = (y_k * splits - y), (x_k * splits - x)
    im = F.pad(im, (0, x_pad, 0, y_pad))
    im = im.reshape(b, c, -1, y_k, splits, x_k).permute(0, 2, 4, 3, 5, 1).reshape(b, -1, y_k * x_k, c)
    return im

class AttnPost(nn.Module):
    def __init__(self, dim, expansion=4, dropout=0.1) -> None:
        super().__init__()
        self.norm2 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * expansion)
        self.linear2 = nn.Linear(dim * expansion, dim)
        self.activation = nn.GELU()
        self.drop = nn.Dropout(dropout)
    
    def forward(self, v):
        v = self.linear2(self.drop(self.activation(self.linear1(v)))) + v
        v = self.norm2(v)
        return v

class SimplifiedAttn(nn.Module):
    def __init__(self, dim, dropout=0.1) -> None:
        super().__init__()
        self.dim_out = dim
        self.norm1 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.pos_emb = PositionEmbeddingSineMaskless(dim // 2)
    
    def forward(self, q, k, v, pos_embed=None):
        if isinstance(pos_embed, tuple):
            pos_embed = self.pos_emb(*pos_embed, q.dtype(), q.device())
        if isinstance(pos_embed, torch.Tensor):
            k += pos_embed
        w = q @ k.transpose(-1, -2) / sqrt(self.dim_out)
        w = self.drop(w.softmax(-1))
        v = w @ v

        v = self.drop(v) + q
        v = self.norm1(v)
        return v

class AttnEmbed(nn.Module):
    def __init__(self, splits, x_dim, dim_in, dim_out, dropout=0.1) -> None:
        super().__init__()
        self.splits = splits
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.x_dim = x_dim
        self.pos_emb = PositionEmbeddingSineMaskless(dim_in // 2)

        self.qw = nn.Linear(dim_out, dim_out)
        self.kw = nn.Linear(dim_in, dim_out)
        self.vw = nn.Linear(dim_in, dim_out)

        self.attn1 = SimplifiedAttn(dim_out, dropout=dropout)
        self.embW = nn.Linear(dim_out, dim_out * 3)
        self.attn2 = SimplifiedAttn(dim_out, dropout=dropout)
        self.attn2_post = AttnPost(dim_out, dropout=dropout)

        self.res = nn.AdaptiveAvgPool2d((x_dim, x_dim))
        self.res_upsample = nn.Linear(dim_in, dim_out)
    
    def attn(self, q, x, pos):
        q = self.qw(q)
        k = self.kw(x + pos)
        v = self.vw(x)

        v = self.attn1(q, k, v)
        return v

    def forward(self, im, emb):
        b, c, y, x = im.shape
        y_k, x_k = ceil(y / self.splits), ceil(x / self.splits)
        im = split_im(im, self.splits)
        pos_emb = self.pos_emb(y_k, x_k, im.dtype, im.device).flatten(0, 1)[None, None]

        res_im = self.res(im.reshape(-1, y_k, x_k, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).reshape(b, -1, self.x_dim ** 2, c)
        res_im = self.res_upsample(res_im)
        
        if emb.dim() < 4:
            emb = emb.unsqueeze(1)
        embW = self.embW(emb).chunk(3, -1)
        emb = self.attn1(*embW) + emb

        attn = self.attn(emb, im, pos_emb) + res_im
        attn = self.attn2_post(attn)
        attn = attn.flatten(1, 2)
        return attn


class FeatureExtractor(nn.Module):
    """Performs decoder attention on input images with queries"""
    def __init__(self, splits, in_embed_heads, dim, dropout=0.1) -> None:
        super().__init__()
        self.embed_head_proj = nn.Conv2d(in_embed_heads, splits, 1)
        self.embed_head_norm = nn.LayerNorm(dim)

        self.attn1 = SimplifiedAttn(dim, dropout)



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


class QueryExtractor(nn.Module):
    def __init__(self, x_dim, dims, depths, min_query_size=None, max_query_size=None) -> None:
        super().__init__()
        self.min_size = min_query_size
        self.max_size = max_query_size
        self.x_dim = x_dim
        self.conv = ConvNeXt(3, depths, dims)
        self.down_sample = nn.AdaptiveAvgPool2d((x_dim, x_dim))

        chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear
        self.out_proc = nn.Identity() #nn.Sequential(
            #PreNormResidual(dims[-1], FeedForward(x_dim ** 2, 4, 0., chan_first)),
            #PreNormResidual(dims[-1], FeedForward(dims[-1], 4, 0., chan_last)))
    
    def forward(self, im):
        y, x = im.shape[2:]
        if self.min_size is not None and min(y, x) < self.min_size:
            y = max(y, self.min_size)
            x = max(x, self.min_size)
            im = F.interpolate(im, size=(y, x))
            #y, x = im.shape[2:]
        if self.max_size is not None and max(y, x) > self.max_size:
            y = min(y, self.max_size)
            x = min(x, self.max_size)
            im = F.interpolate(im, size=(y, x))
            #y, x = im.shape[2:]
        im = self.conv(im)
        h0 = self.down_sample(im).flatten(2, 3).transpose(-1, -2)
        h1 = self.out_proc(h0).reshape(im.shape[0], self.x_dim ** 2, -1)
        return h1


class ARTR(nn.Module):
    def __init__(self, n_pred, x_dim, splits, im_depth, im_dim, qr_depths, qr_dims, mlp_depth, mlp_feats,
            min_qr_size=None, max_qr_size=None) -> None:
        super().__init__()
        self.im_conv = ConvNeXtIsotropic(3, im_depth, im_dim) # convnext_isotropic_small(True)
        self.qr_conv = QueryExtractor(x_dim, qr_dims, qr_depths, min_qr_size, max_qr_size)
        #self.qr_conv.conv = convnext_small(True)
        self.attn = AttnEmbed(splits, x_dim, im_dim, qr_dims[-1]) # splits, x_dim, 384, 768)
        out_features = x_dim ** 2 * splits ** 2
        self.mlp_proj = nn.Conv1d(out_features, mlp_feats, 1)
        self.mlp_res = nn.AdaptiveAvgPool1d(mlp_feats)
        #qr_dims[-1] = 768
        self.mlp_proj_norm = nn.LayerNorm(qr_dims[-1])
        self.mlp = MLPMixer(mlp_feats, mlp_feats, qr_dims[-1], mlp_depth)
        self.out_proj = nn.Conv1d(mlp_feats, n_pred, 1)
        self.out_norm = nn.LayerNorm(qr_dims[-1])
        self.box_proj = nn.Linear(qr_dims[-1], 4)
        self.class_proj = nn.Linear(qr_dims[-1], 2)
    
    def batch(self, x):
        if isinstance(x, list):
            return make_equal_3D(x)
        return x

    def forward(self, im, qr):
        im = self.batch(im)
        qr = self.batch([i[0] for i in qr])
        qr_embed = self.qr_conv(qr)
        nan_guard(qr_embed, "qr_embed")
        #print("qr", qr_embed.shape)
        im_embed = self.im_conv(im)
        nan_guard(im_embed, "im_embed")
        #print("im", im_embed.shape)
        features0 = self.attn(im_embed, qr_embed)
        nan_guard(features0, "attn")
        #print("feats", features.shape)
        features1 = self.mlp_proj(features0)
        res = self.mlp_res(features0.transpose(-1, -2)).transpose(-1, -2)
        features1 = self.mlp_proj_norm(features1) + res
        nan_guard(features1, "mlp_proj")
        features2 = self.mlp(features1)
        nan_guard(features2, "mlp")
        if torch.isnan(features2).any():
            print(features1.max(), features1.min())
            print(features0.max(), features0.min())
        #print("feats2", features.shape)
        features = self.out_proj(features2)
        features = self.out_norm(features)
        nan_guard(features, "out_proj")
        box = self.box_proj(features)
        logits = self.class_proj(features)
        return {'pred_boxes': box.sigmoid(), 'pred_logits': logits}

