# %%
from functools import partial
from math import sqrt, ceil
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.pos_emb import PositionEmbeddingSineMaskless
from model.modelV4.backbone import ConvNeXt, ConvNeXtIsotropic, LayerNorm, Block, convnext_isotropic_small, convnext_small
from model.modelV4.fast_conv import conv2d
from timm.models.layers import trunc_normal_, DropPath

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
    """Performs modulated conv2d on input images with queries"""
    def __init__(self, kernel_size, dim, qr_feats, qr_dim, expansion=4, dropout=0.1, drop_path=0., layer_scale_init_value=1e-6) -> None:
        super().__init__()
        self.qr_proj_dim = nn.Sequential(
            nn.Linear(qr_dim, qr_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(qr_dim * 2, dim),
            nn.Dropout(dropout),
            nn.LayerNorm(dim)
        )
        self.qr_proj_seq = nn.Conv1d(qr_feats, 1, 1)
        self.qr_norm = nn.LayerNorm(dim)

        self.ks = kernel_size
        self.conv = nn.Conv2d(dim, dim, kernel_size, padding='same', groups=dim)
        #self.weight = conv.weight
        #self.bias = conv.bias
        #self.out_proj = PreNormResidual(dim, FeedForward(dim, expansion, dropout, nn.Linear))
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, im, qr):
        # qr.shape = [batch, x1, c]
        batch_size, c, x, y = im.shape
        im_pre = im
        qr = self.qr_proj_seq(qr)
        qr = self.qr_proj_dim(qr)
        st = self.qr_norm(qr)

        #w = self.weight * st.reshape(batch_size, 1, -1, 1, 1)
        #w = w.reshape(-1, c, self.ks, self.ks)
        #im = im.reshape(1, -1, x, y)
        #x = F.conv2d(im, w, padding='same', groups=batch_size).reshape(batch_size, c, x, y).permute(0, 2, 3, 1) + self.bias
        #x = [F.conv2d(im[i:i+1], w[i], padding='same') for i in range(batch_size)]
        #x = torch.cat(x, 0).permute(0, 2, 3, 1) + self.bias
        im = im * st.squeeze()[:, :, None, None]
        #x = F.conv2d(im, self.weight, self.bias, padding='same').permute(0, 2, 3, 1)
        x = self.conv(im).permute(0, 2, 3, 1)

        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = im_pre + self.drop_path(x)
        return x

"""f = FeatureExtractor(7, 512, 50, 128).cuda()
# %%
with torch.cuda.amp.autocast(enabled=True):
    a = f(torch.rand(4, 512, 32, 32, device='cuda'), torch.rand(4, 50, 128, device='cuda'))
    print(a.dtype)
    l = a.sum()
l.backward()"""
# %%
"""a = torch.rand(4).cuda()
b = torch.rand(4, device='cuda', dtype=torch.half)

print(a.dtype, b.dtype)
a = a.to(b.dtype)
print(a.dtype)"""

# %%
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
    def __init__(self, x_dim, dims, depths, mlp_depth, mlp_features, min_query_size=None, max_query_size=None) -> None:
        super().__init__()
        self.min_size = min_query_size
        self.max_size = max_query_size
        self.x_dim = x_dim
        self.conv = ConvNeXt(3, depths, dims)
        self.down_sample = nn.AdaptiveAvgPool2d((x_dim, x_dim))

        self.out_proc = MLPMixer(x_dim ** 2, mlp_features, dims[-1], mlp_depth)
    
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
        h1 = self.out_proc(h0)
        return h1


class ConvNeXtIsotropic(nn.Module):
    """A modulated implementation of isotropic ConvNeXt,
       where 1 is a modulated block while 0 is a normal block
       in block_pattern"""
    def __init__(self, dim, qr_feats, qr_dim, in_chans=3,
                 block_pattern=[0, 0, 1, 0], repeats=4, drop_path_rate=0., 
                 layer_scale_init_value=0
                 ):
        super().__init__()
        depth = len(block_pattern) * repeats
        self.stem = nn.Conv2d(in_chans, dim, kernel_size=16, stride=16)
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, depth)] 
        self.blocks = nn.ModuleList()
        self.block_patterns = block_pattern * repeats
        for i in self.block_patterns:
            if i == 0:
                self.blocks.append(Block(dim, dp_rates[i], layer_scale_init_value))
            else:
                self.blocks.append(FeatureExtractor(7, dim, qr_feats, qr_dim))

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first") # final norm layer
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, im, qr):
        x = self.stem(im)
        for t, layer in zip(self.block_patterns, self.blocks):
            if t == 0:
                x = layer(x)
            else:
                x = layer(x, qr)
        return self.norm(x)


class ARTR1(nn.Module):
    def __init__(
        self,
        n_pred=50,
        im_flatten_dim=8,
        im_block_pattern=[0, 0, 1, 1],
        im_repeats=4,
        im_dim=256,
        qr_depths=[3, 3, 6, 3],
        qr_dims=[64, 128, 256, 512],
        qr_flatten_dim=8,
        qr_feats=64,
        qr_mlp_depth=6,
        mlp_depth=6,
        mlp_feats=256,
        min_qr_size=None,
        max_qr_size=None
    ) -> None:
        super().__init__()
        self.im_conv = ConvNeXtIsotropic(im_dim, qr_feats, qr_dims[-1], 3, im_block_pattern, im_repeats)
        self.qr_conv = QueryExtractor(qr_flatten_dim, qr_dims, qr_depths, qr_mlp_depth, qr_feats, min_qr_size, max_qr_size)
        out_features = im_flatten_dim ** 2
        self.im_downsample = nn.AdaptiveAvgPool2d((im_flatten_dim, im_flatten_dim))
        self.im_dim_inc = nn.Conv2d(im_dim, qr_dims[-1], 1)
        self.mlp_proj = nn.Conv1d(out_features, mlp_feats, 1)
        self.mlp_res = nn.AdaptiveAvgPool1d(mlp_feats)
        self.mlp_proj_norm = nn.LayerNorm(qr_dims[-1])
        self.mlp = MLPMixer(mlp_feats, mlp_feats, qr_dims[-1], mlp_depth)
        self.box_proj1 = nn.Conv1d(mlp_feats, n_pred, 1)
        self.box_proj2 = nn.Linear(qr_dims[-1], 4)
        self.out_norm1 = nn.LayerNorm(qr_dims[-1])
        self.class_proj1 = nn.Conv1d(mlp_feats, n_pred, 1)
        self.class_proj2 = nn.Linear(qr_dims[-1], 2)
        self.out_norm2 = nn.LayerNorm(qr_dims[-1])
    
    def batch(self, x):
        if isinstance(x, list):
            return make_equal_3D(x)
        return x

    def forward(self, im, qr):
        im = self.batch(im)
        if isinstance(qr, list):
            qr = self.batch([i[0] for i in qr])
        qr_embed = self.qr_conv(qr)
        #print("qr", qr_embed.shape)
        im_embed = self.im_conv(im, qr_embed)
        im_embed = self.im_downsample(im_embed)
        im_embed = self.im_dim_inc(im_embed).flatten(2, 3).permute(0, 2, 1)
        #print("im", im_embed.shape)
        #print("feats", features.shape)
        features1 = self.mlp_proj(im_embed)
        res = self.mlp_res(im_embed.transpose(-1, -2)).transpose(-1, -2)
        features1 = self.mlp_proj_norm(features1) + res
        features = self.mlp(features1)
        #print("feats2", features.shape)
        box_features = self.box_proj1(features)
        box_features = self.out_norm1(box_features)
        box = self.box_proj2(box_features)
        class_features = self.class_proj1(features)
        class_features = self.out_norm2(class_features)
        logits = self.class_proj2(class_features) 
        return {'pred_boxes': box.sigmoid(), 'pred_logits': logits}

class ARTR(nn.Module):
    def __init__(self, 
        n_pred=50, 
        x_dim=5, 
        splits=3, 
        im_block_pattern=[0, 0, 1, 1], 
        im_repeats=4, 
        im_dim=256, 
        qr_depths=[3, 3, 6, 3], 
        qr_dims=[64, 128, 256, 512],
        qr_x_dim=8,
        qr_feats=64, 
        qr_mlp_depth=4, 
        mlp_depth=6, 
        mlp_feats=256,
        min_qr_size=None, 
        max_qr_size=None
    ) -> None:
        super().__init__()
        self.im_conv = ConvNeXtIsotropic(im_dim, qr_feats, qr_dims[-1], 3, im_block_pattern, im_repeats)
        self.qr_conv = QueryExtractor(qr_x_dim, qr_dims, qr_depths, qr_mlp_depth, qr_feats, min_qr_size, max_qr_size)
        #self.qr_conv.conv = convnext_small(True)
        self.embed = nn.Parameter(torch.randn(1, splits ** 2, x_dim ** 2, qr_dims[-1]))
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
        if isinstance(qr, list):
            qr = self.batch([i[0] for i in qr])
        qr_embed = self.qr_conv(qr)
        #print("qr", qr_embed.shape)
        im_embed = self.im_conv(im, qr_embed)
        #print("im", im_embed.shape)
        features0 = self.attn(im_embed, self.embed)
        #print("feats", features.shape)
        features1 = self.mlp_proj(features0)
        res = self.mlp_res(features0.transpose(-1, -2)).transpose(-1, -2)
        features1 = self.mlp_proj_norm(features1) + res
        features2 = self.mlp(features1)
        #print("feats2", features.shape)
        features = self.out_proj(features2)
        features = self.out_norm(features)
        box = self.box_proj(features)
        logits = self.class_proj(features)
        return {'pred_boxes': box.sigmoid(), 'pred_logits': logits}


