# %%
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.utils import checkpoint


class AlibiMask(nn.Module):
    def __init__(self, start_ratio=1/2, heads=8, apply=True):
        super().__init__()
        self.use = apply
        self.heads = heads
        self.start_ratio = start_ratio
        self.mask = None
    
    def alibi_mask(self, n, heads):
        a = torch.arange(0.0, n, 1.0).unsqueeze(0).repeat(n, 1)
        b = torch.arange(0.0, n, 1.0).unsqueeze(1)
        a = a - b
        a = a.unsqueeze(0).repeat(heads, 1, 1)
        c = torch.tensor([[[self.start_ratio / 2 ** i]] for i in range(heads)])
        return c * a
    
    def forward(self, x):
        if self.use is False:
            return None
        b, seq_len, dim = x.shape
        if self.mask is None or self.mask.shape[-2] < seq_len:
            self.mask = self.alibi_mask(seq_len, self.heads).to(x)
        if self.mask.device != x.device or self.mask.dtype != x.dtype:
            self.mask = self.mask.to(x)
        return self.mask[:, :seq_len, :seq_len]


class Attention(nn.Module):
    def __init__(self, d_model, heads, bias=False, dropout=0):
        super().__init__()

        self.d_model = d_model
        self.heads = heads
        self.b = bias

        self.qw = nn.Linear(d_model, d_model, bias=bias)
        self.kw = nn.Linear(d_model, d_model, bias=bias)
        self.vw = nn.Linear(d_model, d_model, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
    
    def split_heads(self, x: torch.Tensor):
        # shape = [batch, sequence, features]
        # split features into heads; size = [batch, heads, sequence, depth]
        batch, seq_len, features = x.shape
        x = x.reshape((batch, seq_len, self.heads, features // self.heads))
        return x.permute(0, 2, 1, 3)

    def combine_heads(self, x: torch.Tensor):
        # inverse operation of split heads
        batch, heads, seq_len, depth = x.shape
        x = x.permute(0, 2, 1, 3)
        return x.reshape((batch, seq_len, heads * depth))
    
    def unsqueeze_mask(self, mask: torch.Tensor):
        if mask.dim() == 2:
            return mask.unsqueeze(1).unsqueeze(2)
        raise NotImplementedError(f"mask dim {mask.dim()} not supported")

    def multihead_attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None, alibi_mask=None):
        """
        Most naive implementation
        mask.shape = [bs, k.shape[-2]]; k.shape[-2] = k_seq_len
        """
        depth = q.shape[-1]
        w = q @ k.transpose(-1, -2)
        w = w / math.sqrt(depth)

        if mask is not None:
            w = w.masked_fill(mask, float('-inf'))
        
        if alibi_mask is not None:
            w = w + alibi_mask
        
        a = w.softmax(-1)
        a = self.dropout(a)
        out = a @ v
        return out, a
        
    def attn_proj(self, q, k, v):
        """Transforms the inputs"""
        q = self.qw(q)
        k = self.kw(k)
        v = self.vw(v)
        return q, k, v
    
    def forward(self, q, k, v, mask=None, alibi_mask=None):
        q, k, v = self.attn_proj(q, k, v)
        q, k, v = map(self.split_heads, (q, k, v))
        out, a = self.multihead_attn(q, k, v, mask, alibi_mask)
        return self.combine_heads(out), a
    

def checkpoint_wrapper(func, *args, apply=True):
    if apply:
        return checkpoint.checkpoint(func, *args)
    return func(*args)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, proj_forward, activation=F.relu, dropout=0.1, attn_dropout=0.1, bias=None):
        super().__init__()
        self.attn = Attention(d_model, heads, bias, dropout=attn_dropout)        

        self.linear1 = nn.Linear(d_model, proj_forward)
        self.linear2 = nn.Linear(proj_forward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x, mask=None, alibi_mask=None, pos=None):
        if pos is not None:
            q = k = x + pos
        else:
            q = k = x
        x1 = self.attn(q, k, x, mask, alibi_mask=alibi_mask)[0]
        x = self.drop(x1) + x  # save, non-deterministic
        x = self.norm1(x)
        x1 = self.activation(self.linear1(x))
        x1 = self.drop(x1)  # save, non-deterministic
        x1 = self.linear2(x1)
        x = x + x1
        x = self.norm2(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, proj_forward, activation=F.relu, dropout=0.1, attn_dropout=0.1, bias=None):
        super().__init__()
        self.attn1 = Attention(d_model, heads, bias, attn_dropout)
        self.attn2 = Attention(d_model, heads, bias, attn_dropout)
        
        self.linear1 = nn.Linear(d_model, proj_forward)
        self.linear2 = nn.Linear(proj_forward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
    
    def apply_pos(self, x, pos):
        return x if pos is None else x + pos
    
    def forward(self, x, query=None, memory=None, mask_q=None, mask_k=None, alibi_mask=None, pos_q=None, pos_k=None):
        query = self.apply_pos(query, pos_q)
        q = k = self.apply_pos(x, query)
        x1 = self.attn1(q, k, x, mask=mask_q, alibi_mask=alibi_mask)[0]
        x = x + self.dropout(x1)
        x = self.norm1(x)
        
        x1 = self.attn2(
            self.apply_pos(x, query),
            self.apply_pos(memory, pos_k),
            memory,
            mask=mask_k)[0]
        x = x + self.dropout(x1)
        x = self.norm2(x)
        x1 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout(x1)
        x = self.norm3(x)
        return x


class EncoderLayerV2(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, bias=True, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     alibi=None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, v=src, mask=mask, alibi_mask=alibi)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None, 
                    alibi=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, v=src2,
                              mask=mask, alibi_mask=alibi)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                alibi_mask=None):
        if self.normalize_before:
            return self.forward_pre(src, mask, pos, alibi=alibi_mask)
        return self.forward_post(src, mask, pos, alibi=alibi_mask)


class DecoderLayerV2(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, bias=True, dropout=dropout)
        self.multihead_attn = Attention(d_model, nhead, bias=True, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     alibi=None):
        # tgt = 0, memory = encoder output, memory_key_padding_mask = mask, pos = positional_embedding, query_pos = selected query embeddings
        q = k = self.with_pos_embed(tgt, query_pos)  # q = k = query embeddings
        tgt2 = self.self_attn(q, k, tgt,
                              mask=tgt_key_padding_mask, alibi_mask=alibi)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(q=self.with_pos_embed(tgt, query_pos),
                                   k=self.with_pos_embed(memory, pos),
                                   v=memory,
                                   mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    alibi=None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, tgt2,
                              mask=tgt_key_padding_mask, alibi_mask=alibi)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(q=self.with_pos_embed(tgt2, query_pos),
                                   k=self.with_pos_embed(memory, pos),
                                   v=memory,
                                   mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                mask_q: Optional[Tensor] = None,
                mask_k: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                alibi_mask=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory,
                                    mask_q, mask_k, pos, query_pos, alibi=alibi_mask)
        return self.forward_post(tgt, memory,
                                 mask_q, mask_k, pos, query_pos, alibi=alibi_mask)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if type(activation) != str:
        return activation
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
