# %%
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from torch.utils import checkpoint


def split_heads(x: torch.Tensor, heads: int):
    # shape = [batch, sequence, features]
    # split features into heads; size = [batch, heads, sequence, depth]
    batch, seq_len, features = x.shape
    x = x.reshape((batch, seq_len, heads, features // heads))
    return x.permute(0, 2, 1, 3)


def combine_heads(x: torch.Tensor):
    # inverse operation of split heads
    batch, heads, seq_len, depth = x.shape
    x = x.permute(0, 2, 1, 3)
    return x.reshape((batch, seq_len, heads * depth))


def multihead_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
    """
    Most naive implementation
    mask.shape = [bs, k.shape[-2]]; k.shape[-2] = k_seq_len
    """
    depth = q.shape[-1]
    w = q @ k.transpose(-1, -2)
    w = w / math.sqrt(depth)

    if mask is not None:
        w = w.masked_fill(mask, float('-inf'))
    
    a = w.softmax(-1)
    out = a @ v
    return out, a


class Attention(nn.Module):
    def __init__(self, d_model, heads, bias=None, self_attn=True):
        super().__init__()

        self.d_model = d_model
        self.heads = heads
        self.b = bias
        self.self_attn = self_attn  # if q, k, v are the same

        self.w = nn.Parameter(torch.Tensor(d_model * 3, d_model))
        init.xavier_uniform_(self.w)
        if self.b:
            self.b = nn.Parameter(torch.zeros(d_model * 3))
        self.split_heads = lambda x: split_heads(x, self.heads)
    
    def attn_proj(self, q, k, v):
        """Transforms the inputs"""
        if self.self_attn:  # if positional encoding is not injected in every layer
            # self-attention, singular matrix transform
            q, k, v = F.linear(q, self.w, self.b).chunk(3, dim=-1)
        else:  # multiple matrix transforms, as q != v and k != v
            if self.b is None:
                q, k, v = map(F.linear, (q, k, v), self.w.chunk(3))
            else:
                q, k, v = map(F.linear, (q, k, v), self.w.chunk(3), self.b.chunk(3))
        
        return q, k, v
    
    def forward(self, q, k, v, mask=None):
        q, k, v = self.attn_proj(q, k, v)
        q, k, v = map(self.split_heads, (q, k, v))
        out, a = multihead_attn(q, k, v, mask)
        return combine_heads(out), a


def checkpoint_wrapper(func, *args, apply=True):
    if apply:
        return checkpoint.checkpoint(func, *args)
    return func(*args)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, proj_forward, activation=F.relu, dropout=0.1, bias=None):
        super().__init__()
        self.attn = Attention(d_model, heads, bias, self_attn=False)        

        self.linear1 = nn.Linear(d_model, proj_forward)
        self.linear2 = nn.Linear(proj_forward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x, mask=None, pos=None):
        if pos is not None:
            q = k = x + pos
        else:
            q = k = x
        x1 = self.attn(q, k, x, mask)[0]
        x = self.drop(x1) + x  # save, non-deterministic
        x = self.norm1(x)
        x1 = self.activation(self.linear1(x))
        x1 = self.drop(x1)  # save, non-deterministic
        x1 = self.linear2(x1)
        x = x + x1
        x = self.norm2(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, proj_forward, activation=F.relu, dropout=0.1, bias=None):
        super().__init__()
        self.attn1 = Attention(d_model, heads, bias, self_attn=False)
        self.attn2 = Attention(d_model, heads, bias, self_attn=False)
        
        self.linear1 = nn.Linear(d_model, proj_forward)
        self.linear2 = nn.Linear(proj_forward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
    
    def apply_pos(self, x, pos):
        return x if pos is None else x + pos
    
    def forward(self, x, query, memory, mask_q=None, mask_k=None, pos_q=None, pos_k=None):
        query = self.apply_pos(query, pos_q)
        q = k = self.apply_pos(x, query)
        x1 = self.attn1(q, k, x, mask=mask_q)[0]
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

