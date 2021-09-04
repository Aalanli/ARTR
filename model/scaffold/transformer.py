import torch
import torch.nn as nn
import torch.nn.functional as F

from model.transformer import EncoderLayer, DecoderLayer


class TransformerV1(nn.Module):
    def __init__(
        self, 
        d_model, 
        heads, 
        proj_forward, 
        enc_q_layers,
        enc_t_layers,
        dec_layers,
        activation=F.relu,
        dropout=0.1, 
        bias=None):
        super().__init__()
        if type(activation) == str:
            activation = getattr(F, activation)
        self.d_model = d_model
        args = [d_model, heads, proj_forward, activation, dropout, bias]
        self.enc_q = nn.ModuleList([EncoderLayer(*args) for _ in range(enc_q_layers)])
        self.enc_t = nn.ModuleList([EncoderLayer(*args) for _ in range(enc_t_layers)])

        self.dec_kv = nn.ModuleList([DecoderLayer(*args) for _ in range(dec_layers)])
        self.dec_final = nn.ModuleList([DecoderLayer(*args) for _ in range(dec_layers)])
    
    def forward(self, im_query, im_target, query_embed, mask_q=None, mask_t=None, pos_q=None, pos_t=None):
        key_padding_mask_q = mask_q[:, None, None, :].bool()
        key_padding_mask_t = mask_t[:, None, None, :].bool()
        for layer in self.enc_q:
            im_query = layer(im_query, key_padding_mask_q, pos_q)
        for layer in self.enc_t:
            im_target = layer(im_target, key_padding_mask_t, pos_t)

        x = torch.zeros_like(query_embed)  # flow-through variable
        kv = torch.zeros_like(im_query)    # flow-through variable
        for kv_layer, out_layer in zip(self.dec_kv, self.dec_final):
            kv = kv_layer(kv, im_query, im_target, mask_q=key_padding_mask_q, mask_k=key_padding_mask_t, pos_q=pos_q, pos_k=pos_t)
            x = out_layer(x, query_embed, kv, mask_k=key_padding_mask_q, pos_k=pos_q)
        return x


class TransformerV2(TransformerV1):
    """Cross attends im_target with im_query, the former the query and latter the key"""
    def forward(self, im_query, im_target, query_embed, mask_q=None, mask_t=None, pos_q=None, pos_t=None):
        key_padding_mask_q = mask_q[:, None, None, :].bool()
        key_padding_mask_t = mask_t[:, None, None, :].bool()
        for layer in self.enc_q:
            im_query = layer(im_query, key_padding_mask_q, pos_q)
        for layer in self.enc_t:
            im_target = layer(im_target, key_padding_mask_t, pos_t)

        x = torch.zeros_like(query_embed)  # flow-through variable
        kv = torch.zeros_like(im_target)    # flow-through variable
        for kv_layer, out_layer in zip(self.dec_kv, self.dec_final):
            kv = kv_layer(kv, im_target, im_query, mask_q=key_padding_mask_t, mask_k=key_padding_mask_q, pos_q=pos_t, pos_k=pos_q)
            x = out_layer(x, query_embed, kv, mask_k=key_padding_mask_t, pos_k=pos_t)
        return x


class TransformerV3(TransformerV1):
    def forward(self, im_query, im_target, query_embed, mask_q=None, mask_t=None, pos_q=None, pos_t=None):
        key_padding_mask_q = mask_q[:, None, None, :].bool()
        key_padding_mask_t = mask_t[:, None, None, :].bool()
        for layer in self.enc_q:
            im_query = layer(im_query, key_padding_mask_q, pos_q)
        for layer in self.enc_t:
            im_target = layer(im_target, key_padding_mask_t, pos_t)

        x = torch.zeros_like(query_embed)  # flow-through variable
        kv = torch.zeros_like(im_target)    # flow-through variable
        for kv_layer in self.dec_kv:
            kv = kv_layer(kv, im_target, im_query, mask_q=key_padding_mask_t, mask_k=key_padding_mask_q, pos_q=pos_t, pos_k=pos_q)
        for out_layer in self.dec_final:
            x = out_layer(x, query_embed, kv, mask_k=key_padding_mask_t, pos_k=pos_t)
        return x


variants = [TransformerV1, TransformerV2, TransformerV3]