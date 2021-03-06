# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.transformer import EncoderLayer, EncoderLayerV2, DecoderLayer, DecoderLayerV2, AlibiMask


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
        attn_dropout=0.1,
        bias=False,
        alibi=True,
        alibi_start_ratio=1/2):
        super().__init__()
        if type(activation) == str:
            activation = getattr(F, activation)
        self.d_model = d_model
        args = [d_model, heads, proj_forward, activation, dropout, attn_dropout, bias]
        self.enc_q = nn.ModuleList([EncoderLayer(*args) for _ in range(enc_q_layers)])
        self.enc_t = nn.ModuleList([EncoderLayer(*args) for _ in range(enc_t_layers)])

        self.dec_kv = nn.ModuleList([DecoderLayer(*args) for _ in range(dec_layers)])
        self.dec_final = nn.ModuleList([DecoderLayer(*args) for _ in range(dec_layers)])

        self.alibi_mask = AlibiMask(alibi_start_ratio, heads=heads, apply=alibi)
    
    def forward(self, im_query, im_target, query_embed, mask_q=None, mask_t=None, pos_q=None, pos_t=None):
        key_padding_mask_q = mask_q[:, None, None, :].bool()
        key_padding_mask_t = mask_t[:, None, None, :].bool()
        alibi_mask_q = self.alibi_mask(im_query)
        alibi_mask_t = self.alibi_mask(im_target)
        for layer in self.enc_q:
            im_query = layer(im_query, key_padding_mask_q, pos_q, alibi_mask=alibi_mask_q)
        for layer in self.enc_t:
            im_target = layer(im_target, key_padding_mask_t, pos_t, alibi_mask=alibi_mask_t)
        
        alibi_mask_embed = self.alibi_mask(query_embed)
        x = torch.zeros_like(query_embed)  # flow-through variable
        kv = torch.zeros_like(im_query)    # flow-through variable
        for kv_layer, out_layer in zip(self.dec_kv, self.dec_final):
            kv = kv_layer(kv, im_query, im_target, mask_q=key_padding_mask_q, mask_k=key_padding_mask_t, alibi_mask=alibi_mask_q, pos_q=pos_q, pos_k=pos_t)
            x = out_layer(x, query_embed, kv, mask_k=key_padding_mask_q, alibi_mask=alibi_mask_embed, pos_k=pos_q)
        return x


class TransformerV2(TransformerV1):
    """Cross attends im_target with im_query, the former the query and latter the key"""
    def forward(self, im_query, im_target, query_embed, mask_q=None, mask_t=None, pos_q=None, pos_t=None):
        key_padding_mask_q = mask_q[:, None, None, :].bool()
        key_padding_mask_t = mask_t[:, None, None, :].bool()
        alibi_mask_q = self.alibi_mask(im_query)
        alibi_mask_t = self.alibi_mask(im_target)
        for i, layer in enumerate(self.enc_q):
            im_query = layer(im_query, key_padding_mask_q, alibi_mask_q, pos_q)
            # assert not torch.isnan(im_query).any(), f"im_query layer {i} nan"
        for i, layer in enumerate(self.enc_t):
            im_target = layer(im_target, key_padding_mask_t, alibi_mask_t, pos_t)
            # assert not torch.isnan(im_target).any(), f"im_target layer {i} nan"
        
        alibi_mask_embed = self.alibi_mask(query_embed)
        x = torch.zeros_like(query_embed)  # flow-through variable
        kv = torch.zeros_like(im_target)    # flow-through variable
        for i, (kv_layer, out_layer) in enumerate(zip(self.dec_kv, self.dec_final)):
            kv = kv_layer(kv, im_target, im_query, mask_q=key_padding_mask_t, mask_k=key_padding_mask_q, alibi_mask=alibi_mask_t, pos_q=pos_t, pos_k=pos_q)
            # assert not torch.isnan(kv).any(), f"kv layer {i} nan"
            x = out_layer(x, query_embed, kv, mask_k=key_padding_mask_t, alibi_mask=alibi_mask_embed, pos_k=pos_t)
            # assert not torch.isnan(x).any(), f"x layer {i} nan"
        return x


class TransformerV3(nn.Module):
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
        attn_dropout=0.1,
        bias=False,
        alibi=True,
        alibi_start_ratio=1/2):
        super().__init__()
        self.d_model = d_model
        self.enc_t = nn.ModuleList([EncoderLayerV2(d_model, heads, proj_forward, dropout, activation) for _ in range(enc_t_layers)])
        self.dec_final = nn.ModuleList([DecoderLayerV2(d_model, heads, proj_forward, dropout, activation) for _ in range(dec_layers)])
        self.alibi_mask = AlibiMask(alibi_start_ratio, heads, alibi)
        
    def forward(self, im_query, im_target, query_embed, mask_q=None, mask_t=None, pos_q=None, pos_t=None):
        key_padding_mask_t = mask_t[:, None, None, :].bool()
        alibi_mask_t = self.alibi_mask(im_target)
        for layer in self.enc_t:
            im_target = layer(im_target, mask=key_padding_mask_t, alibi_mask=alibi_mask_t, pos=pos_t)

        x = torch.zeros_like(query_embed)  # flow-through variable
        query_mask = self.alibi_mask(x)
        for out_layer in self.dec_final:
            x = out_layer(x, im_target, mask_k=key_padding_mask_t, alibi_mask=query_mask, pos=pos_t, query_pos=query_embed)
        return x


variants = [TransformerV1, TransformerV2, TransformerV3]

