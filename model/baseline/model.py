import torch
from torch import nn
from model.transformer import EncoderLayer, DecoderLayer


class Transformer(nn.Module):
    def __init__(
        self, 
        d_model, 
        heads, 
        proj_forward, 
        activation,
        enc_q_nlayers,
        enc_t_nlayers,
        dec_nlayers,
        dropout=0.1, 
        bias=None):
        super().__init__()
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