# %%
import torch
from torch import nn

from model.transformer import DecoderLayerV2, EncoderLayerV2
from model.pos_emb import PositionEmbeddingSine


class Detector(nn.Module):
    """
    Detector class guarantees:
    input shape: 
        im_features = [batch, channels, x, y] 
        mask = [batch, x, y]
    
    output:
        a dict containing 2 keys, pred_logits and pred_boxes where
        pred_logits associates with a tensor of shape [batch, queries, 2]
        pred_boxes associates with a tensor of shape [batch, queries, 4]
    
    The portion acts like a detection head, where im_features is already compressed by
    a resnet backbone.
    """
    def __init__(self, n_queries, d_model=512, nhead=8, n_enc_layer=6, n_dec_layer=6, dim_feedforward=2048, 
                 dropout=0.1, activation='relu', normalize_before=False): # TODO add return intermediate
        super().__init__()
        self.encoder = nn.ModuleList([EncoderLayerV2(d_model, nhead, dim_feedforward, dropout, activation, normalize_before) for _ in range(n_enc_layer)])
        self.norm1 = nn.LayerNorm(d_model) if normalize_before else nn.Identity()
        self.decoder = nn.ModuleList([DecoderLayerV2(d_model, nhead, dim_feedforward, dropout, activation, normalize_before) for _ in range(n_dec_layer)])
        self.norm2 = nn.LayerNorm(d_model) if normalize_before else nn.Identity()
        self.d_model = d_model
        self.in_proj = nn.Conv2d(2048, d_model, kernel_size=3, padding=1)
        self.query_embed = nn.Embedding(n_queries, d_model)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)
        self.class_embed = nn.Linear(d_model, 2)
        self.pos_emb = PositionEmbeddingSine(d_model // 2)
    
    def forward(self, im_features: torch.Tensor, mask: torch.Tensor):  # im_features.shape = [B, C, X, Y]
        # squeeze input channels into self.d_model
        im_features = self.in_proj(im_features)
        features = im_features.flatten(2).transpose(-1, -2)

        pos_emb = self.pos_emb(mask)
        mask = mask.flatten(1)[:, None, None, :]
        query_emb = self.query_embed.weight.unsqueeze(0).repeat(features.shape[0], 1, 1)
        pos_emb = pos_emb.flatten(1, 2)
        tgt = torch.zeros_like(query_emb)
        for encoder in self.encoder:
            features = encoder(features, mask=mask, pos=pos_emb)
        memory = self.norm1(features)
        for decoder in self.decoder:
            tgt = decoder(tgt, memory, mask_k=mask, pos=pos_emb, query_pos=query_emb)
        hs = self.norm2(tgt)
        outputs_class = self.class_embed(hs)
        outputs_boxes = self.bbox_embed(hs).sigmoid()
        return {'pred_logits': outputs_class, 'pred_boxes': outputs_boxes}


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

