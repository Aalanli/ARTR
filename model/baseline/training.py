# %%
import torch
import torch.nn.functional as F
from model.baseline.model import ARTR

# backbone_args    = [name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool]
# pos_embed_args   = [num_pos_feats=64, temperature=10000, normalize=False, scale=None]
# transformer_args = [d_model, heads, proj_forward, enc_q_nlayers, enc_t_nlayers, dec_nlayers, activation=F.relu, dropout=0.1, bias=None]

backbone_args    = ['resnet50', True, False, False]
pos_embed_args   = [256, 10000, True]
transformer_args = [512, 8, 1024, 4, 4, 6, F.relu, 0.1, True]

model = ARTR(backbone_args, pos_embed_args, transformer_args, num_queries=100).cuda()