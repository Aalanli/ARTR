import torch
from torch.nn import Module

import data.dataset as data
import utils.misc as misc
from model.trainer import TrainerWandb


def backbone_builder(backbone: Module, args):
    return backbone(args.backbone_name, args.backbone_train_backbone, args.backbone_return_interm_layers,
                    args.backbone_dilation, args.pos_embed_num_pos_feats, args.pos_embed_temperature,
                    args.pos_embed_normalize, args.pos_embed_scale)


def transformer_builder(transformer: Module, args):

    return transformer(
        args.transformer_d_model,
        args.transformer_heads,
        args.transformer_proj_forward,
        args.transformer_enc_q_layers,
        args.transformer_enc_t_layers,
        args.transformer_dec_layers,
        args.transformer_activation,
        args.transformer_dropout,
        args.attn_dropout,
        args.transformer_bias,
        args.alibi,
        args.alibi_start_ratio
    )


def model_builder(artr: Module, transformer: Module, backbone: Module, args):
    return artr(backbone_builder(backbone, args), transformer_builder(transformer, args), args.num_queries)
