# %%
import os
from typing import List

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
        args.transformer_bias
    )


def model_builder(artr: Module, transformer: Module, backbone: Module, args):
    return artr(backbone_builder(backbone, args), transformer_builder(transformer, args), args.num_queries)


def build_loss(HungarianMatcher: Module, SetCriterion: Module, args):
    matcher = HungarianMatcher(args.cost_class, args.cost_bbox, args.cost_giou)
    return SetCriterion(1, matcher, args.cost_eof, args.losses).cuda()


def increment_folder(root_dir):
    return f'v{len(os.listdir(root_dir))}'


def run_sweep(objects: List[dict], args_list: list, epochs, root_dir, data_dir, metric_step, checkpoint_step, results_step):
    for s in range(len(args_list)):
        sweep = objects[s]
        args = args_list[s]

        backbone = sweep['backbone']
        transformer = sweep['transformer']
        artr = sweep['artr']
        HungarianMatcher = sweep['HungarianMatcher']
        SetCriterion = sweep['SetCriterion']

        model = model_builder(artr, transformer, backbone, args).cuda()
        criterion = build_loss(HungarianMatcher, SetCriterion, args).cuda()

        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
        ]

        args.parameter_size = misc.calculate_param_size(model)
        print(f'training model {args.name} with {args.parameter_size} parameters.' )

        optimizer = torch.optim.AdamW(param_dicts, args.lr, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
        trainer = TrainerWandb(model, criterion, optimizer, os.path.join(root_dir, args.name), 
                               metric_step, checkpoint_step, results_step, False, lr_scheduler, 5, config=args)
        
        train_set = data.CocoBoxes(data_dir + '/train2017', data_dir + '/annotations/instances_train2017.json',
                                   data.query_transforms(), data.img_transforms('train'), data_dir + '/train2017_query_pool',
                                   args.mirror_prob, args.case4_sigma, args.case4_mu, args.min_query_dim)
        train_set = torch.utils.data.DataLoader(train_set, args.batch_size, True, num_workers=4, collate_fn=data.CocoBoxes.collate_fn)        
        trainer.train_epochs(epochs, train_set)

        del model, criterion, param_dicts, optimizer, lr_scheduler, train_set, trainer

