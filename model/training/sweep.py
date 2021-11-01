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
        args.attn_dropout,
        args.transformer_bias,
        args.alibi,
        args.alibi_start_ratio
    )


def model_builder(artr: Module, transformer: Module, backbone: Module, args):
    return artr(backbone_builder(backbone, args), transformer_builder(transformer, args), args.num_queries)


def build_loss(HungarianMatcher: Module, SetCriterion: Module, args):
    matcher = HungarianMatcher(args.cost_class, args.cost_bbox, args.cost_giou)
    return SetCriterion(91, matcher, args.cost_eof, args.losses).cuda()


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
                               metric_step, checkpoint_step, results_step, args.mixed_precision, lr_scheduler, 5, config=args)
        
        root = "datasets/coco/"
        proc = 'train'
        query_process = data.GetQuery(data.query_transforms(), args.query_mu, args.query_std, stretch_limit=args.query_stretch_limit, min_size=args.min_query_dim,
                                      query_pool=root + proc + "2017_query_pool", prob_pool=args.query_pool_prob, max_queries=args.max_queries)

        dataset = data.CocoDetection(root + proc + '2017', root + f'annotations/instances_{proc}2017.json', data.img_transforms('train'), None)
        train_set = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=True, num_workers=8, collate_fn=data.collate_fn)        
        trainer.train_epochs(epochs, train_set)

        del model, criterion, param_dicts, optimizer, lr_scheduler, train_set, trainer


def run_test_sweep(objects: List[dict], args_list: list):
    for s in range(len(args_list)):
        sweep = objects[s]
        args = args_list[s]

        backbone = sweep['backbone']
        transformer = sweep['transformer']
        artr = sweep['artr']

        model = model_builder(artr, transformer, backbone, args)
        ims = misc.gen_test_data(512, 64, 4)
        qrs = [misc.gen_test_data(256, 64, 2) for i in range(4)]

        a = model(ims, qrs)
        print(args.name, "passed with no errors")
        print({f"output shape: {k}": v.shape for k, v in a.items()})

        l = sum([a[k].sum() for k in a])
        l.sum().backward()

        grad_parameters = 0
        for n, i in model.named_parameters():
            if i.grad is None:
                print(n, "parameter is None")
            else:
                s = 1
                for h in i.grad.shape: s *= h
                grad_parameters += s
        print("grad parameters", grad_parameters)
    return model


