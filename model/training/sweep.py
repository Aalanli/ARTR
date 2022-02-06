# %%
import os
from typing import Callable, List, Generator, Set, Tuple

import torch
from torch.nn import Module

import data.dataset as data
import utils.misc as misc
from utils.loss import HungarianMatcher, SetCriterion
from model.trainer import TrainerWandb



def build_loss(HungarianMatcher: Module, SetCriterion: Module, args):
    matcher = HungarianMatcher(args.cost_class, args.cost_bbox, args.cost_giou)
    return SetCriterion(args.classes, matcher, args.cost_eof, args.losses).cuda()


def increment_folder(root_dir):
    return f'v{len(os.listdir(root_dir))}'


def run_sweep(objects: Generator[Tuple[Callable, misc.EasyDict], None, None], epochs, root_dir, data_dir, metric_step, checkpoint_step, results_step, qr=None):
    for model, args in objects:
        train_epoch(model, args, epochs, root_dir, data_dir, metric_step, checkpoint_step, results_step, qr=None)


def train_epoch(model, args, epochs, root_dir, data_dir, metric_step, checkpoint_step, results_step, qr=None):
    criterion = build_loss(HungarianMatcher, SetCriterion, args).cuda()
    model = model.cuda()
    args.parameter_size = misc.calculate_param_size(model)
    print(f'training model {args.name} with {args.parameter_size} parameters.' )

    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    trainer = TrainerWandb(model, criterion, optimizer, os.path.join(root_dir, args.name), 
                            metric_step, checkpoint_step, results_step, args.mixed_precision, lr_scheduler, 5, config=args)
    
    root = "datasets/coco/"
    proc = 'train'
    query_process = None
    if qr is not None:
        query_process = data.GetQuery(data.query_transforms(), args.query_mu, args.query_std, stretch_limit=args.query_stretch_limit, min_size=args.min_query_dim,
                                    query_pool=root + proc + "2017_query_pool", prob_pool=args.query_pool_prob, max_queries=args.max_queries)

    dataset = data.CocoDetection(root + proc + '2017', root + f'annotations/instances_{proc}2017.json', data.img_transforms('train'), query_process, args.fix_label)
    train_set = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=True, num_workers=8, collate_fn=data.collate_fn)        
    trainer.train_epochs(epochs, train_set)
    return model


def run_test_sweep(objects: Generator[Tuple[Callable, misc.EasyDict], None, None]):
    for (model, args) in objects:
        
        ims = misc.gen_test_data(512, 64, 4)
        qrs = [misc.gen_test_data(256, 64, 2) for i in range(4)]

        a = model(ims, qrs)
        print(args.name, "passed with no errors")

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


def load_model(name):
    trainer = TrainerWandb(name)
    return trainer.model


def get_dataset(args, qr=None):
    root = "datasets/coco/"
    proc = 'train'
    query_process = None
    if qr is not None:
        query_process = data.GetQuery(data.query_transforms(), args.query_mu, args.query_std, stretch_limit=args.query_stretch_limit, min_size=args.min_query_dim,
                                    query_pool=root + proc + "2017_query_pool", prob_pool=args.query_pool_prob, max_queries=args.max_queries)
    transforms = None
    if "transforms" in args and args.transforms:
        transforms = data.img_transforms('train')
    dataset = data.CocoDetection(root + proc + '2017', root + f'annotations/instances_{proc}2017.json', transforms, query_process, args.fix_label)
    train_set = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=True, num_workers=8, collate_fn=data.collate_fn)
    return train_set 
