# %%
import os

import torch
from torch.utils.data import DataLoader

from model.baseline.model import ARTR
import data.dataset as data

from utils.loss import HungarianMatcher, SetCriterion
from model.trainer import TrainerV1

root = 'datasets/coco/'
if not os.path.exists(root + 'val2017_query_pool'):
    data.make_query_pool(root + 'val2017', root + 'annotations/instances_val2017.json', root + 'val2017_query_pool')
if not os.path.exists(root + 'train2017_query_pool'):
    data.make_query_pool(root + 'train2017', root + 'annotations/instances_train2017.json', root + 'train2017_query_pool')

# backbone_args    = [name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool]
# pos_embed_args   = [num_pos_feats=64, temperature=10000, normalize=False, scale=None]
# transformer_args = [d_model, heads, proj_forward, enc_q_nlayers, enc_t_nlayers, dec_nlayers, activation=F.relu, dropout=0.1, bias=None]

batch_size = 4

cost_class = 1
cost_bbox = 5
cost_giou = 2
cost_eof = 0.1
losses = ['boxes', 'labels']

lr = 1e-4
lr_backbone = 1e-5
weight_decay = 1e-4
lr_drop = 200
weight_dict = {'loss_giou': 2, 'loss_bbox': 5, 'loss_ce': 1}

model_dir = 'experiments/artr/v1'
metric_step = 50
checkpoint_step = 100

model = ARTR(transformer_args={'proj_forward': 512, 'enc_q_layers': 1, 'enc_t_layers': 1, 'dec_layers': 3}, d_model=128, num_queries=20).cuda()

# dataset_args = [image_dir: str, json_file: str, query_transforms=None, transforms=None, query_pool: str=None, case4_prob=0.5, case4_sigma=3]
"""train_set = CocoBoxes(root + 'train2017', root + 'annotations/instances_train2017.json',
                      query_transforms(), img_transforms('train'), root + 'train2017_query_pool',
                      0.7, 3)"""
val_set = data.CocoBoxes(root + 'val2017', root + 'annotations/instances_val2017.json',
                    data.query_transforms(), data.img_transforms('val'), root + 'val2017_query_pool',
                    0.5, 2)

val_set = DataLoader(val_set, batch_size, True, num_workers=2, collate_fn=data.CocoBoxes.collate_fn)
val_set = iter(val_set)

matcher = HungarianMatcher(cost_class, cost_bbox, cost_giou)
criterion = SetCriterion(1, matcher, cost_eof, losses).cuda()

param_dicts = [
    {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
    {
        "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
        "lr": lr_backbone,
    },
]
optimizer = torch.optim.AdamW(param_dicts, lr, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_drop)
trainer = TrainerV1(model, criterion, optimizer, model_dir, metric_step, checkpoint_step, False, None, 10, weight_dict)

trainer.train(val_set)
# %%
a, b, c = next(val_set)
