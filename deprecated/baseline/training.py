# %%
import sys
sys.path[0] = '/home/allan/Programs/ARTR'
import os

import torch
from torch.utils.data import DataLoader

from model.baseline.model import ARTR, ARTRV1, ARTRV2, ARTRV3
import data.dataset as data

from utils.loss import HungarianMatcher, SetCriterion
from utils.misc import calculate_param_size
from model.trainer import TrainerWandb

root = 'datasets/coco/'
if not os.path.exists(root + 'val2017_query_pool'):
    data.make_query_pool(root + 'val2017', root + 'annotations/instances_val2017.json', root + 'val2017_query_pool')
if not os.path.exists(root + 'train2017_query_pool'):
    data.make_query_pool(root + 'train2017', root + 'annotations/instances_train2017.json', root + 'train2017_query_pool')


class EasyDict(dict):
    def __getattr__(self, name):
        return self[name]
    def __setattr__(self, name: str, value) -> None:
        self[name] = value 
    def search_common_naming(self, name, seperator='_'):
        name = name + seperator
        return {k.replace(name, ''): v for k, v in self.items() if name in k}

args = EasyDict()

args.batch_size = 4

args.mirror_prob = 0.9
args.case4_sigma = 1.5
args.case4_mu = 1
args.min_query_dim = None

args.cost_class = 1
args.cost_bbox = 5
args.cost_giou = 2
args.cost_eof = 0.1
args.losses = ['boxes', 'labels']

args.lr = 1e-4
args.lr_backbone = 1e-5
args.weight_decay = 1e-4
args.lr_drop = 200
args.weight_dict = {'loss_giou': 2, 'loss_bbox': 5, 'loss_ce': 1}

args.backbone_name = 'resnet50'
args.backbone_train_backbone = True
args.backbone_return_interm_layers = False
args.backbone_dilation = False

args.d_model = 256
args.num_queries = 3

args.pos_embed_num_pos_feats = args.d_model // 2
args.pos_embed_temperature = 10000
args.pos_embed_normalize = False
args.pos_embed_scale = None

args.transformer_d_model = args.d_model
args.transformer_heads = 8
args.transformer_proj_forward = 1024
args.transformer_enc_q_layers = 4
args.transformer_enc_t_layers = 5
args.transformer_dec_layers = 6
args.transformer_activation = 'relu'
args.transformer_dropout = 0.1
args.transformer_bias = True


model_dir = 'experiments/artr/v9'
metric_step = 500
results_step = 14000
checkpoint_step = 1500

model = ARTRV3(transformer_args=args.search_common_naming('transformer'), 
             backbone_args= args.search_common_naming('backbone'),
             pos_embed_args=args.search_common_naming('pos_embed'),
             d_model=args.d_model, num_queries=args.num_queries).cuda()

args.parameter_size = calculate_param_size(model)
print('training model with', args.parameter_size, 'parameters.' )

# dataset_args = [image_dir: str, json_file: str, query_transforms=None, transforms=None, query_pool: str=None, case4_prob=0.5, case4_sigma=3]
train_set = data.CocoBoxes(root + 'train2017', root + 'annotations/instances_train2017.json',
                           data.query_transforms(), data.img_transforms('train'), root + 'train2017_query_pool',
                           args.mirror_prob, args.case4_sigma, args.case4_mu, args.min_query_dim)
# val_set = data.CocoBoxes(root + 'val2017', root + 'annotations/instances_val2017.json',
#                          data.query_transforms(), data.img_transforms('val'), root + 'val2017_query_pool',
#                          0.5, 2)

train_set = DataLoader(train_set, args.batch_size, True, num_workers=4, collate_fn=data.CocoBoxes.collate_fn)
train_set = iter(train_set)

# val_set = DataLoader(val_set, 1, True, num_workers=1, collate_fn=data.CocoBoxes.collate_fn)
# val_set = iter(val_set)

matcher = HungarianMatcher(args.cost_class, args.cost_bbox, args.cost_giou)
criterion = SetCriterion(1, matcher, args.cost_eof, args.losses).cuda()

param_dicts = [
    {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
    {
        "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
        "lr": args.lr_backbone,
    },
]

optimizer = torch.optim.AdamW(param_dicts, args.lr, weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
trainer = TrainerWandb('singletest_variant3', model, criterion, optimizer, model_dir, metric_step, checkpoint_step, results_step, False, None, 5, config_args=args)

for i in range(3):
    trainer.train(train_data=train_set)
    trainer.regulate_checkpoints()

