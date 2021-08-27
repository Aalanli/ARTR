# %%
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.baseline.model import ARTR
import data.dataset as data

root = 'datasets/coco/'
if not os.path.exists(root + 'val2017_query_pool'):
    data.make_query_pool(root + 'val2017', root + 'annotations/instances_val2017.json', root + 'val2017_query_pool')
if not os.path.exists(root + 'train2017_query_pool'):
    data.make_query_pool(root + 'train2017', root + 'annotations/instances_train2017.json', root + 'train2017_query_pool')

# %%
# backbone_args    = [name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool]
# pos_embed_args   = [num_pos_feats=64, temperature=10000, normalize=False, scale=None]
# transformer_args = [d_model, heads, proj_forward, enc_q_nlayers, enc_t_nlayers, dec_nlayers, activation=F.relu, dropout=0.1, bias=None]

model = ARTR(transformer_args={'proj_forward': 256, 'enc_q_layers': 1, 'enc_t_layers': 1, 'dec_layers': 1}, d_model=64, num_queries=10)

# dataset_args = [image_dir: str, json_file: str, query_transforms=None, transforms=None, query_pool: str=None, case4_prob=0.5, case4_sigma=3]
"""train_set = CocoBoxes(root + 'train2017', root + 'annotations/instances_train2017.json',
                      query_transforms(), img_transforms('train'), root + 'train2017_query_pool',
                      0.7, 3)"""
val_set = data.CocoBoxes(root + 'val2017', root + 'annotations/instances_val2017.json',
                    data.query_transforms(), data.img_transforms('val'), root + 'val2017_query_pool',
                    0.5, 2)

val_set = DataLoader(val_set, 4, True, num_workers=4, collate_fn=data.CocoBoxes.collate_fn)
val_set = iter(val_set)

# %%
im, qim, tar = next(val_set)
print(im[1].shape)
# %%
out = model(im, qim)
# %%
print(im[0].shape)
# %%
