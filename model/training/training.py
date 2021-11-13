# %%
import sys
import copy
sys.path[0] = '/home/allan/Programs/ARTR'

from model.training.sweep import run_sweep, run_test_sweep
from model.modelV2.ARTR import ARTR
from utils.misc import EasyDict

def get_parameters():
    args = EasyDict()

    args.batch_size = 4
    args.mixed_precision = False

    args.query_pool_prob = 0.4
    args.query_std = 2
    args.query_mu = 3
    args.query_stretch_limit = 0.7
    args.min_query_dim = 32
    args.max_queries = 10

    args.cost_class = 1
    args.cost_bbox = 5 * 4
    args.cost_giou = 2 * 4
    args.cost_eof = 0.05
    args.losses = ['boxes', 'labels', 'cardinality']

    args.lr = 1e-4
    args.weight_decay = 1e-4
    args.lr_drop = 200
    args.weight_dict = {'loss_giou': 2, 'loss_bbox': 5, 'loss_ce': 1}
    args.name = 'V2Variant_v3'
    return args

def get_model_parameters():
    args = EasyDict()
    args.layer_maps=[[0, 1], [1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]]
    args.backbone_transformer_kwargs={'nhead': 8, 'dim_feedforward': 1024, 'dropout': 0.1, 'activation': 'relu', 'normalize_before': False}
    args.feat_maps_arch='resnet34'
    args.pretrained=True
    args.n_queries=50
    args.d_model=256
    args.nhead=8
    args.n_enc_layers=3
    args.n_dec_layers=3
    args.dim_feedforward=2048
    args.dropout=0.1
    args.activation='relu'
    args.normalize_before=False
    return args


def model_generator(param_list):
    for p, p1 in param_list:
        model = ARTR(**p)
        p = copy.deepcopy(p)
        p.update(p1)
        yield model, p

param_list = [[get_model_parameters(), get_parameters()]]
model = run_test_sweep(model_generator(param_list))

# %%
run_sweep(model_generator(param_list), epochs=1, root_dir='experiments/artr', data_dir='datasets/coco', metric_step=1200, checkpoint_step=2000,
          results_step=21000)

