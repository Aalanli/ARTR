# %%
import sys
import copy
sys.path[0] = '/home/allan/Programs/ARTR'

from model.training.sweep import run_sweep, run_test_sweep, train_epoch
from model.modelV2.ARTR import ARTR
from model.modelV3.conv import Detector
from utils.misc import EasyDict

def get_parameters():
    args = EasyDict()

    args.batch_size = 4
    args.mixed_precision = True
    args.fix_label = None

    args.query_pool_prob = 0.4
    args.query_std = 2
    args.query_mu = 3
    args.query_stretch_limit = 0.7
    args.min_query_dim = 32
    args.max_queries = 10

    args.cost_class = 1
    args.cost_bbox = 5
    args.cost_giou = 2
    args.cost_eof = 0.05
    args.losses = ['boxes', 'labels', 'cardinality']

    args.lr = 1e-4
    args.weight_decay = 1e-4
    args.lr_drop = 200
    args.weight_dict = {'loss_giou': 2, 'loss_bbox': 5, 'loss_ce': 1}
    args.name = 'modelV3_v7'
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

def get_model_v3_parameters():
    args = EasyDict()
    args.dims = 256
    args.final_dim = 256
    args.depths = [5, 5, 5, 5]
    args.out_sizes = [15, 9, 8, 7]
    args.kernel_sizes = [7, 5, 4, 4]
    args.patch_sizes = [7, 5, 3, 3]
    args.mlp_depth = 5
    args.queries = 50
    args.attn_out_channels = 256
    args.attn_kernel_size = 1
    args.classes = 97
    return args


def model_generator(param_list):
    for p, p1 in param_list:
        model = Detector(**p)
        p = copy.deepcopy(p)
        p.update(p1)
        yield model, p

param_list = [[get_model_v3_parameters(), get_parameters()]]
_ = run_test_sweep(model_generator(param_list))

# %%
model, args = next(model_generator(param_list))
model = train_epoch(model, args, epochs=10, root_dir='experiments/artr', data_dir='datasets/coco', metric_step=1200, checkpoint_step=2000,
                    results_step=21000)

# %%
