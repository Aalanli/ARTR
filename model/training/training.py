# %%
import sys
sys.path[0] = '/home/allan/Programs/ARTR'

from utils.loss import HungarianMatcher, SetCriterion

from model.scaffold.backbone import variants as backbone_variants
from model.scaffold.transformer import variants as transformer_variants
from model.scaffold.artr import ARTR

from model.training.parameters import get_parameters

from model.training.sweep import run_sweep


objects = []
arg_list = []

for i in range(len(backbone_variants)):
    new_args = get_parameters()
    new_args.name = backbone_variants[i].__name__ + '_single'
    arg_list.append(new_args)

    object_dict = {'HungarianMatcher': HungarianMatcher, 
                   'SetCriterion': SetCriterion, 
                   'artr': ARTR,
                   'backbone': backbone_variants[i],
                   'transformer': transformer_variants[0]}
    objects.append(object_dict)


run_sweep(objects, arg_list, epochs=5, root_dir='experiments/artr', data_dir='datasets/coco', metric_step=1200, checkpoint_step=2000,
          results_step=14000)