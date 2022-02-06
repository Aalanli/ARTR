# %%
import os

import torch
from model.modelV4.artr import ARTR
from utils.loss import simple_box_loss, HungarianMatcher, SetCriterion

import data.dataset as data
import utils.misc as misc
from model.trainer import TrainerWandb

def get_parameters():
    args = misc.EasyDict()

    args.batch_size = 4
    args.mixed_precision = True
    args.fix_label = None

    args.min_qr_area = 32 ** 2

    args.cost_class = 1
    args.cost_bbox = 5
    args.cost_giou = 2
    args.cost_eof = 0.05
    args.losses = ['boxes', 'labels', 'cardinality']

    args.lr = 1e-4
    args.weight_decay = 1e-4
    args.lr_drop = 200
    args.weight_dict = {'loss_giou': 2, 'loss_bbox': 5, 'loss_ce': 1}
    args.name = 'ModelV4-tst'
    return args

args = misc.EasyDict(
    n_pred      = 50,
    x_dim       = 7,
    splits      = 4,
    im_dim      = 256,
    im_depth    = 15,
    qr_dims     = [64, 128, 256, 512],
    qr_depths   = [3, 3, 6, 3],
    mlp_depth   = 3,
    mlp_feats   = 256,
    min_qr_size = 64,
    max_qr_size = 96
)

model = ARTR(**args).cuda()
args.update(get_parameters())
matcher = HungarianMatcher(args.cost_class, args.cost_bbox, args.cost_giou)
criterion = SetCriterion(1, matcher, args.cost_eof, args.losses).cuda()

root_dir = 'experiments/artr'

metric_step = 1200
checkpoint_step = 2000
results_step = 20000
qr = True

epochs = 20
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
    query_process = data.SingleQuery(args.min_qr_area)

dataset = data.CocoDetection(root + proc + '2017', root + f'annotations/instances_{proc}2017.json', data.img_transforms('train'), query_process, args.fix_label)
train_set = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=True, num_workers=8, collate_fn=data.collate_fn)        
# %%
trainer.train_epochs(epochs, train_set)



# %%
#########################################
## Testing qualitatively
#########################################
if __name__ == "__main__":
    from utils.misc import visualize_model_output

    #query_process = GetQuery(query_transforms(), 3, 2, stretch_limit=0.7, min_size=32,
    #                                query_pool=root + proc + "2017_query_pool", prob_pool=0.3, max_queries=10)
    cpu_model = model.cpu()
# %%
    im, qr, y = next(iter(train_set))
    pred = cpu_model(im, qr)
    pred_box = pred['pred_boxes'][0].detach()
    true_box = y[0]['boxes']
    visualize_model_output(im[0], qr[0], true_box, pred_box)

