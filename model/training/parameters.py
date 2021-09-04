from utils.misc import EasyDict

def get_parameters():
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
    args.num_queries = 5

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
    return args
