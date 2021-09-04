# %%
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from model.transformer import EncoderLayer, DecoderLayer, PositionEmbeddingSine, PositionEmbeddingSineMaskless, FixedPositionalEmbedding
from model.resnet_parts import Backbone
from utils.misc import make_equal, make_equal1D
from utils.ops import nested_tensor_from_tensor_list

class Transformer(nn.Module):
    def __init__(
        self, 
        d_model, 
        heads, 
        proj_forward, 
        enc_q_layers,
        enc_t_layers,
        dec_layers,
        activation=F.relu,
        dropout=0.1, 
        bias=None):
        super().__init__()
        self.d_model = d_model
        args = [d_model, heads, proj_forward, activation, dropout, bias]
        self.enc_q = nn.ModuleList([EncoderLayer(*args) for _ in range(enc_q_layers)])
        self.enc_t = nn.ModuleList([EncoderLayer(*args) for _ in range(enc_t_layers)])

        self.dec_kv = nn.ModuleList([DecoderLayer(*args) for _ in range(dec_layers)])
        self.dec_final = nn.ModuleList([DecoderLayer(*args) for _ in range(dec_layers)])
    
    def forward(self, im_query, im_target, query_embed, mask_q=None, mask_t=None, pos_q=None, pos_t=None):
        key_padding_mask_q = mask_q[:, None, None, :].bool()
        key_padding_mask_t = mask_t[:, None, None, :].bool()
        for layer in self.enc_q:
            im_query = layer(im_query, key_padding_mask_q, pos_q)
        for layer in self.enc_t:
            im_target = layer(im_target, key_padding_mask_t, pos_t)

        x = torch.zeros_like(query_embed)  # flow-through variable
        kv = torch.zeros_like(im_query)    # flow-through variable
        for kv_layer, out_layer in zip(self.dec_kv, self.dec_final):
            kv = kv_layer(kv, im_query, im_target, mask_q=key_padding_mask_q, mask_k=key_padding_mask_t, pos_q=pos_q, pos_k=pos_t)
            x = out_layer(x, query_embed, kv, mask_k=key_padding_mask_q, pos_k=pos_q)
        return x


class TransformerV2(Transformer):
    """Cross attends im_target with im_query, the former the query and latter the key"""
    def forward(self, im_query, im_target, query_embed, mask_q=None, mask_t=None, pos_q=None, pos_t=None):
        key_padding_mask_q = mask_q[:, None, None, :].bool()
        key_padding_mask_t = mask_t[:, None, None, :].bool()
        for layer in self.enc_q:
            im_query = layer(im_query, key_padding_mask_q, pos_q)
        for layer in self.enc_t:
            im_target = layer(im_target, key_padding_mask_t, pos_t)

        x = torch.zeros_like(query_embed)  # flow-through variable
        kv = torch.zeros_like(im_target)    # flow-through variable
        for kv_layer, out_layer in zip(self.dec_kv, self.dec_final):
            kv = kv_layer(kv, im_target, im_query, mask_q=key_padding_mask_t, mask_k=key_padding_mask_q, pos_q=pos_t, pos_k=pos_q)
            x = out_layer(x, query_embed, kv, mask_k=key_padding_mask_t, pos_k=pos_t)
        return x


class TransformerV3(Transformer):
    def forward(self, im_query, im_target, query_embed, mask_q=None, mask_t=None, pos_q=None, pos_t=None):
        key_padding_mask_q = mask_q[:, None, None, :].bool()
        key_padding_mask_t = mask_t[:, None, None, :].bool()
        for layer in self.enc_q:
            im_query = layer(im_query, key_padding_mask_q, pos_q)
        for layer in self.enc_t:
            im_target = layer(im_target, key_padding_mask_t, pos_t)

        x = torch.zeros_like(query_embed)  # flow-through variable
        kv = torch.zeros_like(im_target)    # flow-through variable
        for kv_layer in self.dec_kv:
            kv = kv_layer(kv, im_target, im_query, mask_q=key_padding_mask_t, mask_k=key_padding_mask_q, pos_q=pos_t, pos_k=pos_q)
        for out_layer in self.dec_final:
            x = out_layer(x, query_embed, kv, mask_k=key_padding_mask_t, pos_k=pos_t)
        return x


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ARTR(nn.Module):
    def __init__(
        self, 
        backbone_args:    Dict = {'name': 'resnet50', 'train_backbone': True, 'return_interm_layers': False, 'dilation': False}, 
        pos_embed_args:   Dict = {'temperature': 10000, 'normalize': False, 'scale': None}, 
        transformer_args: Dict = {'heads': 8, 'proj_forward': 1024, 'enc_q_layers': 3,
                                  'enc_t_layers': 3, 'dec_layers': 3, 'activation': 'relu', 'dropout': 0.1, 'bias': None},
        d_model:          int = 256,
        num_queries:      int  = 50
        ):
        super().__init__()
        # default args
        backbone_args_ = {'name': 'resnet50', 'train_backbone': True, 'return_interm_layers': False, 'dilation': False}
        pos_embed_args_ = {'num_pos_feats': d_model // 2, 'temperature': 10000, 'normalize': False, 'scale': None}
        transformer_args_ = {'d_model': d_model, 'heads': 8, 'proj_forward': 1024, 'enc_q_layers': 3,
                             'enc_t_layers': 3, 'dec_layers': 3, 'activation': 'relu', 'dropout': 0.1, 'bias': None}
        
        backbone_args_.update(backbone_args)
        self.backbone_args = backbone_args_
        pos_embed_args_.update(pos_embed_args)
        self.pos_embed_args = pos_embed_args_
        transformer_args_.update(transformer_args)
        self.transformer_args = transformer_args_
        self.config_ = [self.backbone_args.copy(), self.pos_embed_args.copy(), self.transformer_args.copy(), num_queries]
        self.transformer_args['activation'] = getattr(F, transformer_args_['activation'])

        self.backbone = Backbone(**self.backbone_args)
        self.pos_embed = PositionEmbeddingSineMaskless(**self.pos_embed_args)
        self.transformer = Transformer(**self.transformer_args)
        d_model = self.transformer.d_model

        self.class_embed = nn.Linear(d_model, 2)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)
        self.query_embed = nn.Parameter(torch.Tensor(num_queries, d_model))
        torch.nn.init.normal_(self.query_embed)        
        self.input_proj = nn.Conv2d(self.backbone.num_channels, d_model, kernel_size=1)
    
    def make_equal_masks(self, ims: List[torch.Tensor]):
        """Accepts a list of flattened images and produces binary masks for attention"""
        shapes = [i.shape[-1] for i in ims]
        batch = len(ims)
        max_len = max(shapes)
        mask = torch.zeros(batch, max_len, dtype=ims[0].dtype, device=ims[0].device)
        for i, s in enumerate(shapes):
            mask[i, s:].fill_(1)
        return mask.bool()
    
    def make_pos_embed(self, x: List[torch.Tensor]):
        """Makes a 2D positional embedding from a 2D image"""
        # [embed_dim, x * y]
        return self.pos_embed(x).permute(2, 0, 1).flatten(1)

    def make_equal_ims(self, ims: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Unrolls resnet features and calculates masks and positional embed,
        vectorizes all.
        """
        pos_embed = []
        for i in range(len(ims)):
            ims[i] = self.backbone(ims[i].unsqueeze_(0))['0']
            ims[i] = self.input_proj(ims[i]).squeeze(0)
            pos_embed.append(self.make_pos_embed(ims[i]))
            ims[i] = ims[i].flatten(1)
        pos_embed = make_equal1D(pos_embed)  # no grad
        masks = self.make_equal_masks(ims)   # no grad
        ims = make_equal(*ims)               # with grad
        return ims, masks, pos_embed
    
    def make_equal_queries(self, qrs: List[List[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Same as make_equal_ims method, but unrolls multiple queries into
        one for each sample in a batch
        """
        pos_embed = []
        for j in range(len(qrs)):
            pos_embed_ = []
            for i in range(len(qrs[j])):
                qrs[j][i] = self.backbone(qrs[j][i].unsqueeze_(0))['0']
                qrs[j][i] = self.input_proj(qrs[j][i]).squeeze(0)                
                pos_embed_.append(self.make_pos_embed(qrs[j][i]))
                qrs[j][i] = qrs[j][i].flatten(1)
            # [embed_dim, n * x * y]
            pos_embed.append(torch.cat(pos_embed_, dim=-1))
            qrs[j] = torch.cat(qrs[j], dim=-1)
        pos_embed = make_equal1D(pos_embed)
        masks = self.make_equal_masks(qrs)
        qrs = make_equal(*qrs)
        return qrs, masks, pos_embed
    
    def backbone_processing(self, tar_im: List[torch.Tensor], query_im: List[List[torch.Tensor]]) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Takes unprocessed images and queries and returns feature vectors of shape [batch, l, dim] for the transformer
        returns: (tar_im features, tar_im mask, tar_im pos_emb), (qr features, qr mask, qr pos_emb)
        """
        t_features, t_mask, t_pos = self.make_equal_ims(tar_im)
        q_features, q_mask, q_pos = self.make_equal_queries(query_im)
        t_features = t_features.transpose(-1, -2)
        q_features = q_features.transpose(-1, -2)
        t_pos.transpose_(-1, -2)
        q_pos.transpose_(-1, -2)
        return (t_features, t_mask, t_pos), (q_features, q_mask, q_pos)

    def forward(self, tar_im: List[torch.Tensor], query_im: List[List[torch.Tensor]]):
        """
        len(tar_im) = batch_size, len(query_im) = batch_size, len(query_im[i]) = # of query images for tar_im[i]
        dist.shape = [batch_size]; a scaler difference between the bounding boxes and query images 
        """
        # backbone portion
        (t_features, t_mask, t_pos), (q_features, q_mask, q_pos) = self.backbone_processing(tar_im, query_im)
         
        # transformer portion
        batch_size = t_pos.shape[0]
        query_embed = self.query_embed.unsqueeze(0).repeat(batch_size, 1, 1)
        out = self.transformer(q_features, t_features, query_embed, q_mask, t_mask, q_pos, t_pos)
        output_class = self.class_embed(out)
        output_bbox = self.bbox_embed(out).sigmoid()
        return {'pred_logits': output_class, 'pred_boxes': output_bbox}
    
    @property
    def config(self):
        return self.config_


class ARTRV1(ARTR):
    """
    Variant 1 of ARTR, treats queries and target images before backbone,
    same pos embed for every query instance 
    """
    def __init__(self, backbone_args: Dict, pos_embed_args: Dict, transformer_args: Dict, d_model: int, num_queries: int):
        super().__init__(backbone_args=backbone_args, pos_embed_args=pos_embed_args, transformer_args=transformer_args, d_model=d_model, num_queries=num_queries)
        self.pos_embed = PositionEmbeddingSine(**self.pos_embed_args)

    def make_equal_ims(self, ims: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ims, mask = nested_tensor_from_tensor_list(ims)
        ims, mask = self.backbone(ims, mask)["0"]
        pos_emb = self.pos_embed(mask)
        ims = self.input_proj(ims)
        # [batch, n, d_model], [batch, n], [batch, n, d_model]
        return ims.flatten(2).transpose(-1, -2), mask.flatten(1), pos_emb.flatten(1, 2)
    
    def make_equal_queries(self, qrs: List[List[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, masks, pos_emb = [], [], []
        for i in range(len(qrs)):
            f, m, p = self.make_equal_ims(qrs[i])
            f = f.flatten(0, 1)
            m = m.flatten(0)
            p = p.flatten(0, 1)
            features.append(f); masks.append(m); pos_emb.append(p)
        
        max_len = max([f.shape[0] for f in features])
        for i in range(len(features)):
            pad_amount = max_len - features[i].shape[0]
            features[i] = F.pad(features[i], pad=(0, 0, 0, pad_amount), value=0)
            masks[i] = F.pad(masks[i], pad=(0, pad_amount), value=1).bool()
            pos_emb[i] = F.pad(pos_emb[i], pad=(0, 0, 0, pad_amount), value=0)
        features, masks, pos_emb = map(lambda x: torch.stack(x, dim=0), (features, masks, pos_emb))
        return features, masks, pos_emb
       
    def backbone_processing(self, tar_im: List[torch.Tensor], query_im: List[List[torch.Tensor]]) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return self.make_equal_ims(tar_im), self.make_equal_queries(query_im)


class ARTRV2(ARTRV1):
    """Concat each query along either the x plane"""
    def make_equal_queries(self, qrs: List[List[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for i in range(len(qrs)):
            y_dims = [im.shape[-2] for im in qrs[i]]
            x_dims = [im.shape[-1] for im in qrs[i]]
            y_dim = max(y_dims)
            x_dim = sum(x_dims)
            image = torch.zeros(3, y_dim, x_dim, device=qrs[i][0].device)
            x_pos = 0
            for j in range(len(x_dims)):
                image[:, :y_dims[j], x_pos:x_dims[j] + x_pos].copy_(qrs[i][j])
                x_pos += x_dims[j]
            qrs[i] = image
        return self.make_equal_ims(qrs)
            

class ARTRV3(ARTR):
    """Implements 1D positional embedding"""
    def __init__(self, backbone_args: Dict, pos_embed_args: Dict, transformer_args: Dict, d_model: int, num_queries: int):
        super().__init__(backbone_args=backbone_args, pos_embed_args=pos_embed_args, transformer_args=transformer_args, d_model=d_model, num_queries=num_queries)
        self.pos_embed = FixedPositionalEmbedding(self.transformer.d_model)
    
    def make_equal_ims(self, ims: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Unrolls resnet features and calculates masks and positional embed,
        vectorizes all.
        """
        for i in range(len(ims)):
            ims[i] = self.backbone(ims[i].unsqueeze_(0))['0']
            ims[i] = self.input_proj(ims[i]).squeeze(0).flatten(1)

        masks = self.make_equal_masks(ims)   # no grad
        ims = make_equal(*ims)               # with grad
        ims = ims.transpose(-1, -2)
        pos_embed = self.pos_embed(ims)
        return ims, masks, pos_embed
    
    def make_equal_queries(self, qrs: List[List[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Same as make_equal_ims method, but unrolls multiple queries into
        one for each sample in a batch
        """
        for j in range(len(qrs)):
            for i in range(len(qrs[j])):
                qrs[j][i] = self.backbone(qrs[j][i].unsqueeze_(0))['0']
                qrs[j][i] = self.input_proj(qrs[j][i]).squeeze(0).flatten(1)
            # [embed_dim, n * x * y]
            qrs[j] = torch.cat(qrs[j], dim=-1)
        masks = self.make_equal_masks(qrs)
        qrs = make_equal(*qrs)
        qrs = qrs.transpose(-1, -2)
        pos_embed = self.pos_embed(qrs)
        return qrs, masks, pos_embed

    def backbone_processing(self, tar_im: List[torch.Tensor], query_im: List[List[torch.Tensor]]) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return self.make_equal_ims(tar_im), self.make_equal_queries(query_im)
    

if __name__ == "__main__":
    from utils.misc import gen_test_data
    import copy
    ims = gen_test_data(256, 64, 4)
    qrs = [gen_test_data(128, 64, 2) for i in range(4)]
    artr2 = ARTRV3({}, {}, {}, 128, 50)


    a = artr2(ims, qrs)

