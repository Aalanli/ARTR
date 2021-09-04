# %%
from typing import List, Tuple

import torch
import torch.nn as nn

import model.pos_emb as pos
from model.resnet_parts import Backbone
from utils.misc import make_equal, make_equal1D
from utils.ops import nested_tensor_from_tensor_list



class BackboneProcessV1(nn.Module):
    """Memory efficient backbone, applies backbone before concatenation, with 2D positional embedding"""
    def __init__(self, name='resnet50', train_backbone=True, return_interm_layers=False, dilation=False,
                 n_pos_feats=128, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.backbone = Backbone(name, train_backbone, return_interm_layers, dilation)
        self.pos_embed = pos.PositionEmbeddingSineMaskless(n_pos_feats, temperature, normalize, scale)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, n_pos_feats * 2, kernel_size=1)

    def make_equal_masks(self, ims: List[torch.Tensor]):
        """Accepts a list of flattened images and produces binary masks for attention"""
        shapes = [i.shape[-1] for i in ims]
        batch = len(ims)
        max_len = max(shapes)
        mask = torch.zeros(batch, max_len, dtype=ims[0].dtype, device=ims[0].device)
        for i, s in enumerate(shapes):
            mask[i, s:].fill_(1)
        return mask.bool()
    
    def make_pos_embed(self, x: torch.Tensor):
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
    
    def forward(self, tar_im: List[torch.Tensor], query_im: List[List[torch.Tensor]]) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
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


class BackboneProcessV2(BackboneProcessV1):
    """
    Concat each query onto a singular image, thereby making positional embedding
    different for each query in a batch. Least memory efficient.
    """
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


class BackboneProcessV3(nn.Module):
    """
    Memory inefficient variant of backbone, concatenates images before backbone 
    treatment, with 2D positional embedding, same for all queries in a batch
    """
    def __init__(self, name='resnet50', train_backbone=True, return_interm_layers=False, dilation=False,
                 n_pos_feats=128, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.backbone = Backbone(name, train_backbone, return_interm_layers, dilation)
        self.pos_embed = pos.PositionEmbeddingSine(n_pos_feats, temperature, normalize, scale)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, n_pos_feats * 2, kernel_size=1)
    
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
            features[i] = nn.functional.pad(features[i], pad=(0, 0, 0, pad_amount), value=0)
            masks[i] = nn.functional.pad(masks[i], pad=(0, pad_amount), value=1).bool()
            pos_emb[i] = nn.functional.pad(pos_emb[i], pad=(0, 0, 0, pad_amount), value=0)
        features, masks, pos_emb = map(lambda x: torch.stack(x, dim=0), (features, masks, pos_emb))
        return features, masks, pos_emb
       
    def forward(self, tar_im: List[torch.Tensor], query_im: List[List[torch.Tensor]]) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return self.make_equal_ims(tar_im), self.make_equal_queries(query_im)


class BackboneProcessV4(BackboneProcessV1):
    def __init__(self, name='resnet50', train_backbone=True, return_interm_layers=False, dilation=False,
                 n_pos_feats=128, *args):
        super().__init__()
        self.backbone = Backbone(name, train_backbone, return_interm_layers, dilation)
        self.pos_embed = pos.FixedPositionalEmbedding(n_pos_feats * 2)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, n_pos_feats * 2, kernel_size=1)
    
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

    def forward(self, tar_im: List[torch.Tensor], query_im: List[List[torch.Tensor]]) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return self.make_equal_ims(tar_im), self.make_equal_queries(query_im)

variants = [BackboneProcessV1, BackboneProcessV2, BackboneProcessV3, BackboneProcessV4]

if __name__ == "__main__":
    from utils.misc import gen_test_data
    import copy
    ims = gen_test_data(256, 64, 4)
    qrs = [gen_test_data(128, 64, 2) for i in range(4)]

    for h in variants:
        ims_ = copy.deepcopy(ims)
        qrs_ = copy.deepcopy(qrs)

        b = h()
        (a, b, c), (e, d, f) = b(ims_, qrs_)

        print(a.shape, b.shape, c.shape)
        print(e.shape, d.shape, f.shape)
