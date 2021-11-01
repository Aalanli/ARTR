# %%
from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor

from model.transformer import DecoderLayerV2


class DynamicConv2D(nn.Module):
    def __init__(self, attn, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.attn = attn
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(1, kernel_size * kernel_size, in_channels))
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, x, features, mask=None, pos=None, alibi=None):
        batch, c, x1, y1 = x.shape
        w = self.weight.repeat(features.shape[0], 1, 1)
        w = self.attn(w, features, mask_k=mask, pos=pos, alibi_mask=alibi)
        # w.shape = [batch, kernel_size * kernel_size, in_channels]
        # x.shape = [batch, channels, x, y]
        w = w.transpose(-1, -2).reshape(batch * self.in_channels, 1, self.kernel_size, self.kernel_size)
        x = x.view(1, batch * c, x1, y1)
        x = F.conv2d(x, w, self.bias, self.stride, self.padding, groups=batch * self.in_channels)
        x = x.view(batch, c, x1, y1)
        return self.conv(x)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden,
        out_channels,
        stride: int = 1,
        dilation: int = 1,
        norm_layer = None,
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_channels, hidden)
        self.bn1 = norm_layer(hidden)
        self.conv2 = conv3x3(hidden, hidden, stride, 1, dilation)
        self.bn2 = norm_layer(hidden)
        self.conv3 = conv1x1(hidden, out_channels)
        self.bn3 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        
        if out_channels != in_channels or stride != 1:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
                norm_layer(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DynamicBlock(nn.Module):
    def __init__(
        self,
        attn,
        in_channels,
        hidden,
        out_channels,
        stride=1,
        norm_layer = None,
    ) -> None:
        super(DynamicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_channels, hidden)
        self.bn1 = norm_layer(hidden)
        self.conv2 = DynamicConv2D(attn, hidden, hidden)
        self.bn2 = norm_layer(hidden)
        self.conv3 = conv3x3(hidden, hidden, stride)
        self.bn3 = norm_layer(hidden)
        self.conv4 = conv1x1(hidden, out_channels)
        self.bn4 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if out_channels != in_channels or stride != 1:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
                norm_layer(out_channels)
            )
        else:
            self.downsample = None


    def forward(self, x: Tensor, feat: Tensor, mask: Tensor = None, pos: Tensor = None) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out, feat, mask, pos)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.bn4(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out


class DynamicResnetLayer(nn.Module):
    def __init__(self, layer_map, in_channel, hidden, expansion=4,
                 stride=1, transformer_kwargs=None, dilation=1, norm_layer=None):
        super().__init__()
        previous_dilation = dilation
        if dilation > 1:
            dilation *= stride
            stride = 1
        
        if transformer_kwargs is None:
            transformer_kwargs = {'nhead': 8, 'dim_feedforward': 1024, 'dropout': 0.1, 'activation': 'relu', 'normalize_before': False}
        if 1 in layer_map:
            self.attn = DecoderLayerV2(hidden, **transformer_kwargs)

        self.layer_map = layer_map
        self.layers = nn.ModuleList()
        if layer_map[0] == 1:
            self.layers.append(DynamicBlock(self.attn, in_channel, hidden, hidden * expansion, stride=stride, norm_layer=norm_layer))
        else:
            self.layers.append(Bottleneck(in_channel, hidden, hidden * expansion, stride=stride,
                                          dilation=previous_dilation, norm_layer=norm_layer))
        for i in layer_map[1:]:
            if i == 0:
                self.layers.append(Bottleneck(hidden * expansion, hidden, hidden * expansion, dilation=dilation, 
                                            norm_layer=norm_layer))
            else:
                self.layers.append(DynamicBlock(self.attn, hidden * expansion, hidden, hidden * expansion, norm_layer=norm_layer))
        
    def forward(self, x, feat_map, mask=None, pos=None):
        for i, b in enumerate(self.layer_map):
            if b == 0:
                x = self.layers[i](x)
            else:
                x = self.layers[i](x, feat_map, mask, pos)
        return x


class SpatialFeatureExtractor(nn.Module):
    def __init__(
        self,
        layer_maps: List[List[int]],
        transformer_kwargs = None,
        dilation = None,
        norm_layer = None,
        zero_init_residual=False
    ) -> None:
        super(SpatialFeatureExtractor, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        if dilation is None:
            dilation = [1, 1, 1]
        elif not isinstance(dilation, (list, tuple)):
            dilation = [2, 4, 8]
    
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = DynamicResnetLayer(layer_maps[0], 64, 64, transformer_kwargs=transformer_kwargs, norm_layer=norm_layer)
        self.layer2 = DynamicResnetLayer(layer_maps[1], 256, 128, stride=2, transformer_kwargs=transformer_kwargs, dilation=dilation[0], norm_layer=norm_layer)
        self.layer3 = DynamicResnetLayer(layer_maps[2], 512, 256, stride=2, transformer_kwargs=transformer_kwargs, dilation=dilation[1], norm_layer=norm_layer)
        self.layer4 = DynamicResnetLayer(layer_maps[3], 1024, 512, stride=2, transformer_kwargs=transformer_kwargs, dilation=dilation[2], norm_layer=norm_layer)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]


    def forward(self, x: Tensor, feat_maps: List[Tensor], masks: List[Tensor] = None, pos: List[Tensor] = None) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if masks is None:
            masks = [None] * 4
        if pos is None:
            pos = [None] * 4
        x = self.layer1(x, feat_maps[0], masks[0], pos[0])
        x = self.layer2(x, feat_maps[1], masks[1], pos[1])
        x = self.layer3(x, feat_maps[2], masks[2], pos[2])
        x = self.layer4(x, feat_maps[3], masks[3], pos[3])

        return x

