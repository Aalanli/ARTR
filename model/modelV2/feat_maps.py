# %%
from typing import Any, Union, Optional, Callable, List

from torch import nn
from torch import Tensor
import torch

from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck, load_state_dict_from_url, model_urls

from utils.misc import make_equal
from utils.ops import make_equal_3D


class DepthWiseConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.dim_proj = nn.Conv3d(in_channels, in_channels, 3, stride, padding=[1, 1, 1], groups=in_channels)
        self.channel_proj = nn.Conv3d(in_channels, out_channels, 1)
    
    def forward(self, x):
        return self.channel_proj(self.dim_proj(x))


class QueryResnet(ResNet):
    def __init__(
        self, 
        block, 
        layers: List[int], 
        num_classes: int = 1000, 
        zero_init_residual: bool = False, 
        groups: int = 1, 
        width_per_group: int = 64, 
        replace_stride_with_dilation: Optional[List[bool]] = None, 
        norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__(
            block, 
            layers, 
            num_classes=num_classes, 
            zero_init_residual=zero_init_residual, 
            groups=groups, width_per_group=width_per_group, 
            replace_stride_with_dilation=replace_stride_with_dilation, 
            norm_layer=norm_layer)
        
        del self.avgpool, self.fc
        self.conv3d_1 = nn.Identity() #DepthWiseConv3D(64, 64)
        self.conv3d_2 = nn.Identity() #DepthWiseConv3D(128, 128)
        self.conv3d_3 = nn.Identity() #DepthWiseConv3D(256, 256)
        self.conv3d_4 = nn.Identity() #DepthWiseConv3D(512, 512)
        self.feature_dims = [64, 128, 256, 512]
    
    def flatten(self, x):
        B, N, C, X1, X2 = x.shape
        return x.flatten(0, 1), (B, N)
    
    def inflate(self, x, B, N):
        BN, C, X1, X2 = x.shape
        return x.reshape(B, N, C, X1, X2)
    
    def _forward_impl(self, x: Tensor) -> List[Tensor]:
        x, (B, N) = self.flatten(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outputs = []
        x = self.layer1(x)
        x = self.inflate(x, B, N).transpose(2, 1)
        x = self.conv3d_1(x)        
        outputs.append(x)

        x, (B, N) = self.flatten(x.transpose(2, 1))
        x = self.layer2(x)
        x = self.inflate(x, B, N).transpose(2, 1)
        x = self.conv3d_2(x)        
        outputs.append(x)

        x, (B, N) = self.flatten(x.transpose(2, 1))
        x = self.layer3(x)
        x = self.inflate(x, B, N).transpose(2, 1)
        x = self.conv3d_3(x)
        outputs.append(x)

        x, (B, N) = self.flatten(x.transpose(2, 1))
        x = self.layer4(x)
        x = self.inflate(x, B, N).transpose(2, 1)
        x = self.conv3d_4(x)
        outputs.append(x)

        return outputs


def _resnet(
    arch: str,
    block: Union[BasicBlock, Bottleneck],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs
) -> QueryResnet:
    model = QueryResnet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


class QueryBackbone(nn.Module):
    def __init__(self, arch='resnet34', pretrained=False):
        super().__init__()
        if arch == 'resnet34':
            self.resnet = resnet34(pretrained)
        elif arch == 'resnet18':
            self.resnet = resnet18(pretrained)
        else:
            raise NameError(f'architecture {arch} not in supported choices [resnet34, resnet18]')
    
    def forward(self, x: List[List[Tensor]]) -> List[Tensor]:
        features = [[] for i in range(4)]

        for sample in x:
            # qim.shape = [N, 3, X1, X2]
            qim = make_equal_3D(sample)
            qr = self.resnet(qim.unsqueeze(0))
            for i in range(4):
                features[i].append(qr[i].squeeze(0).flatten(1))
        for k in range(4):
            features[k] = make_equal(*features[k])

        return features

"""# %%
res = resnet34()
a = torch.rand(4, 1, 3, 256, 256)
print([i.shape for i in res(a)])"""

