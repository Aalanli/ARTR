# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def ConvMixer(dim, depth, out_size, kernel_size=9, patch_size=7):
    return nn.Sequential(
        nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((out_size, out_size)),
        nn.Flatten(),
    )


class AttnMap(nn.Module):
    def __init__(self, channels, out_channels=1, kernel_size=1):
        super().__init__()
        self.norm = nn.InstanceNorm2d(1)
        self.feats = nn.Parameter(torch.empty(out_channels, channels, kernel_size, kernel_size))
        nn.init.normal_(self.feats)
        self.pad = (kernel_size - 1) // 2
    
    def forward(self, x):
        # x.shape = [B, N, X1, X2]
        x = x.softmax(1).log()
        feats = self.feats.softmax(1)
        attn_map = F.conv2d(x, feats, padding=self.pad)
        return self.norm(attn_map).squeeze()


class ExtractCoord(nn.Module):
    def __init__(self, dim, depth, out_size, out_proj, kernel_size, patch_size):
        super().__init__()
        self.in_proj = ConvMixer(dim, depth, out_size, kernel_size, patch_size)
        self.lin1 = nn.Linear(out_size ** 2, out_proj)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(out_proj, out_proj)
    
    def forward(self, x):
        x = self.in_proj(x)
        x = self.lin2(self.relu(self.lin1(x)) + x)
        return x.transpose(-1, -2)

