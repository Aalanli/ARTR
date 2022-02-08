# %%
import torch
import torch.nn.functional as F


im = torch.rand(1, 128, 64, 64, dtype=torch.float64)
k = torch.rand(1, 128, 5, 5, dtype=torch.float64)

def feature_similarity1(im, features):  # im.shape = [B, C, X1, X2], features.shape = [B, C, Y1, Y2]
    im = im.softmax(1)
    features = features.softmax(1).log()
    features = -F.adaptive_avg_pool2d(features, (1, 1))
    return F.conv2d(im, features)

def feature_similarity2(im, features):  # im.shape = [B, C, X1, X2], features.shape = [B, C, Y1, Y2]
    im = im.softmax(1)
    features = features.softmax(1).log()
    features = -features.flatten(2).sum(2)[:, :, None, None]
    return features * im

a = feature_similarity1(im, k)
b = feature_similarity2(im, k)
print(torch.allclose(a, b))

# %%
import torch
import torch.nn as nn
from torch.nn import functional as F

w = torch.rand(4, 512, 128, 3, 3)
a = torch.rand(4, 128, 64, 64)
a1 = a.reshape(1, 4 * 128, 64, 64)
w1 = w.reshape(4 * 512, 128, 3, 3)

x1 = F.conv2d(a1, w1, groups=4)
x2 = torch.cat([F.conv2d(a[i:i+1], w[i]) for i in range(4)], dim=0)

print(x1.shape)
print(x2.shape)
# %%
torch.allclose(x1.reshape(4, 512, 62, 62), x2)