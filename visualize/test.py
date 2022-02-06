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

a = torch.rand(4, 3, 120)
print(a.min(-1).values.shape)