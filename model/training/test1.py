# %%
import torch
import torch.nn.functional as F

a = torch.rand(3, 4, 512, 128)
b = torch.rand(3, 4, 128, 125)
print((a @ b).shape)
