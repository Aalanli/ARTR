# %%
import torch
import model.modelV4.fast_conv as fs

w = torch.rand(512 * 4, 128, 3, 3, dtype=torch.half, requires_grad=True).cuda()
im = torch.rand(1, 4 * 128, 64, 64, dtype=torch.half, requires_grad=True).cuda()

a = fs.conv2d(im, w, padding=1, groups=4)
a.sum().backward()