# %%
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_padding(kernel_size, stride):
    """
    Ensures same padding for stride of 1
    otherwise ensures sufficient padding that no edges are skipped
    """
    greatest_remainder = stride - 1
    pad = max(greatest_remainder, (kernel_size - 1))
    return math.ceil(pad / 2)


def format_arguments(x):
        a = [None, 2, 3]
        for i in range(len(x)): a[i] = x[i]
        return a


def pad(x, h=0, w=0):
    h_r = h % 2
    h = h // 2
    w_r = w % 2
    w = w // 2
    paddings = (w, w + w_r, h, h + h_r)
    return F.pad(x, paddings)


def cross_entropy(input, target, reduction='mean'):
    """continous cross entropy"""
    if reduction == 'mean':
        return torch.mean(torch.sum(-target * F.log_softmax(input, dim=-1), dim=1))
    else:
        return torch.sum(torch.sum(-target * F.log_softmax(input, dim=-1), dim=1))


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResizeConv3x3(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.conv = conv3x3(in_planes, out_planes)
    def forward(self, x, scale_factor=None, shape=None):
        if shape is None:
            scale_factor = 1
        x = F.interpolate(x, size=shape, scale_factor=scale_factor)
        return self.conv(x)


class ResizeConv1x1(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.conv = conv1x1(in_planes, out_planes)
    def forward(self, x, scale_factor=None, shape=None):
        if shape is None:
            scale_factor = 1
        x = F.interpolate(x, size=shape, scale_factor=scale_factor)
        return self.conv(x)


class EncoderBlock(nn.Module):
    """
    ResNet block, copied from
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L35
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DecoderBlock(nn.Module):
    """
    ResNet block, but convs replaced with resize convs, and channel increase is in
    second conv, not first
    """

    expansion = 1

    def __init__(self, inplanes, planes, upsample=False):
        super().__init__()
        self.conv1 = ResizeConv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = ResizeConv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        if self.upsample:
            self.residual = ResizeConv1x1(inplanes, planes)
            self.bn3 = nn.BatchNorm2d(planes)

    def forward(self, x, shape=None):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, shape=shape)
        out = self.bn2(out)

        if self.upsample:
            identity = self.residual(x, shape=shape)
            identity = self.bn3(identity)

        out += identity
        out = self.relu(out)

        return out


class ResNetEncoder(nn.Module):

    def __init__(self, block, layers, first_conv=False, maxpool1=False):
        super().__init__()

        self.inplanes = 64
        self.first_conv = first_conv
        self.maxpool1 = maxpool1

        if self.first_conv:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        if self.maxpool1:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, blocks, stride=1):
        if blocks < 1:
            return lambda x: x
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        shapes = [x.shape[-2:]]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        shapes.append(x.shape[-2:])
        x = self.layer2(x)
        shapes.append(x.shape[-2:])

        x = self.layer3(x)
        shapes.append(x.shape[-2:])
        x = self.layer4(x)
        shapes.append(x.shape[-2:])
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x, shapes


class ResNetDecoder(nn.Module):
    """
    Resnet in reverse order
    """

    def __init__(self, block, layers, latent_dim, sample_dim=512, first_conv=False, maxpool1=False):
        super().__init__()
        self.inplanes = sample_dim
        self.first_conv = first_conv
        self.maxpool1 = maxpool1

        self.linear = nn.Linear(latent_dim, self.inplanes * 4 * 4)

        self.layer1 = self._make_layer(block, self.inplanes, 256, layers[0])
        self.layer2 = self._make_layer(block, 256, 128, layers[1])
        self.layer3 = self._make_layer(block, 128, 64, layers[2])

        if self.maxpool1:
            self.layer4 = self._make_layer(block, 64, 64, layers[3])
            self.upscale_factor *= 2
        else:
            self.layer4 = self._make_layer(block, 64, 64, layers[3])

        self.conv1 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False)
    
    def _make_layer(self, block, in_dim, out_dim, layers):
        if layers < 1:
            return [lambda x: x]  # returns identity function
        layer = nn.ModuleList([block(in_dim, out_dim, True)])
        for _ in range(1, layers):
            layer.append(block(out_dim, out_dim))
        return layer

    def _module_wrapper(self, x, module_list, shape):
        x = module_list[0](x, shape=shape)
        for m in module_list[1:]:
            x = m(x)
        return x

    def forward(self, x, shapes):
        # shapes = [before layer1, bef. layer 2, bef. layer 3, bef. layer 4]
        x = self.linear(x)

        # NOTE: replaced this by Linear(in_channels, 514 * 4 * 4)
        # x = F.interpolate(x, scale_factor=4)

        x = x.view(x.size(0), 512, 4, 4)
        
        x = self._module_wrapper(x, self.layer1, shapes[3])
        x = self._module_wrapper(x, self.layer2, shapes[2])
        x = self._module_wrapper(x, self.layer3, shapes[1])
        x = self._module_wrapper(x, self.layer4, shapes[0])

        x = self.conv1(x)
        return x

class VAE(nn.Module):
    def __init__(self, layers, latent_dim):
        super().__init__()
        self.layers = layers
        self.latent_dim = latent_dim

        self.encoder = ResNetEncoder(EncoderBlock, layers)
        layers.reverse()
        self.decoder = ResNetDecoder(DecoderBlock, layers, latent_dim)        

        self.fc_var = nn.Linear(latent_dim, latent_dim)
        self.fc_mu = nn.Linear(latent_dim, latent_dim)

        self.log_scale = nn.Parameter(torch.zeros([1]))
    
    @staticmethod
    def gaussian_likelihood(x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2, 3))
    
    @staticmethod
    def kl_divergence(z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl
    
    def reconstruction_loss(self, x):
        # encode x to get the mu and variance parameters
        x_encoded, shapes = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)
        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z, shapes=shapes)
        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)

        # kl
        kl = self.kl_divergence(z, mu, std)

        # elbo
        elbo = (kl - recon_loss)
        elbo = elbo.mean()

        log_dict = {
            'elbo': elbo,
            'kl': kl.mean(),
            'recon_loss': recon_loss.mean(),
            'kl': kl.mean(),
            'reconstructed_im': x_hat
        }

        return elbo, log_dict
    
    def mse_reconstruction_loss(self, x):
        encoded, shapes = self.encoder(x)
        x_hat = self.decoder(encoded, shapes)
        loss = F.mse_loss(x_hat, x)
        return loss, {'mse_loss': loss, 'im': x_hat}
    
    def similarity_loss(self, im, im_transformed):
        """treat encoder output as logits, and lower the cross entropy between 
        images and transformed images"""

        logit_x, shapes = self.encoder(im)
        logit_y, shapes = self.encoder(im_transformed)
        loss = cross_entropy(logit_x, logit_y.softmax(-1))
        
        log_dict = {'cross_entropy': loss}
        return loss, log_dict
    
    def train_step(self, im, im_transformed):
        elbo, log_dict = self.reconstruction_loss(im)
        similarity, log_dict1 = self.similarity_loss(im, im_transformed)

        loss = elbo + similarity
        log_dict.update(log_dict1)
        return loss, log_dict
    
    @property
    def config(self):
        return {
            'operations': self.layers,
            'latent_dim': self.latent_dim
        }

