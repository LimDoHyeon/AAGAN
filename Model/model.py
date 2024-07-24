import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm
import math
from utils import init_weights, get_padding

LRELU_SLOPE = 0.1


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(
                Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d))
            ) for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(
                Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
            ) for _ in dilation
        ])
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class Generator(nn.Module):
    def __init__(self, in_ch, ch_init_upsample, upsample_rates, upsample_kernel_sizes, resblock_kernel_sizes,
                 resblock_dilation_sizes, out_ch):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.conv_pre = weight_norm(Conv1d(in_ch, ch_init_upsample // 2, 7, 1, padding=3))
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(ConvTranspose1d(ch_init_upsample // (2 ** i),
                                                        ch_init_upsample // (2 ** (i + 1)), k, u,
                                                        padding=(k - u) // 2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = ch_init_upsample // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, out_ch, 7, 1, padding=3))
        self.conv_pre.apply(init_weights)
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)  # torch.nn.utils.remove_weight_norm
        for l in self.resblocks:
            l.remove_weight_norm()  # Recursive call
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class Discriminator(nn.Module):
    def __init__(self, in_ch, ch_init_downsample, downsample_rates, downsample_kernel_sizes, resblock_kernel_sizes,
                 resblock_dilation_sizes):
        super(Discriminator, self).__init__()
        self.ch_init_downsample = ch_init_downsample
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_downsamples = len(downsample_rates)

        self.conv_pre = weight_norm(Conv1d(in_ch, ch_init_downsample, 7, 1, padding=3))
        self.downs = nn.ModuleList()
        for i, (u, k) in enumerate(zip(downsample_rates, downsample_kernel_sizes)):
            self.downs.append(weight_norm(Conv1d(ch_init_downsample * (2 ** i),
                                                 ch_init_downsample * (2 ** (i + 1)), k, u,
                                                 padding=math.ceil((k - u) / 2))))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.downs)):
            ch = ch_init_downsample * (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.conv_pre.apply(init_weights)
        self.downs.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_downsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.downs[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.sigmoid(x)
        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.downs:
            remove_weight_norm(l)  # torch.nn.utils.remove_weight_norm
        for l in self.resblocks:
            l.remove_weight_norm()  # Recursive call
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
