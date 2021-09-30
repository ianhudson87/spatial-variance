# source: https://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA%3D%3D&mid=2247494694&idx=1&sn=ed29071f700b129534beb649a04b3b97&scene=45#wechat_redirect
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class UDVD(nn.Module):
    def __init__(self, k, in_channels, depth=15):
        # k = size of dynamic kernel
        super().__init__()
        self.in_channels = in_channels
        self.head = nn.Conv2d(16 + in_channels, 128, 3, 1, 1)
        body = [ResBlock(128, 3, 0.1) for _ in range(depth)]
        self.body = nn.Sequential(*body)
        self.ComDyConv1 = CommonDynamicConv(k, in_channels)

    def forward(self, image, kernel, noise):
        assert image.size(1) == self.in_channels, 'Channels of Image should be in_channels, not {}'.format(image.size(1))
        assert kernel.size(1) == 15, 'Channels of kernel should be 15, not {}'.format(kernel.size(1))
        assert noise.size(1) == 1, 'Channels of noise should be 1, not {}'.format(noise.size(1))
        inputs = torch.cat([image, kernel, noise], 1)
        head = self.head(inputs)
        body = self.body(head) + head
        print("here", body.shape)
        output2 = self.ComDyConv1(image, body)
        return  output2

class UDVD_upscale(nn.Module):
    def __init__(self, k, r, in_channels, depth=15):
        # k = size of dynamic kernel
        # r = upscaling rate
        super().__init__()
        self.in_channels = in_channels
        self.head = nn.Conv2d(16 + in_channels, 128, 3, 1, 1)
        body = [ResBlock(128, 3, 0.1) for _ in range(depth)]
        self.body = nn.Sequential(*body)
        # self.UpDyConv = UpDynamicConv(k, r, in_channels)
        self.ComDyConv1 = CommonDynamicConv(k, in_channels)

    def forward(self, image, kernel, noise):
        assert image.size(1) == self.in_channels, 'Channels of Image should be in_channels, not {}'.format(image.size(1))
        assert kernel.size(1) == 15, 'Channels of kernel should be 15, not {}'.format(kernel.size(1))
        assert noise.size(1) == 1, 'Channels of noise should be 1, not {}'.format(noise.size(1))
        inputs = torch.cat([image, kernel, noise], 1)
        head = self.head(inputs)
        body = self.body(head) + head
        output1 = self.UpDyConv(image, body)
        return  output1


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, res_scale=1.0):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, 1, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size, 1, padding)
        )
        self.res_scale = res_scale

    def forward(self, inputs):
        return inputs + self.conv(inputs) * self.res_scale


class PixelConv(nn.Module):
    def __init__(self, scale=2, depthwise=False):
        super().__init__()
        self.scale = scale
        self.depthwise = depthwise

    def forward(self, feature, kernel):
        NF, CF, HF, WF = feature.size()
        NK, ksize, HK, WK = kernel.size()
        assert NF == NK and HF == HK and WF == WK
        if self.depthwise:
            ink = CF
            outk = 1
            ksize = int(np.sqrt(int(ksize // (self.scale ** 2))))
            pad = (ksize - 1) // 2
        else:
            ink = 1
            outk = CF
            ksize = int(np.sqrt(int(ksize // CF // (self.scale ** 2))))
            pad = (ksize - 1) // 2

        # features unfold and reshape, same as PixelConv
        feat = F.pad(feature, [pad, pad, pad, pad])
        feat = feat.unfold(2, ksize, 1).unfold(3, ksize, 1)
        feat = feat.permute(0, 2, 3, 1, 5, 4).contiguous()
        feat = feat.reshape(NF, HF, WF, ink, -1)

        # kernel
        kernel = kernel.permute(0, 2, 3, 1).reshape(NK, HK, WK, ksize * ksize, self.scale ** 2 * outk)

        output = torch.matmul(feat, kernel)
        output = output.permute(0, 3, 4, 1, 2).view(NK, -1, HF, WF)
        if self.scale > 1:
            output = F.pixel_shuffle(output, self.scale)
        return output


class CommonDynamicConv(nn.Module):
    def __init__(self, k, in_channels):
        super().__init__()
        self.image_conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 1, 1)
        )
        self.feat_residual = nn.Sequential(
            nn.Conv2d(160, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, in_channels, 3, 1, 1)
        )
        self.feat_kernel = nn.Conv2d(160, k**2, 3, 1, 1)
        self.pixel_conv = PixelConv(scale=1, depthwise=True)

    def forward(self, image, features):
        image_conv = self.image_conv(image)
        print("here2", image_conv.shape)
        cat_inputs = torch.cat([image_conv, features], 1)
        print("here3", cat_inputs.shape)

        kernel = self.feat_kernel(cat_inputs)
        print("here4", kernel.shape)
        output = self.pixel_conv(image, kernel)
        print("here6", output.shape)

        residual = self.feat_residual(cat_inputs)
        print("here5", residual.shape)
        return output + residual


class UpDynamicConv(nn.Module):
    # conv used to generate the dynamic kernel
    def __init__(self, k, r, in_channels):

        super().__init__()
        self.image_conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 1, 1)
        )
        self.feat_residual = nn.Sequential(
            nn.Conv2d(160, 16*(r**2), 3, 1, 1), # create enough channels so that when PixelShuffle happens we have 16 channels
            nn.ReLU(inplace=True),
            nn.PixelShuffle(upscale_factor=r),
            nn.Conv2d(16, in_channels, 3, 1, 1)
        )
        self.feat_kernel = nn.Conv2d(160, k**2 * r**2, 3, 1, 1) # 160 = 32 + 128 = # channels from image_conv + # channels from feature maps, F
        self.pixel_conv = PixelConv(scale=r, depthwise=True)

    def forward(self, image, features):
        image_conv = self.image_conv(image)
        cat_inputs = torch.cat([image_conv, features], 1)

        kernel = self.feat_kernel(cat_inputs) # generated per-pixel kernel
        output = self.pixel_conv(image, kernel) # applying per-pixel kernel to original image

        residual = self.feat_residual(cat_inputs)
        return output + residual


def demo():
    net = UDVD(k=5, in_channels=1)

    inputs = torch.randn(1, 1, 64, 64)
    kernel = torch.randn(1, 15, 64, 64)
    noise = torch.randn(1, 1, 64, 64)

    with torch.no_grad():
        output2 = net(inputs, kernel, noise)

    print(output2.size())


if __name__ == '__main__':
    demo()