import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from model_zoo.udvd_model import PixelConv

class DnCNNDynamicSpecnorm(nn.Module):
    def __init__(self, channels, num_of_layers=14, verbose=False):
        super(DnCNNDynamicSpecnorm, self).__init__()
        print("Using DnCNN with (deep) dynamic convolution (at tail) and spectral normalization")
        kernel_size = 3 # of convolution
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-4):
            layers.append(spectral_norm(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)


        feat_kernel_layers = []
        feat_kernel_layers.append(nn.Conv2d(channels, 5**2, 3, 1, 1))
        for _ in range(5):
            feat_kernel_layers.append(nn.Conv2d(5**2, 5**2, 3, 1, 1))
            layers.append(nn.BatchNorm2d(5**2))
            layers.append(nn.ReLU(inplace=True))
        feat_kernel_layers.append(nn.Conv2d(5**2, 5**2, 3, 1, 1))
        self.feat_kernel = nn.Sequential(*feat_kernel_layers)
        self.pixel_conv = PixelConv(scale=1, depthwise=True)

        self.verbose = verbose
    def forward(self, x):
        x = self.dncnn(x)

        kernel = self.feat_kernel(x)
        out = self.pixel_conv(x, kernel)

        return out

class DnCNNDynamicSpecnormMoreOutputLayers(nn.Module):
    def __init__(self, channels, num_of_layers=14, verbose=False):
        super(DnCNNDynamicSpecnormMoreOutputLayers, self).__init__()
        print("Using DnCNN with (deep) dynamic convolution (at tail) with spectral normalization and more layers from cnn output")
        kernel_size = 3 # of convolution
        padding = 1
        features = 64
        cnn_out_features = 25
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-4):
            layers.append(spectral_norm(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)))
            layers.append(nn.ReLU(inplace=True))
        self.dncnn = nn.Sequential(*layers)

        # input to the dynamic kernel
        self.dncnn_out1 = spectral_norm(nn.Conv2d(in_channels=features, out_channels=cnn_out_features, kernel_size=kernel_size, padding=padding, bias=False))

        # output that the dynamic kernel is applied to
        self.dncnn_out2 = spectral_norm(nn.Conv2d(in_channels=features, out_channels=1))


        feat_kernel_layers = []
        feat_kernel_layers.append(nn.Conv2d(channels, 5**2, 3, 1, 1))
        for _ in range(5):
            feat_kernel_layers.append(nn.Conv2d(5**2, 5**2, 3, 1, 1))
            layers.append(nn.BatchNorm2d(5**2))
            layers.append(nn.ReLU(inplace=True))
        feat_kernel_layers.append(nn.Conv2d(5**2, 5**2, 3, 1, 1))
        self.feat_kernel = nn.Sequential(*feat_kernel_layers)
        self.pixel_conv = PixelConv(scale=1, depthwise=True)

        self.verbose = verbose
    def forward(self, x):
        x = self.dncnn(x)

        x1 = self.dncnn_out1(x)
        x2 = self.dncnn_out2(x)

        kernel = self.feat_kernel(x1)
        out = self.pixel_conv(x2, kernel)

        return out

class DnCNNDynamicSpecnormMoreDynamicLayers(nn.Module):
    def __init__(self, channels, num_of_layers=14, verbose=False):
        super(DnCNNDynamicSpecnormMoreDynamicLayers, self).__init__()
        print("Using DnCNN with many shallow dynamic convolution layers at the tail. with spectral normalization and 1 layer from cnn output")
        kernel_size = 3 # of convolution
        padding = 1
        features = 64
        num_dynamic_layers = 5

        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-4):
            layers.append(spectral_norm(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

        self.dynamic_layers = []
        for _ in range(num_dynamic_layers):
            self.dynamic_layers.append(nn.Sequential([
                nn.Conv2d(5**2, 5**2, 3, 1, 1),
                nn.BatchNorm2d(5**2),
                nn.ReLU(inplace=True),
            ]))
        
        self.pixel_conv = PixelConv(scale=1, depthwise=True)

        self.verbose = verbose
    def forward(self, x):
        x = self.dncnn(x)

        for i in range(len(self.dynamic_layers)):
            kernel = self.dynamic_layers[i](x)
            x = self.pixel_conv(x, kernel) # might need to change this

        out = x
        return out