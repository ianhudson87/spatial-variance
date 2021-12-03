import torch
import torch.nn as nn
from model_zoo.udvd_model import PixelConv

class DnCNNablationHead(nn.Module):
    def __init__(self, channels, num_of_layers=14, verbose=False):
        super(DnCNNablationHead, self).__init__()
        print("Using DnCNN ablation Head model")
        kernel_size = 3 # of convolution
        padding = 1
        features = 64

        self.feat_kernel = nn.Conv2d(channels, 5**2, 3, 1, 1)
        self.pixel_conv = PixelConv(scale=1, depthwise=True)

        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-3):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self.verbose = verbose

    def forward(self, x):

        kernel = self.feat_kernel(x)
        x = self.pixel_conv(x, kernel)
        
        out = self.dncnn(x)
        return out

class DnCNNablationMiddle(nn.Module):
    def __init__(self, channels, num_of_layers=14, verbose=False):
        super(DnCNNablationMiddle, self).__init__()
        print("Using DnCNN ablation Middle model")
        kernel_size = 3 # of convolution
        padding = 1
        features = 64

        num_conv_blocks = num_of_layers-5

        mid = int(num_conv_blocks/2)

        layers1 = []
        layers1.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers1.append(nn.ReLU(inplace=True))
        for _ in range(mid):
            layers1.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers1.append(nn.BatchNorm2d(features))
            layers1.append(nn.ReLU(inplace=True))
        layers1.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        
        self.feat_kernel = nn.Conv2d(channels, 5**2, 3, 1, 1)
        self.pixel_conv = PixelConv(scale=1, depthwise=True)

        layers2 = []
        layers2.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers2.append(nn.ReLU(inplace=True))
        for _ in range(mid, num_conv_blocks):
            layers2.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers2.append(nn.BatchNorm2d(features))
            layers2.append(nn.ReLU(inplace=True))
        layers2.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        
        self.seq1 = nn.Sequential(*layers1)
        self.seq2 = nn.Sequential(*layers2)
        self.verbose = verbose
    def forward(self, x):
        x = self.seq1(x)

        kernel = self.feat_kernel(x)
        x = self.pixel_conv(x, kernel)

        out = self.seq2(x)
        return out

class DnCNNablationTail(nn.Module):
    def __init__(self, channels, num_of_layers=14, verbose=False):
        super(DnCNNablationTail, self).__init__()
        print("Using DnCNN ablation Tail model")
        kernel_size = 3 # of convolution
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-3):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))

        self.feat_kernel = nn.Conv2d(channels, 5**2, 3, 1, 1)
        self.pixel_conv = PixelConv(scale=1, depthwise=True)

        self.dncnn = nn.Sequential(*layers)
        self.verbose = verbose
    def forward(self, x):
        x = self.dncnn(x)

        kernel = self.feat_kernel(x)
        out = self.pixel_conv(x, kernel)

        return out

class DnCNNablation_more_dyn(nn.Module):
    def __init__(self, channels, num_of_layers=14, verbose=False):
        super(DnCNNablation_more_dyn, self).__init__()
        print("Using DnCNN better dynamic tail model")
        kernel_size = 3 # of convolution
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-4):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)


        feat_kernel_layers = []
        feat_kernel_layers.append(nn.Conv2d(channels, 5**2, 3, 1, 1))
        for _ in range(6):
            feat_kernel_layers.append(nn.Conv2d(5**2, 5**2, 3, 1, 1))
        self.feat_kernel = nn.Sequential(*feat_kernel_layers)
        self.pixel_conv = PixelConv(scale=1, depthwise=True)

        self.verbose = verbose
    def forward(self, x):
        x = self.dncnn(x)

        kernel = self.feat_kernel(x)
        out = self.pixel_conv(x, kernel)

        return out

class DnCNNablationFull(nn.Module):
    def __init__(self, channels, num_of_layers=14, verbose=False):
        super(DnCNNablationFull, self).__init__()
        print("Using DnCNN ablation Full model")
        padding = 1
        features = 64

        self.blocks = nn.ModuleList()
        for _ in range(num_of_layers):
            block = nn.ModuleDict({
                "feat_kernel": nn.Conv2d(channels, 5**2, 3, 1, 1),
                "pixel_conv": PixelConv(scale=1, depthwise=True),
                "batch_norm": nn.BatchNorm2d(channels),
                "relu": nn.ReLU(inplace=True)
            })
            self.blocks.append(block)
        # layers = []
        # layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        # layers.append(nn.ReLU(inplace=True))
        # for _ in range(num_of_layers):
        #     layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        #     layers.append(nn.BatchNorm2d(features))
        #     layers.append(nn.ReLU(inplace=True))
        # layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))

        # self.feat_kernel = nn.Conv2d(channels, 5**2, 3, 1, 1)
        # self.pixel_conv = PixelConv(scale=1, depthwise=True)

        self.verbose = verbose
    def forward(self, x):
        for block in self.blocks:
            kernel = block["feat_kernel"](x)
            x = block["pixel_conv"](x, kernel)
            x = block["batch_norm"](x)
            x = block["relu"](x)
        
        return x