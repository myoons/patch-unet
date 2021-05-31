import torch.nn.functional as F
from .parts import *


class WordSegmenter(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(WordSegmenter, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 64)
        self.down4 = Down(64, 128 // self.factor)
        self.up1 = Up(128, 64 // self.factor, bilinear)
        self.up2 = Up(64, 32 // self.factor, bilinear)
        self.up3 = Up(32, 16 // self.factor, bilinear)
        self.up4 = Up(16, 8, bilinear)
        self.outc = OutConv(8, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
