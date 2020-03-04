"""
    D-UNet: a dimension-fusion U shape network for chronic stroke lesion segmentation
    https://arxiv.org/pdf/1908.05104.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from DUnet_parts import *

class DUnet(nn.Module):
    def __init__(self, in_channels):
        super(DUnet, self).__init__()

        self.in_channels = in_channels
        self.BN_block3d = BN_block3d
        self.BN_block2d = BN_block2d
        self.D_SE_Add = D_SE_Add
        self.MaxPool3d = nn.MaxPool3d
        self.MaxPool2d = nn.MaxPool2d
        self.Dropout = nn.Dropout
        self.Expand = Expand
        self.Conv2d = nn.Conv2d
        self.Sigmoid = nn.Sigmoid

    def forward(self, x):
        input3d = self.Expand()(x) # 1, batch_size, 4, 192, 192
        input3d = input3d.permute(1, 0, 2, 3, 4) # batch, 1, 4, 192, 192

        in_channels = input3d.size(1)

        print(input3d.size())

        # 3d Stream
        conv3d1 = self.BN_block3d(in_channels, in_channels * 32)(input3d)
        pool3d1 = self.MaxPool3d(kernel_size=2)(conv3d1)

        conv3d2 = self.BN_block3d(in_channels * 32, in_channels * 64)(pool3d1)
        pool3d2 = self.MaxPool3d(kernel_size=2)(conv3d2)

        conv3d3 = self.BN_block3d(in_channels * 64, in_channels * 128)(pool3d2)
        
        # 2d Encoding
        in_channels = self.in_channels

        conv1 = self.BN_block2d(in_channels, in_channels * 8)(x)
        pool1 = self.MaxPool2d(kernel_size=2)(conv1)

        conv2 = self.BN_block2d(in_channels * 8, in_channels * 16)(pool1)
        conv2 = self.D_SE_Add(in_channels * 16, in_channels * 16)(conv3d2, conv2)
        pool2 = self.MaxPool2d(kernel_size=2)(conv2)

        conv3 = self.BN_block2d(in_channels * 16, in_channels * 32)(pool2)
        conv3 = self.D_SE_Add(in_channels * 32, in_channels * 32)(conv3d3, conv3)
        pool3 = self.MaxPool2d(kernel_size=2)(conv3)

        conv4 = self.BN_block2d(in_channels * 32, in_channels * 64)(pool3)
        conv4 = self.Dropout(0.3)(conv4)
        pool4 = self.MaxPool2d(kernel_size=2)(conv4)

        conv5 = self.BN_block2d(in_channels * 64, in_channels * 128)(pool4)
        conv5 = self.Dropout(0.3)(conv5)

        # Decoding

        up6 = self.up_block(in_channels * 128, in_channels * 64)(conv5)
        merge6 = conv4 + up6
        conv6 = self.BN_block2d(in_channels * 64, in_channels * 64)(merge6)

        up7 = self.up_block(in_channels * 64, in_channels * 32)(conv6)
        merge7 = conv3 + up7
        conv7 = self.BN_block2d(in_channels * 32, in_channels * 32)(merge7)

        up8 = self.up_block(in_channels * 32, in_channels * 16)(conv7)
        merge8 = conv2 + up8
        conv8 = self.BN_block2d(in_channels * 16, in_channels * 16)(merge8)

        up9 = self.up_block(in_channels * 16, in_channels * 8)(conv8)
        merge9 = conv1 + up9
        conv9 = self.BN_block2d(in_channels * 8, in_channels * 8)(merge9)

        conv10 = self.Conv2d(in_channels * 8, 1, kernel_size=1, padding=0)(conv9)
        conv10 = self.Sigmoid()(conv10)

        return conv10