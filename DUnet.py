import torch
import torch.nn as nn
import torch.nn.functional as F

from DUnet_parts import *

class DUnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DUnet, self).__init__()

    def down_block_2d(self, in_channels, out_channels):
        return nn.Sequential(
            BN_block2d(in_channels, in_channels * 8),
            nn.MaxPool2d(kernel_size=2)
        )

    def conv_block_3d(self, in_channels, out_channels):
        return nn.Sequential(
            BN_block3d(in_channels, in_channels * 8),
            nn.MaxPool3d(kernel_size=2),
            BN_block3d(in_channels * 8, in_channels * 16),
            nn.MaxPool3d(kernel_size=2),
            BN_block3d(in_channels * 16, in_channels * 32)
        )

    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x):
        in_channels = x.size(0)

        input3d = Expand(x) # x.size = 4 * 192 * 192

        # 3d Stream
        conv3d1 = BN_block3d(in_channels, in_channels * 8)(input3d)
        pool3d1 = nn.MaxPool3d(kernel_size=2)(conv3d1)

        conv3d2 = BN_block3d(in_channels * 8, in_channels * 16)(pool3d1)
        pool3d2 = nn.MaxPool3d(kernel_size=2)(conv3d2)

        conv3d3 = BN_block3d(in_channels * 16, in_channels * 32)(pool3d2)

        # 2d Encoding

        conv1 = BN_block2d(in_channels, in_channels * 8)(x)
        #conv1 = D_Add(32, conv3d1, conv1)
        pool1 = nn.MaxPool2d(kernel_size=2)(conv1)

        conv2 = BN_block2d(in_channels * 8, in_channels * 16)(pool1)
        conv2 = D_SE_Add(in_channels * 8, in_channels * 16)(conv3d2, conv2)
        pool2 = nn.MaxPool2d(kernel_size=2)(conv2)

        conv3 = BN_block2d(in_channels * 16, in_channels * 32)(pool2)
        conv3 = D_SE_Add(in_channels * 16, in_channels * 32)(conv3d3, conv3)
        pool3 = nn.MaxPool2d(kernel_size=2)(conv3)

        conv4 = BN_block2d(in_channels * 32, in_channels * 64)(pool3)
        conv4 = nn.Dropout(0.3)(conv4)
        pool4 = nn.MaxPool2d(kernel_size=2)(conv4)

        conv5 = BN_block2d(in_channels * 64, in_channels * 128)(pool4)
        conv5 = nn.Dropout(0.3)(conv5)

        #TODO :  Decoding ( keras -> PyTorch)

        up6 = self.up_block(in_channels * 128, in_channels * 64)(conv5)
        merge6 = torch.cat(conv4, up6)
        conv6 = BN_block2d(in_channels * 64, in_channels * 64)(merge6)

        up7 = self.up_block(in_channels * 64, in_channels * 32)(conv6)
        merge7 = torch.cat(conv3, up7)
        conv7 = BN_block2d(in_channels * 32, in_channels * 32)(merge7)

        up8 = self.up_block(in_channels * 32, in_channels * 16)(conv7)
        merge8 = torch.cat(conv2, up8)
        conv8 = BN_block2d(in_channels * 16, in_channels * 16)(merge8)

        up9 = self.up_block(in_channels * 32, in_channels * 16)(conv8)
        merge9 = torch.cat(conv1, up9)
        conv9 = BN_block2d(in_channels * 16, in_channels * 8)(merge9)

        conv10 = nn.Conv2d(in_channels * 8, in_channels * 8, kernel_size=1)(conv9)
        conv10 = nn.Sigmoid()(conv10)

        return conv10