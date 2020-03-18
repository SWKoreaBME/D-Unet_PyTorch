"""
    D-UNet: a dimension-fusion U shape network for chronic stroke lesion segmentation
    https://arxiv.org/pdf/1908.05104.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from DUnet_parts import *

def weights_init_he(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
    if isinstance(m, nn.Conv3d):
        torch.nn.init.kaiming_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

class DUnet(nn.Module):
    def __init__(self, in_channels, weights_init = True):
        super().__init__()

        self.in_channels = in_channels
        in_channels_3d = 1
        
        self.Expand = Expand
        self.MaxPool3d = nn.MaxPool3d(kernel_size=2)
        self.MaxPool2d = nn.MaxPool2d(kernel_size=2)
        self.Dropout = nn.Dropout(0.3)
        
        # 3d down
        self.bn_3d_1 = BN_block3d(in_channels_3d, in_channels_3d * 32)
        self.bn_3d_2 = BN_block3d(in_channels_3d * 32, in_channels_3d * 64)
        self.bn_3d_3 = BN_block3d(in_channels_3d * 64, in_channels_3d * 128)
        
        # 2d down
        
        self.bn_2d_1 = BN_block2d(in_channels, in_channels * 8)

        self.bn_2d_2 = BN_block2d(in_channels * 8, in_channels * 16)
        self.se_add_2 = D_SE_Add(in_channels * 16, in_channels * 16, 2)
        
        self.bn_2d_3 = BN_block2d(in_channels * 16, in_channels * 32)
        self.se_add_3 = D_SE_Add(in_channels * 32, in_channels * 32, 1)
        
        self.bn_2d_4 = BN_block2d(in_channels * 32, in_channels * 64)

        self.bn_2d_5 = BN_block2d(in_channels * 64, in_channels * 128)
        
        # up

        self.up_block_1 = up_block(in_channels * 128, in_channels * 64)
        self.bn_2d_6 = BN_block2d(in_channels * 128, in_channels * 64)
        
        self.up_block_2 = up_block(in_channels * 64, in_channels * 32)
        self.bn_2d_7 = BN_block2d(in_channels * 64, in_channels * 32)
        
        self.up_block_3 = up_block(in_channels * 32, in_channels * 16)
        self.bn_2d_8 = BN_block2d(in_channels * 32, in_channels * 16)
        
        self.up_block_4 = up_block(in_channels * 16, in_channels * 8)
        self.bn_2d_9 = BN_block2d(in_channels * 16, in_channels * 8)
        
        self.conv_10 = nn.Sequential(
            nn.Conv2d(in_channels * 8, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        # He initialization stated in the original paper
        if weights_init:
            self.apply(weights_init_he)

    def forward(self, x):
        input3d = self.Expand()(x) # 1, batch_size, 4, 192, 192
        input3d = input3d.permute(1, 0, 2, 3, 4) # batch_size, 1, 4, 192, 192

        # 3d Stream
        conv3d1 = self.bn_3d_1(input3d)
        pool3d1 = self.MaxPool3d(conv3d1)

        conv3d2 = self.bn_3d_2(pool3d1)
        pool3d2 = self.MaxPool3d(conv3d2)

        conv3d3 = self.bn_3d_3(pool3d2)
        
        # 2d Encoding
        in_channels = self.in_channels

        conv1 = self.bn_2d_1(x)
        pool1 = self.MaxPool2d(conv1)

        conv2 = self.bn_2d_2(pool1)
        conv2 = self.se_add_2(conv3d2, conv2)
        pool2 = self.MaxPool2d(conv2)

        conv3 = self.bn_2d_3(pool2)
        conv3 = self.se_add_3(conv3d3, conv3)
        pool3 = self.MaxPool2d(conv3)

        conv4 = self.bn_2d_4(pool3)
        conv4 = self.Dropout(conv4)
        pool4 = self.MaxPool2d(conv4)

        conv5 = self.bn_2d_5(pool4)
        conv5 = self.Dropout(conv5)

        # Decoding

        up6 = self.up_block_1(conv5)
        merge6 = torch.cat(([conv4, up6]), 1)
        conv6 = self.bn_2d_6(merge6)

        up7 = self.up_block_2(conv6)
        merge7 = torch.cat(([conv3, up7]), 1)
        conv7 = self.bn_2d_7(merge7)

        up8 = self.up_block_3(conv7)
        merge8 = torch.cat(([conv2, up8]), 1)
        conv8 = self.bn_2d_8(merge8)

        up9 = self.up_block_4(conv8)
        merge9 = torch.cat(([conv1, up9]), 1)
        conv9 = self.bn_2d_9(merge9)

        conv10 = self.conv_10(conv9)

        return conv10

if __name__ == "__main__":

    model = DUnet(in_channels=4)

    BATCH_SIZE = 1
    input_batch = torch.Tensor(BATCH_SIZE, 4, 192, 192)
    output_batch = model(input_batch)

    print(output_batch.size()) # BATCH_SIZE, 1, 192, 192