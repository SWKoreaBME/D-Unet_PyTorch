import torch
import torch.nn as nn
import torch.nn.functional as F

class Expand(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.unsqueeze(x, dim=0)

class Squeeze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.squeeze(x, dim=1)

class SE_block(nn.Module):
    """[summary]
    
    Squeeze Excite block

    """
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        self.avg_2d = F.adaptive_avg_pool2d
        self.dense_block = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // ratio, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        filters = x.size(1)
        reshape_size = (x.size(0), 1, 1, filters)
        se = self.avg_2d(x, (1, 1))
        se = torch.reshape(se, reshape_size)
        se = self.dense_block(se)
        se = se.permute(0, 3, 1, 2)
        return x * se

class BN_block2d(nn.Module):
    """
        2-d batch-norm block
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.bn_block(x)

class BN_block3d(nn.Module):
    """
        3-d batch-norm block
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.bn_block(x)

class D_SE_Add(nn.Module):
    """
        D_SE_Add block
    """
    def __init__(self, in_channels, out_channels, mid_channels):
        super().__init__()
        
        self.conv3d_ = nn.Conv3d(in_channels, 1, kernel_size=1, padding=0)
        self.Squeeze = Squeeze()
        self.conv2d_ = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.ReLU = nn.ReLU()
        
        self.SE_block = SE_block(in_channels)
        
        self.squeeze_block_3d = nn.Sequential(
            nn.Conv3d(in_channels, 1, kernel_size=1, padding=0),
            Squeeze(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            SE_block(in_channels)
        )
        
    def forward(self, in_3d, in_2d):
        in_2d = self.SE_block(in_2d)
        in_3d = self.squeeze_block_3d(in_3d)

        return in_3d + in_2d

def up_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU()
    )