import torch
import torch.nn as nn
import torch.nn.functional as F

class Expand(nn.Module):
    def __init__(self):
        super(Expand, self).__init__()

    def forward(self, x):
        return torch.unsqueeze(x, dim=0)

class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        return torch.squeeze(x, dim=1)

class SE_block(nn.Module):
    """[summary]
    
    Squeeze Excite block

    """
    def __init__(self, ratio=16):
        super(SE_block, self).__init__()
        self.ratio = ratio

    def dense_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Linear(in_channels, in_channels // self.ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // self.ratio, out_channels, bias=False),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        filters = x.size(1) # 64
        reshape_size = (x.size(0), 1, 1, filters)
        se = F.adaptive_avg_pool2d(x, (1, 1))
        se = torch.reshape(se, reshape_size)
        se = self.dense_block(in_channels=filters, out_channels=filters)(se)
        se = se.permute(0, 3, 1, 2)
        return x * se

class BN_block2d(nn.Module):
    """
        2-d batch-norm block
    """
    def __init__(self, in_channels, out_channels):
#TODO: Supposed to be a padding 'same' not padding 1 => calculate the padding and jot it down
        super(BN_block2d, self).__init__()
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
        super(BN_block3d, self).__init__()
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
    def __init__(self, in_channels, out_channels):
        super(D_SE_Add, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.squeeze_block_3d = nn.Sequential(
            nn.Conv3d(in_channels, 1, kernel_size=1, padding=0),
            Squeeze()
        )

    def squeeze_block_2d(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            SE_block()
        )
        
    def forward(self, in_3d, in_2d):
        in_3d = self.squeeze_block_3d(self.in_channels)(in_3d)
        in_3d = self.squeeze_block_2d(in_3d.size(1), self.out_channels)(in_3d)
        in_2d = SE_block()(in_2d)
        return in_3d + in_2d

def up_block(self, in_channels, out_channels):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU()
    )