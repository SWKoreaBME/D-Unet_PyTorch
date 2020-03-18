import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def enhanced_mixing_loss(y_true, y_pred):
    # Code written by Seung hyun Hwang
    gamma = 1.1
    alpha = 0.48
    smooth = 1.
    epsilon = 1e-7
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)

    # dice loss
    intersection = (y_true * y_pred).sum()
    dice_loss = (2. * intersection + smooth) / ((y_true * y_true).sum() + (y_pred * y_pred).sum() + smooth)

    # focal loss
    y_pred = torch.clamp(y_pred, epsilon)

    pt_1 = torch.where(torch.eq(y_true, 1), y_pred, torch.ones_like(y_pred))
    pt_0 = torch.where(torch.eq(y_true, 0), y_pred, torch.zeros_like(y_pred))
    focal_loss = -torch.mean(alpha * torch.pow(1. - pt_1, gamma) * torch.log(pt_1)) - \
                 torch.mean((1 - alpha) * torch.pow(pt_0, gamma) * torch.log(1. - pt_0))
    return focal_loss - torch.log(dice_loss)