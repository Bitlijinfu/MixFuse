import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import SpatialGradient
from kornia.losses import SSIMLoss


class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv = SpatialGradient('sobel', normalized=False)

    def gradient(self, x, eps=1e-6):
        s = self.sobelconv(x)
        dx, dy = s[:, :, 0, :, :], s[:, :, 1, :, :]
        #u = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2) + eps)
        u = torch.abs(dx)+torch.abs(dy)
        return u

    def forward(self, image_A, image_B, image_fused):
        gradient_A = self.gradient(image_A)
        gradient_B = self.gradient(image_B)
        gradient_fused = self.gradient(image_fused)
        gradient_joint = torch.max(gradient_A, gradient_B)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        return Loss_gradient

class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()
        
    def forward(self, image_A, image_B, image_fused):
        # vis ir fused
        intensity_joint = torch.max(image_A, image_B)
        Loss_intensity = F.l1_loss(image_fused, intensity_joint)
        return Loss_intensity


class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()
        self.ssim = SSIMLoss(window_size=11, reduction='mean')

    def forward(self, image_A, image_B, image_fused):
        Loss_SSIM = 0.5 * self.ssim(image_A, image_fused) + \
            0.5 * self.ssim(image_B, image_fused)
        return Loss_SSIM

class fusion_loss_vif(nn.Module):
    def __init__(self):
        super(fusion_loss_vif, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        self.L_SSIM = L_SSIM()
        
    def forward(self, image_A, image_B, image_fused):

        loss_l1 = 5 * self.L_Inten(image_A, image_B, image_fused)
        loss_gradient = 10 * self.L_Grad(image_A, image_B, image_fused) 
        loss_SSIM = 4 * self.L_SSIM(image_A, image_B, image_fused)
        fusion_loss = loss_gradient + loss_l1 + loss_SSIM
        return fusion_loss, loss_gradient, loss_l1, loss_SSIM


