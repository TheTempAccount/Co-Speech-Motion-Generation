import os
import sys
from typing import ForwardRef

sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F

class KeypointLoss(nn.Module):
    def __init__(self):
        super(KeypointLoss, self).__init__()
    
    def forward(self, pred_seq, gt_seq, gt_conf=None):
        #pred_seq: (B, C, T)
        if gt_conf is not None:
            gt_conf = gt_conf >= 0.01
            return F.mse_loss(pred_seq[gt_conf], gt_seq[gt_conf], reduction='mean')
        else:
            return F.mse_loss(pred_seq, gt_seq)

class VelocityLoss(nn.Module):
    def __init__(self,
        velocity_length
    ):
        super(VelocityLoss, self).__init__()
        self.velocity_length = velocity_length

    def forward(self, pred_seq, gt_seq, prev_seq):
        B, C, T = pred_seq.shape
        last_frame = prev_seq[:, :, -1:]
        gt_velocity = gt_seq - torch.cat([last_frame, gt_seq[:, :, :-1]], dim=-1)
        pred_velocity = pred_seq - torch.cat([last_frame, pred_seq[:, :, :-1]], dim=-1)

        assert gt_velocity.shape[0] == B and gt_velocity.shape[1] == C and gt_velocity.shape[2] == T
        gt_velocity = gt_velocity[:, :, :self.velocity_length]
        pred_velocity = pred_velocity[:, :, :self.velocity_length]
        return F.mse_loss(pred_velocity, gt_velocity)

class KLLoss(nn.Module):
    def __init__(self, kl_tolerance):
        super(KLLoss, self).__init__()
        self.kl_tolerance = kl_tolerance

    def forward(self, mu, var):
        kld_loss = -0.5 * torch.sum(1 + var - mu**2 -var.exp(), dim=1)
        if self.kl_tolerance is not None:
            above_line = kld_loss[kld_loss > self.kl_tolerance]
            if len(above_line) > 0:
                kld_loss = torch.mean(kld_loss, dim=0)
            else:
                kld_loss = 0
        else:
            kld_loss = torch.mean(kld_loss, dim=0)
        return kld_loss

class L2RegLoss(nn.Module):
    def __init__(self):
        super(L2RegLoss, self).__init__()
    
    def forward(self, x):
        #TODO: check
        return torch.sum(x**2)

class AudioLoss(nn.Module):
    def __init__(self):
        super(AudioLoss, self).__init__()
    
    def forward(self, dynamics, gt_poses):
        mean = torch.mean(gt_poses, dim=-1).unsqueeze(-1)
        gt = gt_poses - mean
        return F.mse_loss(dynamics, gt_poses)

L1Loss = nn.L1Loss