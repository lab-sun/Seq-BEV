import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as PLT
import numpy as np
import cv2
from pytorch_toolbelt import losses as L

NO_LABEL = None  # 背景类也算是语义类

class compute_losses(nn.Module):
    def __init__(self, device='GPU'):
        super(compute_losses, self).__init__()
        self.device = device
        self.seg_criterion_dice = L.DiceLoss(mode='multiclass', ignore_index=NO_LABEL).cuda()   # 分割dice损失
        self.seg_criterion = L.SoftCrossEntropyLoss(reduction='mean', smooth_factor = 0.1, ignore_index=NO_LABEL).cuda()  # 分割CE损失
        #self.seg_criterion_focal = L.FocalLoss(reduction="mean", gamma=2, ignore_index=NO_LABEL).cuda()  # 分割focal_loss
        self.seg_criterion_focal = L.FocalLoss(reduction="mean", gamma=2).cuda()

    def forward(self, opt, labels, outputs):
        #_, _, labels, road_layout_labels, _ = inputs
        loss_items = outputs.keys()
        if 'bev' in loss_items:
            bev_label = labels['bev']
            bev_pred = outputs['bev']
        if 'layout' in loss_items:
            layoput_label = labels['layout']
            layout_pred = outputs['layout']
        #bev_gt = torch.autograd.Variable(bev_label).cuda().long()   # 在传入之前就已经转成tensor.cuda().long()类型了
        #layout_gt = torch.autograd.Variable(layoput_label).cuda().long()

        dice_weight = opt.dice_weight
        layout_weight = opt.layout_weight
        loss_type = opt.loss_type
        roadLayout_loss_type = opt.roadLayout_loss_type
        losses = {}
        losses['bev'] = 0
        losses['layout'] = 0

        # 用ce loss和dice loss同时计算bev loss，通过dice_weight调整两个losses的比例
        # print(">>>> in losses.py, torch.unique(bev_pred): ", torch.unique(bev_pred))
        # print(">>>> in losses.py, bev_pred.shape: ", bev_pred.shape)
        # print(">>>> in losses.py, torch.unique(bev_label): ", torch.unique(bev_label))
        # print(">>>> in losses.py, bev_label.shape: ", bev_label.shape)
        # print(">>>> in losses.py, torch.unique(layout_pred): ", torch.unique(layout_pred))
        # print(">>>> in losses.py, layout_pred.shape: ", layout_pred.shape)
        # print(">>>> in losses.py, torch.unique(layoput_label): ", torch.unique(layoput_label))
        # print(">>>> in losses.py, layoput_label.shape: ", layoput_label.shape)
        #loss_1 = self.seg_criterion(bev_pred, bev_label)
        #loss_2 = self.seg_criterion_dice(bev_pred, bev_label)
        #loss_3 = self.seg_criterion(layout_pred, layoput_label)
        #print("in losses.py: loss_1: {}, loss_2: {}, loss_3: {}".format("loss_1", "loss_2", loss_3))

        if 'bev' in loss_items:
            #losses['bev'] = (self.seg_criterion(bev_pred, bev_label) + dice_weight * self.seg_criterion_dice(bev_pred, bev_label))/(200*200)  # 平均到每个像素上的loss
            if loss_type == 'focal':
                losses['bev'] = self.seg_criterion_focal(bev_pred, bev_label) + dice_weight * self.seg_criterion_dice(bev_pred, bev_label)  # 使用focal loss 和 dice loss
            else:
                losses['bev'] = (self.seg_criterion(bev_pred, bev_label) + dice_weight * self.seg_criterion_dice(bev_pred, bev_label))
            
        # 用ce loss计算layout loss
        if 'layout' in loss_items:
            #losses['layout'] = self.seg_criterion(layout_pred, layoput_label)/(240*400)  # 平均到每个像素上的loss
            if roadLayout_loss_type == 'focal':
                losses['layout'] = self.seg_criterion_focal(layout_pred, layoput_label)  + dice_weight * self.seg_criterion_dice(layout_pred, layoput_label)
            else:
                losses['layout'] = self.seg_criterion(layout_pred, layoput_label)
        # 整体loss为bev loss和layout loss的加权和
        losses['loss'] = losses['bev'] + layout_weight * losses['layout']
        #print("in loss.py, losses['loss']: ", losses['loss'].item())
        
        return losses


