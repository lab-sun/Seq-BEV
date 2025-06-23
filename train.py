import argparse
from doctest import FAIL_FAST
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from pyexpat import model
import random
from sre_parse import State
import time
from pathlib import Path
from cv2 import moments
import shutil

import numpy as np
import copy
from datetime import datetime
import skimage
#from plotly import data

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import autograd
import torchvision.utils
from pytorch_toolbelt import losses as L
# from torch.optim.lr_scheduler import ExponentialLR
# from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import MultiStepLR
# from torch.optim.lr_scheduler import CosineAnnealingLR
from tensorboardX import SummaryWriter

from PIL import Image
import matplotlib.pyplot as PLT
import matplotlib.cm as mpl_color_map

import tqdm
from sklearn.metrics import confusion_matrix
import glob

from configs.opt import get_args

# 需要自己写的文件
from losses.losses import compute_losses
from src.utils.confusion import compute_results

#from src.data.data_factory import build_dataloaders
from src.data.seqBEV_dataset import seqBEV_dataset, Img_ColorJitter
from src.seqBEV.deeplabv3_plus_STM_multi_inputs_decoder_TFM_ablation_selfAdoption_GPU3 import DeepLab

#from src.seqBEV.TS_fusion import TS_Fusion
from src.utils.configs import get_default_configuration, load_config
from src.utils.confusion import BinaryConfusionMatrix
from src.utils.lr_update import set_optimizer_lr, get_lr_scheduler
from src.data.nuscenes.utils import NUSCENES_CLASS_NAMES
from src.data.argoverse.utils import ARGOVERSE_CLASS_NAMES
from src.utils.visualise import colorise
from src.data.utils import create_visual_anno
# from src.seqBEV.TS_Encorder import TS_Encoder
# from src.seqBEV.TS_fusion import TS_Fusion
# from src.seqBEV.Road_layout import road_layout
# from src.seqBEV.STM import STM
# from src.seqBEV.BEV_transformer import build_obj_transformer
# from src.seqBEV.Decoder import Decoder

# 自己的module文件夹
#import src.models

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

verbose = False
global epoch_idx
start_datetime = datetime.now().replace(microsecond=0)  # 全局起始时间
label_list = ['background', 'drivable_area', 'ped_crossing', 'walkway', 'movable_object', 'vehicle', 'predestrian']
global is_best
global best_iou
global best_epoch
global is_best_test
global best_test_iou
global best_test_epoch


class Trainer:
    def __init__(self, args):
        self.opt = args
        #self.epoch_idx = epoch_idx
        self.global_step = 0
        self.models = {}
        self.device = "cuda"
        #self.seg_criterion = L.SoftCrossEntropyLoss(reduction='sum', smooth_factor = 0.1).cuda()
        self.parameters_to_train = []  # 不同模块对应的不同学习率列表，cross-view中写了好几个？
        self.backbone_parameters = []
        self.head_parameters = []

        self.base_parameters_to_train = []
        self.T_encoder_parameter_to_train = []
        self.S_encoder_parameter_to_train = []
        self.rest_parameters_to_train = []
        self.criterion = compute_losses() # 多个loss的计算，compute_losses()函数在losses.py中
        self.create_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        self.epoch = 0
        self.start_epoch = 0
        self.scheduler = 0  # 学习率调整策略
        # Save log and models path
        self.log_root = os.path.join(self.opt.log_root, self.opt.model_name)
        #self.opt.save_path = os.path.join(self.opt.save_path, self.opt.split)
        self.save_path = self.opt.save_path
        self.writer = SummaryWriter(os.path.join(self.opt.log_root, self.opt.model_name, self.create_time))
        self.log = open(os.path.join(self.opt.log_root, self.opt.model_name, self.create_time,
                                     '%s.csv' % self.opt.model_name), 'w')
        
        # Initializing models
        self.models['deeplabv3_plus'] = DeepLab(num_classes=7, backbone="mobilenet", downsample_factor=8, pretrained=self.opt.pretrained_backbone)


        # 载入deeplabv3_plus的预训练参数
        if self.opt.deeplabv3Plus_model_path != '':
            print('Load weights {}'.format(self.opt.deeplabv3Plus_model_path))
            # 根据预训练权重的Key和模型的Key进行加载
            model_dict = self.models['deeplabv3_plus'].state_dict()
            pretrained_dict = torch.load(self.opt.deeplabv3Plus_model_path, map_location=self.device)
            load_key, no_load_key, temp_dict = [], [], {}
            temporal_layer = [2, 4, 7, 14]
            for k, v in pretrained_dict.items():
                if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v): 
                    if 'backbone' in k:
                        InvertedResidual_layer = k.split('.')[2]
                        conv_layer = k.split('.')[4]
                        if int(InvertedResidual_layer) in temporal_layer and int(conv_layer) == 3:  # 载入temporal部分参数 # 时间融合模块位置修改后，也需要修改这里
                            print("!!!!!!!find one temporal_layer!!!!!!!")
                            print("backbone for spatial: ", k)
                            k_temporal = k.split('.')[0] + '_temporal.' + '.'.join(k.split('.')[1:-1]) + '.net.weight'
                            print("backbone for temporal: ", k_temporal)
                        else:
                            k_temporal = k.split('.')[0] + '_temporal.' + '.'.join(k.split('.')[1:])
                        temp_dict[k] = v
                        temp_dict[k_temporal] = v
                    temp_dict[k] = v
                    load_key.append(k)
                else:
                    no_load_key.append(k)
            model_dict.update(temp_dict)
            self.models['deeplabv3_plus'].load_state_dict(model_dict)
            # 显示没有匹配上的Key
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))


        # 设置优化参数
        total_param = []
        for key in self.models.keys():
            if 'deeplab' in key:
                self.models[key].to(self.device)
                total_param = list(self.models[key].parameters())
                self.backbone_parameters += list(self.models[key].backbone.parameters())
                self.head_parameters += list(set(total_param)^set(self.backbone_parameters))
        self.parameters_to_train = [{'params': self.backbone_parameters, 'lr':self.opt.lr*0.1},   # 设置backbone的学习率为整体学习率的0.1倍
                                    {'params':self.head_parameters, 'lr': self.opt.lr}]

        #-------------------------------------------------------------------#
        #   判断当前batch_size，自适应调整学习率
        #-------------------------------------------------------------------#
        nbs = 16
        Init_lr         = self.opt.lr
        Min_lr          = Init_lr * 0.01
        lr_limit_max    = self.opt.lr 
        lr_limit_min    = 0.01 *  lr_limit_max
        Init_lr_fit     = min(max(self.opt.batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(self.opt.batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        
        # 设置优化函数
        #self.model_optimizer = optim.Adam(self.parameters_to_train)
        self.model_optimizer = optim.AdamW(self.parameters_to_train, lr=self.opt.lr, weight_decay=self.opt.weight_decay)
        #self.model_optimizer = optim.SGD(self.parameters_to_train, lr = self.opt.lr, momentum=0.9, weight_decay=self.opt.weight_decay)  # TODO 权重衰减  权重衰减直接在括号里加上weight_decay=self.opt.weight_decay
        # 学习率调整策略
        self.lr_scheduler_func = get_lr_scheduler(lr_decay_type='cos', lr=Init_lr, min_lr=Min_lr_fit, total_iters=self.opt.num_epochs)  # warmup+余弦退火
        #self.scheduler = optim.lr_scheduler.ExponentialLR(self.model_optimizer, gamma=0.95)   # TODO  改成SGD+ExponentialLR
        #self.scheduler = StepLR(self.model_optimizer, step_size=step_size, gamma=0.65)
        #self.scheduler = MultiStepLR(self.model_optimizer, milestones=self.opt.lr_steps, gamma=0.1)
        # self.scheduler = CosineAnnealingLR(self.model_optimizer, T_max=15)  # iou 35.55

        # Data Loaders
        train_csv_file = self.opt.train_csv
        train_dataset = seqBEV_dataset(train_csv_file, transform=Img_ColorJitter())
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.opt.batch_size,   
            shuffle=True,
            num_workers=2*self.opt.num_workers,  # Needs images twice as fast
            pin_memory=True,
            drop_last=False)

        val_csv_file = self.opt.val_csv
        val_dataset = seqBEV_dataset(val_csv_file)
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.opt.val_batch_size,   
            shuffle=True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=False)

        test_csv_file = self.opt.test_csv
        test_dataset = seqBEV_dataset(test_csv_file)
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.opt.val_batch_size,   
            shuffle=True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=False)
        print(
            "There are {:d} training items, {:d} validation items, and {:d} test items\n".format(
                len(self.train_loader)*self.opt.batch_size,
                len(self.val_loader)*self.opt.batch_size,
                len(self.test_loader)*self.opt.batch_size))

    
    def get_visual_img(self, img, IF_Train=False):  # 调用create_visual_anno函数，将pred或者label添加颜色
        if IF_Train:
            num_to_show = self.opt.batch_size  # 在tensorboard中显示的图像数
        else:
            num_to_show = self.opt.val_batch_size
        cur_bev_list = img['bev'][:num_to_show].clone().cpu().data  # 传入的img是包含bev tensor和layout tensor的字典，取前三个frame. 该函数可以接收模型计算得到的labels和outputs作为输入
        IF_LABEL = True if len(cur_bev_list.shape)==3 else False
        # if IF_LABEL:
        #     cur_layout_list = img['layout'][:num_to_show].clone().cpu().data  # 传入label时，shape为[3, h, w]
        cur_layout_list = img['layout'][:num_to_show].clone().cpu().data
        num_img = 3
        # IF_LABEL = True if len(cur_bev_list.shape)==3 else False
        if not IF_LABEL:  # 若是pred
            cur_bev_list = np.reshape(np.argmax(cur_bev_list.numpy().transpose((0,2,3,1)), axis=3), [num_to_show, 150, 150]).astype(np.uint8)
            cur_layout_list = np.reshape(np.argmax(cur_layout_list.numpy().transpose((0,2,3,1)), axis=3), [num_to_show, 256, 512]).astype(np.uint8)
        else:
            cur_bev_list = cur_bev_list.numpy()
            cur_layout_list = cur_layout_list.numpy()
        if verbose: print("<<< in train.py, cur_bev_list.shape: ", cur_bev_list.shape)
        bev_c = []
        layout_c = []
        
        for i in range(num_to_show):
            #bev_pred.argmax(1).cpu().numpy().squeeze().flatten()
            #print("<<< in train.py, cur_bev_list[i].shape: ", torch.unique(cur_bev_list[i]))
            if verbose: print("<<< in train.py, np.max(cur_bev_list[i]): ", np.max(cur_bev_list[i]))
            
            bev_img = torch.from_numpy(create_visual_anno(cur_bev_list[i]).transpose((2, 0, 1)))
            bev_c.append(bev_img)
            # if IF_LABEL:
            #     layout_img = torch.from_numpy(create_visual_anno(cur_layout_list[i]).transpose((2, 0, 1)))
            #     layout_c.append(layout_img)
            layout_img = torch.from_numpy(create_visual_anno(cur_layout_list[i]).transpose((2, 0, 1)))
            layout_c.append(layout_img)
        if verbose: print("<<< in train.py, bev_c[0].shape: ", bev_c[0].shape)
        return bev_c, layout_c

    


    def train(self):
        global epoch_idx

        conf_total = np.zeros((self.opt.num_class, self.opt.num_class))  # 整体混淆矩阵
        loss = {          # "loss"为整体loss和，bev为BEV loss，layout为road layout loss  loss在self.criterion中计算
            "loss": 0.0,
            "bev": 0.0,
            "layout": 0.0
        }
        
        set_optimizer_lr(self.model_optimizer, self.lr_scheduler_func, epoch_idx)
        # iteration
        for batch_idx, input_data in tqdm.tqdm(enumerate(self.train_loader)):
            # 获取输入图像
            inputs = {}
            labels = {}
            img_seq, bev_labels, road_layout_labels, cur_name = input_data.values()
            img_seq = torch.autograd.Variable(img_seq).to(self.device)  # img_squence [B, 3*num_seq, H, W]  # 直接从dataloader中读入的类型为torch.tensor的输入和torch.autograd.Variable是一样的
            bev_labels = torch.autograd.Variable(bev_labels).to(self.device).long()
            road_layout_labels = torch.autograd.Variable(road_layout_labels).to(self.device).long()
            labels['bev'] = bev_labels
            labels['layout'] = road_layout_labels
            current_frame = torch.split(img_seq, self.opt.num_squence, dim=1)[1]  # 当前帧输入
            inputs['current_frame'] = current_frame
            inputs['seq_frames'] = img_seq
            
            # 进行迭代
            start_time = time.time()
            outputs, losses = self.process_batch(inputs, labels) # 返回一次mini_batch的iteration的outputs和losses
            bev_pred = outputs['bev']  
            layout_pred = outputs['layout']
            
            # 梯度更新
            self.model_optimizer.zero_grad()
            losses['loss'].backward()
            self.model_optimizer.step()
            self.global_step += 1
            #print("!!!! current lr", self.model_optimizer.state_dict()['param_groups'][0]['lr'])
 
            
            # 计算混淆矩阵
            label = labels['bev'].cpu().numpy().squeeze().flatten()
            prediction = bev_pred.argmax(1).cpu().numpy().squeeze().flatten()
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0,1,2,3,4,5,6])  # 标签0为background，在计算loss时是ignore_index
            conf_total += conf

            # 打印输出
            if batch_idx % self.opt.log_frequency == 0:  # 每隔log_frequency打印一次
                print_output = ("Epoch: [{epoch}/{total_epoch}] | Iter: [{global_step}/{total_step}] | Time: {time:.4f} | Basic lr: {basic_lr:.7f} | Loss: {loss:.8f} | BEV Loss: {bev:.8f} | Layout Loss: {layout:.8f}]".format(
                                epoch=epoch_idx, total_epoch=self.opt.num_epochs, global_step=batch_idx, total_step=len(self.train_loader), time=(time.time()-start_time), 
                                basic_lr=self.model_optimizer.param_groups[-1]['lr'], loss=losses['loss'], bev = losses['bev'], layout = losses['layout']))
                print(print_output)  # 屏幕输出结构
                self.log.write(print_output + '\n')  # 结果写入文件
                self.log.flush()

            # loss和lr写入tensorboard
            for loss_name in loss:  
                self.writer.add_scalar('train/'+loss_name, losses[loss_name], global_step=self.global_step)
            self.writer.add_scalar('train/lr', self.model_optimizer.param_groups[-1]['lr'], global_step=self.global_step)
            # tensorboard 显示图片
            img_1, img_2, img_3 = torch.split(img_seq, self.opt.num_squence, dim=1)  # 将[bs, c*num_seq, h, w]的img_seg拆成num_seq个[bs, c, h, w]维张量

            if batch_idx % (len(self.train_loader)//1) == 0:
                grid_image = torchvision.utils.make_grid(img_1[:self.opt.batch_size], 3, normalize=False)  #tensorboard显示输入图像(current frame)
                #print("*** in train.py train(), grid_image.shape: ", grid_image.shape)
                self.writer.add_image('train/current_frame', grid_image, self.global_step)
                grid_image = torchvision.utils.make_grid(img_2[:self.opt.batch_size], 3, normalize=False)  #tensorboard显示输入图像(previous frame)
                self.writer.add_image('train/pre_frame', grid_image, self.global_step)
                grid_image = torchvision.utils.make_grid(img_3[:self.opt.batch_size], 3, normalize=False)  #tensorboard显示输入图像(pre_previous frame)
                self.writer.add_image('train/pre_pre_frame', grid_image, self.global_step)
                # show BEV and layout label
                color_bev_gt, color_layout_gt = self.get_visual_img(labels, IF_Train=True)   # torchvision.utils.make_grid的输入是list或者是tensor，现在从self.get_visual_labels返回的是list
                grid_image = torchvision.utils.make_grid(color_bev_gt, 3, normalize=False)  #tensorboard显示bev gt图像
                #print("*** in train.py train(), grid_image.shape: ", grid_image.shape)
                self.writer.add_image('train/bev_ground_truth', grid_image, self.global_step)
                grid_image = torchvision.utils.make_grid(color_layout_gt, 3, normalize=False)  #tensorboard显示layout gt图像
                self.writer.add_image('train/layout_ground_truth', grid_image, self.global_step)
                # show BEV and layout pred
                color_bev_pred, color_layout_pred = self.get_visual_img(outputs, IF_Train=True)
                grid_image = torchvision.utils.make_grid(color_bev_pred, 3, normalize=False)  #tensorboard显示bev pred图像
                self.writer.add_image('train/bev_pred', grid_image, self.global_step)
                grid_image = torchvision.utils.make_grid(color_layout_pred, 3, normalize=False)  #tensorboard显示layout pred图像
                self.writer.add_image('train/layout_pred', grid_image, self.global_step)
                # show the attention map
                # TODO

            for loss_name in losses:
                # if not 'layout' in loss_name:
                #     loss[loss_name] += losses[loss_name].item() 
                loss[loss_name] += losses[loss_name].item()

        # 学习率更新
        #self.scheduler.step()  # torch自带的学习率更新
        # set_optimizer_lr(self.model_optimizer, self.lr_scheduler_func, epoch_idx)
        # 计算一个epoch中所有的mini_batch的loss均值
        for loss_name in loss:  # 求一个epoch中的每个mini_batch loss的平均
            loss[loss_name] /= len(self.train_loader)
        # 计算评价指标
        precision_per_class, recall_per_class, iou_per_class = compute_results(conf_total)  # TODO 计算每一个语义类别的iou
        average_precision = precision_per_class.mean()
        average_recall = recall_per_class.mean()
        average_IoU = iou_per_class.mean()
        # 将一个epoch的评价指标写入tensorboard中
        self.writer.add_scalar('train/average_loss', loss['loss'], epoch_idx)
        self.writer.add_scalar('train/average_precision', average_precision, epoch_idx)
        self.writer.add_scalar('train/average_recall', average_recall, epoch_idx)
        self.writer.add_scalar('train/average_IoU', average_IoU, epoch_idx)
        # 将IoU等写入文件
        # metirc_outputs = (' *val* average_precision {average_precision:.6f}\taverage_recall {average_recall:.6f}\taverage_IoU {average_IoU:.6f}\n'
        # .format(average_precision=average_precision, average_recall=average_recall, average_IoU=average_IoU))
        metirc_outputs = (">>>average_precision: {mAP:.8f} | average_recall: {average_recall:.8f} | average_IoU: {mIoU:.8f}\n>>>Loss: {loss:.8f} | BEV Loss: {bev_loss:.8f} | Layout Loss: {layout_loss:.8f}".format(
            mAP=average_precision, average_recall=average_recall, mIoU=average_IoU, loss=loss['loss'], bev_loss=loss['bev'], layout_loss=loss['layout']
        ))
        print(metirc_outputs)  # 屏幕输出结构
        self.log.write(metirc_outputs + '\n')  # 结果写入文件
        self.log.flush()


        

    def process_batch(self, inputs, labels, IF_Train=True):  # 每次epoch中的mini_batch的iteration
        global epoch_idx
    
        img_seq = inputs['seq_frames']
        current_frame = inputs['current_frame']
        outputs = {}

        # 冻结参数
        if self.opt.freeze_train and epoch_idx < self.opt.Freeze_Epoch:  # 当epoch小于Freeze Epoch时，冻结backbone参数
            #print("in train_with_STM_freeze, process batch, Freezing backbone")
            for param in self.models['deeplabv3_plus'].backbone.parameters():
                param.requires_grad = False
        elif self.opt.freeze_train and epoch_idx > self.opt.Freeze_Epoch:
            #print("in train_with_STM_freeze, process batch, ***UN***Freezing backbone")
            for param in self.models['deeplabv3_plus'].backbone.parameters():
                param.requires_grad = True

        # 网络搭建
        if not IF_Train:
            for key in self.models.keys():  # 将model转化成eval模式，使用固定的dropout网络和bn层的均值方差
                self.models[key].eval()
        else:
            for key in self.models.keys():  # 将model转化成train模式。注意在TS_Encoder中重写了.train()函数，需要显式声明为train模式
                self.models[key].train()
        # 先是输入current_frame
        outputs['bev'], outputs['layout'] = self.models['deeplabv3_plus'](img_seq, epoch_idx+1)  # 送入序列图像

        losses = self.criterion(self.opt, labels, outputs)  # labels为包含bev_labels和layout_labels的字典，outputs为包含bev_pred和layout_pred的字典
        
        return outputs, losses  # 一次mini_batch的outputs和losses，在val传回时可以先在epoch内进行类加，然后在除len(val_dataloader)做平均


    def validation(self):  # TODO 能在tensorboard中看到IOU和loss
        global epoch_idx, is_best, best_iou, best_epoch

        conf_total = np.zeros((self.opt.num_class, self.opt.num_class))  # 整体混淆矩阵
        loss = {          # "loss"为整体loss和，bev为BEV loss，layout为road layout loss  loss在self.criterion中计算
            "loss": 0.0,
            "bev": 0.0,
            "layout": 0.0
        }
        
        # iteration
        for batch_idx, input_data in tqdm.tqdm(enumerate(self.val_loader)):  
            # 获取输入图像
            inputs = {}
            labels = {}
            img_seq, bev_labels, road_layout_labels, cur_name = input_data.values()
            img_seq = torch.autograd.Variable(img_seq).to(self.device)  # img_squence [B, 3*num_seq, H, W]  # 类型为torch.tensor的输入和torch.autograd.Variable是一样的
            bev_labels = torch.autograd.Variable(bev_labels).to(self.device).long()
            road_layout_labels = torch.autograd.Variable(road_layout_labels).to(self.device).long()
            labels['bev'] = bev_labels
            labels['layout'] = road_layout_labels
            current_frame = torch.split(img_seq, self.opt.num_squence, dim=1)[1]  # 当前帧输入
            inputs['current_frame'] = current_frame
            inputs['seq_frames'] = img_seq
            
            # 进行迭代
            start_time = time.time()
            with torch.no_grad():   # TODO with torch.no_grad()，以节省GPU算力和显存
                outputs, losses = self.process_batch(inputs, labels, IF_Train=False) # 返回一次mini_batch的iteration的outputs和losses
            bev_pred = outputs['bev']  
            layout_pred = outputs['layout']

            # 计算混淆矩阵
            label = labels['bev'].cpu().numpy().squeeze().flatten()
            prediction = bev_pred.argmax(1).cpu().numpy().squeeze().flatten()
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0,1,2,3,4,5,6])  # 标签0为background，在计算loss时是ignore_index
            #conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0,1,2,3])  #FIXME 只保留layout类
            conf_total += conf

            # 打印输出
            if batch_idx % self.opt.log_frequency == 0:  # 每隔log_frequency打印一次
                print_output = ("--validating-- Epoch: [{epoch}/{total_epoch}] | Iter: [{global_step}/{total_step}] | Time: {time:.4f} | Img/Sec: {num_img:.2f} | Loss: {loss:.8f} | BEV Loss: {bev:.8f} | Layout Loss: {layout:.8f}]".format(
                                epoch=epoch_idx, total_epoch=self.opt.num_epochs, global_step=batch_idx, total_step=len(self.val_loader), time=(time.time()-start_time), 
                                num_img=self.opt.val_batch_size/(time.time()-start_time), loss=losses['loss'], bev = losses['bev'], layout = losses['layout']))
                print(print_output)  # 屏幕输出结构
                self.log.write(print_output + '\n')  # 结果写入文件
                self.log.flush()

            # loss和lr写入tensorboard
            for loss_name in loss:  
                self.writer.add_scalar('val/'+loss_name, losses[loss_name], global_step=self.global_step)

            # tensorboard 显示图片
            img_1, img_2, img_3 = torch.split(img_seq, self.opt.num_squence, dim=1)  # 将[bs, c*num_seq, h, w]的img_seg拆成num_seq个[bs, c, h, w]维张量
            if batch_idx % (len(self.val_loader)//1) == 0:
                grid_image = torchvision.utils.make_grid(img_1[:self.opt.val_batch_size], 3, normalize=False)  #tensorboard显示输入图像(current frame)
                #print("*** in train.py train(), grid_image.shape: ", grid_image.shape)
                self.writer.add_image('val/current_frame', grid_image, self.global_step)
                grid_image = torchvision.utils.make_grid(img_2[:self.opt.val_batch_size], 3, normalize=False)  #tensorboard显示输入图像(previous frame)
                self.writer.add_image('val/pre_frame', grid_image, self.global_step)
                grid_image = torchvision.utils.make_grid(img_3[:self.opt.val_batch_size], 3, normalize=False)  #tensorboard显示输入图像(pre_previous frame)
                self.writer.add_image('val/pre_pre_frame', grid_image, self.global_step)
                # show BEV and layout label
                color_bev_gt, color_layout_gt = self.get_visual_img(labels, IF_Train=False)   # torchvision.utils.make_grid的输入是list或者是tensor，现在从self.get_visual_labels返回的是list
                grid_image = torchvision.utils.make_grid(color_bev_gt, 3, normalize=False)  #tensorboard显示bev gt图像
                #print("*** in train.py train(), grid_image.shape: ", grid_image.shape)
                self.writer.add_image('val/bev_ground_truth', grid_image, self.global_step)
                grid_image = torchvision.utils.make_grid(color_layout_gt, 3, normalize=False)  #tensorboard显示layout gt图像
                self.writer.add_image('val/layout_ground_truth', grid_image, self.global_step)
                # show BEV and layout pred
                color_bev_pred, color_layout_pred = self.get_visual_img(outputs, IF_Train=False)
                grid_image = torchvision.utils.make_grid(color_bev_pred, 3, normalize=False)  #tensorboard显示bev pred图像
                self.writer.add_image('val/bev_pred', grid_image, self.global_step)
                grid_image = torchvision.utils.make_grid(color_layout_pred, 3, normalize=False)  #tensorboard显示layout pred图像
                self.writer.add_image('val/layout_pred', grid_image, self.global_step)

            for loss_name in losses:
                # if not 'layout' in loss_name:
                #     loss[loss_name] += losses[loss_name].item()
                loss[loss_name] += losses[loss_name].item()
        # 计算一个epoch中所有的mini_batch的loss均值
        for loss_name in loss:  # 求一个epoch中的每个mini_batch loss的平均
            loss[loss_name] /= len(self.val_loader)

        # 计算评价指标
        precision_per_class, recall_per_class, iou_per_class = compute_results(conf_total)  
        average_precision = precision_per_class.mean()
        average_recall = recall_per_class.mean()
        average_IoU = iou_per_class.mean()
        # 将一个epoch的评价指标写入tensorboard中
        self.writer.add_scalar('val/average_loss', loss['loss'], epoch_idx)
        self.writer.add_scalar('val/average_precision', average_precision, epoch_idx)
        self.writer.add_scalar('val/average_recall', average_recall, epoch_idx)
        self.writer.add_scalar('val/average_IoU', average_IoU, epoch_idx)
        # 将IoU等写入文件
        metirc_outputs = (">>>Val: average_precision: {mAP:.8f} | average_recall: {average_recall:.8f} | average_IoU: {mIoU:.8f}\n>>>     Loss: {loss:.8f} | BEV Loss: {bev_loss:.8f} | Layout Loss: {layout_loss:.8f} | Time Used: {time}".format(
            mAP=average_precision, average_recall=average_recall, mIoU=average_IoU, loss=loss['loss'], bev_loss=loss['bev'], layout_loss=loss['layout'], time=datetime.now().replace(microsecond=0)-start_datetime
        ))
        precision_record = {}  # 记录每个语义类的评价指标
        recall_record = {}
        iou_record = {}
        for i in range(len(iou_per_class)):  
            precision_record[label_list[i]] = precision_per_class[i]
            recall_record[label_list[i]] = recall_per_class[i]
            iou_record[label_list[i]] = iou_per_class[i]
        print(metirc_outputs)  # 屏幕输出结构
        metirc_each_class = ("precision for each class: {}\nrecall for each class: {}\niou for each class: {}".format(precision_record, recall_record, iou_record))
        print(metirc_each_class)
        self.log.write(metirc_outputs + '\n')  # 结果写入文件
        self.log.write(metirc_each_class + '\n')
        self.log.flush()

        # 计算best iou
        is_best = average_IoU > best_iou
        best_iou = max(average_IoU, best_iou)
        if is_best:
            best_epoch = epoch_idx
        best_content = ('*** best VAL iou is {best_iou:.4f} at {epoch} epoch'.format(best_iou=best_iou, epoch=best_epoch))
        print(best_content)
        self.log.write(best_content + '\n')  # 结果写入文件
        self.log.flush()

    def test(self):
        global epoch_idx, is_best_test, best_test_iou, best_test_epoch
        IF_save = self.opt.if_save_img

        conf_total = np.zeros((self.opt.num_class, self.opt.num_class))  # 整体混淆矩阵
        loss = {          # "loss"为整体loss和，bev为BEV loss，layout为road layout loss  loss在self.criterion中计算
            "loss": 0.0,
            "bev": 0.0,
            "layout": 0.0
        }
        
        # iteration
        for batch_idx, input_data in tqdm.tqdm(enumerate(self.test_loader)):
            # 获取输入图像
            inputs = {}
            labels = {}
            img_seq, bev_labels, road_layout_labels, cur_name = input_data.values()
            img_seq = torch.autograd.Variable(img_seq).to(self.device)  # img_squence [B, 3*num_seq, H, W]  # 类型为torch.tensor的输入和torch.autograd.Variable是一样的
            bev_labels = torch.autograd.Variable(bev_labels).to(self.device).long()
            road_layout_labels = torch.autograd.Variable(road_layout_labels).to(self.device).long()
            labels['bev'] = bev_labels
            labels['layout'] = road_layout_labels
            current_frame = torch.split(img_seq, self.opt.num_squence, dim=1)[1]  # 当前帧输入
            inputs['current_frame'] = current_frame
            inputs['seq_frames'] = img_seq
            
            # 进行迭代
            start_time = time.time()
            with torch.no_grad():
                outputs, losses = self.process_batch(inputs, labels, IF_Train=False) # 返回一次mini_batch的iteration的outputs和losses
            bev_pred = outputs['bev']  
            layout_pred = outputs['layout']

            # 计算混淆矩阵
            label = labels['bev'].cpu().numpy().squeeze().flatten()
            prediction = bev_pred.argmax(1).cpu().numpy().squeeze().flatten()
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0,1,2,3,4,5,6])  # 标签0为background，在计算loss时是ignore_index
            #conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0,1,2,3])  #FIXME 只保留layout类
            conf_total += conf

            # 在最后一个epoch保存test的图片
            if epoch_idx == (self.opt.num_epochs-1) and IF_save:
                color_bev_pred, color_layout_pred = self.get_visual_img(outputs, IF_Train=False)
                color_bev_gt, color_layout_gt = self.get_visual_img(labels, IF_Train=False)
                img_path = self.opt.save_img_path + self.opt.model_name + '/'
                if not os.path.exists(img_path):
                    os.makedirs(img_path)
                print("saving test image... ")
                print("outputs['bev'].shape: ", np.argmax(outputs['bev'].cpu().detach().numpy().transpose((0,2,3,1)), axis=3).shape)
                map_to_save = np.reshape(np.argmax(outputs['bev'].cpu().detach().numpy().transpose((0,2,3,1)), axis=3), [self.opt.val_batch_size, 150, 150]).astype(np.uint8)
                for i in range(self.opt.val_batch_size):
                    sample_token = ''.join(cur_name[i].split('.')[0].split('_')[-1])
                    scene_name = cur_name[i].split('.')[0].split('_')[0]
                    img_path_num = os.path.join(img_path, scene_name)
                    img_path_num = img_path_num + '/'
                    if not os.path.exists(img_path_num):
                        os.makedirs(img_path_num)
                    skimage.io.imsave(img_path_num + sample_token + '.png', current_frame[i].cpu().detach().numpy().transpose((1, 2, 0)))  # 保存current_frame
                    skimage.io.imsave(img_path_num + sample_token + '_label.png', color_bev_gt[i].cpu().detach().numpy().transpose((1, 2, 0)))  # 保存gt
                    skimage.io.imsave(img_path_num + sample_token + '_nn_pred.png', map_to_save[i])  # 保存单通道prediction
                    skimage.io.imsave(img_path_num + sample_token + '_nn_pred_c.png', color_bev_pred[i].cpu().detach().numpy().transpose((1, 2, 0)))  # 保存彩色prediction
                    

            # 打印输出
            if batch_idx % self.opt.log_frequency == 0:  # 每隔log_frequency打印一次
                print_output = ("--testing-- Epoch: [{epoch}/{total_epoch}] | Iter: [{global_step}/{total_step}] | Time: {time:.4f} | Img/Sec: {num_img:.2f} | Loss: {loss:.8f} | BEV Loss: {bev:.8f} | Layout Loss: {layout:.8f}]".format(
                                epoch=epoch_idx, total_epoch=self.opt.num_epochs, global_step=batch_idx, total_step=len(self.test_loader), time=(time.time()-start_time), 
                                num_img=self.opt.val_batch_size/(time.time()-start_time), loss=losses['loss'], bev = losses['bev'], layout = losses['layout']))
                print(print_output)  # 屏幕输出结构
                self.log.write(print_output + '\n')  # 结果写入文件
                self.log.flush()

            # loss和lr写入tensorboard
            for loss_name in loss:  
                self.writer.add_scalar('test/'+loss_name, losses[loss_name], global_step=self.global_step)
            # tensorboard 显示图片
            img_1, img_2, img_3 = torch.split(img_seq, self.opt.num_squence, dim=1)  # 将[bs, c*num_seq, h, w]的img_seg拆成num_seq个[bs, c, h, w]维张量
            if batch_idx % (len(self.test_loader)//1) == 0:
                grid_image = torchvision.utils.make_grid(img_1[:self.opt.val_batch_size], 3, normalize=False)  #tensorboard显示输入图像(current frame)
                #print("*** in train.py train(), grid_image.shape: ", grid_image.shape)
                self.writer.add_image('test/current_frame', grid_image, self.global_step)
                grid_image = torchvision.utils.make_grid(img_2[:self.opt.val_batch_size], 3, normalize=False)  #tensorboard显示输入图像(previous frame)
                self.writer.add_image('test/pre_frame', grid_image, self.global_step)
                grid_image = torchvision.utils.make_grid(img_3[:self.opt.val_batch_size], 3, normalize=False)  #tensorboard显示输入图像(pre_previous frame)
                self.writer.add_image('test/pre_pre_frame', grid_image, self.global_step)
                # show BEV and layout label
                color_bev_gt, color_layout_gt = self.get_visual_img(labels, IF_Train=False)   # torchvision.utils.make_grid的输入是list或者是tensor，现在从self.get_visual_labels返回的是list
                grid_image = torchvision.utils.make_grid(color_bev_gt, 3, normalize=False)  #tensorboard显示bev gt图像
                #print("*** in train.py train(), grid_image.shape: ", grid_image.shape)
                self.writer.add_image('test/bev_ground_truth', grid_image, self.global_step)
                grid_image = torchvision.utils.make_grid(color_layout_gt, 3, normalize=False)  #tensorboard显示layout gt图像
                self.writer.add_image('test/layout_ground_truth', grid_image, self.global_step)
                # show BEV and layout pred
                color_bev_pred, color_layout_pred = self.get_visual_img(outputs, IF_Train=False)
                grid_image = torchvision.utils.make_grid(color_bev_pred, 3, normalize=False)  #tensorboard显示bev pred图像
                self.writer.add_image('test/bev_pred', grid_image, self.global_step)
                grid_image = torchvision.utils.make_grid(color_layout_pred, 3, normalize=False)  #tensorboard显示layout pred图像
                self.writer.add_image('test/layout_pred', grid_image, self.global_step)
                # show the attention map
                # TODO

            for loss_name in losses:
                # if not 'layout' in loss_name:
                #     loss[loss_name] += losses[loss_name].item() 
                loss[loss_name] += losses[loss_name].item()

        # 计算一个epoch中所有的mini_batch的loss均值
        for loss_name in loss:  # 求一个epoch中的每个mini_batch loss的平均
            loss[loss_name] /= len(self.val_loader)

        # 计算评价指标
        precision_per_class, recall_per_class, iou_per_class = compute_results(conf_total)  # TODO 计算每一个语义类别的iou
        average_precision = precision_per_class.mean()
        average_recall = recall_per_class.mean()
        average_IoU = iou_per_class.mean()
        # 将一个epoch的评价指标写入tensorboard中
        self.writer.add_scalar('test/average_loss', loss['loss'], epoch_idx)
        self.writer.add_scalar('test/average_precision', average_precision, epoch_idx)
        self.writer.add_scalar('test/average_recall', average_recall, epoch_idx)
        self.writer.add_scalar('test/average_IoU', average_IoU, epoch_idx)
        # 将IoU等写入文件
        metirc_outputs = (">>>Test: average_precision: {mAP:.8f} | average_recall: {average_recall:.8f} | average_IoU: {mIoU:.8f}\n>>>      Loss: {loss:.8f} | BEV Loss: {bev_loss:.8f} | Layout Loss: {layout_loss:.8f} | Time Used: {time}".format(
            mAP=average_precision, average_recall=average_recall, mIoU=average_IoU, loss=loss['loss'], bev_loss=loss['bev'], layout_loss=loss['layout'], time=datetime.now().replace(microsecond=0)-start_datetime
        ))
        precision_record = {}  # 记录每个语义类的评价指标
        recall_record = {}
        iou_record = {}
        for i in range(len(iou_per_class)):  
            precision_record[label_list[i]] = precision_per_class[i]
            recall_record[label_list[i]] = recall_per_class[i]
            iou_record[label_list[i]] = iou_per_class[i]
        print(metirc_outputs)  # 屏幕输出结构
        metirc_each_class = ("precision for each class: {}\nrecall for each class: {}\niou for each class: {}".format(precision_record, recall_record, iou_record))
        print(metirc_each_class)
        self.log.write(metirc_outputs + '\n')  # 结果写入文件
        self.log.write(metirc_each_class + '\n')
        self.log.flush()

        # 计算best iou
        is_best_test = average_IoU > best_test_iou
        best_test_iou = max(average_IoU, best_test_iou)
        if is_best_test:
            best_test_epoch = epoch_idx 
        best_content = ('*** best TEST iou is {best_iou:.4f} at {epoch} epoch'.format(best_iou=best_test_iou, epoch=best_test_epoch))
        print(best_content)
        self.log.write(best_content + '\n')  # 结果写入文件
        self.log.flush()


    # 保存模型
    def save_model(self):
        global epoch_idx, is_best, best_iou, best_epoch, is_best_test, best_test_iou, best_test_epoch
        
        save_path = os.path.join(self.save_path, self.opt.model_name, self.create_time)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        checkpoint = save_path + '/checkpoint.pth'
        best_path = save_path + '/val_best.pth'
        best_test_path = save_path + '/test_best.pth'

        model_state_dicts = {}
        for model_name, model in self.models.items():
            model_path = os.path.join(save_path, "{}.pth".format(model_name))
            state_dict = model.state_dict()
            # state_dict['epoch'] = self.epoch
            # if "branch" in model_name:
            #     state_dict["height"] = self.opt.height
            #     state_dict["width"] = self.opt.width
            model_state_dicts[model_name] = state_dict
            torch.save(state_dict, model_path)

        # 定义存储状态
        save_state = {}
        save_state['epoch'] = epoch_idx
        save_state['state_dict'] = model_state_dicts
        save_state['best_iou'] = best_iou
        save_state['best_epoch'] = best_epoch
        save_state['best_test_iou'] = best_test_iou
        save_state['best_test_epoch'] = best_test_epoch
        save_state['optimizer'] = self.model_optimizer.state_dict()
        optim_path = os.path.join(save_path, "{}.pth".format("optim"))
        torch.save(self.model_optimizer.state_dict(), optim_path)

        # 保存最新一次的模型和best_iou时的模型
        torch.save(save_state, checkpoint)
        print("--- checkpoint saved to %s ---" % checkpoint)
        if is_best:
            shutil.copyfile(checkpoint, best_path)
            print("--- checkpoint copied to %s ---" % best_path)
        if is_best_test:
            shutil.copyfile(checkpoint, best_test_path)
            print("--- checkpoint copied to %s ---" % best_test_path)

    def load_model(self):
        """
        Load models from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(
            self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print(
            "loading model from folder {}".format(
                self.opt.load_weights_folder)) 

        for key in self.models.keys():
            print("Loading {} weights...".format(key))
            path = os.path.join(
                self.opt.load_weights_folder,
                "{}.pth".format(key))
            model_dict = self.models[key].state_dict()
            pretrained_dict = torch.load(path)
            if 'epoch' in pretrained_dict:
                self.start_epoch = pretrained_dict['epoch']
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[key].load_state_dict(model_dict)

        # loading optim state
        if self.opt.load_weights_folder == "":
            optimizer_load_path = os.path.join(
                self.opt.load_weights_folder, "optim.pth")
            if os.path.isfile(optimizer_load_path):
                print("Loading Optim weights")
                optimizer_dict = torch.load(optimizer_load_path)
                self.model_optimizer.load_state_dict(optimizer_dict)
            else:
                print("Cannot find Adam weights so Adam is randomly initialized")

    def adjust_learning_rate(self, optimizer, epoch, lr_steps):
        """Sets the learning rate to the initial LR decayed by 10 every 25 epochs"""
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        decay = round(decay, 2)
        lr = self.opt.lr * decay
        decay = self.opt.weight_decay
        # 需要根据optimizer的具体情况修改param_groups[0]
        optimizer.param_groups[0]['lr'] = lr
        optimizer.param_groups[0]['weight_decay'] = decay


            


# 代码测试
if __name__ == '__main__':
    epoch_idx = 0
    args = get_args()  # 倒入参数列表
    if not os.path.isdir(args.log_root):
        os.mkdir(args.log_root)
    epoch_idx = args.start_epoch
    trainer = Trainer(args)

    # 开始训练
    best_iou = 0
    best_epoch = 0
    best_test_iou = 0
    best_test_epoch = 0
    is_best = False
    is_best_test = False
    start_epoch = epoch_idx
    for epoch in range(start_epoch, args.num_epochs):
        epoch_idx = epoch
        print("!!!!! epoch_idx: ", epoch_idx)
        print("Training the model...")
        start_time = time.time()
        trainer.train()
        print("--- training epoch in %s seconds ---"%(time.time()-start_time))

        print("Evaluating the model...")
        start_time = time.time()
        trainer.validation()
        print("--- validating epoch in %s seconds ---"%(time.time()-start_time))

        print("testing the model...")
        start_time = time.time()
        trainer.test()
        print("--- testing epoch in %s seconds ---"%(time.time()-start_time))

        trainer.save_model()



    