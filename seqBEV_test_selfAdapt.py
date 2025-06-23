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
from src.data.seqBEV_dataset import seqBEV_dataset
#from src.seqBEV.deeplabv3_plus_SEAttention import DeepLab
from src.seqBEV.deeplabv3_plus_STM_multi_inputs_decoder_TFM_ablation_selfAdoption_GPU3 import DeepLab

#from src.seqBEV.TS_fusion import TS_Fusion
from src.utils.configs import get_default_configuration, load_config
from src.utils.confusion import BinaryConfusionMatrix
from src.data.nuscenes.utils import NUSCENES_CLASS_NAMES
from src.data.argoverse.utils import ARGOVERSE_CLASS_NAMES
from src.utils.visualise import colorise
from src.data.utils import create_visual_anno

IF_SAVE = False
print('current device: ', torch.cuda.current_device())

def get_visual_img(img, IF_Train=True):  # 调用create_visual_anno函数，将pred或者label添加颜色
    if IF_Train:
        num_to_show = 3  # 在tensorboard中显示的图像数
    else:
        num_to_show = opt.val_batch_size
    cur_bev_list = img['bev'][:num_to_show].clone().cpu().data  # 传入的img是包含bev tensor和layout tensor的字典，取前三个frame. 该函数可以接收模型计算得到的labels和outputs作为输入
    cur_layout_list = img['layout'][:num_to_show].clone().cpu().data  # 传入label时，shape为[3, h, w]
    IF_LABEL = True if len(cur_bev_list.shape)==3 else False
    if not IF_LABEL:  # 若是pred
        #print("<<< in train.py, its pred!!!")
        cur_bev_list = np.reshape(np.argmax(cur_bev_list.numpy().transpose((0,2,3,1)), axis=3), [num_to_show, 150, 150]).astype(np.uint8)
        cur_layout_list = np.reshape(np.argmax(cur_layout_list.numpy().transpose((0,2,3,1)), axis=3), [num_to_show, 256, 512]).astype(np.uint8)
    else:
        #print("<<< in train.py, its label!!!")
        cur_bev_list = cur_bev_list.numpy()
        cur_layout_list = cur_layout_list.numpy()
    #if verbose: print("<<< in train.py, cur_bev_list.shape: ", cur_bev_list.shape)
    bev_c = []
    layout_c = []
    
    for i in range(num_to_show):  # TODO 检查为什么val和test时会显示两张图像？
        #print("<<< in train.py, cur_bev_list[i].shape: ", torch.unique(cur_bev_list[i]))
        #if verbose: print("<<< in train.py, np.max(cur_bev_list[i]): ", np.max(cur_bev_list[i]))
        
        bev_img = torch.from_numpy(create_visual_anno(cur_bev_list[i]).transpose((2, 0, 1)))
        layout_img = torch.from_numpy(create_visual_anno(cur_layout_list[i]).transpose((2, 0, 1)))
        bev_c.append(bev_img)
        layout_c.append(layout_img)
    #print("<<< in train.py, len(bev_c): ", len(bev_c))
    #if verbose: print("<<< in train.py, bev_c[0].shape: ", bev_c[0].shape)
    return bev_c, layout_c


if __name__ == "__main__":
    opt = get_args()
    device = 'cuda'
    models = DeepLab(num_classes=7, backbone="mobilenet", downsample_factor=8, pretrained=opt.pretrained_backbone).to(device)
    pretrained_model_path = './best_weights/2022-11-20-03-52/test_best.pth'
    start_datetime = datetime.now().replace(microsecond=0)
    label_list = ['background', 'drivable_area', 'ped_crossing', 'walkway', 'movable_object', 'vehicle', 'predestrian']


# 载入deeplabv3_plus的预训练参数
    if pretrained_model_path != '':
        print('Load weights {}'.format(pretrained_model_path))
        # 根据预训练权重的Key和模型的Key进行加载
        model_dict = models.state_dict()
        checkpoint = torch.load(pretrained_model_path, map_location=device)
        pretrained_dict = checkpoint['state_dict']
        #pretrained_dict = torch.load(pretrained_model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        temporal_layer = [2, 4, 7, 14]
        for k, v in pretrained_dict['deeplabv3_plus'].items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        models.load_state_dict(model_dict)
        # 显示没有匹配上的Key
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))



    # test data loader
    test_csv_file = opt.test_csv
    test_dataset = seqBEV_dataset(test_csv_file)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.val_batch_size,
        #batch_size=8,   
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True)

    print("loading {} test data: ".format(len(test_loader)*opt.val_batch_size))
    
    # 开始测试
    conf_total = np.zeros((opt.num_class, opt.num_class))  # 整体混淆矩阵
    print("!!!!!!! in seqBEV_test_selfAdapt.py!!!!!!!")
    models.eval()
    print(models)
    with torch.no_grad():
        for batch_idx, input_data in tqdm.tqdm(enumerate(test_loader)):
            # 获取输入图像
            inputs = {}
            labels = {}
            img_seq, bev_labels, road_layout_labels, cur_name = input_data.values()
            img_seq = torch.autograd.Variable(img_seq).to(device)  # img_squence [B, 3*num_seq, H, W]  # 类型为torch.tensor的输入和torch.autograd.Variable是一样的
            bev_labels = torch.autograd.Variable(bev_labels).to(device).long()
            road_layout_labels = torch.autograd.Variable(road_layout_labels).to(device).long()
            labels['bev'] = bev_labels
            labels['layout'] = road_layout_labels
            current_frame = torch.split(img_seq, opt.num_squence, dim=1)[1]  # 当前帧输入
            inputs['current_frame'] = current_frame
            inputs['seq_frames'] = img_seq
            
            # 进行迭代
            start_time = time.time()
            outputs = {}
            outputs['bev'], outputs['layout'] = models(img_seq, 34) # 返回一次mini_batch的iteration的outputs和losses
            bev_pred = outputs['bev']  
            layout_pred = outputs['layout']

            # 计算test的mIoU等参数
            # 计算混淆矩阵
            label = labels['bev'].cpu().numpy().squeeze().flatten()
            prediction = bev_pred.argmax(1).cpu().numpy().squeeze().flatten()
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0,1,2,3,4,5,6])  # 标签0为background，在计算loss时是ignore_index
            conf_total += conf
        
            if IF_SAVE:
                # 保存test的图片
                color_bev_pred, color_layout_pred = get_visual_img(outputs, IF_Train=False)
                color_bev_gt, color_layout_gt = get_visual_img(labels, IF_Train=False)
                img_path = opt.save_img_path + opt.model_name + '/'
                if not os.path.exists(img_path):
                    os.makedirs(img_path)
                print("saving test image... ")
                print("outputs['bev'].shape: ", np.argmax(outputs['bev'].cpu().detach().numpy().transpose((0,2,3,1)), axis=3).shape)
                map_to_save = np.reshape(np.argmax(outputs['bev'].cpu().detach().numpy().transpose((0,2,3,1)), axis=3), [opt.val_batch_size, 150, 150]).astype(np.uint8)

                for i in range(opt.val_batch_size):
                #for i in range(8):
                    sample_token = ''.join(cur_name[i].split('.')[0].split('_')[-1])
                    sample_idx = cur_name[i].split('_')[1]
                    sample_name = sample_idx + '_' + sample_token
                    scene_name = cur_name[i].split('.')[0].split('_')[0]
                    img_path_num = os.path.join(img_path, scene_name)
                    img_path_num = img_path_num + '/'
                    if not os.path.exists(img_path_num):
                        os.makedirs(img_path_num)
                    skimage.io.imsave(img_path_num + sample_name + '.png', current_frame[i].cpu().detach().numpy().transpose((1, 2, 0)))  # 保存current_frame
                    skimage.io.imsave(img_path_num + sample_name + '_label.png', color_bev_gt[i].cpu().detach().numpy().transpose((1, 2, 0)))  # 保存gt
                    skimage.io.imsave(img_path_num + sample_name + '_nn_pred.png', map_to_save[i])  # 保存单通道prediction
                    skimage.io.imsave(img_path_num + sample_name + '_nn_pred_c.png', color_bev_pred[i].cpu().detach().numpy().transpose((1, 2, 0)))  # 保存彩色prediction
    # 计算评价指标
    precision_per_class, recall_per_class, iou_per_class = compute_results(conf_total)  # TODO 计算每一个语义类别的iou
    average_precision = precision_per_class.mean()
    average_recall = recall_per_class.mean()
    average_IoU = iou_per_class.mean()
    metirc_outputs = (">>>Test: average_precision: {mAP:.8f} | average_recall: {average_recall:.8f} | average_IoU: {mIoU:.8f} Time Used: {time}".format(
                      mAP=average_precision, average_recall=average_recall, mIoU=average_IoU, time=datetime.now().replace(microsecond=0)-start_datetime
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
    print("Finish")
        

