import pandas as pd
import os
import torch
import random
import math
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

#from src.seqBEV.Road_layout import road_layout

verbose = False

class seqBEV_dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.examples = pd.read_csv(csv_file, header=None)
        self.transform = transform

        imgs = []
        for i in range(len(self.examples)):
            pre_img_path = self.examples.iloc[i, 0]
            cur_img_path = self.examples.iloc[i, 1]
            next_img_path = self.examples.iloc[i, 2]
            bev_map_path = self.examples.iloc[i, 3]
            rl_map_path = self.examples.iloc[i, 4]
            img = (pre_img_path, cur_img_path, next_img_path, bev_map_path, rl_map_path)
            imgs.append(img)
        self.imgs = imgs

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        pre_img = np.asarray(io.imread(self.examples.iloc[item, 0]))
        cur_img = np.asarray(io.imread(self.examples.iloc[item, 1]))
        next_img = np.asarray(io.imread(self.examples.iloc[item, 2]))
        bev_map_orig = np.asarray(io.imread(self.examples.iloc[item, 3]))
        #bev_map = np.where(bev_map_orig>=4, 0, bev_map_orig)  # 7个类别，其中0为背景类  labels = np.where(labels_orig==7, 0, labels_orig)
        bev_map = np.where(bev_map_orig==7, 0, bev_map_orig)
        rl_map = np.asarray(io.imread(self.examples.iloc[item, 4]))

        pre_img_name = self.examples.iloc[item, 0].split('/')[-1]
        cur_img_name = self.examples.iloc[item, 1].split('/')[-1]
        cur_scene = self.examples.iloc[item, 1].split('/')[-3]
        cur_name = cur_scene + '_' + cur_img_name
        next_img_name = self.examples.iloc[item, 2].split('/')[-1]
        bev_map_name = self.examples.iloc[item, 3].split('/')[-1]
        rl_map_name = self.examples.iloc[item, 4].split('/')[-1]
        if verbose: print('in seqBEV_dataset, pre_name:{}\ncur_name:{}\nnext_name:{}\nbev_map_name:{}\nrl_map_name:{}'.format(pre_img_name, cur_img_name, next_img_name, bev_map_name, rl_map_name))
        
        example  ={'pre_img': pre_img,
                   'cur_img': cur_img,
                   'next_img': next_img,
                   'bev_map': bev_map,
                   'rl_map': rl_map,
                   'cur_name': cur_name}

        if self.transform:
            example = self.transform(example)
        sample = stack()(example)
        sample = ToTensor()(sample)
        #print('in seqBEV_dataset.py, after ToTensor type(image_squence)', sample['img_seq'].shape)

        # sample = ToTensor()(example)
        # sample = stack()(sample)


        if verbose: print("!!!we are in seqBEV_dataset.py, let see the shape of \nsample['img_seq']:{}, \nsample['bev_map']:{}, \nsample['rl_map']:{}".format(sample['img_seq'].shape, sample['bev_map'].shape, sample['rl_map'].shape))
        return sample


class ToTensor(object):
    def __call__(self, sample):

        trans = transforms.Compose([transforms.ToTensor()])
        cur_name = sample['cur_name']
        sample_tensor = {}
        for key in sample.keys():
            if 'img' in key:
                sample_tensor[key] = trans(sample[key])
            elif 'map' in key:
                sample_tensor[key] = torch.from_numpy(sample[key])
        sample_tensor['cur_name'] = cur_name
        return sample_tensor

class stack(object):
    def __call__(self, sample):
        image_sequence = []
        bev_map = sample['bev_map']
        rl_map = sample['rl_map']
        cur_name = sample['cur_name']
        for key in sample.keys():
            if 'img' in key:
                #print('the sample key is ', key)
                image_sequence.append(sample[key])

        #print('in seqBEV_dataset.py, before stack type(image_squence)', image_sequence[0].shape)
        img_stacked = np.concatenate(image_sequence, axis=2)  # 图像的shape由（w,h,3）--> (w,h,9)
        #print('in seqBEV_dataset.py, after stack type(image_squence)', img_stacked.shape)
        return {'img_seq': img_stacked,
                'bev_map': bev_map,
                'rl_map': rl_map,
                'cur_name': cur_name}

class Img_ColorJitter(object):
    def __init__(self, brightness=0.5, prob=0.9) -> None:
        self.brightness = brightness
        self.prob = prob
    def __call__(self, sample):
        pre_img = sample['pre_img']
        cur_img = sample['cur_img']
        next_img = sample['next_img']
        bev_map = sample['bev_map']
        rl_map = sample['rl_map']
        cur_name = sample['cur_name']

        if np.random.rand() < self.prob:
            bright_factor = np.random.uniform(1-self.brightness, 0.8+self.brightness)
            pre_img = (pre_img * bright_factor).astype(pre_img.dtype)
            cur_img = (cur_img * bright_factor).astype(cur_img.dtype)
            next_img = (next_img * bright_factor).astype(next_img.dtype)
        
        sample = {'pre_img': pre_img,
                'cur_img': cur_img,
                'next_img': next_img,
                'bev_map': bev_map,
                'rl_map': rl_map,
                'cur_name': cur_name}
        return sample



if __name__ == '__main__':
    import time
    csv_file = '/workspace/dataset/nuscenes/seq_bev_dataset/csv_files/test.csv'
    test_dataset = seqBEV_dataset(csv_file, transform=Img_ColorJitter())
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=2,   
        shuffle=True,
        num_workers=2,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False)

    print('The length of test_loader is ',len(test_loader))
    start_time = time.time()
    test_data = next(iter(test_loader))
    end_time = time.time()
    duration_1 = end_time - start_time

    start = time.time()
    for i in range(1):
        for batch_idx, input_data in enumerate(test_loader):
            print('this is {} epoch, input_img shape is {}'.format(i, input_data['img_seq'].shape))
            img_seq = input_data['img_seq']
            bev_label = input_data['bev_map']
            road_layout_label = input_data['rl_map']
            cur_name = input_data['cur_name']
            print('shape of img_seq:{}, bev_label:{}, road_layout_label:{}'.format(img_seq.shape, bev_label.shape, road_layout_label.shape))
            print('the num class of bev_label:{}, road_layout_label:{}'.format(np.unique(bev_label), np.unique(road_layout_label)))
    end = time.time()
    duration = end - start
    print("test load duration: ", duration)

    print('type of test_data: ', type(test_data))
    print('test load duration: ', duration_1)
    print('The length of test_dataset is ',len(test_dataset))
    print('The length of test_loader is ',len(test_loader))

    # 检查读入的数据
    # img_seq = test_data['img_seq']
    # bev_label = test_data['bev_map']
    # road_layout_label = test_data['rl_map']
    # cur_name = test_data['cur_name']
    # print('shape of img_seq:{}, bev_label:{}, road_layout_label:{}'.format(img_seq.shape, bev_label.shape, road_layout_label.shape))
    # print('the num class of bev_label:{}, road_layout_label:{}'.format(np.unique(bev_label), np.unique(road_layout_label)))
