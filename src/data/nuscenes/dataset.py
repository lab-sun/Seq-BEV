import os
from tabnanny import verbose
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageFile
from skimage import io
from nuscenes import NuScenes
from torchvision.transforms.functional import to_tensor

from src.data.nuscenes.utils import CAMERA_NAMES, NUSCENES_CLASS_NAMES, iterate_samples
from src.data.utils import decode_binary_labels

verbose = False

class NuScenesMapDataset(Dataset):

    def __init__(self, nuscenes, map_root, road_layout_root, image_size=(800, 450), 
                 scene_names=None, num_squence=3):
        
        self.nuscenes = nuscenes
        self.map_root = os.path.expandvars(map_root)
        self.road_layout_root = os.path.expandvars(road_layout_root)
        self.image_size = image_size

        # Preload the list of tokens in the dataset
        self.get_tokens(scene_names)

        # Allow PIL to load partially corrupted images
        # (otherwise training crashes at the most inconvenient possible times!)
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
        self.squence_num = num_squence
    

    def get_tokens(self, scene_names=None):
        
        self.tokens = list()
        print("in NuScenesMapDataset")

        # Iterate over scenes
        if verbose: print("in NuScenesMapDataset, the num of the scene is ", len(self.nuscenes.scene))
        for scene in self.nuscenes.scene:
            
            
            
            # Ignore scenes which don't belong to the current split
            if scene_names is not None and scene['name'] not in scene_names:
                continue
            if verbose: print("in NuScenesMapDataset, the name of the scene is ", scene['name']) 
            # Iterate over samples
            # modified by GS, exculde the first_sample
            first_sample = self.nuscenes.get('sample', scene['first_sample_token'])
            second_sample = self.nuscenes.get('sample', first_sample['next'])
            third_sample = self.nuscenes.get('sample', second_sample['next'])
            start_token = third_sample['next']
            # for sample in iterate_samples(self.nuscenes, 
            #                               scene['first_sample_token']):
            for sample in iterate_samples(self.nuscenes, 
                                          start_token):
                
                # Iterate over cameras
                for camera in CAMERA_NAMES:
                    self.tokens.append(sample['data'][camera])
        
        if verbose: print("in NuScenesMapDataset, the len of the tokens is ", len(self.tokens))
        return self.tokens


    # added by GS
    def get_scene_token(self, token):
        sample_data = self.nuscenes.get('sample_data', token)
        sample_token = sample_data['sample_token']
        sample = self.nuscenes.get('sample', sample_token)
        scene_token = sample['scene_token']
        scene = self.nuscenes.get('scene', scene_token)
        scene_token = scene['token']
        return scene_token
    
    # added by GS
    def get_prev_token(self, index, token):
        squence_token = []
        current_token = token
        if verbose: print("in get_prev_token, {} current_toke: {}".format(index, current_token))
        squence_token.append(current_token)
        num_interval = 5
        for i in range(self.squence_num-1):
            sample_data = self.nuscenes.get('sample_data', current_token)
            
            prev_token_list = []
            new_token = sample_data['prev']
            for frame in range(num_interval):
                new_sample_data = self.nuscenes.get('sample_data', new_token)
                new_prev_token = new_sample_data['prev']
                new_token = new_prev_token
                prev_token_list.append(new_prev_token)
            prev_token = prev_token_list[-1]

            if prev_token == '':
                current_token = token
                
            current_token = prev_token
            squence_token.append(current_token)
        if verbose: print("in get_prev_token, {} the len of squence is {}".format(index, len(squence_token)) )
        if verbose: print("in get_prev_token, {} current frame:{}, prev frame:{}, preprev frame: {}".
                          format(index, squence_token[0], squence_token[1], squence_token[2]))
        return squence_token
    
    
    # added by GS
    def get_name(self, token):
        sample_data = self.nuscenes.get('sample_data', token)
        filename = sample_data['filename'].split('/')[-1]
        return filename
    
    # added by GS
    def get_scene(self, token):
        sample_data = self.nuscenes.get('sample_data', token)
        sample = self.nuscenes.get('sample', sample_data['sample_token'])
        scene = self.nuscenes.get('scene', sample['scene_token'])
        scene_name = scene['name']
        return scene_name
    
    # added by GS
    def get_squence(self, squence_token):
        squence_len = len(squence_token)
        img_squence = []
        calib_squence = []
        name_squence = []
        for i in range(squence_len):
            token = squence_token[i]
            image = self.load_image(token)
            calib = self.load_calib(token)
            scene = self.get_scene(token)
            # filename = token
            filename = scene + '_' + token
            img_squence.append(image)
            calib_squence.append(calib)
            name_squence.append(filename)
        return img_squence, calib_squence, name_squence
            
    
    
    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):    # sequence中按照current frame, pre_frame, prepre_frame顺序组织
        
        token = self.tokens[index]
        squence_token = self.get_prev_token(index, token)
        
        img_squence, calib_squence, name_squence = self.get_squence(squence_token)
        labels = self.load_labels(token)
        road_layout_labels = self.load_road_layout_labels(token)

        return img_squence, calib_squence, labels, road_layout_labels, name_squence
        

    
    def load_image(self, token):

        # Load image as a PIL image
        image = Image.open(self.nuscenes.get_sample_data_path(token))

        # Resize to input resolution
        image = image.resize(self.image_size)

        # Convert to a torch tensor
        return to_tensor(image)
    

    def load_calib(self, token):

        # Load camera intrinsics matrix
        sample_data = self.nuscenes.get('sample_data', token)
        sensor = self.nuscenes.get(
            'calibrated_sensor', sample_data['calibrated_sensor_token'])
        intrinsics = torch.tensor(sensor['camera_intrinsic'])

        # Scale calibration matrix to account for image downsampling
        intrinsics[0] *= self.image_size[0] / sample_data['width']
        intrinsics[1] *= self.image_size[1] / sample_data['height']
        return intrinsics
    

    def load_labels(self, token):

        # Load label image as a torch tensor
        label_path = os.path.join(self.map_root, token + '.png')
        #print("in dataset, label_path: ", label_path)
        #labels = to_tensor(Image.open(label_path)).long()
        labels_orig = io.imread(label_path)
        #labels = np.where(labels_orig==7, 0, labels_orig)  # 7个类别，其中0为背景类
        labels = np.where(labels_orig>=4, 0, labels_orig)  # FIXME 4个类别，只保留layout的类别
        labels = torch.from_numpy(labels).long()
        #labels = torch.from_numpy(io.imread(label_path)).long()
        #print("in dataset, label type: ", labels.shape, np.max(labels.numpy()))
        #print(labels)
        

        # # Decode to binary labels
        # num_class = len(NUSCENES_CLASS_NAMES)
        # labels = decode_binary_labels(labels, num_class + 1)
        # labels, mask = labels[:-1], ~labels[-1]
        
        return labels

    def load_road_layout_labels(self, token):

        # Load label image as a torch tensor
        label_path = os.path.join(self.road_layout_root, token + '.png')
        #print("in dataset, label_path: ", label_path)
        #labels = to_tensor(Image.open(label_path)).long()
        road_layout_labels = torch.from_numpy(io.imread(label_path)).long()
        #print("in dataset, label type: ", road_layout_labels.shape, np.max(road_layout_labels.numpy()))
        #print(labels)
        

        # # Decode to binary labels
        # num_class = len(NUSCENES_CLASS_NAMES)
        # labels = decode_binary_labels(labels, num_class + 1)
        # labels, mask = labels[:-1], ~labels[-1]
        
        return road_layout_labels
