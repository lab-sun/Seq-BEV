import torch
from torch.utils.data import Dataset
import numpy as np

class AugmentedMapDataset(Dataset):

    def __init__(self, dataset, hflip=True, stack=True):
        self.dataset = dataset
        self.hflip = hflip
        self.stack = stack
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image_squence, calib_squence, labels, road_layout_labels, name_squence = self.dataset[index]

        # Apply data augmentation
        if self.hflip:
            image_squence, labels, road_layout_labels = random_hflip(image_squence, labels, road_layout_labels)
        
        if self.stack:
            image_squence = stack(image_squence)

        return image_squence, calib_squence, labels, road_layout_labels, name_squence

    
def random_hflip(image_squence, labels, road_layout_labels):
    image_hflip_squence = []
    for image in image_squence:
        image = torch.flip(image, (-1,))
        image_hflip_squence.append(image)
    
    labels = torch.flip(labels.int(), (-1,))
    road_layout_labels = torch.flip(road_layout_labels.int(), (-1,))

    #mask = torch.flip(mask.int(), (-1,)).bool()
    return image_hflip_squence, labels, road_layout_labels

def stack(image_squence):
    #print('in augmentation.py, before stack type(image_squence)', image_squence[0].shape)
    img_stacked = np.concatenate(image_squence, axis=0)
    #print('in augmentation.py, after stack type(img_stacked)', img_stacked.shape)
    return img_stacked