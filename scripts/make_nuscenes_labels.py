import io
import os
import sys
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from collections import OrderedDict

import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

from shapely.strtree import STRtree
from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))

from src.utils.configs import get_default_configuration
from src.data.utils import get_visible_mask, get_occlusion_mask, transform, \
    encode_binary_labels, create_visual_anno, get_onehot_mask
import src.data.nuscenes.utils as nusc_utils



def process_scene(nuscenes, map_data, scene, config):

    # Get the map corresponding to the current sample data
    log = nuscenes.get('log', scene['log_token'])
    location = log['location']
    nusc_map = NuScenesMap(dataroot, location)
    scene_map_data = map_data[log['location']]  # get the centarin map for the scene type is a orderedDict including shapely.strtree type

    # Iterate over samples
    # modified by GS, exclude the first_sample_token
    first_sample_token = scene['first_sample_token']
    first_sample = nuscenes.get('sample', first_sample_token)
    start_token = first_sample['next']
    for sample in nusc_utils.iterate_samples(nuscenes, start_token):
        process_sample(nuscenes, scene_map_data, sample, config)
        #get_road_layout(nuscenes, nusc_map, sample, config)
        get_road_layout_multicls(nuscenes, nusc_map, sample, config)

# add by GS, get road layout mask
def get_road_layout(nuscenes, nusc_map, sample, config):
    sample_token = sample['token']
    layer_names = ['road_segment', 'ped_crossing', 'walkway', 'lane', 'carpark_area']
    camera_channel = 'CAM_FRONT'
    fig, ax = nusc_map.render_map_in_image(nuscenes, sample_token, layer_names=layer_names, camera_channel=camera_channel)
    patches_list = ax.patches
    label_mat = np.zeros([900,1600,3])
    fig_new = plt.figure(figsize=(9,16))
    ax_new = fig_new.add_axes([0, 0, 1, 1])
    ax_new.imshow(label_mat)
    for i in range(len(patches_list)):
        patch = patches.PathPatch(patches_list[i].get_path(), facecolor='orange', lw=0)
        ax_new.add_patch(patch)
    plt.axis('off')
    plt.close('all')

    # 申请缓存
    buffer_ = io.BytesIO()
    fig_new.savefig(buffer_, format='png', bbox_inches='tight', dpi=200, pad_inches=0.0)
    buffer_.seek(0)
    image = Image.open(buffer_)
    ar = np.asarray(image)  # 转成numpy矩阵
    # 释放缓存
    buffer_.close()
    # RGBA to RGB
    resized_img = cv2.resize(ar, (400, 240), interpolation=cv2.INTER_AREA)
    rgb = np.zeros((240, 400, 3), dtype='float32')
    r,g,b,a = resized_img[:,:,0], resized_img[:,:,1], resized_img[:,:,2], resized_img[:,:,3]

    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b

    rgb = np.asarray(rgb, dtype='uint8')
    gt = np.where(rgb[:,:,0]!=0, 1, 0)

    camera = 'CAM_FRONT'  # 'CAM_FRONT'
    sample_data = nuscenes.get('sample_data', sample['data'][camera])
    gt_output_path = os.path.join(os.path.expandvars(config.road_layout_root),
                                  sample_data['token'] + '.png')
    color_output_path = os.path.join(os.path.expandvars(config.road_layout_root),
                               sample_data['token'] + '_c.png')

    cv2.imwrite(color_output_path, cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    cv2.imwrite(gt_output_path, gt)
    print("saving {}.png".format(sample_data['token']))

# add by GS, get road layout mask with multi-classes
def get_road_layout_multicls(nuscenes, nusc_map, sample, config):
    sample_token = sample['token']
    drivable_area = ['road_segment', 'lane', 'carpark_area']
    ped_crossing = ['ped_crossing']
    walkway = ['walkway']
    layer_names = [drivable_area, ped_crossing, walkway]
    gt_layers = []
    camera_channel = 'CAM_FRONT'
    plt.close('all')
    for area in layer_names:
        fig, ax = nusc_map.render_map_in_image(nuscenes, sample_token, layer_names=area, camera_channel=camera_channel)
        patches_list = ax.patches
        label_mat = np.zeros([900, 1600, 3])
        fig_new = plt.figure(figsize=(9,16))
        ax_new = fig_new.add_axes([0,0,1,1])
        ax_new.imshow(label_mat)
        for i in range(len(patches_list)):
            patch = patches.PathPatch(patches_list[i].get_path(), facecolor='orange', lw=0)
            ax_new.add_patch(patch)
        plt.axis('off')
        # plt.clear()
        plt.close('all')

        # 申请缓存
        buffer_ = io.BytesIO()
        fig_new.savefig(buffer_, format='png', bbox_inches='tight', dpi=200, pad_inches=0.0)
        buffer_.seek(0)
        image = Image.open(buffer_)
        ar = np.asarray(image)  # 转成numpy矩阵
        
        fig_new.clear()
        plt.close(fig_new)

        # 释放缓存
        # buffer_.close()
        # RGBA to RGB
        resized_img = cv2.resize(ar, (400, 240), interpolation=cv2.INTER_AREA)
        rgb = np.zeros((240, 400, 3), dtype='float32')
        r, g, b, a = resized_img[:,:,0], resized_img[:,:,1], resized_img[:,:,2], resized_img[:,:,3]

        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        gt_single_layer = np.where(rgb[:,:,0]!=0, 1, 0)
        gt_layers.append(gt_single_layer)  # 顺序是drivable_area, ped_crossing, walkway

    plt.axis('off')
    plt.close('all')
    gt_drivable = np.where(gt_layers[0]!=gt_layers[1], gt_layers[0], 0)
    gt_drivable = np.where(gt_drivable!=gt_layers[2], gt_drivable, 0)
    gt_ped = gt_layers[1]
    gt_ped = np.where(gt_ped!=gt_layers[2], gt_ped, 0)
    gt_walkway = gt_layers[2]

    gt = gt_drivable + 2*gt_ped + 3*gt_walkway
    rgb_gt = create_visual_anno(gt)

    camera = 'CAM_FRONT'  # 'CAM_FRONT'
    sample_data = nuscenes.get('sample_data', sample['data'][camera])
    gt_output_path = os.path.join(os.path.expandvars(config.road_layout_root),
                                  sample_data['token'] + '.png')
    color_output_path = os.path.join(os.path.expandvars(config.road_layout_root),
                               sample_data['token'] + '_c.png')

    cv2.imwrite(color_output_path, cv2.cvtColor(rgb_gt, cv2.COLOR_BGR2RGB))
    cv2.imwrite(gt_output_path, gt)
    print(gt_output_path)
    print("saving {}.png".format(sample_data['token']))
    plt.axis('off')
    plt.close('all')
    buffer_.close()
    

def process_sample(nuscenes, map_data, sample, config):

    # Load the lidar point cloud associated with this sample
    lidar_data = nuscenes.get('sample_data', sample['data']['LIDAR_TOP'])
    lidar_pcl = nusc_utils.load_point_cloud(nuscenes, lidar_data)

    # Transform points into world coordinate system
    lidar_transform = nusc_utils.get_sensor_transform(nuscenes, lidar_data)
    lidar_pcl = transform(lidar_transform, lidar_pcl)

    # Iterate over sample data
    for camera in nusc_utils.CAMERA_NAMES:
        sample_data = nuscenes.get('sample_data', sample['data'][camera])
        process_sample_data(nuscenes, map_data, sample_data, lidar_pcl, config)


def process_sample_data(nuscenes, map_data, sample_data, lidar, config):

    # Render static road geometry masks
    map_masks = nusc_utils.get_map_masks(nuscenes, 
                                         map_data, 
                                         sample_data, 
                                         config.map_extents, 
                                         config.map_resolution)
    
    # Render dynamic object masks
    obj_masks = nusc_utils.get_object_masks(nuscenes, 
                                            sample_data, 
                                            config.map_extents, 
                                            config.map_resolution)
    masks = np.concatenate([map_masks, obj_masks], axis=0)  # mask.shape (15, 196, 200)

    # Ignore regions of the BEV which are outside the image
    sensor = nuscenes.get('calibrated_sensor', 
                          sample_data['calibrated_sensor_token'])
    intrinsics = np.array(sensor['camera_intrinsic'])
    #masks[-1] |= ~get_visible_mask(intrinsics, sample_data['width'], 
    #                               config.map_extents, config.map_resolution)
    
    # Transform lidar points into camera coordinates
    cam_transform = nusc_utils.get_sensor_transform(nuscenes, sample_data)
    cam_points = transform(np.linalg.inv(cam_transform), lidar)
    #masks[-1] |= get_occlusion_mask(cam_points, config.map_extents,
    #                                config.map_resolution)
    
    # Encode masks as integer bitmask
    # save gt. gt label is a tensor and shape is (img_height, img_width), which is a single channel image
    #print("in make nuscenes labels, masks shape is ", masks.shape)
    gt = get_onehot_mask(masks)
    
    # flip the gt make it bottom to top
    flip_gt = gt.reshape(gt.size)
    flip_gt = gt[::-1]
    gt = flip_gt.reshape(gt.shape)
    # get the visualized gt, which is a (img_height, img_width, 3) dim image
    color_gt = create_visual_anno(gt)
    #print("color_gt shape: ", color_gt.shape)

    # Save outputs to disk
    gt_output_path = os.path.join(os.path.expandvars(config.label_root),
                               sample_data['token'] + '.png')
    #Image.fromarray(gt.astype(np.int32), mode='I').save(gt_output_path)
    cv2.imwrite(gt_output_path, gt)

    color_output_path = os.path.join(os.path.expandvars(config.label_root),
                               sample_data['token'] + '_c.png')
    #Image.fromarray(color_gt.astype(np.int32), mode='I').save(color_output_path)
    cv2.imwrite(color_output_path, cv2.cvtColor(color_gt, cv2.COLOR_BGR2RGB))


def load_map_data(dataroot, location):

    # Load the NuScenes map object
    nusc_map = NuScenesMap(dataroot, location)

    map_data = OrderedDict()
    for layer in nusc_utils.STATIC_CLASSES:
        
        # Retrieve all data associated with the current layer
        records = getattr(nusc_map, layer)
        polygons = list()

        # Drivable area records can contain multiple polygons
        if layer == 'drivable_area':
            for record in records:

                # Convert each entry in the record into a shapely object
                for token in record['polygon_tokens']:
                    poly = nusc_map.extract_polygon(token)
                    if poly.is_valid:
                        polygons.append(poly)
        else:
            for record in records:

                # Convert each entry in the record into a shapely object
                poly = nusc_map.extract_polygon(record['polygon_token'])
                if poly.is_valid:
                    polygons.append(poly)

        
        # Store as an R-Tree for fast intersection queries
        map_data[layer] = STRtree(polygons)
    
    return map_data





if __name__ == '__main__':

    # Load the default configuration
    config = get_default_configuration()
    config.merge_from_file('configs/datasets/nuscenes.yml')

    # Load NuScenes dataset
    dataroot = os.path.expandvars(config.dataroot)
    nuscenes = NuScenes(config.nuscenes_version, dataroot)

    # Preload all 4 NuScenes map data
    map_data = { location : load_map_data(dataroot, location) 
                 for location in nusc_utils.LOCATIONS }
    
    # Create a directory for the generated labels
    output_root = os.path.expandvars(config.label_root)
    os.makedirs(output_root, exist_ok=True)
    
   # print(nuscenes.scene)
    # Iterate over NuScene scenes
    print("\nGenerating labels...")
    for scene in tqdm(nuscenes.scene):
        process_scene(nuscenes, map_data, scene, config)




    


    





    

