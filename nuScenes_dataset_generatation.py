# 按照距离序列划分序列
# load nuscenes 数据集和 map 

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import points_in_box
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud 
from nuscenes.utils.geometry_utils import transform_matrix
from PIL import Image 
import numpy as np 
import shutil, os

import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from collections import OrderedDict
from shapely.strtree import STRtree
from nuscenes.map_expansion.map_api import NuScenesMap

from src.utils.configs import get_default_configuration
from src.data.utils import transform, create_visual_anno, get_onehot_mask
import src.data.nuscenes.utils as nusc_utils

import cv2
import io



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



raw_data_dir = '/home/gs/workspace/datasets/nuScenes/trainval/'
save_data_dir = '/home/gs/workspace/datasets/nuScenes/seq_bev_dataset/'

#dataroot='/home/gs/Datasets/nuScenes/mini'
dataroot='/home/gs/workspace/datasets/nuScenes/trainval/'
#map_name='singapore-onenorth'

nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)  # 'v1.0-trainval' 'v1.0-mini'

# Preload all 4 NuScenes map data
LOCATIONS = ['boston-seaport', 'singapore-onenorth', 'singapore-queenstown',
             'singapore-hollandvillage']
map_data = { location : load_map_data(raw_data_dir, location) 
             for location in nusc_utils.LOCATIONS }

# 生成BEV gt和roadLayout gt

map_extents = [-25., 5., 25., 55.]  # Top-left and bottom right coordinates of map region, in meters
map_resolution = 0.25  # # Spacing between adjacent grid cells in the map, in meters
map_config = {}
map_config['map_extents'] = [-15., 5., 15., 35.]
map_config['map_resolution'] = 0.2

def process_scene(nuscenes, map_data, scene, map_config):

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
        gt, color_gt = process_sample(nuscenes, scene_map_data, sample, map_config)
        #get_road_layout(nuscenes, nusc_map, sample, config)
        #get_road_layout_multicls(nuscenes, nusc_map, sample, config)
        return gt, color_gt

def process_sample(nuscenes, map_data, sample, map_config):

    # Load the lidar point cloud associated with this sample
    lidar_data = nuscenes.get('sample_data', sample['data']['LIDAR_TOP'])
    lidar_pcl = nusc_utils.load_point_cloud(nuscenes, lidar_data)

    # Transform points into world coordinate system
    lidar_transform = nusc_utils.get_sensor_transform(nuscenes, lidar_data)
    lidar_pcl = transform(lidar_transform, lidar_pcl)

    # Iterate over sample data
    for camera in nusc_utils.CAMERA_NAMES:
        sample_data = nuscenes.get('sample_data', sample['data'][camera])
        gt, color_gt = process_sample_data(nuscenes, map_data, sample_data, lidar_pcl, map_config)
        return gt, color_gt

def process_sample_data(nuscenes, map_data, sample_data, lidar, map_config):

    # Render static road geometry masks
    map_masks = nusc_utils.get_map_masks(nuscenes, 
                                         map_data, 
                                         sample_data, 
                                         map_config['map_extents'], 
                                         map_config['map_resolution'])
    
    # Render dynamic object masks
    obj_masks = nusc_utils.get_object_masks(nuscenes, 
                                            sample_data, 
                                            map_config['map_extents'], 
                                            map_config['map_resolution'])
    masks = np.concatenate([map_masks, obj_masks], axis=0)  # mask.shape (15, 196, 200)

    # Ignore regions of the BEV which are outside the image
    sensor = nuscenes.get('calibrated_sensor', 
                          sample_data['calibrated_sensor_token'])
    intrinsics = np.array(sensor['camera_intrinsic'])
    #masks[-1] |= ~get_visible_mask(intrinsics, sample_data['width'], 
    #                               map_config['map_extents'], map_config['map_resolution'])
    
    # Transform lidar points into camera coordinates
    cam_transform = nusc_utils.get_sensor_transform(nuscenes, sample_data)
    cam_points = transform(np.linalg.inv(cam_transform), lidar)
    #masks[-1] |= get_occlusion_mask(cam_points, map_config['map_extents'],
    #                                map_config['map_resolution'])
    
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

    # # Save outputs to disk
    # gt_output_path = os.path.join(os.path.expandvars(config.label_root),
    #                            sample_data['token'] + '.png')
    # #Image.fromarray(gt.astype(np.int32), mode='I').save(gt_output_path)
    # cv2.imwrite(gt_output_path, gt)
# 
    # color_output_path = os.path.join(os.path.expandvars(config.label_root),
    #                            sample_data['token'] + '_c.png')
    # #Image.fromarray(color_gt.astype(np.int32), mode='I').save(color_output_path)
    # cv2.imwrite(color_output_path, cv2.cvtColor(color_gt, cv2.COLOR_BGR2RGB))
    
    print("in process_sample_data the type of gt is {}, and color_gt is {}".format(type(gt), type(color_gt)))
    return gt, color_gt


# add by GS, get road layout mask with multi-classes
def get_road_layout_multicls(nuscenes, nusc_map, sample):
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
        buffer_.close()
        # RGBA to RGB
        resized_img = cv2.resize(ar, (512, 256), interpolation=cv2.INTER_AREA)
        rgb = np.zeros((256, 512, 3), dtype='float32')
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
    # gt_output_path = os.path.join(os.path.expandvars(config.road_layout_root),
    #                               sample_data['token'] + '.png')
    # color_output_path = os.path.join(os.path.expandvars(config.road_layout_root),
    #                            sample_data['token'] + '_c.png')

    #cv2.imwrite(color_output_path, cv2.cvtColor(rgb_gt, cv2.COLOR_BGR2RGB))
    #cv2.imwrite(gt_output_path, gt)
    #print(gt_output_path)
    #print("saving {}.png".format(sample_data['token']))
    plt.axis('off')
    plt.close('all')
    buffer_.close()
    return gt, rgb_gt


# 计算每个scene中相邻帧的距离和偏角

import csv
import cv2

raw_data_dir = '/home/gs/workspace/datasets/nuScenes/trainval/'
save_data_dir = '/home/gs/workspace/datasets/nuScenes/seq_bev_dataset/'


# check and create folders
def check_and_create_dirs(paths):
    for path in paths:        
        if not os.path.isdir(path):
            print('%s does not exist, create the folder' %path)
            os.makedirs(path)
        else:
            print('%s exist, will delete and create the folder' %path)
            shutil.rmtree(path) 
            os.makedirs(path)
            
def get_dist(cur_pose, next_pose, CAM):
    # cur_camera_data = nusc.get('sample_data', current_sample['data'][CAM])
    # cur_pose = nusc.get('ego_pose', cur_camera_data['ego_pose_token'])
    # next_camera_data = nusc.get('sample_data', next_sample['data'][CAM])
    # next_pose = nusc.get('ego_pose', next_camera_data['ego_pose_token'])
    cur_t = np.array(cur_pose['translation'].copy())
    next_t = np.array(next_pose['translation'].copy())
    dist = np.sqrt(np.sum(np.square(cur_t-next_t)))
    return dist

def get_angle(cur_pose, next_pose, CAM):
    cur_p = cur_pose['rotation'].copy()   # nuscenes的四元数顺序wxyz
    next_p = next_pose['rotation'].copy()
    
    cur_p.insert(len(cur_p), cur_p[0])  # 交换四元数顺序，以便用scipy包进行欧拉角变换
    cur_p.remove(cur_p[0])
    next_p.insert(len(next_p), next_p[0])
    next_p.remove(next_p[0])
    
    cur_r = R.from_quat(cur_p)  # 转成旋转矩阵
    next_r = R.from_quat(next_p)
    cur_euler = cur_r.as_euler('zxy', degrees=True)  # 转成欧拉角
    next_euler = next_r.as_euler('zxy', degrees=True)
    rotation = abs((cur_euler-next_euler)[0])
    if rotation > 180:
        rotation = 360 - rotation
    return rotation
    
# 将不同的scene中的序列文档汇总成一个
from src.data.nuscenes.splits import TRAIN_SCENES, VAL_SCENES, TEST_SCENES
def get_total_csv():
    seq_bev_dataset_path = '/home/gs/workspace/datasets/nuScenes/seq_bev_dataset'
    print(os.path.join(seq_bev_dataset_path, 'csv_files/'))
    check_and_create_dirs([os.path.join(seq_bev_dataset_path, 'csv_files/')])

    # create train.csv, val.csv, test.csv
    train_csv = os.path.join(seq_bev_dataset_path, 'csv_files/train.csv')
    val_csv = os.path.join(seq_bev_dataset_path, 'csv_files/val.csv')
    test_csv = os.path.join(seq_bev_dataset_path, 'csv_files/test.csv')
    with open(train_csv, 'w') as train_f:
        train_csv_write = csv.writer(train_f)
    with open(val_csv, 'w') as val_f:
        val_csv_write = csv.writer(val_f)
    with open(test_csv, 'w') as test_f:
        test_csv_write = csv.writer(test_f)
        
    # split scenes into different lists
    train_list = TRAIN_SCENES
    val_list = VAL_SCENES
    test_list = TEST_SCENES
    print('the length of train_list: {}, val_list: {}, test_list:{}'.format(len(train_list), len(val_list), len(test_list)))

    #scene_lists = [train_list, val_list, test_list]
    scene_dict = {}
    scene_dict['train'] = train_list
    scene_dict['val'] = val_list
    scene_dict['test'] = test_list

    for key in scene_dict.keys():
        for scene in scene_dict[key]:
            csv_file_path = os.path.join(seq_bev_dataset_path, scene) + '/{}.csv'.format(scene)
            CAM_FRONT_path = os.path.join(seq_bev_dataset_path, scene) + '/cam_front'
            BEV_gt_path = os.path.join(seq_bev_dataset_path, scene) + '/BEV_gt'
            roadLayout_gt_path = os.path.join(seq_bev_dataset_path, scene) + '/roadLayout_gt'
            print('csv_file_path: ', csv_file_path)
            csvFile = open(csv_file_path, 'r')
            reader = csv.reader(csvFile)
            for item in reader:
                pre_frame = item[0]
                cur_frame = item[1]
                next_frame = item[2]
                print(item)
                print(CAM_FRONT_path)
                csv_row = {}
                #csv_row_write = []
                for i, j, k in os.walk(CAM_FRONT_path):
                    for name in k:
                        if pre_frame in name:
                            csv_row['pre'] = name
                        elif cur_frame in name:
                            #print("!!!name:", name)
                            csv_row['cur'] = name
                            csv_row['BEV_gt'] = name.split('.')[0] + '_BEVmask.png'
                            csv_row['roadLayout_gt'] = name.split('.')[0] + '_RLmask.png'
                            #print("!!!csv_row['roadLayout_gt']:", csv_row['roadLayout_gt'])
                        elif next_frame in name:
                            csv_row['next'] = name
                pre_frame_path = os.path.join(CAM_FRONT_path, csv_row['pre'])
                cur_frame_path = os.path.join(CAM_FRONT_path, csv_row['cur'])
                next_frame_path = os.path.join(CAM_FRONT_path, csv_row['next'])
                cur_BEV_gt_path = os.path.join(BEV_gt_path, csv_row['BEV_gt'])
                cur_roadLayout_gt_path = os.path.join(roadLayout_gt_path, csv_row['roadLayout_gt'])
                csv_row_write = [pre_frame_path, cur_frame_path, next_frame_path, cur_BEV_gt_path, cur_roadLayout_gt_path]
                if 'train' in key:
                    print("YES")
                    with open(train_csv, 'a+') as train_f:
                        csv_write = csv.writer(train_f)
                        csv_write.writerow(csv_row_write)
                elif 'val' in key:
                    with open(val_csv, 'a+') as val_f:
                        csv_write = csv.writer(val_f)
                        csv_write.writerow(csv_row_write)
                elif 'test' in key:
                    with open(test_csv, 'a+') as test_f:
                        csv_write = csv.writer(test_f)
                        csv_write.writerow(csv_row_write)
                print(csv_row_write)
                #break
            #break
        #break



# 循环scenes
for scene_id in range(len(nusc.scene)):
#for scene_id in range(0,len(nusc.scene)):
    print(scene_id)
    scene = nusc.scene[scene_id]
    log = nusc.get('log', scene['log_token'])
    location = log['location']
    nusc_map = NuScenesMap(raw_data_dir, location)
    # get the centarin map for the scene type is a orderedDict including shapely.strtree type
    scene_map_data = map_data[log['location']]  
    #print(scene)
    
    # 保存路径
    save_root = os.path.join(save_data_dir, scene['name'])
    save_image_path = os.path.join(save_root, 'cam_front')
    #print(">>>>>>>>>>>>save_image_path: ", save_image_path)
    save_BEVgt_path = os.path.join(save_root, 'BEV_gt')
    #print(">>>>>>>>>>>>save_gt_path: ", save_gt_path)
    save_roadLayoutGT_path = os.path.join(save_root, 'roadLayout_gt')
    paths_list = [save_root, save_image_path, save_BEVgt_path, save_roadLayoutGT_path]
    # TODO 檢查其他文件夾
    # paths_list = [save_roadLayoutGT_path]
    check_and_create_dirs(paths_list)  # create folders
    scene_file_path = save_root + '/{}.csv'.format(scene['name'])  # create csv file
    with open(scene_file_path, 'w') as f:
        csv_write = csv.writer(f)
    
    CAM = 'CAM_FRONT'
    
    sample_num = 0
    
    first_sample_token = scene['first_sample_token']
    current_sample = nusc.get('sample', first_sample_token)
    #cur_sample = current_sample
    
    while(True):
        print('Scene No. {}/{}, Sample No. {}/{}: '.format(scene_id+1, len(nusc.scene), sample_num+1, scene['nbr_samples']))
        
        ################ Camera Data #################
        camera_data = nusc.get('sample_data', current_sample['data'][CAM])
        image_filename = camera_data['filename']  # 存图时的图像位置img_front = Image.open(os.path.join(raw_data_dir, image_filename))
        sample_token = camera_data['sample_token']  # sample_token作为全局索引
        save_image_name = '{:0>3}'.format(sample_num)+'_'+sample_token+'.png' 
        print('save image name: ', save_image_name)
        img_front = Image.open(os.path.join(raw_data_dir, image_filename))
        img_front_resized = img_front.resize((512,256),Image.LANCZOS) # 相比于原图缩小一倍
        ego_pose = nusc.get('ego_pose', camera_data['ego_pose_token'])
        cur_sample = current_sample
        #sample_token_row = []
        num_seq = 3
        
        # TODO 保存原始图像
        img_front_resized.save(os.path.join(save_image_path,'{:0>3}'.format(sample_num)+'_'+sample_token+'.png'))
        
        # 向csv文件中记录前后3帧的sample_token
        for num_frame in range(num_seq-1):   # 前后3帧
            dist = 0
            angle = 0
            num_next = 0
            print('*'*60)
            while(True):
                
                cur_sample_token = cur_sample['token']
                #print('cur_sample_token: {}'.format(cur_sample_token))
                cur_camera_data = nusc.get('sample_data', cur_sample['data'][CAM])
                cur_pose = nusc.get('ego_pose', cur_camera_data['ego_pose_token'])

                if num_next == 0:
                    next_sample_token = cur_sample['next']
                if next_sample_token == '':
                    break
                #print('next sample token: {}'.format(next_sample_token))
                next_sample = nusc.get('sample', next_sample_token)
                next_camera_data = nusc.get('sample_data', next_sample['data'][CAM])
                next_pose = nusc.get('ego_pose', next_camera_data['ego_pose_token'])
                
                if dist < 10 and angle < 30:
                    dist = get_dist(cur_pose, next_pose, CAM)
                    angle = get_angle(cur_pose, next_pose, CAM)
                    print('dist: {}, angle: {}'.format(dist, angle))
                    next_sample_token = next_sample['next']
                    #print('new next sample token: {}'.format(next_sample_token))
                else:
                    # TODO record next_sample_token into the csv file
                    print('***fond great dist or angle***')
                    print('cur_sample_token: {}; next_sample_token:{}'.format(cur_sample_token, next_sample['prev']))  # next_sample已经指向下一个sample，为得到满足条件的sample,所以用next_sample['prev']
                    if num_frame == 0:
                        sample_token_row = [cur_sample_token, next_sample['prev']]
                    else:
                        sample_token_row.append(next_sample['prev'])
                    if len(sample_token_row)==num_seq:
                        with open(scene_file_path, 'a+') as f:
                            csv_write = csv.writer(f)
                            print("<<<<<<<<<<<<<sample_token_row: ", sample_token_row)
                            csv_write.writerow(sample_token_row)
              
                    cur_sample = nusc.get('sample', next_sample['prev'])
                    break
                num_next += 1
    
        # TODO 保存generated ground truth
        import time
        start = time.time()
        BEV_gt, BEV_color_gt = process_sample(nusc, scene_map_data, current_sample, map_config)
        BEVgt_name = os.path.join(save_BEVgt_path,'{:0>3}'.format(sample_num)+'_'+sample_token+'_BEVmask.png')
        color_BEVgt_name = os.path.join(save_BEVgt_path,'{:0>3}'.format(sample_num)+'_'+sample_token+'_BEVmask_c.png')
        cv2.imwrite(BEVgt_name, BEV_gt)  # 保存BEV gt
        cv2.imwrite(color_BEVgt_name, cv2.cvtColor(BEV_color_gt, cv2.COLOR_BGR2RGB))  # 保存color BEV gt
        
        # 保存roadLayput ground truth
        plt.close('all')
        RL_gt, RL_color_gt = get_road_layout_multicls(nusc, nusc_map, current_sample)
        RLgt_name = os.path.join(save_roadLayoutGT_path,'{:0>3}'.format(sample_num)+'_'+sample_token+'_RLmask.png')
        color_RLgt_name = os.path.join(save_roadLayoutGT_path,'{:0>3}'.format(sample_num)+'_'+sample_token+'_RLmask_c.png')
        cv2.imwrite(RLgt_name, RL_gt)
        cv2.imwrite(color_RLgt_name, cv2.cvtColor(RL_color_gt, cv2.COLOR_BGR2RGB))  # 保存color RoadLayout gt
        plt.close('all')
        
        
        end = time.time()
        duration = end - start
        print('duration is ', duration)
        
        sample_num += 1
        if (current_sample['next'] == ''):
            print('!There is no more next!')
            break
        current_sample = nusc.get('sample', current_sample['next'])

    # TODO 得到全部的csv
    get_total_csv()

