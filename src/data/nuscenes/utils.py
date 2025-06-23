import os
from matplotlib.pyplot import cla
import numpy as np
from shapely import geometry, affinity
from pyquaternion import Quaternion

#from nuscenes.eval.detection.utils import category_to_detection_name
#from nuscenes.eval.detection.constants import DETECTION_NAMES
from typing import List, Optional
from nuscenes.utils.data_classes import LidarPointCloud

from ..utils import transform_polygon, render_polygon, transform

# CAMERA_NAMES = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
#                 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']
CAMERA_NAMES = ['CAM_FRONT']

# NUSCENES_CLASS_NAMES = [
#     'drivable_area', 'ped_crossing', 'walkway', 'carpark', 'car', 'truck', 
#     'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 
#     'bicycle', 'traffic_cone', 'barrier'
# ]

# 语义类别，包含STATIC_CLASSES和DETECTION_OBJECTS, 共6类
# 其中STATIC_CLASSES有'drivable_area', 'ped_crossing', 'walkway'； DETECTION_OBJECTS有'movable_object', 'vehicle', 'pedestrian'
DETECTION_NAMES = ['movable_object', 'vehicle', 'pedestrian']

def category_to_detection_name(category_name: str) -> Optional[str]:
    """
    Default label mapping from nuScenes to nuScenes detection classes.
    Note that pedestrian does not include personal_mobility, stroller and wheelchair.
    :param category_name: Generic nuScenes class.
    :return: nuScenes detection class.
    """
    detection_mapping = {
        'movable_object.barrier': 'movable_object',
        'vehicle.bicycle': 'vehicle',
        'vehicle.bus.bendy': 'vehicle',
        'vehicle.bus.rigid': 'vehicle',
        'vehicle.car': 'vehicle',
        'vehicle.construction': 'vehicle',
        'vehicle.motorcycle': 'vehicle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'movable_object',
        'vehicle.trailer': 'vehicle',
        'vehicle.truck': 'vehicle'
    }

    if category_name in detection_mapping:
        return detection_mapping[category_name]
    else:
        return None

# NUSCENES_CLASS_NAMES = [
#     'drivable_area', 'ped_crossing', 'walkway', 'car', 'truck', 
#     'bus', 'trailer', 'pedestrian', 
#     'bicycle', 'traffic_cone', 'barrier'
# ]

NUSCENES_CLASS_NAMES = [
    'drivable_area', 'ped_crossing', 'walkway', 
    'movable_object', 'vehicle', 'pedestrian'
]

#STATIC_CLASSES = ['drivable_area', 'ped_crossing', 'walkway', 'carpark_area']
STATIC_CLASSES = ['drivable_area', 'ped_crossing', 'walkway']

LOCATIONS = ['boston-seaport', 'singapore-onenorth', 'singapore-queenstown',
             'singapore-hollandvillage']


def iterate_samples(nuscenes, start_token):
    sample_token = start_token
    while sample_token != '':
        sample = nuscenes.get('sample', sample_token)
        yield sample
        sample_token = sample['next']
    

def get_map_masks(nuscenes, map_data, sample_data, extents, resolution):

    # Render each layer sequentially
    layers = [get_layer_mask(nuscenes, polys, sample_data, extents, 
              resolution) for layer, polys in map_data.items()]
    return np.stack(layers, axis=0)


def get_layer_mask(nuscenes, polygons, sample_data, extents, resolution):

    # Get the 2D affine transform from bev coords to map coords
    tfm = get_sensor_transform(nuscenes, sample_data)[[0, 1, 3]][:, [0, 2, 3]]
    inv_tfm = np.linalg.inv(tfm)

    # Create a patch representing the birds-eye-view region in map coordinates
    map_patch = geometry.box(*extents)
    map_patch = transform_polygon(map_patch, tfm)

    # Initialise the map mask
    x1, z1, x2, z2 = extents
    mask = np.zeros((int((z2 - z1) / resolution), int((x2 - x1) / resolution)),
                    dtype=np.uint8)

    # Find all polygons which intersect with the area of interest
    for polygon in polygons.query(map_patch):

        polygon = polygon.intersection(map_patch)
        
        # Transform into map coordinates
        polygon = transform_polygon(polygon, inv_tfm)

        # Render the polygon to the mask
        render_shapely_polygon(mask, polygon, extents, resolution)
    
    return mask.astype(np.bool)




def get_object_masks(nuscenes, sample_data, extents, resolution):

    # Initialize object masks
    nclass = len(DETECTION_NAMES) + 1
    grid_width = int((extents[2] - extents[0]) / resolution)
    grid_height = int((extents[3] - extents[1]) / resolution)
    masks = np.zeros((nclass, grid_height, grid_width), dtype=np.uint8)

    # Get the 2D affine transform from bev coords to map coords
    tfm = get_sensor_transform(nuscenes, sample_data)[[0, 1, 3]][:, [0, 2, 3]]
    inv_tfm = np.linalg.inv(tfm)

    for box in nuscenes.get_boxes(sample_data['token']):

        # Get the index of the class
        det_name = category_to_detection_name(box.name)
        if det_name not in DETECTION_NAMES:
            class_id = -1
        else:
            class_id = DETECTION_NAMES.index(det_name)
            #print("in util.py, box.name is {}, det_name is {}, class_id is {}".format(box.name, det_name, class_id))
        
        # Get bounding box coordinates in the grid coordinate frame
        bbox = box.bottom_corners()[:2]
        local_bbox = np.dot(inv_tfm[:2, :2], bbox).T + inv_tfm[:2, 2]

        # Render the rotated bounding box to the mask
        render_polygon(masks[class_id], local_bbox, extents, resolution)
    
    return masks.astype(np.bool)


def get_sensor_transform(nuscenes, sample_data):

    # Load sensor transform data
    sensor = nuscenes.get(
        'calibrated_sensor', sample_data['calibrated_sensor_token'])
    sensor_tfm = make_transform_matrix(sensor)

    # Load ego pose data
    pose = nuscenes.get('ego_pose', sample_data['ego_pose_token'])
    pose_tfm = make_transform_matrix(pose)

    return np.dot(pose_tfm, sensor_tfm)


def load_point_cloud(nuscenes, sample_data):

    # Load point cloud
    lidar_path = os.path.join(nuscenes.dataroot, sample_data['filename'])
    pcl = LidarPointCloud.from_file(lidar_path)
    return pcl.points[:3, :].T


def make_transform_matrix(record):
    """
    Create a 4x4 transform matrix from a calibrated_sensor or ego_pose record
    """
    transform = np.eye(4)
    transform[:3, :3] = Quaternion(record['rotation']).rotation_matrix
    transform[:3, 3] = np.array(record['translation'])
    return transform


def render_shapely_polygon(mask, polygon, extents, resolution):

    if polygon.geom_type == 'Polygon':

        # Render exteriors
        render_polygon(mask, polygon.exterior.coords, extents, resolution, 1)

        # Render interiors
        for hole in polygon.interiors:
            render_polygon(mask, hole.coords, extents, resolution, 0)
    
    # Handle the case of compound shapes
    else:
        #for poly in polygon:
        for poly in polygon.geoms:
            render_shapely_polygon(mask, poly, extents, resolution)




