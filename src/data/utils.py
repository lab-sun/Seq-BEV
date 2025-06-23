import cv2
import numpy as np
import torch
from shapely import affinity
from .nuscenes.combinators import Rasterizer

def decode_binary_labels(labels, nclass):
    bits = torch.pow(2, torch.arange(nclass))
    return (labels & bits.view(-1, 1, 1)) > 0


def encode_binary_labels(masks):
    bits = np.power(2, np.arange(len(masks), dtype=np.int32))
    return (masks.astype(np.int32) * bits.reshape(-1, 1, 1)).sum(0)


def transform(matrix, vectors):
    vectors = np.dot(matrix[:-1, :-1], vectors.T)
    vectors = vectors.T + matrix[:-1, -1]
    return vectors


def transform_polygon(polygon, affine):
    """
    Transform a 2D polygon
    """
    a, b, tx, c, d, ty = affine.flatten()[:6]
    return affinity.affine_transform(polygon, [a, b, c, d, tx, ty])


def render_polygon(mask, polygon, extents, resolution, value=1):
    if len(polygon) == 0:
        return
    polygon = (polygon - np.array(extents[:2])) / resolution
    polygon = np.ascontiguousarray(polygon).round().astype(np.int32)
    #cv2.fillConvexPoly(mask, polygon, value)
    #print(polygon)
    cv2.fillPoly(mask, np.int32([polygon]), value)


def get_visible_mask(instrinsics, image_width, extents, resolution):

    # Get calibration parameters
    fu, cu = instrinsics[0, 0], instrinsics[0, 2]

    # Construct a grid of image coordinates
    x1, z1, x2, z2 = extents
    x, z = np.arange(x1, x2, resolution), np.arange(z1, z2, resolution)
    ucoords = x / z[:, None] * fu + cu

    # Return all points which lie within the camera bounds
    return (ucoords >= 0) & (ucoords < image_width)


def get_occlusion_mask(points, extents, resolution):

    x1, z1, x2, z2 = extents

    # A 'ray' is defined by the ratio between x and z coordinates
    ray_width = resolution / z2
    ray_offset = x1 / ray_width
    max_rays = int((x2 - x1) / ray_width)

    # Group LiDAR points into bins
    rayid = np.round(points[:, 0] / points[:, 2] / ray_width - ray_offset)
    depth = points[:, 2]

    # Ignore rays which do not correspond to any grid cells in the BEV
    valid = (rayid > 0) & (rayid < max_rays) & (depth > 0)
    rayid = rayid[valid]
    depth = depth[valid]

    # Find the LiDAR point with maximum depth within each bin
    max_depth = np.zeros((max_rays,))
    np.maximum.at(max_depth, rayid.astype(np.int32), depth)

    # For each bev grid point, sample the max depth along the corresponding ray
    x = np.arange(x1, x2, resolution)
    z = np.arange(z1, z2, resolution)[:, None]
    grid_rayid = np.round(x / z / ray_width - ray_offset).astype(np.int32)
    grid_max_depth = max_depth[grid_rayid]

    # A grid position is considered occluded if the there are no LiDAR points
    # passing through it
    occluded = grid_max_depth < z
    return occluded

# added by GS
# get one_hot label
def get_onehot_mask(mask):
    one_hot = np.array(mask.transpose(1,2,0)).astype(int)  # shape is (width, heigh, nbr_class), 0,1tensor
    nbr_class = one_hot.shape[2]
    images = []
    map_mask = np.zeros(one_hot.shape, dtype=np.int32)
    
    #expand every semantic layer mask into 3 channels by repeating
    for i in range(nbr_class):
        map_mask[:,:,i] = np.where(one_hot[:,:,i]==1, i+1, 0)
        image = np.repeat(np.expand_dims(map_mask[:,:,i], axis=2), 3, axis=2).astype(np.uint8)
        images.append(image)
    
    combinator = Rasterizer()
    img_combinator = combinator.combine(data=images)
    GT = img_combinator[:,:,0]
    return GT



# added by GS
# mask visualization
def create_visual_anno(anno):
    # print("in src/data/utils.py, anno.shape:", anno.shape)
    assert np.max(anno) <= 15, "only 15 classes are supported, add new color in label2color_dict"
    labels = {
        0: "empty_area",
        1: "drivable_area",
        2: "ped_crossing",
        3: "walkway",
        4: "movable_object",
        5: "vehicle",  
        6: "pedestrian",
        7: "mask"
    }

    label2color_dict = {
        # 0: [0, 0, 0],  #empty area
        # 1: [255, 0, 255], # driveable_area  cimare
        # 2: [150, 240, 80], # ped_crossing green
        # 3: [127, 0, 255], # walkway  purple  100, 80, 250
        # 4: [255, 255, 0], # movable_object yellow  
        # 5: [0, 127, 255], # vehicle  blue
        # 6: [255, 0, 0], # pedestrian red
        # 7: [0, 0, 0] #mask black

        # blue style
        0: [0, 0, 0],  #empty area
        1: [0, 49, 118], # driveable_area  blue
        2: [0, 71, 172], # ped_crossing blue
        3: [33, 70, 156], # walkway  blue
        4: [172, 86, 0], # movable_object orange  
        5: [170, 3, 18], # vehicle  brown
        6: [255, 195, 0], # pedestrian yellow
        7: [0, 0, 0] #mask black
    }

    #visual
    # anno.shape: (196,200)
    visual_anno = np.zeros((anno.shape[0], anno.shape[1], 3), dtype=np.uint8)
    for i in range(visual_anno.shape[0]):
        for j in range(visual_anno.shape[1]):
            color = label2color_dict[anno[i, j]]
            visual_anno[i, j, 0] = color[0]
            visual_anno[i, j, 1] = color[1]
            visual_anno[i, j, 2] = color[2]
    return visual_anno





    




    






