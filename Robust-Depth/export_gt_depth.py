
# python vddepth/export_gt_depth.py --split nuScenes_test_night --data_path /media/kieran/SSDNEW/Base-Model/data/nuScenes_RAW

import os
from PIL import Image
from tqdm import tqdm
import argparse
import numpy as np
import PIL.Image as pil
from pyquaternion import Quaternion
from utils import readlines
from kitti_utils import generate_depth_map
import os.path as osp
from datasets.nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from datasets.nuscenes.nuscenes import NuScenes
from datasets.nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box

def export_gt_depths_kitti():

    parser = argparse.ArgumentParser(description='export_gt_depth')

    parser.add_argument('--data_path',
                        type=str,
                        help='path to the root of the data',
                        required=True)
    parser.add_argument('--split',
                        type=str,
                        help='which split to export gt from',
                        required=True,
                        choices=["eigen", "eigen_benchmark", "eigen_zhou", "nuScenes_test", "nuScenes_val", "nuScenes_test_night"])
    opt = parser.parse_args()

    #eigen_zhou represents validation splits

    split_folder = os.path.join("/media/kieran/SSDNEW/Base-Model/splits", opt.split)
    if opt.split in ["eigen", "eigen_benchmark", "nuScenes_test"]:
        lines = readlines(os.path.join(split_folder, "test_files.txt"))
    elif opt.split == "nuScenes_test_night":
        lines = readlines(os.path.join(split_folder, "night_lidar.txt"))
    else:
        lines = readlines(os.path.join(split_folder, "val_files.txt"))

    if opt.split == "nuScenes_test":
        nusc = NuScenes(version="v1.0-test", dataroot=opt.data_path, verbose=False)
    else:
        nusc = NuScenes(version='v1.0-trainval', dataroot=opt.data_path, verbose=False)

    print("Exporting ground truth depths for {}".format(opt.split))

    gt_depths = []
    for line in tqdm(lines):

        if opt.split == "nuScenes_val":
            cam_index, lidar_index = line.split()
        elif opt.split =="nuScenes_test" or opt.split =="nuScenes_test_night":
            lidar_index = line
        else:
            folder, frame_id, _ = line.split()
            frame_id = int(frame_id)

        if opt.split == "eigen":
            calib_dir = os.path.join(opt.data_path, folder.split("/")[0])
            velo_filename = os.path.join(opt.data_path, folder,
                                         "velodyne_points/data", "{:010d}.bin".format(frame_id))
            gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)
        elif opt.split == "eigen_benchmark":
            gt_depth_path = os.path.join(opt.data_path, folder, "proj_depth",
                                         "groundtruth", "image_02", "{:010d}.png".format(frame_id))
            gt_depth = np.array(pil.open(gt_depth_path)).astype(np.float32) / 256
        elif opt.split == "eigen_zhou":
            calib_dir = os.path.join(opt.data_path, folder.split("/")[0])
            velo_filename = os.path.join(opt.data_path, folder,
                                         "velodyne_points/data", "{:010d}.bin".format(frame_id))
            gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)
        elif opt.split in ["nuScenes_test", "nuScenes_val", "nuScenes_test_night"]:
            if opt.split == "nuScenes_test" or opt.split == "nuScenes_test_night":
                sample = nusc.sample[int(lidar_index)]
                pointsensor_token = sample['data']['LIDAR_TOP']
                camsensor_token = sample['data']["CAM_FRONT"]
            else:
                sample_lidar = nusc.sample_data[int(lidar_index)]
                sample_cam = nusc.sample_data[int(cam_index)]
                pointsensor_token = sample_lidar
                camsensor_token = sample_cam
            
            gt_depth = get_depth(nusc, pointsensor_token, camsensor_token)
            
        gt_depths.append(gt_depth.astype(np.float32))

    output_path = os.path.join(split_folder, "gt_depths.npz")

    print("Saving to {}".format(opt.split))

    np.savez_compressed(output_path, data=np.array(gt_depths))

def get_depth(nusc, pointsensor_token, camsensor_token):
    pts, depth, img = map_pointcloud_to_image(nusc, pointsensor_token, camsensor_token)
    depth_gt = np.zeros((img.size[0], img.size[1]))
    pts_int = np.array(pts, dtype=int)
    depth_gt[pts_int[0,:], pts_int[1,:]] = depth
    return np.transpose(depth_gt, (1,0))

def map_pointcloud_to_image(nusc,
                            pointsensor_token: str,
                            camera_token: str,
                            min_dist: float = 1.0,
                            render_intensity: bool = False,
                            show_lidarseg: bool = False):

    parser = argparse.ArgumentParser(description='export_gt_depth')

    parser.add_argument('--data_path',
                        type=str,
                        help='path to the root of the data',
                        required=True)
    parser.add_argument('--split',
                        type=str,
                        help='which split to export gt from',
                        required=True,
                        choices=["eigen", "eigen_benchmark", "eigen_zhou", "nuScenes_test", "nuScenes_val", "nuScenes_test_night"])
    opt = parser.parse_args()

    if opt.split == "nuScenes_test" or opt.split == "nuScenes_test_night":
        cam = nusc.get('sample_data', camera_token)
        pointsensor = nusc.get('sample_data', pointsensor_token)
    else:
        cam = camera_token
        pointsensor = pointsensor_token
    pcl_path = osp.join(nusc.dataroot, pointsensor['filename'])
    if pointsensor['sensor_modality'] == 'lidar':
        # Ensure that lidar pointcloud is from a keyframe.
        assert pointsensor['is_key_frame'], \
            'Error: Only pointclouds which are keyframes have lidar segmentation labels. Rendering aborted.'
        pc = LidarPointCloud.from_file(pcl_path)
    else:
        pc = RadarPointCloud.from_file(pcl_path)
    im = Image.open(osp.join(nusc.dataroot, cam['filename']))

    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    depths = pc.points[2, :]

    if render_intensity:
        assert pointsensor['sensor_modality'] == 'lidar', 'Error: Can only render intensity for lidar, ' \
                                                            'not %s!' % pointsensor['sensor_modality']
        intensities = pc.points[3, :]
        intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
        intensities = intensities ** 0.1
        intensities = np.maximum(0, intensities - 0.5)
        coloring = intensities

    else:
        coloring = depths

    points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
    points = points[:, mask]
    coloring = coloring[mask]

    return points, coloring, im



if __name__ == "__main__":
    export_gt_depths_kitti()
