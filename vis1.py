#! coding: utf-8

# pandar128

import os
import sys

import numpy as np

print(os.path.dirname(__file__))
print(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tools'))

import torch
import argparse
from pcdet.config import cfg, cfg_from_yaml_file

from pathlib import Path
from tools.visual_utils.open3d_vis_utils import draw_scenes, draw_box
from tools.demo_all_in_one import DemoDatasetWaymo
from pcdet.utils import common_utils
from pcdet.models import build_network, load_data_to_gpu

from load_gt import get_gtbox_at128, lidar2ego, get_gtbox_pandar128, get_gtbox_pandar128_wt, get_gtbox

pandar_Extrinsics = np.array([[1.06805951e-03,-9.99857909e-01,-1.68729842e-02,1.07098034e+00],
                        [9.99810288e-01,7.39604419e-04,1.94593057e-02,-4.65892236e-02],
                        [-1.94438655e-02,-1.68905455e-02,9.99667673e-01,1.95128934e+00],
                        [0.00000000e+00,0.00000000e+00,0.00000000e+00,1.00000000e+00]])

at_Extrinsics = np.array([[0.0171819663, -0.999852354, 0.0002251974, -0.0876900181],
                       [0.999802211, 0.01718337, 0.0100130518, 1.78330076],
                       [-0.0100154395,  0.0000530988688, 0.999949851, 1.16920168],
                       [0.00000000e+00,0.00000000e+00,0.00000000e+00,1.00000000e+00]])

pp128_E = np.array([[0.99989665,  0.00246708, -0.01416377, 1.005],
                    [-0.01413665, -0.01068225, -0.99984301, 0],
                    [-0.00261799,  0.9999399,  -0.01064627, 1.54],
                    [0.00000000e+00, 0.00000000e+00,0.00000000e+00,1.00000000e+00]])

Extrinsics = at_Extrinsics

import json
from scipy.spatial.transform import Rotation as R


def get_bbox(json_file):
    gt_vehicle, gt_pedestrian, gt_cyclist = 0, 0, 0
    fp = open(json_file)
    contents = json.load(fp)
    fp.close()

    boxes3d_center = []
    labels = []
    for frame_id, frame_res in contents.items():
        for anno in frame_res:
            size = anno['size']
            rotation = anno['rotation']
            translation = anno['translation']
            velocity = anno['velocity']
            category = anno['detection_name']
            score = anno['detection_score']

            rot_mat = R.from_quat(rotation).as_matrix()
            euler = np.array(R.from_matrix(rot_mat).as_euler('zxy', degrees=False))

            yaw_angle = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])

            size = [size[1], size[0], size[2]]
            gt_boxes = translation + size + [yaw_angle]

            labels.append(category)
            boxes3d_center.append(gt_boxes)

    return np.array(boxes3d_center), np.array(labels)

def get_gt_bbox(json_file):
    gt_vehicle, gt_pedestrian, gt_cyclist = 0, 0, 0
    fp = open(json_file)
    contents = json.load(fp)
    fp.close()

    # meta = contents['meta']
    # ego_pose = meta['ego_pose']
    # sensor = meta['sensor']
    # pandar128 = sensor[7]
    # R = pandar128['sensor_param']['sensor2ego_rotation']
    # T = pandar128['sensor_param']['sensor2ego_translation']
    # R, T = rotation_translation(np.array(R), np.array(T))

    # RT = np.hstack((np.array(R), np.array(T)))
    # Extrinsics = np.vstack((RT, np.zeros((1, 4), dtype=np.float32)))
    # Extrinsics[3, 3] = 1.0

    annotations = contents['annotations']

    boxes3d_corner = []
    boxes3d_center = []

    labels = []
    ids = []
    for anno in annotations:
        # corner
        if anno['labeling_type'] != 'PC_3D':
            continue
        # id = anno['id']
        # if id in ids:
        #     continue

        bboxes = anno['PC_3D']
        category = anno['category']

        # boxes3d_center.append(bboxes[:7])

        # 只选择某一类别，例如cyclist
        # if category not in ['cyclist', 'pedestrian', 'car']:
        #     continue
        # if category not in ['cyclist']:
        #     continue
        # if category in ['MotorVehicle'] and sub_category in ['Trailer', 'CommercialTruck']:
        #     continue

        labels.append(category)

        # if category in ['cyclist']:

        # center + lwh + heading
        # xyz = [bboxes[0], bboxes[1], bboxes[2]]
        # xyz_ = np.hstack((xyz, [[1.]]))
        # new_xyz = Extrinsics.dot(xyz_.T)
        # gt_boxes = [new_xyz[0], new_xyz[1], new_xyz[2], bboxes[4], bboxes[3], bboxes[5], bboxes[6]]
        gt_boxes = [bboxes[0], bboxes[1], bboxes[2], bboxes[4], bboxes[3], bboxes[5], bboxes[8]]
        # gt_boxes = [bboxes[0] - float(ego_pose['x']), bboxes[1] - float(ego_pose['y']), bboxes[2] - float(ego_pose['z']),
        #             bboxes[3], bboxes[4], bboxes[5],
        #             bboxes[6] + float(ego_pose['yaw'])]

        boxes3d_center.append(gt_boxes)
        # ids.append(id)

    return np.array(boxes3d_center), np.array(labels)

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_type', type=str, default='kitti',
                        help='specify the data type for demo (kitti, waymo, nuscences)')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    args.cfg_file = '../tools/cfgs/waymo_models/pv_rcnn_plusplus_resnet_infer.yaml'
    # args.data_path = '/opt/dataset/3dod_debug/52497/'  # at128
    # args.data_path = '/opt/dataset/lidar20221213/Lidar/Pandar128/'  # zhp
    # args.data_path = '/opt/dataset/lidar20221213/Lidar/mytest/'
    # args.data_path = '/opt/dataset/tmp_debug/lidar_compensate'
    args.data_path = '/home/sczone/Downloads/20230518172402149 (1).pcd'
    args.ckpt = '/opt/code/point_cloud/pcdet-pvrcnnplusplus-master/tools/model/waymo_pv_rcnn_pp_d1.pth'  # zhp
    args.ext = '.pcd'  # zhp
    args.data_type = 'waymo'  # zhp

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

args, cfg = parse_config()
logger = common_utils.create_logger()

demo_dataset = DemoDatasetWaymo(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )

logger.info(f'Total number of samples: \t{len(demo_dataset)}')

model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
model.cuda()
model.eval()

# json_file = '/opt/dataset/3dod_debug/52497.json'
# json_file = '/opt/dataset/lidar20221213/Lidar/mytest/n000001_2022-12-13-10-18-46-200156_Pandar128.json'
# json_file = '/opt/dataset/post-fusion-debug/8/20231123_162458_n000001_out_searching_8/LSJWK409XNS001953+n000001_2023-11-23-16-29-00-500062_Fusion.pcd.json'
# json_file = '/home/sczone/Downloads/20231123_162458_n000001_out_searching_8.json'
# json_file = '/home/sczone/Downloads/data_track_new.json'
# pred_json_file = '/home/sczone/Downloads/20230730152113557.json'
pred_json_file = None
json_file = '/home/sczone/Downloads/20230518172402149_new_anno (1).json'

for idx in range(len(demo_dataset)):
    with torch.no_grad():
        data_dict = demo_dataset[idx]
        logger.info(f'Visualized sample index: \t{idx + 1}')
        data_dict = demo_dataset.collate_batch([data_dict])
        # load_data_to_gpu(data_dict)
        # pred_dicts, _ = model.forward(data_dict)
        # print(pred_dicts)  # zhp

        import open3d

        filepath = demo_dataset.sample_file_list[idx]
        # for hil debug
        base_dir, filename = os.path.split(filepath)
        # json_file = os.path.join(base_dir.replace('lidar_compensate', 'od'), 'LSJWK4095NS119733+' + filename + '.json')
        if not os.path.exists(json_file):
            logger.error(f'file not exists: {json_file}')
            continue

        pcd = open3d.t.io.read_point_cloud(filename=str(filepath))
        positions = pcd.point.positions.numpy()
        # ego2lidar = np.array(
        #     [[-0.005235734090522084, 0.9998620648788563, -0.015761925793596133, 0.03733606366289205],
        #      [-0.9999314620548416, -0.005399856626071972, -0.010388105760180765, 1.1008060715971923],
        #      [-0.01047178501499049, 0.015706456144125976, 0.999821808600909, -1.9983323074716361], [0.0, 0.0, 0.0, 1.0]]
        # )
        # positions = np.concatenate((positions, np.ones((positions.shape[0], 1))), axis=1).dot(ego2lidar.T)

        intensity = pcd.point.intensity.numpy()
        points = np.hstack((positions, intensity)).astype(np.float64)

        # points = data_dict['points'][:, 1:]
        # json_file = demo_dataset.sample_file_list[idx].replace('.pcd', '.json')
        # gt_boxes, labels, (gt_v, gt_p, gt_c) = np.array(get_gtbox_pandar128_wt(json_file, idx))
        if pred_json_file is not None:
            pred_boxes, pred_labels = get_bbox(pred_json_file)
        else:
            pred_boxes, pred_labels = None, None
        gt_boxes, labels = get_gt_bbox(json_file)
        # r, t, gt_boxes, labels, (gt_v, gt_p, gt_c) = np.array(get_gtbox(json_file, idx))
        # r = np.array(r).reshape(3, 3)
        # t = np.array(t).reshape(1, -1)
        gt_boxes = np.array(gt_boxes)
        # gt_boxes = np.array([lidar2ego(r, t, x) for x in gt_boxes])
        # mask = (np.array(labels) == 'TwoWheels') | (np.array(labels) == 'Pedestrian')
        # gt_boxes = gt_boxes[mask]

        # gt_xyz = gt_boxes.copy()
        # gt_xyz_ = np.concatenate((gt_xyz[:, :3], np.ones((gt_xyz.shape[0], 1))), axis=1)
        # gt_xyz_ = gt_xyz_.dot(Extrinsics.T)
        # gt_xyz_[:, 3] = gt_xyz_[:, 3] / 255.0
        # gt_xyz[:, :3] = gt_xyz_[:, :3]
        # gt_xyz[:, 6] = -gt_xyz[:, 6] + np.pi / 2
        #
        # print({'gt_boxes': gt_boxes})
        # ref_boxes = pred_dicts[0]['pred_boxes']
        # ref_labels = pred_dicts[0]['pred_labels']
        # ref_scores = pred_dicts[0]['pred_scores']

        # mask = ref_scores > 0.8
        # ref_boxes = ref_boxes[mask]
        # ref_labels = ref_labels[mask]
        # ref_scores = ref_scores[mask]

        # draw_scenes(points, gt_boxes, ref_boxes, ref_labels, ref_scores)
        # draw_scenes(points, gt_xyz, ref_boxes, ref_labels, ref_scores)  #
        draw_scenes(points, gt_boxes, ref_boxes=pred_boxes)

