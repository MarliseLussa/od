import os
import pickle
import numpy as np
import open3d as o3d

# Step 1: Load the data from pkl files
def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# 使用示例
base_file = '01161336'
root_file = '/clever/volumes/label-train/xyzt/pvrcnn_plusplus/xyztdir/'

det_annos_path=os.path.join(root_file,base_file,'eval_det_annos.pkl')
pcd_paths_path = os.path.join(root_file,base_file,'eval_pcd_paths.pkl')
gt_annos_path = os.path.join(root_file,base_file,'eval_gt_annos.pkl')
pcd_base_path = "/clever/volumes/label-train/datasets/bevod/ac_pandar128_datasets_dev"

# Updated paths
# det_annos_path = "/clever/volumes/label-train/xyzt/pvrcnn_plusplus/xyztdir/eval_det_annos.pkl"
# pcd_paths_path = "/clever/volumes/label-train/xyzt/pvrcnn_plusplus/xyztdir/eval_pcd_paths.pkl"
# gt_annos_path = "/clever/volumes/label-train/xyzt/pvrcnn_plusplus/xyztdir/eval_gt_annos.pkl"
# pcd_base_path = "/clever/volumes/label-train/datasets/bevod/ac_pandar128_datasets_dev"

# Load data
pre = load_pkl(det_annos_path)
pcd_paths = load_pkl(pcd_paths_path)
gt = load_pkl(gt_annos_path)

# Step 2: Add absolute path prefix to pcd paths
full_pcd_paths = [os.path.join(pcd_base_path, pcd_path) for pcd_path in pcd_paths]

# Step 3: Define a function to read point cloud and boxes, then save as independent files
def save_pcd_with_boxes(pcd_file, gt_boxes, pre_boxes, save_dir, idx):
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Save paths
    pcd_save_path = os.path.join(save_dir, f"visualization_result_frame_{idx}_pointcloud.ply")
    gt_bbox_save_path = os.path.join(save_dir, f"visualization_result_frame_{idx}_gt_bbox.ply")
    pre_bbox_save_path = os.path.join(save_dir, f"visualization_result_frame_{idx}_pre_bbox.ply")

    # Load point cloud
    pcd = o3d.io.read_point_cloud(pcd_file)
    
    # Save point cloud
    o3d.io.write_point_cloud(pcd_save_path, pcd)
    print(f"Point cloud saved to {pcd_save_path}")
    
    # Save Ground Truth bounding boxes
    gt_lines = []
    for box in gt_boxes:
        bbox = create_open3d_box(box, color=[0, 1, 0])  # Green for Ground Truth
        gt_lines.append(bbox)
    if len(gt_lines) > 0:
        gt_line_set = merge_line_sets(gt_lines)  # Combine all GT boxes
        o3d.io.write_line_set(gt_bbox_save_path, gt_line_set)
        print(f"Ground Truth bounding boxes saved to {gt_bbox_save_path}")
    else:
        print("No Ground Truth bounding boxes to save.")

    # Save Predicted bounding boxes
    pre_lines = []
    for box in pre_boxes:
        bbox = create_open3d_box(box, color=[1, 0, 0])  # Red for Predictions
        pre_lines.append(bbox)
    if len(pre_lines) > 0:
        pre_line_set = merge_line_sets(pre_lines)  # Combine all predicted boxes
        o3d.io.write_line_set(pre_bbox_save_path, pre_line_set)
        print(f"Predicted bounding boxes saved to {pre_bbox_save_path}")
    else:
        print("No Predicted bounding boxes to save.")

# Helper function to create an Open3D 3D box
def create_open3d_box(box, color=[1, 0, 0]):
    # Box is expected in the format (x, y, z, dx, dy, dz, yaw)
    x, y, z, dx, dy, dz, yaw = box
    corners = get_3d_box_corners(x, y, z, dx, dy, dz, yaw)
    
    # Define the lines to connect box corners
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    ]
    
    # Convert to Open3D LineSet object
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    return line_set

# Helper function to combine multiple LineSet objects into one
def merge_line_sets(line_sets):
    merged_line_set = o3d.geometry.LineSet()
    points = []
    lines = []
    colors = []
    offset = 0
    for line_set in line_sets:
        points.extend(np.asarray(line_set.points))
        lines.extend(np.asarray(line_set.lines) + offset)
        colors.extend(np.asarray(line_set.colors))
        offset += len(line_set.points)
    merged_line_set.points = o3d.utility.Vector3dVector(points)
    merged_line_set.lines = o3d.utility.Vector2iVector(lines)
    merged_line_set.colors = o3d.utility.Vector3dVector(colors)
    return merged_line_set

# Compute 3D box corners
def get_3d_box_corners(x, y, z, dx, dy, dz, yaw):
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                  [np.sin(yaw),  np.cos(yaw), 0],
                  [0,            0,           1]])
    corners = np.array([[dx/2, dy/2, dz/2],
                        [-dx/2, dy/2, dz/2],
                        [-dx/2, -dy/2, dz/2],
                        [dx/2, -dy/2, dz/2],
                        [dx/2, dy/2, -dz/2],
                        [-dx/2, dy/2, -dz/2],
                        [-dx/2, -dy/2, -dz/2],
                        [dx/2, -dy/2, -dz/2]])
    corners = np.dot(corners, R.T) + np.array([x, y, z])
    return corners

# Step 4: Batch visualize and save for a range of indices
start_idx = 20  # Starting index
end_idx = 40  # Ending index (exclusive)

for idx in range(start_idx, end_idx):
    print(f"Processing index: {idx}")
    pcd_file = full_pcd_paths[idx]
    gt_boxes = gt[idx]['gt_boxes_lidar']
    pre_boxes = pre[idx]['boxes_lidar']
    
    # Create a subdirectory for each index
    save_dir = f"/clever/volumes/label-train/xyzt/pvrcnn_plusplus/xyztdir/visual-result-{base_file}/{idx}"
    # save_dir = os.path.join('/clever/volumes/label-train/xyzt/pvrcnn_plusplus/xyztdir/visual-result/',base_file,)
    # Save point cloud and bounding boxes
    save_pcd_with_boxes(pcd_file, gt_boxes, pre_boxes, save_dir, idx)