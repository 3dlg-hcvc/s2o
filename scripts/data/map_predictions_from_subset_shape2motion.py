import argparse
import copy
import glob
import os
import pdb

import h5py
import numpy as np
import open3d as o3d
from tqdm import tqdm


def knn_assign_instances(original_points, subset_points, subset_instance_labels, fps_mask):
    subset_pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(subset_points))

    unassigned_idx = np.where(fps_mask == 0)[0]

    full_instance_labels = -np.ones(np.shape(fps_mask), dtype=int)
    full_instance_labels[fps_mask == 1] = subset_instance_labels

    search_tree = o3d.geometry.KDTreeFlann(subset_pcd)
    k = 3

    for idx in unassigned_idx:
        result_k, result_indexes, _ = search_tree.search_knn_vector_3d(np.array(original_points[idx]).reshape((3, 1)).astype(np.float64), k)
        closest_instance_ids = subset_instance_labels[result_indexes]
        most_popular_label = np.argmax(np.bincount(closest_instance_ids))

        full_instance_labels[idx] = most_popular_label

    return full_instance_labels


def compute_iou(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)

    iou = intersection.sum() / union.sum()
    return iou


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_dir', type=str, help='Experiment directory')
    parser.add_argument('-d', '--data_path', type=str, help="Data path")
    parser.add_argument('-o', '--output_path', type=str, help="Output path")
    parser.add_argument('-p', '--processed_data', type=str, help="Path to shape2motion prepocessed data", default="../shape2motion-pytorch/results/preprocess/val.h5")
    args = parser.parse_args()

    os.makedirs(f"{args.output_path}/pred_mask", exist_ok=True)
    os.makedirs(f"{args.output_path}/predicted_masks", exist_ok=True)

    original_h5 = h5py.File(f"{args.data_path}/downsample.h5", "r")
    processed_h5 = h5py.File(f"{args.processed_data}", "r")

    model_idx_map = {}
    for i, id in enumerate(original_h5["model_ids"]):
        correct_id = id.decode("utf-8")
        model_idx_map[correct_id] = i

    args.predict_dir = f"{args.exp_dir}"

    model_ids = [path.split('/')[-1].split('.')[0] for path in glob.glob(f"{args.predict_dir}/*.txt")]
    print(model_ids)

    subset_points_len = processed_h5[f"{model_ids[0]}_0"]["input_pts"].shape[0]

    for correct_id in tqdm(model_ids):
        model_id = f"{correct_id}_0"
        with open(f"{args.predict_dir}/{correct_id}.txt", "r") as f:
            lines = f.readlines()

        correct_id = model_id.split("_")[0]

        subset_points = processed_h5[model_id]["input_pts"][:, :3]
        if "n_points" in original_h5.keys():
            original_points = original_h5["points"][model_idx_map[correct_id]].reshape([original_h5["n_points"][model_idx_map[correct_id]], 3])
        else:
            original_points = original_h5["points"][model_idx_map[correct_id]]

        point_idx = processed_h5[model_id]["point_idx"]
        fps_mask = np.zeros(original_points.shape[0], dtype=int)
        fps_mask[point_idx] = 1
        instance_mask = -np.ones(subset_points_len, dtype=int)
        confidence_map = {}
        semantic_map = {}

        for instance_id, line in enumerate(lines):
            mask_file, prediction, conf = line.split(" ")
            confidence_map[str(instance_id)] = conf
            semantic_map[str(instance_id)] = prediction
            with open(f"{args.predict_dir}/{mask_file}", "r") as f:
                point_masked = f.readlines()
            prediction_mask = np.asarray(point_masked, dtype=int)
            if (instance_mask[prediction_mask == 1] != -1).any():
                overlapping_instances = np.unique(instance_mask[prediction_mask == 1])
                for overlapping_instance in overlapping_instances:
                    if overlapping_instance != -1:
                        iou = compute_iou(prediction_mask == 1, instance_mask == overlapping_instance)
                        if iou > IOU_THRESHOLD:
                            if confidence_map[str(instance_id)] > confidence_map[str(overlapping_instance)]:
                                instance_mask[instance_mask == overlapping_instance] = -1
                            else:
                                prediction_mask[prediction_mask == 1] = 0
                        else:
                            overlapping_mask = np.logical_and(prediction_mask == 1, instance_mask == overlapping_instance)
                            if confidence_map[str(instance_id)] > confidence_map[str(overlapping_instance)]:
                                instance_mask[overlapping_mask] = -1
                            else:
                                prediction_mask = np.logical_xor(prediction_mask == 1, overlapping_mask).astype(int)
            instance_mask[prediction_mask == 1] = instance_id

        valid_idx = np.where(instance_mask != -1)[0]
        subset_points = subset_points[valid_idx]
        non_valid_idx = np.where(instance_mask == -1)[0]
        non_valid_mask = np.asarray(instance_mask == -1, dtype=int)

        valid_instance_mask = instance_mask[valid_idx]
        fps_idx = np.where(fps_mask == 1)[0]
        combined_mask = np.zeros(fps_mask.shape)
        combined_mask[fps_idx] = non_valid_mask
        fps_mask[combined_mask == 1] = 0

        full_instance_labels = knn_assign_instances(original_points, subset_points, valid_instance_mask, fps_mask)

        present_instances = np.unique(full_instance_labels)
        updated_lines = []
        updated_lines_with_id = []
        for instance_id, line in enumerate(lines):
            if instance_id in full_instance_labels:
                updated_lines_with_id.append((instance_id, line))
                updated_lines.append(line)

        with open(f"{args.output_path}/{correct_id}.txt", "w+") as f:
            f.writelines(updated_lines)

        for (instance_id, line) in updated_lines_with_id:
            mask_file, _, _ = line.split(" ")
            with open(f"{args.output_path}/{mask_file}", "w+") as f:
                cur_instance_mask = np.asarray(np.asarray(full_instance_labels == instance_id, dtype=int), dtype=str)
                cur_instance_mask = [str(e) + "\n" for e in cur_instance_mask]
                f.writelines(cur_instance_mask)
