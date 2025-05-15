import argparse
import copy
import glob
import os
import pdb

import h5py
import numpy as np
import open3d as o3d
from tqdm import tqdm

IOU_THRESHOLD = 0.8


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

    parser.add_argument('-e ', '--exp_dir', type=str, default=None, help='Experiment directory')
    parser.add_argument('-d', '--data_path', type=str, help="Data path")
    parser.add_argument('-s', '--subset_path', type=str, help="Subset data path")
    parser.add_argument('-o', '--output_path', type=str, help="Output path")
    args = parser.parse_args()

    os.makedirs(f"{args.output_path}/pred_mask", exist_ok=True)
    os.makedirs(f"{args.output_path}/predicted_masks", exist_ok=True)

    original_h5 = h5py.File(f"{args.data_path}", "r")
    subset_h5 = h5py.File(f"{args.subset_path}", "r")

    model_idx_map = {}
    for i, id in enumerate(subset_h5["model_ids"]):
        model_idx_map[id.decode("utf-8")] = i

    if os.path.exists(f"{args.exp_dir}/inference/val/predictions/instance"):
        args.predict_dir = f"{args.exp_dir}/inference/val/predictions/instance"
    else:
        args.predict_dir = f"{args.exp_dir}"

    model_ids = [path.split('/')[-1].split('.')[0] for path in glob.glob(f"{args.predict_dir}/*.txt")]

    subset_points_len = subset_h5["n_points"][0]

    for model_id in tqdm(model_ids):
        with open(f"{args.predict_dir}/{model_id}.txt", "r") as f:
            lines = f.readlines()

        if model_id not in model_idx_map:
            print(f"Model {model_id} not in subset")
            continue
        subset_points = subset_h5["points"][model_idx_map[model_id]]
        original_points = original_h5["points"][model_idx_map[model_id]].reshape([original_h5["n_points"][model_idx_map[model_id]], 3])
        fps_mask = np.asarray(subset_h5["fps_mask"][model_idx_map[model_id]], dtype=int)
        instance_mask = -np.ones(subset_points_len, dtype=int)
        confidence_map = {}
        semantic_map = {}

        for instance_id, line in enumerate(lines):
            mask_file, prediction, conf = line.split(" ")
            confidence_map[str(instance_id)] = float(conf)
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

        with open(f"{args.output_path}/{model_id}.txt", "w+") as f:
            f.writelines(updated_lines)

        for (instance_id, line) in updated_lines_with_id:
            mask_file, _, _ = line.split(" ")
            with open(f"{args.output_path}/{mask_file}", "w+") as f:
                cur_instance_mask = np.asarray(np.asarray(full_instance_labels == instance_id, dtype=int), dtype=str)
                cur_instance_mask = [str(e) + "\n" for e in cur_instance_mask]
                f.writelines(cur_instance_mask)
