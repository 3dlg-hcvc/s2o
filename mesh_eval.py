import argparse
import glob
import json
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.append("../../proj-opmotion")
import pygltftoolkit as pygltk


def compute_bbox_iou(aabb1, aabb2):
    # https://pbr-book.org/3ed-2018/Geometry_and_Transformations/Bounding_Boxes#:~:text=The%20intersection%20of%20two%20bounding,Intersection%20of%20Two%20Bounding%20Boxes.
    (center1, dim1, cat1) = aabb1
    (center2, dim2, cat2) = aabb2
    if int(cat1) != int(cat2):
        return -np.inf

    aabb1_min, aabb1_max = center1 - dim1 / 2, center1 + dim1 / 2
    aabb2_min, aabb2_max = center2 - dim2 / 2, center2 + dim2 / 2
    max_min = np.maximum(aabb1_min, aabb2_min)
    min_max = np.minimum(aabb1_max, aabb2_max)

    intersection_dims = np.maximum(0, min_max - max_min)
    intersection_volume = np.prod(intersection_dims)

    gt_volume = np.prod(aabb1_max - aabb1_min)
    pred_volume = np.prod(aabb2_max - aabb2_min)
    union_volume = gt_volume + pred_volume - intersection_volume

    return intersection_volume / union_volume

def greedy_matching(list1, list2, iou_threshold):
    num_boxes_list1 = len(list1)
    num_boxes_list2 = len(list2)
    max_dim = max(num_boxes_list1, num_boxes_list2)
    distance_matrix = -np.ones((max_dim, max_dim))
    row_indices, col_indices = -np.ones(max_dim, dtype=int), -np.ones(max_dim, dtype=int)

    for i, bbox1 in enumerate(list1):
        current_matching_iou = -np.inf
        current_matching_index = -1
        for j, bbox2 in enumerate(list2):
            iou = compute_bbox_iou(bbox1, bbox2)
            if iou < iou_threshold:
                continue
            if iou > current_matching_iou and j not in col_indices:
                current_matching_index = j
                current_matching_iou = iou
            distance_matrix[i][j] = iou 
        row_indices[i] = i
        col_indices[i] = current_matching_index
    return row_indices, col_indices, distance_matrix

def precision_recall_f1(args):
    tp = 0
    fp = 0
    fn = 0

    macro_precisions = []
    macro_recalls = []
    macro_f1s = []

    for model_id in tqdm(args.model_ids):
        gt = np.load(f"{args.gt_path}/gt/{model_id}.npz", allow_pickle=True)
        pred = np.load(f"{args.predict_dir}/pred/{model_id}.npz", allow_pickle=True)

        unique_gt_instances = np.unique(gt["instance"])
        gltf = pygltk.load(f"{args.glb_path}/{model_id}/{model_id}.glb")

        gt_bboxes = []
        instance_id_idx_map_gt = {}
        for idx, instance_id in enumerate(unique_gt_instances):
            sem_label = gt["semantic"][gt["instance"] == instance_id][0]
            if int(sem_label) != 3:
                instance_id_idx_map_gt[instance_id] = idx
                triangles = gltf.vertices[gltf.faces][gt["instance"] == instance_id]
                vertices = triangles.reshape(-1, 3)
                bbox_min = np.min(vertices, axis=0)
                bbox_max = np.max(vertices, axis=0)

                center = (bbox_max + bbox_min) / 2
                dim = bbox_max - bbox_min

                gt_bboxes.append((center, dim, sem_label))

        unique_pred_instances = np.unique(pred["instance"])
        pred_bboxes = []
        instance_id_idx_map_pred = {}
        for idx, pred_instance_id in enumerate(unique_pred_instances):
            if "semantic" in pred:
                sem_label = pred["semantic"][pred["instance"] == pred_instance_id][0]
            else:
                # MeshWalker case
                sem_label = pred_instance_id
            if int(sem_label) != 3:
                instance_id_idx_map_pred[pred_instance_id] = idx
                triangles = gltf.vertices[gltf.faces][pred["instance"] == pred_instance_id]
                vertices = triangles.reshape(-1, 3)
                bbox_min = np.min(vertices, axis=0)
                bbox_max = np.max(vertices, axis=0)

                center = (bbox_max + bbox_min) / 2
                dim = bbox_max - bbox_min

                pred_bboxes.append((center, dim, sem_label))

        matched_gt_indices, matching_pred_indices, distance_matrix = greedy_matching(gt_bboxes, pred_bboxes, args.iou)
        current_tp = np.sum(matching_pred_indices >= 0)
        current_fp = len(pred_bboxes) - current_tp
        current_fn = len(gt_bboxes) - current_tp

        tp += current_tp
        fp += current_fp
        fn += current_fn

        per_object_p = tp / (tp + fp) if (tp + fp) > 0 else 0
        per_object_r = tp / (tp + fn) if (tp + fn) > 0 else 0
        per_object_f1 = 2 * (per_object_p * per_object_r) / (per_object_p + per_object_r) if (per_object_p + per_object_r) > 0 else 0

        macro_precisions.append(per_object_p)
        macro_recalls.append(per_object_r)
        macro_f1s.append(per_object_f1)

    macro_precision = np.mean(macro_precisions)
    macro_recall = np.mean(macro_recalls)
    macro_f1 = np.mean(macro_f1s)

    micro_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    micro_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    return {"Micro Prec": micro_precision, "Micro Rec": micro_recall, "Micro F1": micro_f1,
            "Macro Prec": macro_precision, "Macro Rec": macro_recall, "Macro F1": macro_f1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--predict_dir', type=str, required=True)
    parser.add_argument('-g', '--gt_path', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('--iou', default=0.5, type=float)
    parser.add_argument('--data_json', type=str, required=True)
    parser.add_argument('--glb_path', type=str, required=True)

    args = parser.parse_args()
    with open(args.data_json, "r") as f:
        metadata = json.load(f)
        args.model_ids = metadata["val"].keys()
    os.makedirs(f"{args.output_dir}/{args.iou}", exist_ok=True)

    metrics = precision_recall_f1(args)
    print(metrics)

    with open(f"{args.output_dir}/{args.iou}/map.json", "w+") as f:
        json.dump(metrics, f)