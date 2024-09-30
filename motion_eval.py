import argparse
import copy
import json
import os
import random
import sys
import time
from glob import glob

import numpy as np
import open3d as o3d
import pygltftoolkit as pygltk
import trimesh
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from opmotion import Evaluator

# Set the random seed
RANDOM_SEED = 521
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
LABEL_ID_LABEL_MAP = {0: "drawer", 1: "door", 2: "lid", 3: "base"}
IOU_THRESHOLD = 0.5


def existDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def aabb_distance(aabb1, aabb2):
    (center1, dim1, cat1, _) = aabb1[:3]
    (center2, dim2, cat2, _) = aabb2[:3]

    if cat1 != cat2:
        # Dummy value, gets rejected later on. If np.inf is used - linear_sum_assignment returns error in case there are 
        # rows of distance_matrix consisting solely out of np.inf
        distance = 123456789
    else:
        dists = np.maximum(0, np.abs(center1 - center2) - (dim1 + dim2) / 2)
        distance = np.linalg.norm(dists)
    return distance


def compute_bbox_iou(aabb1, aabb2):
    # https://pbr-book.org/3ed-2018/Geometry_and_Transformations/Bounding_Boxes#:~:text=The%20intersection%20of%20two%20bounding,Intersection%20of%20Two%20Bounding%20Boxes.
    (center1, dim1, cat1) = aabb1[:3]
    (center2, dim2, cat2) = aabb2[:3]
    if cat1 != cat2:
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


def hungarian_matching(list1, list2):
    num_boxes_list1 = len(list1)
    num_boxes_list2 = len(list2)
    distance_matrix = np.zeros((num_boxes_list1, num_boxes_list2))

    for i, bbox1 in enumerate(list1):
        for j, bbox2 in enumerate(list2):
            distance_matrix[i, j] = aabb_distance(bbox1, bbox2)
    row_indices, col_indices = None, None
    row_indices, col_indices = linear_sum_assignment(distance_matrix)

    return row_indices, col_indices, distance_matrix


def greedy_matching(list1, list2):
    global IOU_THRESHOLD
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
            if iou < IOU_THRESHOLD:
                continue
            if iou > current_matching_iou and j not in col_indices:
                current_matching_index = j
                current_matching_iou = iou
            distance_matrix[i][j] = iou
        row_indices[i] = i
        col_indices[i] = current_matching_index
    return row_indices, col_indices, distance_matrix


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_json",
        required=True,
        metavar="FILE",
    )
    parser.add_argument(
        "--glb_path",
        required=True,
        metavar="DIR",
    )
    parser.add_argument(
        "--predict_dir",
        required=True,
        metavar="DIR",
    )
    parser.add_argument(
        "--motion_type",
        default="opmotion",
    )
    parser.add_argument(
        "--inference_file",
        default=None,
        metavar="FILE",
        help="path to the inference file. If this value is not None, then the program will use existing predictions instead of inferencing again",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        metavar="DIR",
    )
    return parser


if __name__ == "__main__":
    start = time.time()

    args = get_parser().parse_args()

    if args.inference_file is not None:
        print("Skip the inference step and use the existing inference results")
        with open(args.inference_file) as f:
            results = json.load(f)
    else:
        existDir(f"{args.output_dir}")
        with open(args.data_json) as f:
            data = json.load(f)
        split = data[args.split]
        model_ids = list(split.keys())
        results = {}

        n_preds = 0

        for model_id in tqdm(model_ids):
            results[model_id] = {"gt": {}, "pred": {}}
            model_path = f"{args.glb_path}/{model_id}/{model_id}.glb"
            anno_path = f"{args.glb_path}/{model_id}/{model_id}.art-stk.json"

            gltf = pygltk.load(model_path)
            gltf.load_stk_segmentation_openable(anno_path)
            gltf.load_stk_articulation(anno_path)

            pred_boxes = []
            pred_categories = []
            pred_results = []

            prediction_paths = glob(f"{args.predict_dir}/{args.motion_type}/{model_id}-*.json")
            for prediction_path in prediction_paths:
                prediction_id = prediction_path.split("/")[-1].split(".")[0].split("-")[1]
                if os.path.exists(f"{args.predict_dir}/pcd/{model_id}-{prediction_id}.npz"):
                    n_preds += 1
                    pcd_data = np.load(f"{args.predict_dir}/pcd/{model_id}-{prediction_id}.npz")
                    points = np.asarray(pcd_data["points"])

                    with open(prediction_path) as f:
                        prediction = json.load(f)

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                    # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame()
                    # o3d.visualization.draw_geometries([pcd] + [coordinate])

                    aabb = pcd.get_axis_aligned_bounding_box()
                    min_bound = aabb.get_min_bound()
                    max_bound = aabb.get_max_bound()
                    center = (max_bound + min_bound) / 2
                    dim = max_bound - min_bound
                    prediction_cat = LABEL_ID_LABEL_MAP[int(pcd_data["semantic"])]
                    pred_boxes.append((center, dim, prediction_cat))
                    pred_categories.append(prediction_cat)
                    if "mextrinsic" in prediction.keys():
                        maxis = np.asarray(prediction["maxis"] + [1])
                        morigin = np.asarray(prediction["morigin"] + [1])
                        motion_end = np.asarray((morigin[:3] + maxis[:3]).tolist() + [1])

                        extr = np.asarray(prediction["gtextrinsic"]).reshape([4, 4]).T
                        world_morigin = np.dot(extr, morigin)
                        world_end = np.dot(extr, motion_end)

                        world_maxis = (world_end - world_morigin)[:-1]
                        world_morigin = world_morigin[:-1]

                        pred_results.append({"id": prediction_id, "parent": "base",
                                             "cat": prediction_cat,
                                             "motionType": "revolute" if int(prediction["mtype"]) == 0 else "prismatic",
                                             "motionAxis": world_maxis,
                                             "motionOrigin": world_morigin,
                                             "gtextrinsic": prediction["gtextrinsic"]})
                    else:
                        pred_results.append({"id": prediction_id, "parent": "base",
                                             "cat": prediction_cat,
                                             "motionType": "revolute" if int(prediction["mtype"]) == 0 else "prismatic",
                                             "motionAxis": prediction["maxis"],
                                             "motionOrigin": prediction["morigin"]})

            gt_boxes = []
            gt_categories = []
            for pid, part in gltf.segmentation_parts.items():
                if part.label == "base":
                    continue
                current_pid = part.pid
                part_triangles = gltf.vertices[gltf.faces[gltf.segmentation_map == current_pid]]
                bbox_min = np.min(part_triangles, axis=0)[0]
                bbox_max = np.max(part_triangles, axis=0)[0]
                center = (bbox_min + bbox_max) / 2
                dim = bbox_max - bbox_min
                gt_boxes.append((center, dim, part.label, current_pid))
                gt_categories.append(part.label)
            # matched_gt_indices, matching_pred_indices, distance_matrix = hungarian_matching(gt_boxes, pred_boxes)
            matched_gt_indices, matching_pred_indices, distance_matrix = greedy_matching(gt_boxes, pred_boxes)
            if isinstance(matched_gt_indices, np.ndarray):
                for idx, matched_gt_index in enumerate(matched_gt_indices):
                    matched_pred_index = matching_pred_indices[idx]
                    if matched_pred_index != -1 and matched_gt_index != -1:
                        # Hungarian matching
                        # distance = distance_matrix[matched_gt_index, matched_pred_index]
                        # if distance != 123456789:
                        iou = distance_matrix[matched_gt_index][matched_pred_index]
                        gt_part = gltf.articulation_parts[gt_boxes[matched_gt_index][3]]
                        pred_result = pred_results[matched_pred_index]

                        results[model_id]["pred"][pred_result["id"]] = pred_result

                        results[model_id]["gt"][pred_result["id"]] = {"id": pred_result["id"], "parent": "base",
                                                                      "cat": pred_result["cat"],
                                                                      "motionType": gt_part.type,
                                                                      "motionAxis": gt_part.axis,
                                                                      "motionOrigin": gt_part.origin,
                                                                      "diagonal": (gt_boxes[matched_gt_index][:3][1]**2).sum() ** 0.5,
                                                                      "gt_id": gt_part.pid}
    end = time.time()
    print(f"Time for inference: {end - start}")

    # Save the evaluation results
    start = time.time()
    existDir(args.output_dir)
    existDir(f"{args.output_dir}/motion_evaluation")
    evaluator = Evaluator(
        results,
    )

    evaluator.evaluate()
    performance = evaluator.summarize()

    total_n_preds = n_preds
    total_matches = sum([value for value in performance["part_num"].values()])

    metrics = ["M", "MA", "MAO"]

    n_gt = 0
    for model_id in results.keys():
        n_gt += len(results[model_id]["gt"])

    for metric in metrics:
        p = performance[f"micro_{metric}"]
        precision = p * total_matches / total_n_preds
        recall = p * total_matches / n_gt
        performance[f"micro_{metric}"] = precision
        performance[f"micro_{metric}_recall"] = recall

    with open(f"{args.output_dir}/{args.split}_performance.json", "w") as f:
        json.dump(performance, f, indent=4)

    end = time.time()
    print(f"Time for evaluation: {end - start}")
