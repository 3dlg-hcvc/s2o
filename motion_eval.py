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

RANDOM_SEED = 521
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
LABEL_ID_LABEL_MAP = {0: "drawer", 1: "door", 2: "lid", 3: "base"}
IOU_THRESHOLD = 0.5
MOTION_CODES = {0: "revolute", 1: "prismatic", 2: "fixed"}
MOTION_LOOKUP = {"rotation": "revolute", "translation": "prismatic", "fixed": "fixed"}

def existDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def compute_triangle_areas(gltf):
    triangles = gltf.vertices[gltf.faces]
    v0 = triangles[:, 0, :]
    v1 = triangles[:, 1, :]
    v2 = triangles[:, 2, :]
    cross_prod = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross_prod, axis=1)
    return areas


def compute_triangle_area_iou(inst1, inst2, face_areas):
    area1, mask1, cat1 = inst1[:3]
    area2, mask2, cat2 = inst2[:3]
    if cat1 != cat2:
        return -np.inf
    intersection_mask = mask1 & mask2
    intersection_area = np.sum(face_areas[intersection_mask])
    union_area = area1 + area2 - intersection_area
    return intersection_area / union_area


def greedy_matching(list1, list2, iou_threshold, face_areas):
    num_boxes_list1 = len(list1)
    num_boxes_list2 = len(list2)
    max_dim = max(num_boxes_list1, num_boxes_list2)
    distance_matrix = -np.ones((max_dim, max_dim))
    row_indices, col_indices = -np.ones(max_dim, dtype=int), -np.ones(max_dim, dtype=int)
    for i, inst1 in enumerate(list1):
        current_matching_iou = -np.inf
        current_matching_index = -1
        for j, inst2 in enumerate(list2):
            iou = compute_triangle_area_iou(inst1, inst2, face_areas)
            if iou < iou_threshold:
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
        n_gt = 0
        model_gt_n = {}

        for model_id in tqdm(model_ids):
            results[model_id] = {"gt": {}, "pred": {}}
            model_path = f"{args.glb_path}/{model_id}/{model_id}.glb"
            anno_path = f"{args.glb_path}/{model_id}/{model_id}.art-stk.json"

            gltf = pygltk.load(model_path)
            gltf.load_stk_segmentation_openable(anno_path)
            gltf.load_stk_articulation(anno_path)

            face_areas = compute_triangle_areas(gltf)

            pred_boxes = []
            pred_categories = []
            pred_results = []

            prediction_paths = glob(f"{args.predict_dir}/{args.motion_type}/{model_id}-*.json")
            for prediction_path in prediction_paths:
                prediction_id = prediction_path.split("/")[-1].split(".")[0].split("-")[-1]
                if os.path.exists(f"{args.predict_dir}/pcd/{model_id}-{prediction_id}.npz"):
                    n_preds += 1
                    pcd_data = np.load(f"{args.predict_dir}/pcd/{model_id}-{prediction_id}.npz")
                    points = np.asarray(pcd_data["points"])

                    with open(prediction_path) as f:
                        prediction = json.load(f)

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)

                    aabb = pcd.get_axis_aligned_bounding_box()
                    min_bound = aabb.get_min_bound()
                    max_bound = aabb.get_max_bound()
                    center = (max_bound + min_bound) / 2
                    dim = max_bound - min_bound
                    diag = np.linalg.norm(dim)
                    prediction_cat = LABEL_ID_LABEL_MAP[int(pcd_data["semantic"])]

                    # load pred or gt
                    if os.path.exists(f"{args.predict_dir}/pred"):
                        pred = np.load(f"{args.predict_dir}/pred/{model_id}.npz", allow_pickle=True)
                        instance_map = pred["instance"]
                        triangles = gltf.vertices[gltf.faces]
                        mask = (instance_map == int(prediction_id))
                        area = np.sum(face_areas[mask])
                    else:
                        gt = np.load(f"{args.predict_dir}/gt/{model_id}.npz", allow_pickle=True)
                        instance_map = gt["instance"]
                        triangles = gltf.vertices[gltf.faces]
                        mask = (instance_map == int(prediction_id))
                        area = np.sum(face_areas[mask])

                    pred_boxes.append((area, mask, prediction_cat, prediction_id, diag))
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

                        world_maxis = world_maxis / np.linalg.norm(world_maxis)
                        pred_results.append({"id": prediction_id, "parent": "base",
                                             "cat": prediction_cat,
                                             "motionType": MOTION_CODES[int(prediction["mtype"])],
                                             "motionAxis": world_maxis,
                                             "motionOrigin": world_morigin,
                                             "gtextrinsic": prediction["gtextrinsic"]})
                    else:
                        pred_results.append({"id": prediction_id, "parent": "base",
                                             "cat": prediction_cat,
                                             "motionType": MOTION_CODES[int(prediction["mtype"])],
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
                diag = np.linalg.norm(bbox_max - bbox_min)
                mask = (gltf.segmentation_map == current_pid)
                area = np.sum(face_areas[mask])
                gt_boxes.append((area, mask, part.label, current_pid, diag))
                gt_categories.append(part.label)
                n_gt += 1
                model_gt_n[model_id] = model_gt_n.get(model_id, 0) + 1
            matched_gt_indices, matching_pred_indices, distance_matrix = greedy_matching(gt_boxes, pred_boxes, IOU_THRESHOLD, face_areas)
            if isinstance(matched_gt_indices, np.ndarray):
                for idx, matched_gt_index in enumerate(matched_gt_indices):
                    matched_pred_index = matching_pred_indices[idx]
                    if matched_pred_index != -1 and matched_gt_index != -1:
                        iou = distance_matrix[matched_gt_index][matched_pred_index]
                        if gt_boxes[matched_gt_index][3] not in gltf.articulation_parts.keys():
                            print(f"SKIP")
                            continue
                        gt_part = gltf.articulation_parts[gt_boxes[matched_gt_index][3]]
                        pred_result = pred_results[matched_pred_index]
                        results[model_id]["pred"][pred_result["id"]] = pred_result
                        results[model_id]["gt"][pred_result["id"]] = {"id": pred_result["id"], "parent": "base",
                                                                      "cat": pred_result["cat"],
                                                                      "motionType": gt_part.type if gt_part.type in MOTION_CODES.values() else MOTION_LOOKUP[gt_part.type],
                                                                      "motionAxis": gt_part.axis,
                                                                      "motionOrigin": gt_part.origin,
                                                                      "diagonal": gt_boxes[matched_gt_index][4],
                                                                      "gt_id": gt_part.pid}
    end = time.time()
    print(f"Time for inference: {end - start}")

    start = time.time()
    existDir(args.output_dir)
    existDir(f"{args.output_dir}/motion_evaluation")
    evaluator = Evaluator(
        results,
        save_path=f"{args.output_dir}/motion_evaluation/{args.split}_{args.motion_type}_evaluation.json",
        save=True
    )

    evaluator.evaluate()
    performance = evaluator.summarize()

    with open(f"{args.output_dir}/motion_evaluation/{args.split}_{args.motion_type}_evaluation.json", "r") as f:
        evaluation = json.load(f)

    total_n_preds = n_preds
    total_matches = sum([value for value in performance["part_num"].values()])
    performance["n_preds"] = total_n_preds
    performance["n_matches"] = total_matches
    performance["n_gt"] = n_gt

    metrics = ["M", "MA", "MAO"]

    for metric in metrics:
        p = performance[f"micro_{metric}"]
        precision = p * total_matches / total_n_preds
        recall = p * total_matches / n_gt
        performance[f"micro_{metric}"] = precision
        performance[f"micro_{metric}_recall"] = recall

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        performance[f"micro_{metric}_f1"] = f1

        # Add macro precision, recall, and F1 calculations
        macro_precisions = []
        macro_recalls = []

        for model_id in results.keys():
            if model_id not in evaluation["model_performance"]:
                continue

            # Count predictions and ground truths for this model
            model_preds = sum([1 for path in glob(f"{args.predict_dir}/{args.motion_type}/{model_id}-*.json") 
                               if os.path.exists(f"{args.predict_dir}/pcd/{model_id}-{path.split('-')[-1].split('.')[0]}.npz")])

            model_matches = len(results[model_id]["gt"])
            model_gt = model_gt_n[model_id]

            # Calculate model precision and recall
            model_acc = evaluation["model_performance"][model_id][f"eval_{metric}"]
            model_precision = model_acc * model_matches / model_preds if model_preds > 0 else 0
            model_recall = model_acc * model_matches / model_gt if model_gt > 0 else 0

            macro_precisions.append(model_precision)
            macro_recalls.append(model_recall)

        # Average precision and recall across models
        macro_precision = np.sum(macro_precisions) / len(model_ids) if len(model_ids) > 0 else 0
        macro_recall = np.sum(macro_recalls) / len(model_ids) if len(model_ids) > 0 else 0

        # Calculate macro F1
        if macro_precision + macro_recall > 0:
            macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall)
        else:
            macro_f1 = 0

        performance[f"macro_{metric}_precision"] = macro_precision
        performance[f"macro_{metric}_recall"] = macro_recall
        performance[f"macro_{metric}_f1"] = macro_f1

    with open(f"{args.output_dir}/motion_evaluation/{args.split}_{args.motion_type}_performance.json", "w") as f:
        json.dump(performance, f, indent=4)

    end = time.time()
    print(f"Time for evaluation: {end - start}")
