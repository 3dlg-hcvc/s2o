import argparse
import json
import os
import sys

import numpy as np
import sklearn
from tqdm import tqdm

sys.path.append("../../proj-opmotion")
import pygltftoolkit as pygltk


def triangle_area_by_coords(coords):
    coords_arr = np.asarray(coords, dtype=float)
    vec1 = coords_arr[1] - coords_arr[0]
    vec2 = coords_arr[2] - coords_arr[0]
    return 0.5 * np.linalg.norm(np.cross(vec1, vec2))


def classification_error(args):
    acc_error = 0.0
    n_scenes = len(args.model_ids)

    for scene_id in tqdm(args.model_ids):
        gt = np.load(f"{args.gt_path}/gt/{scene_id}.npz", allow_pickle=True)
        pred = np.load(f"{args.predict_dir}/pred/{scene_id}.npz", allow_pickle=True)

        gltf = pygltk.load(f"{args.glb_path}/{scene_id}/{scene_id}.glb")

        triangle_areas = [triangle_area_by_coords(triangle) for triangle in gltf.vertices[gltf.faces]]
        total_area = sum(triangle_areas)

        matched_area = np.sum((gt["semantic"] == pred["semantic"]).astype(int) * np.asarray(triangle_areas))

        acc_error += matched_area / total_area

    acc_error /= n_scenes
    return round(acc_error, 6)


def classification_error_nb(args):
    acc_error = 0.0
    n_scenes = len(args.model_ids)

    for scene_id in tqdm(args.model_ids):
        gt = np.load(f"{args.gt_path}/gt/{scene_id}.npz", allow_pickle=True)
        pred = np.load(f"{args.predict_dir}/pred/{scene_id}.npz", allow_pickle=True)

        gltf = pygltk.load(f"{args.glb_path}/{scene_id}/{scene_id}.glb")

        triangle_areas = np.asarray([triangle_area_by_coords(triangle) for triangle in gltf.vertices[gltf.faces]])

        gt_openable_mask = gt['semantic'] != 3
        pred_openable_mask = pred['semantic'] != 3
        union_mask = gt_openable_mask | pred_openable_mask
        correct_mask = gt['semantic'] == pred['semantic']
        correct_openable_mask = correct_mask & union_mask

        scene_error = np.sum(triangle_areas[correct_openable_mask]) / np.sum(triangle_areas[union_mask])
        print(f"{scene_id}, metric - {np.sum(triangle_areas[correct_openable_mask])}, {np.sum(triangle_areas[union_mask])}, {scene_error}")

        acc_error += scene_error

    acc_error /= n_scenes
    return round(acc_error, 6)


def segment_weighted_error(args):
    acc_error = 0.0
    acc_error_per_class = {}
    n_scenes = 0
    n_scenes_per_class = {}
    for scene_id in tqdm(args.model_ids):
        n_scenes += 1
        gt = np.load(f"{args.gt_path}/gt/{scene_id}.npz", allow_pickle=True)
        pred = np.load(f"{args.predict_dir}/pred/{scene_id}.npz", allow_pickle=True)

        scene_error = 0.0
        scene_error_per_class = {}

        gt_area_per_class = {}

        gltf = pygltk.load(f"{args.glb_path}/{scene_id}/{scene_id}.glb")

        triangle_areas = np.asarray([triangle_area_by_coords(triangle) for triangle in gltf.vertices[gltf.faces]])

        for instance_id in np.unique(gt['instance']):
            gt_segm_triangles = gt['instance'][gt['instance'] == instance_id]
            gt_sem_label = gt['semantic'][gt['instance'] == instance_id][0]

            if gt_sem_label not in gt_area_per_class.keys():
                gt_area_per_class[gt_sem_label] = 0.0

            for triangleidx in gt_segm_triangles:
                gt_area_per_class[gt_sem_label] += triangle_areas[triangleidx]

        for instance_id in np.unique(gt['instance']):
            gt_segm_triangles = gt['instance'][gt['instance'] == instance_id]
            gt_sem_label = gt['semantic'][gt['instance'] == instance_id][0]
            semantic_error = 0.0

            for idx, triangleidx in enumerate(gt_segm_triangles):
                pred_sem_label = pred['semantic'][idx]
                face_area = triangle_areas[triangleidx]
                err_val = 0.0
                if gt_sem_label == pred_sem_label:
                    err_val = 1.0
                scene_error += round(err_val * face_area / gt_area_per_class[gt_sem_label], 6)
                semantic_error += round(err_val * face_area / gt_area_per_class[gt_sem_label], 6)
            if gt_sem_label in scene_error_per_class.keys():
                scene_error_per_class[gt_sem_label] += round(semantic_error, 6)
            else:
                scene_error_per_class[gt_sem_label] = round(semantic_error, 6)
        scene_error /= np.unique(list(gt_area_per_class.keys())).shape[0]
        for key in scene_error_per_class.keys():
            if key in n_scenes_per_class.keys():
                n_scenes_per_class[key] += 1
            else:
                n_scenes_per_class[key] = 1

            if key in acc_error_per_class.keys():
                acc_error_per_class[key] += scene_error_per_class[key]
            else:
                acc_error_per_class[key] = scene_error_per_class[key]

        acc_error += scene_error
    acc_error /= n_scenes
    print("Segment-Weighted Accuracy Per Class:")

    for key in acc_error_per_class.keys():
        acc_error_per_class[key] /= n_scenes_per_class[key]
        print(f"\t{args.id_class_map[key]}: {round(acc_error_per_class[key], 6)}")

    return round(acc_error, 6), acc_error_per_class


def rand_index(args):
    n_scenes = 0
    acc_ari = 0.0
    for scene_id in tqdm(args.model_ids):
        n_scenes += 1
        gt = np.load(f"{args.gt_path}/gt/{scene_id}.npz", allow_pickle=True)
        pred = np.load(f"{args.predict_dir}/pred/{scene_id}.npz", allow_pickle=True)

        pred_labels = pred['instance']
        gt_labels = gt['instance']

        acc_ari += sklearn.metrics.adjusted_rand_score(gt_labels, pred_labels)
    acc_ari /= n_scenes
    return acc_ari


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--predict_dir', type=str,
                        required=True)
    parser.add_argument('-g', '--gt_path', type=str,
                        required=True)
    parser.add_argument('-o', '--output_dir', type=str,
                        required=True)
    parser.add_argument('--glb_path', type=str, 
                        required=True)
    parser.add_argument('--data_json', type=str,
                        required=True)

    args = parser.parse_args()
    with open(args.data_json) as f:
        data = json.load(f)
        args.model_ids = [model for model in data['val'].keys()]
    args.id_class_map = {0: "drawer", 1: "door", 2: "lid", 3: "base"}
    os.makedirs(args.output_dir, exist_ok=True)

    ce = classification_error(args)
    print(f"Average Classification accuracy: {ce}")
    ce_nb = classification_error_nb(args)
    print(f"Average Classification accuracy without base: {ce_nb}")
    swe, swe_cls = segment_weighted_error(args)
    print(f"Average Segment-weighted accuracy: {swe}")
    ri = rand_index(args)
    print(f"Average Adjusted Rand Index: {ri}")

    eval_dict = {"Classification Accuracy": {"average": float(ce)}, "Classification Accuracy (no base)": {"average": float(ce_nb)}, "Normalized Classification Accuracy": {"average": float(swe), "class": [float(swec) for swec in swe_cls]}, "ARI": ri}
    with open(f"{args.output_dir}/eval_dict.json", "w+") as f:
        json.dump(eval_dict, f)
