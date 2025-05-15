import argparse
import json
import multiprocessing
import os

import h5py
import numpy as np
import torch

MOTION_CODES = {"revolute": 0, "prismatic": 1, "fixed": 2, "rotation": 0, "translation": 1}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--data_json", type=str, required=True)
    parser.add_argument("--glbs_path", type=str, required=True)
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--from_yup", action="store_true")
    args = parser.parse_args()
    with open(args.data_json) as f:
        splits_data = json.load(f)

    splits = ["val"]

    downsample_data = h5py.File(f"{args.data_path}", "a")
    data_name = args.data_name

    if "n_points" in downsample_data.keys():
        n_points = downsample_data["n_points"]
    else:
        n_points = None
    points = downsample_data["points"]
    instance_ids = downsample_data["instance_ids"]
    colors = downsample_data["colors"]
    normals = downsample_data["normals"]
    downsample_model_ids = downsample_data["model_ids"]
    semantic_ids = downsample_data["semantic_ids"]
    barycentric_coordinates = downsample_data["barycentric_coordinates"]

    # Get the map between the model id and the index
    num_models = downsample_model_ids.shape[0]
    model_idx_map = {}
    for i in range(num_models):
        model_idx_map[downsample_model_ids[i].decode("utf-8")] = i

    os.makedirs(f"../../internal_pg/data/{data_name}/metadata")

    for split in splits:

        with open(f"../../internal_pg/data/{data_name}/metadata/{split}.txt", "w+") as f:
            for model_id in splits_data[split].keys():
                f.write(f"{str(model_id)}\n")
            f.truncate(f.tell() - len(os.linesep))

        os.mkdir(f"../../internal_pg/data/{data_name}/{split}")

        for model_id in splits_data[split].keys():
            if n_points != None:
                cur_n_points = n_points[model_idx_map[model_id]]
            else:
                cur_n_points = points[model_idx_map[model_id]].shape[0]
            cur_points = np.asarray(points[model_idx_map[model_id]], dtype=np.float32).reshape([cur_n_points, 3])
            cur_instance_ids = np.asarray(instance_ids[model_idx_map[model_id]], dtype=np.int16)
            if np.asarray(colors[model_idx_map[model_id]]).shape != (cur_n_points, 3):
                # Add fake channels
                cur_colors = np.ones((cur_n_points, 3), dtype=np.uint8)
            else:
                cur_colors = np.asarray(colors[model_idx_map[model_id]] * 255, dtype=np.uint8).reshape([cur_n_points, 3])
            cur_normals = np.asarray(normals[model_idx_map[model_id]], dtype=np.float32).reshape([cur_n_points, 3])
            cur_barycentric = np.asarray(barycentric_coordinates[model_idx_map[model_id]], dtype=np.float32).reshape([cur_n_points, 3])
            cur_sem_labels = np.asarray(semantic_ids[model_idx_map[model_id]], dtype=np.int16)

            # Load and add mobility data
            art_anno_path = f"{args.glbs_path}/{model_id}/{model_id}.art-stk.json"
            with open(art_anno_path) as f:
                art_anno = json.load(f)

            art_anno_lookup = {}
            for art_info in art_anno["articulations"]:
                if not args.from_yup:
                    art_anno_lookup[art_info["pid"]] = art_info
                else:
                    # Modify from y-up z-front to z-up -x-front
                    art_info["axis"] = [-art_info["axis"][2], art_info["axis"][0], art_info["axis"][1]]
                    art_info["origin"] = [-art_info["origin"][2], art_info["origin"][0], art_info["origin"][1]]
                    art_anno_lookup[art_info["pid"]] = art_info

            cur_instance_motion_types = np.zeros((cur_n_points, 1), dtype=np.int16)
            cur_instance_axis_offsets = np.zeros((cur_n_points, 3), dtype=np.float32)
            cur_instance_axis_directions = np.zeros((cur_n_points, 3), dtype=np.float32)
            cur_instance_origin_offsets = np.zeros((cur_n_points, 3), dtype=np.float32)

            instance_axis_map = {}
            instance_origin_map = {}

            for inst_id in np.unique(cur_instance_ids):
                if inst_id == -1:
                    continue
                if int(model_id) == 47443:
                    continue
                inst_id = int(inst_id)

                if inst_id not in art_anno_lookup:
                    # Base part
                    # Simulate art_info
                    art_info = {
                        "type": "fixed",
                        "axis": [0, 0, 1],
                        "origin": [0, 0, 0]
                    }
                else:
                    art_info = art_anno_lookup[inst_id]

                inst_mask = cur_instance_ids == inst_id
                cur_instance_motion_types[inst_mask] = MOTION_CODES[art_info["type"]]
                cur_instance_axis_directions[inst_mask] = art_info["axis"]

                pts = cur_points[inst_mask]  # shape: (N_points, 3)

                # Compute GT offsets
                # Offsets are computed as projection vectors from the points to the axis
                part_sem_label = cur_sem_labels[inst_mask][0]
                if MOTION_CODES[art_info["type"]] != 1:
                    origin = np.array(art_info["origin"], dtype=np.float32)
                else:
                    # Use the center of the bbox
                    # Origin for prismatic might be not set properly
                    pts_bbox_min, pts_bbox_max = pts.min(axis=0), pts.max(axis=0)
                    origin = (pts_bbox_min + pts_bbox_max) / 2

                axis = np.array(art_info["axis"], dtype=np.float32)
                axis_norm = np.linalg.norm(axis)
                axis = axis / axis_norm

                v = pts - origin
                proj_v = np.sum(v * axis, axis=1, keepdims=True) * axis
                offset = proj_v - v
                cur_instance_axis_offsets[inst_mask] = offset

                # Compute offsets to the origin directly

                cur_instance_origin_offsets[inst_mask] = origin - pts

                instance_axis_map[inst_id] = axis
                instance_origin_map[inst_id] = origin

            torch.save({
                'xyz': cur_points,
                'rgb': cur_colors,
                'normal': cur_normals,
                'barycentric_coordinates': cur_barycentric,
                'sem_labels': cur_sem_labels,
                'instance_ids': cur_instance_ids,
                "instance_motion_types": cur_instance_motion_types,
                "instance_axis_offsets": cur_instance_axis_offsets,
                "instance_axis_directions": cur_instance_axis_directions,
                "instance_origin_offsets": cur_instance_origin_offsets,
                "instance_axis_map": instance_axis_map,
                "instance_origin_map": instance_origin_map
            }, os.path.join(f"../../internal_pg/data/{data_name}", split, f"{model_id}.pth"))
