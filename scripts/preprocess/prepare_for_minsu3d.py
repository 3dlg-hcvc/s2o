import argparse
import json
import multiprocessing
import os

import h5py
import numpy as np
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--data_json", type=str, required=True)
    args = parser.parse_args()
    with open(args.data_json) as f:
        splits_data = json.load(f)

    splits = ["val"]

    downsample_data = h5py.File(f"{args.data_path}", "a")
    data_name = args.data_path.split("/")[-1].split(".")[0]

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

    os.makedirs(f"../../minsu3d/data/{data_name}/metadata/{data_name}/metadata")

    for split in splits:

        with open(f"../../minsu3d/data/{data_name}/metadata/{split}.txt", "w+") as f:
            for model_id in splits_data[split].keys():
                f.write(f"{str(model_id)}\n")
            f.truncate(f.tell() - len(os.linesep))

        os.mkdir(f"../../minsu3d/data/{data_name}/{split}")

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

            torch.save({'xyz': cur_points, 'rgb': cur_colors, 'normal': cur_normals, 'barycentric_coordinates': cur_barycentric, 'sem_labels': cur_sem_labels, 'instance_ids': cur_instance_ids},
                       os.path.join(f"../../minsu3d/data/{data_name}", split, f"{model_id}.pth"))
