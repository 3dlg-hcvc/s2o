import argparse
import codecs
import json
import os

import h5py
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--data_json", type=str, required=True)
    parser.add_argument("--art_path", type=str, required=True)
    args = parser.parse_args()

    splits = ["val"]
    with open(args.data_json) as f:
        splits_data = json.load(f)

    downsample_data = h5py.File(f"{args.data_path}/downsample.h5", "a")
    data_name = args.data_path.split("/")[-1]

    shape2motion_path = f"../../shape2motion-pytorch/dataset/{data_name}"
    os.makedirs(shape2motion_path)

    points = downsample_data["points"]
    instance_ids = downsample_data["instance_ids"]
    colors = downsample_data["colors"]
    normals = downsample_data["normals"]
    downsample_model_ids = downsample_data["model_ids"]
    semantic_ids = downsample_data["semantic_ids"]
    num_models = downsample_model_ids.shape[0]
    num_points = int(points[0].shape[0])

    model_idx_map = {}
    for i in range(num_models):
        model_idx_map[downsample_model_ids[i].decode("utf-8")] = i

    for split in splits:
        num_split_models = len(splits_data[split].keys())
        h5file = h5py.File(f"{shape2motion_path}/{split}.h5","w")
        dset_points = h5file.create_dataset(
            "points",
            shape=(num_split_models, num_points, 3),
            dtype="float64",
            compression="gzip",
            compression_opts=9,
        )
        dset_colors = h5file.create_dataset(
            "colors",
            shape=(num_split_models, num_points, 3),
            dtype="float64",
            compression="gzip",
            compression_opts=9,
        )
        dset_normals = h5file.create_dataset(
            "normals",
            shape=(num_split_models, num_points, 3),
            dtype="float64",
            compression="gzip",
            compression_opts=9,
        )

        dset_instance_ids = h5file.create_dataset(
            "instance_ids",
            shape=(num_split_models, num_points),
            dtype="int64",
            compression="gzip",
            compression_opts=9,
        )

        dset_semantic_ids = h5file.create_dataset(
            "semantic_ids",
            shape=(num_split_models, num_points),
            dtype="int64",
            compression="gzip",
            compression_opts=9,
        )

        dset_model_ids = h5file.create_dataset(
            "model_ids", shape=(num_split_models,), dtype=h5py.string_dtype(encoding="utf-8")
        )

        jsonpath = f"{shape2motion_path}/{split}.json"
        jsonfile = {}
        for j, model_id in enumerate(splits_data[split].keys()):
            with open(f"{args.art_path}/{model_id}/{model_id}.art-stk.json") as f:
                try:
                    model_info = json.load(f)
                except json.JSONDecodeError:
                    try:
                        model_info = json.load(codecs.open(f"{args.art_path}/{model_id}/{model_id}.art-stk.json", 'r', 'utf-8-sig'))
                    except json.JSONDecodeError:
                        raise ValueError("Invalid JSON format.")
            jsonfile[model_id] = model_info["annotation"] if "annotation" in model_info else (model_info["data"] if "data" in model_info else model_info) 
            id = model_idx_map[model_id]
            dset_points[j] = points[id].reshape((num_points, 3))
            dset_instance_ids[j] = instance_ids[id]
            dset_colors[j] = colors[id].reshape((num_points, 3))
            dset_normals[j] = normals[id].reshape((num_points, 3))
            dset_semantic_ids[j] = semantic_ids[id]
            dset_model_ids[j] = model_id

        h5file.close()
        with open(jsonpath, "w+") as f:
            json.dump(jsonfile, f)
            json.dump(jsonfile, f)
