import argparse
import json
import os

import h5py
import numpy as np
import open3d as o3d
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--data_json", type=str, required=True)
    args = parser.parse_args()

    original_data_path = args.data_path
    output_path = f"{original_data_path}-subset"
    os.mkdir(output_path)

    with open(args.data_json) as f:
        splits_data = json.load(f)

    downsample_data = h5py.File(f"{original_data_path}/downsample.h5", "a")

    splits = ["val"]

    points = downsample_data["points"]
    instance_ids = downsample_data["instance_ids"]
    colors = downsample_data["colors"]
    normals = downsample_data["normals"]
    downsample_model_ids = downsample_data["model_ids"]
    semantic_ids = downsample_data["semantic_ids"]
    barycentric_coordinates = downsample_data["barycentric_coordinates"]
    face_indexes = downsample_data["face_indexes"]
    n_points = None
    if "n_points" in downsample_data.keys():
        n_points = downsample_data["n_points"]

    # Get the list of model IDs in the original order
    model_ids = [mid.decode("utf-8") for mid in downsample_model_ids]

    # Create a set of model IDs to process (from the splits)
    model_ids_to_process = set()
    for split in splits:
        model_ids_to_process.update(splits_data[split].keys())

    # Filter model_ids to only include those in the splits
    model_ids = [mid for mid in model_ids if mid in model_ids_to_process]

    num_models = len(model_ids)

    h5file = h5py.File(f"{output_path}/downsample.h5", "w")
    # Create the points dataset
    dset_points = h5file.create_dataset(
        "points",
        shape=(num_models, 20000, 3),
        dtype=np.dtype('float64'),
        compression="gzip",
        compression_opts=9,
        shuffle=True,
    )
    dset_n_points = h5file.create_dataset(
        "n_points",
        shape=(num_models,),
        dtype=int,
        compression="gzip",
        compression_opts=9,
        shuffle=True,
    )
    dset_colors = h5file.create_dataset(
        "colors",
        shape=(num_models, 20000, 3),
        dtype=np.dtype('float64'),
        compression="gzip",
        compression_opts=9,
        shuffle=True,
    )
    dset_normals = h5file.create_dataset(
        "normals",
        shape=(num_models, 20000, 3),
        dtype=np.dtype('float64'),
        compression="gzip",
        compression_opts=9,
        shuffle=True,
    )
    dset_instance_ids = h5file.create_dataset(
        "instance_ids",
        shape=(num_models, 20000),
        dtype=np.dtype('int64'),
        compression="gzip",
        compression_opts=9,
        shuffle=True,
    )

    dset_face_indexes = h5file.create_dataset(
        "face_indexes", 
        shape=(num_models, 20000),
        dtype=np.dtype('int64'),
        compression="gzip",
        compression_opts=9,
        shuffle=True,
    )

    dset_barycentric_coordinates = h5file.create_dataset(
        "barycentric_coordinates", 
        shape=(num_models, 20000, 3),
        dtype=np.dtype('float64'),
        compression="gzip",
        compression_opts=9,
        shuffle=True,
    )

    dset_model_ids = h5file.create_dataset(
        "model_ids", shape=(num_models,), dtype=h5py.string_dtype(encoding="utf-8")
    )

    dset_semantic_ids = h5file.create_dataset(
        "semantic_ids",
        shape=(num_models, 20000),
        dtype=np.dtype('int64'),
        compression="gzip",
        compression_opts=9,
        shuffle=True,
    )
    dset_fps_mask = h5file.create_dataset(
        "fps_mask",
        shape=(num_models),
        dtype=h5py.vlen_dtype(np.dtype('int64')),
        compression="gzip",
        compression_opts=9,
        shuffle=True,
    )

    for i, model_id in tqdm(enumerate(model_ids)):
        if n_points is not None:
            cur_n_points = n_points[i]
        else:
            cur_n_points = points[i].shape[0]
        cur_points = np.asarray(points[i], dtype=np.float32).reshape([cur_n_points, 3])

        cur_pcd = o3d.geometry.PointCloud()
        cur_pcd.points = o3d.utility.Vector3dVector(cur_points)
        coord_index_map = {}
        for index in range(len(cur_points)):
            coord_index_map[tuple(cur_points[index])] = index

        fps_pcd = cur_pcd.farthest_point_down_sample(20000)
        fps_points = fps_pcd.points

        fps_indexes = np.array([coord_index_map[tuple(fps_points[index])] for index in range(len(fps_points))])

        fps_mask = np.zeros(np.shape(cur_points)[0], dtype=int)
        fps_mask[fps_indexes] = 1

        dset_instance_ids[i] = np.asarray(instance_ids[i], dtype=np.int16)[fps_mask == 1]
        dset_colors[i] = np.asarray(colors[i]).reshape([cur_n_points, 3])[fps_mask == 1, :]
        dset_normals[i] = np.asarray(normals[i], dtype=np.float32).reshape([cur_n_points, 3])[fps_mask == 1, :]
        dset_semantic_ids[i] = np.asarray(semantic_ids[i], dtype=np.int16)[fps_mask == 1]
        dset_barycentric_coordinates[i] = np.asarray(barycentric_coordinates[i], dtype=np.float32).reshape([cur_n_points, 3])[fps_mask == 1, :]
        dset_face_indexes[i] = np.asarray(face_indexes[i], dtype=np.int16)[fps_mask == 1]
        dset_fps_mask[i] = fps_mask
        dset_n_points[i] = 20000
        dset_points[i] = np.asarray(fps_points)
        dset_model_ids[i] = model_id

    h5file.close()
    downsample_data.close()
