import argparse
import json
import multiprocessing
import os
import sys
import time

import h5py
import numpy as np
from tqdm import tqdm

sys.path.append("../../../../proj-opmotion")
import pygltftoolkit as pygltk


def save_h5(output_path, num_models, results):
    h5file = h5py.File(f"{output_path}/downsample.h5", "w")
    # Create the points dataset
    dset_points = h5file.create_dataset(
        "points",
        shape=(num_models, ),
        dtype=h5py.vlen_dtype(np.dtype('float64')),
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
        shape=(num_models,),
        dtype=h5py.vlen_dtype(np.dtype('float64')),
        compression="gzip",
        compression_opts=9,
        shuffle=True,
    )
    dset_normals = h5file.create_dataset(
        "normals",
        shape=(num_models,),
        dtype=h5py.vlen_dtype(np.dtype('float64')),
        compression="gzip",
        compression_opts=9,
        shuffle=True,
    )
    # Create the instance ids dataset
    dset_instance_ids = h5file.create_dataset(
        "instance_ids",
        shape=(num_models,),
        dtype=h5py.vlen_dtype(np.dtype('int64')),
        compression="gzip",
        compression_opts=9,
        shuffle=True,
    )
    
    dset_face_indexes = h5file.create_dataset(
        "face_indexes", 
        shape=(num_models,),
        dtype=h5py.vlen_dtype(np.dtype('int64')),
        compression="gzip",
        compression_opts=9,
        shuffle=True,
    )

    dset_face_indexes_local = h5file.create_dataset(
        "face_indexes_local", 
        shape=(num_models,),
        dtype=h5py.vlen_dtype(np.dtype('int64')),
        compression="gzip",
        compression_opts=9,
        shuffle=True,
    )

    dset_node_indices = h5file.create_dataset(
        "node_indices", 
        shape=(num_models,),
        dtype=h5py.vlen_dtype(np.dtype('int64')),
        compression="gzip",
        compression_opts=9,
        shuffle=True,
    )

    dset_mesh_indices = h5file.create_dataset(
        "mesh_indices", 
        shape=(num_models,),
        dtype=h5py.vlen_dtype(np.dtype('int64')),
        compression="gzip",
        compression_opts=9,
        shuffle=True,
    )

    dset_primitive_indices = h5file.create_dataset(
        "primitive_indices", 
        shape=(num_models,),
        dtype=h5py.vlen_dtype(np.dtype('int64')),
        compression="gzip",
        compression_opts=9,
        shuffle=True,
    )

    dset_seg_indices = h5file.create_dataset(
        "seg_indices", 
        shape=(num_models,),
        dtype=h5py.vlen_dtype(np.dtype('int64')),
        compression="gzip",
        compression_opts=9,
        shuffle=True,
    )

    dset_barycentric_coordinates = h5file.create_dataset(
        "barycentric_coordinates", 
        shape=(num_models,),
        dtype=h5py.vlen_dtype(np.dtype('float64')),
        compression="gzip",
        compression_opts=9,
        shuffle=True,
    )

    dset_semantic_ids = h5file.create_dataset(
        "semantic_ids",
        shape=(num_models,),
        dtype=h5py.vlen_dtype(np.dtype('int64')),
        compression="gzip",
        compression_opts=9,
        shuffle=True,
    )
    # create model ids dataset
    dset_model_ids = h5file.create_dataset(
        "model_ids", shape=(num_models,), dtype=h5py.string_dtype(encoding="utf-8")
    )

    for i, (model_id, result) in enumerate(results.items()):
        dset_points[i] = result["points"].flatten()
        dset_n_points[i] = result["points"].shape[0]
        dset_colors[i] = result["colors"].flatten()
        dset_normals[i] = result["normals"].flatten()
        dset_instance_ids[i] = result["instance_ids"]
        dset_face_indexes[i] = result["global_tri_indices"]
        dset_face_indexes_local[i] = result["local_tri_indices"]
        dset_node_indices[i] = result["node_indices"]
        dset_mesh_indices[i] = result["mesh_indices"]
        dset_primitive_indices[i] = result["primitive_indices"]
        dset_seg_indices[i] = result["seg_indices"]
        dset_barycentric_coordinates[i] = result["barycentric_coords"].flatten()
        dset_semantic_ids[i] = result["semantic_ids"]
        dset_model_ids[i] = model_id

    h5file.close()


def sample_model(model_id, input_path, precomputed_path, vertices):
    glb_path = f"{input_path}/{model_id}/{model_id}.glb"
    gltf = pygltk.load(glb_path)
    gltf.load_stk_segmentation_openable(f"{input_path}/{model_id}/{model_id}.art-stk.json", )
    gltf.load_stk_articulation(f"{input_path}/{model_id}/{model_id}.art-stk.json")
    if precomputed_path is not None:
        gltf.load_stk_precomputed_segmentation_flattened(f"{precomputed_path}/{model_id}.connectivity.segs.json")

    return gltf.sample_uniform(
        20000,
        oversample=1000000,
        recenter=False,
        rescale=False,
        fpd=True,
        vertices=vertices,
        semantic_map={"drawer": 0, "door": 1, "lid": 2, "base": 3},
        allow_nonuniform=True,
        voxel=False
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--precomputed_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--data_json", type=str, required=True)
    parser.add_argument("--vertices", action="store_true")

    args = parser.parse_args()
    with open(args.data_json, "r") as f:
        data_json = json.load(f)

    os.makedirs(args.output_path, exist_ok=True)

    raw_results = {}

    print("Loading and starting sampling")

    splits = ["train", "val"]
    for split in splits:
        model_ids = list(data_json[split].keys())
        for model_id in tqdm(model_ids):
            print(model_id)

            start = time.time()
            raw_results[model_id] = sample_model(model_id, args.input_path, args.precomputed_path, args.vertices)
            print(str(time.time() - start) + "\n")

    print("Getting sampling results")

    results = raw_results
    save_h5(args.output_path, len(raw_results), results)
    save_h5(args.output_path, len(raw_results), results)
