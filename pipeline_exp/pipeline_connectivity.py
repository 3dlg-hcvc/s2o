import argparse
import os
import sys

sys.path.append("../internal_pg")
sys.path.append("../")
import copy
import json
from glob import glob

import numpy as np
import open3d as o3d
import pygltftoolkit as pygltk
import seaborn as sns
import torch
import trimesh
from Helper3D import (
    SampleSurfaceFromTrimeshScene,
    SampleSurfaceFromTrimeshSceneParallel,
)
from hydra import compose, initialize
from omegaconf import OmegaConf
from scripts.postprocess.map_pc_to_mesh import (
    sample_and_export_points,
    save_nonindexed_geometry,
)
from scripts.postprocess.map_predictions_from_subset_points import (
    compute_iou,
    knn_assign_instances,
)
from tqdm import tqdm

from internal_pg.minsu3d.evaluation.instance_segmentation import rle_decode
from internal_pg.minsu3d.model import PointGroup
from Pointcept.libs.pointops.functions.sampling import farthest_point_sampling

S2O_COLOR_MAP = {
    0: (0.0, 107.0 / 255.0, 164.0 / 255.0),
    1: (255.0 / 255.0, 128.0 / 255.0, 14.0 / 255.0),
    2: (44.0 / 255.0, 160.0 / 255.0, 44.0 / 255.0),
    3: (171.0 / 255.0, 171.0 / 255.0, 171.0 / 255.0),
}

S2O_COLOR_MAP_HEX = {
    0: "#006BA4",
    1: "#FF800E",
    2: "#2CA02C",
    3: "#ABABAB",
}

S2O_SEM_REF = {"drawer": 0,
               "door": 1,
               "lid": 2,
               "base": 3}

IOU_THRESHOLD = 0.8


def debug_vis(points, masks):
    import seaborn as sns
    semantic_colors = np.zeros((points.shape[0], 3))
    palette = sns.color_palette("tab10", len(masks))
    for i, mask in enumerate(masks):
        mask = mask.astype(bool)
        semantic_colors[mask] = np.array(palette[i])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(semantic_colors)
    o3d.visualization.draw_geometries([pcd])


def farthest_point_downsampling_idx_pointops(points, n_samples):
    points_tensor = torch.from_numpy(points).float().cuda()

    N = points.shape[0]
    offset = torch.tensor([N], device='cuda')

    new_offset = torch.tensor([n_samples], device='cuda')

    sampled_indices = farthest_point_sampling(points_tensor, offset, new_offset)

    sampled_indices_np = sampled_indices.cpu().numpy()

    return sampled_indices_np


def map_single_mesh_connectivity(mesh, masks_dict, face_indices, base_pred_idx, output_dir, model_id, all_masks, precomputed_segmentation_map):
    precomputed_segment_faces_map = {}
    for precomputed_idx in np.unique(precomputed_segmentation_map):
        precomputed_segment_faces_map[precomputed_idx] = precomputed_segmentation_map == precomputed_idx

    mapped_faces = -np.ones(len(mesh.faces), dtype=np.int32)
    mapped_pred_conf = {}
    for precomputed_idx, face_mask in precomputed_segment_faces_map.items():
        aggregated_face_mask = np.zeros_like(all_masks, dtype=np.bool_)
        for face_idx in np.where(face_mask)[0]:
            aggregated_face_mask = np.logical_or(aggregated_face_mask, face_indices == face_idx)
        face_preds = all_masks[aggregated_face_mask]
        if len(face_preds) == 0:
            continue
        preds = np.concatenate([face_preds])
        most_common_count = np.bincount(preds).max()
        most_common_preds = np.where(np.bincount(preds) == most_common_count)[0]
        if len(most_common_preds) == 1:
            most_common_pred = most_common_preds[0]
        else:
            most_common_pred = most_common_preds[0]
            for pred in most_common_preds:
                if masks_dict[pred]['conf'] > masks_dict[most_common_pred]['conf']:
                    most_common_pred = pred
        mapped_faces[face_mask] = most_common_pred
        mapped_pred_conf[most_common_pred] = masks_dict[most_common_pred]['conf']
    mapped_faces[mapped_faces == -1] = base_pred_idx

    exisitng_conf = {}
    face_colors = []
    # Use PARTNETSIM_COLOR_MAP as base color and generate subcolors based on number of instances of each semantic class
    palette = [None, None, None, None]
    sem_inst_counters = [0, 0, 0]
    for inst in np.unique(mapped_faces):
        exisitng_conf[inst] = mapped_pred_conf[inst]
        sem_label = masks_dict[inst]['semantic']
        if sem_label != 3:
            sem_inst_counters[sem_label] += 1
    for idx, sem_inst_count in enumerate(sem_inst_counters):
        if sem_inst_count > 0:
            palette[idx] = sns.light_palette(S2O_COLOR_MAP_HEX[idx], n_colors=sem_inst_count + 1)[1:]
    palette[3] = sns.light_palette(S2O_COLOR_MAP_HEX[3], n_colors=2)[1:]
    instance_color_map = {}
    for instance in np.unique(mapped_faces):
        sem_label = masks_dict[instance]['semantic']
        if sem_label == 3:
            instance_color_map[instance] = palette[3][0]
        else:
            sem_inst_counters[sem_label] -= 1
            instance_color_map[instance] = palette[sem_label][sem_inst_counters[sem_label]]
    semantic_map = []
    sem_inst_counters = [0, 0, 0]
    for face_idx in range(len(mesh.faces)):
        semantic_label = masks_dict[mapped_faces[face_idx]]['semantic']
        semantic_map.append(semantic_label)
        face_colors.append(instance_color_map[mapped_faces[face_idx]])

    mesh.visual.face_colors = np.array(face_colors)
    os.makedirs(f"{output_dir}/obj", exist_ok=True)
    nonindexed_mesh = save_nonindexed_geometry(mesh, f"{output_dir}/obj/{model_id}.obj")

    os.makedirs(f"{output_dir}/pred", exist_ok=True)
    np.savez(f"{output_dir}/pred/{model_id}", semantic=semantic_map, instance=mapped_faces, confs=exisitng_conf)

    # Export parts
    for part_idx in np.unique(mapped_faces):
        part_mesh = copy.deepcopy(nonindexed_mesh)
        part_mesh.update_faces(mapped_faces == part_idx)
        part_mesh.remove_unreferenced_vertices()
        part_mesh.visual = nonindexed_mesh.visual.face_subset(mapped_faces == part_idx)
        os.makedirs(f"{output_dir}/parts/{model_id}", exist_ok=True)
        _ = save_nonindexed_geometry(part_mesh, f"{output_dir}/parts/{model_id}/{part_idx}.obj")

    args = args = argparse.Namespace(output_dir=output_dir, id=model_id)
    # Sample points from parts
    sample_and_export_points(nonindexed_mesh, mapped_faces, masks_dict, args)


def postporcess_masks(masks, masks_dict, original_points, subset_points, fps_mask):
    instance_mask = -np.ones(20000, dtype=int)
    confidence_map = {}
    semantic_map = {}
    for instance_id, (mask_info, prediction_mask) in enumerate(zip(masks_dict.values(), masks)):
        confidence_map[str(instance_id)] = float(mask_info['conf'])
        semantic_map[str(instance_id)] = mask_info['semantic']
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
    non_valid_mask = np.asarray(instance_mask == -1, dtype=int)

    valid_instance_mask = instance_mask[valid_idx]
    fps_idx = np.where(fps_mask == 1)[0]
    combined_mask = np.zeros(fps_mask.shape)
    combined_mask[fps_idx] = non_valid_mask
    fps_mask[combined_mask == 1] = 0

    full_instance_labels = knn_assign_instances(original_points, subset_points, valid_instance_mask, fps_mask)
    return full_instance_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_path", type=str, default="pipeline_output")
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--cfg_path", type=str, default="../internal_pg/config")

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    print("Begin processing", args.input_dir, "...")

    print("Loading the model...")
    # Initialize the model
    with initialize(version_base=None, config_path=args.cfg_path):
        cfg = compose(config_name="config", overrides=["model=pointgroup", 'data=partnetsim', "model.network.prepare_epochs=-1", "model.inference.evaluate=False"])

    # Seed everything
    torch.manual_seed(42)
    np.random.seed(42)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    input_mesh_paths = glob(f"{args.input_dir}/*.obj") + glob(f"{args.input_dir}/*.ply") + glob(f"{args.input_dir}/*.glb") + glob(f"{args.input_dir}/*.gltf")
    for input_mesh_path in tqdm(input_mesh_paths):
        try:
            args.input = input_mesh_path
            model_id = os.path.basename(args.input).split(".")[0]
            if os.path.exists(f"{args.output_path}/pred/{model_id}.npz"):
                continue
            args.connectivity_segmentation = f"{args.input_dir}/connectivity_segmentation/{model_id}.connectivity.segs.json"
            if not (args.input.endswith(".obj") or args.input.endswith(".ply") or args.input.endswith(".glb") or args.input.endswith(".gltf")):
                print("Input file must be .obj, .ply, .glb or .gltf")
                sys.exit(1)

            print("Loading the mesh...")
            is_trimesh = True
            if args.connectivity_segmentation:
                source = trimesh.load(args.input, force="mesh", process=False)
                with open(args.connectivity_segmentation, "r") as f:
                    connectivity_segmentation = json.load(f)
                mesh_dict = {}
                abstraction_source_map_face_idx = np.zeros(len(source.faces), dtype=bool)
                precomputed_segmentation_map = np.zeros(len(source.faces), dtype=int)
                for tri_segment in connectivity_segmentation["segmentation"]:
                    triIndex_map = np.zeros(len(source.faces), dtype=bool)
                    for triIndex in tri_segment["triIndex"]:
                        if type(triIndex) is int:
                            triIndex_map[triIndex] = 1
                        elif type(triIndex) is list:
                            triIndex_map[triIndex[0]:triIndex[1]] = 1
                    faces = source.faces[triIndex_map]
                    precomputed_segmentation_map[triIndex_map] = tri_segment["segIndex"]
                    vertices = source.vertices
                    normals = source.face_normals
                    reindexed_faces = np.zeros_like(faces)
                    reindexed_vertices = np.zeros((len(np.unique(faces)), 3))
                    for i, vertex in enumerate(np.unique(faces)):
                        reindexed_faces[faces == vertex] = i
                        reindexed_vertices[i] = vertices[vertex]
                    mesh_dict[str(tri_segment["segIndex"])] = trimesh.Trimesh(vertices=reindexed_vertices, faces=reindexed_faces, face_normals=normals[triIndex_map], process=False)
                abstraction_scene = trimesh.Scene(mesh_dict)
                abstraction = abstraction_scene.dump(concatenate=True)
            else:
                abstraction_scene = trimesh.load(args.input, process=False)
                abstraction = abstraction_scene.dump(concatenate=True)
                vertices = abstraction.vertices
                normals = abstraction.vertex_normals
            if normals.size == 0:
                print("No normals found in the mesh.")
                sys.exit(1)
            model_name = os.path.basename(args.input).split(".")[0]

            # Sample 200k points

            if os.path.exists(f"{args.input_dir}/precomputed_inference_connectivity/{model_id}.npz"):
                print("Loading precomputed connectivity inference...")
                try:
                    npz = np.load(f"{args.input_dir}/precomputed_inference_connectivity/{model_id}.npz", allow_pickle=True)
                    sampled_points = npz["sampled_points"]
                    sampled_normals = npz["sampled_normals"]
                    points_faces = npz["points_faces"]
                    true_face_indices = npz["true_face_indices"]
                    fps_mask = npz["fps_mask"]
                    downsampled_points = sampled_points[fps_mask == 1]
                    downsampled_normals = sampled_normals[fps_mask == 1]
                except:
                    print("Sampling the points...")
                    sampled_points, _, sampled_normals, points_faces, _, geometry_map = SampleSurfaceFromTrimeshSceneParallel(abstraction_scene, num_points=200000)
                    true_face_indices = -np.ones(len(sampled_points))
                    if connectivity_segmentation:
                        for geometry_key in np.unique(geometry_map):
                            # print(f"Processing geometry key: {geometry_key}")
                            geometry_face_idx_source_mesh_idx_correspondence = np.zeros(len(abstraction_scene.geometry[geometry_key].faces))
                            first_triIndex = connectivity_segmentation["segmentation"][int(geometry_key)]["triIndex"][0]
                            start_index = 0
                            if type(first_triIndex) is int:
                                start_index = first_triIndex
                            elif type(first_triIndex) is list:
                                start_index = first_triIndex[0]
                            prev_index = start_index
                            accumulated_diff = 0
                            for triIndex in connectivity_segmentation["segmentation"][int(geometry_key)]["triIndex"]:
                                if type(triIndex) is int:
                                    accumulated_diff += triIndex - prev_index
                                    geometry_face_idx_source_mesh_idx_correspondence[triIndex - start_index - accumulated_diff] = triIndex
                                    prev_index = triIndex
                                elif type(triIndex) is list:
                                    accumulated_diff += triIndex[0] - prev_index
                                    geometry_face_idx_source_mesh_idx_correspondence[triIndex[0] - start_index - accumulated_diff:triIndex[1] - start_index - accumulated_diff] = np.arange(triIndex[0], triIndex[1])
                                    prev_index = triIndex[1]
                            true_face_indices[geometry_map == geometry_key] = geometry_face_idx_source_mesh_idx_correspondence[points_faces[geometry_map == geometry_key]]
                    else:
                        true_face_indices = np.arange(len(source.faces))

                    print("Downsampling the points...")

                    sampled_pcd = o3d.geometry.PointCloud()
                    sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)
                    sampled_pcd.normals = o3d.utility.Vector3dVector(sampled_normals)

                    coord_index_map = {}
                    for index in range(len(sampled_points)):
                        coord_index_map[tuple(sampled_points[index])] = index

                    pcd = sampled_pcd.farthest_point_down_sample(20000)
                    downsampled_points = np.asarray(pcd.points)
                    downsampled_normals = np.asarray(pcd.normals)

                    downsample_idx = np.array(
                        [coord_index_map[tuple(downsampled_points[index])] for index in range(len(downsampled_points))]
                    ).astype(dtype=int).tolist()

                    fps_mask = np.zeros(np.shape(sampled_points)[0], dtype=int)
                    fps_mask[downsample_idx] = 1

                    # Process
                    downsampled_pcd = o3d.geometry.PointCloud()
                    downsampled_pcd.points = o3d.utility.Vector3dVector(downsampled_points)
                    downsampled_pcd.normals = o3d.utility.Vector3dVector(downsampled_normals)
                    downsampled_pcd.scale(2 / (((np.asarray(downsampled_points).max(0) - np.asarray(downsampled_points).min(0)) ** 2).sum()) ** 0.5 , center=tuple([2 * c for c in downsampled_points.mean(0)]))

                    downsampled_points = np.asarray(downsampled_pcd.points)
                    downsampled_normals = np.asarray(downsampled_pcd.normals)
                    downsampled_normals = downsampled_normals / np.linalg.norm(downsampled_normals, axis=1).reshape(-1, 1)

                    os.makedirs(f"{args.input_dir}/precomputed_inference_connectivity", exist_ok=True)
                    np.savez(f"{args.input_dir}/precomputed_inference_connectivity/{model_id}", sampled_points=sampled_points, sampled_normals=sampled_normals, points_faces=points_faces, true_face_indices=true_face_indices, fps_mask=fps_mask)
            else:
                print("Sampling the points...")
                sampled_points, _, sampled_normals, points_faces, _, geometry_map = SampleSurfaceFromTrimeshSceneParallel(abstraction_scene, num_points=200000)
                true_face_indices = -np.ones(len(sampled_points))
                if connectivity_segmentation:
                    for geometry_key in np.unique(geometry_map):
                        geometry_face_idx_source_mesh_idx_correspondence = np.zeros(len(abstraction_scene.geometry[geometry_key].faces))
                        first_triIndex = connectivity_segmentation["segmentation"][int(geometry_key)]["triIndex"][0]
                        start_index = 0
                        if type(first_triIndex) is int:
                            start_index = first_triIndex
                        elif type(first_triIndex) is list:
                            start_index = first_triIndex[0]
                        prev_index = start_index
                        accumulated_diff = 0
                        for triIndex in connectivity_segmentation["segmentation"][int(geometry_key)]["triIndex"]:
                            if type(triIndex) is int:
                                accumulated_diff += triIndex - prev_index
                                geometry_face_idx_source_mesh_idx_correspondence[triIndex - start_index - accumulated_diff] = triIndex
                                prev_index = triIndex
                            elif type(triIndex) is list:
                                accumulated_diff += triIndex[0] - prev_index
                                geometry_face_idx_source_mesh_idx_correspondence[triIndex[0] - start_index - accumulated_diff:triIndex[1] - start_index - accumulated_diff] = np.arange(triIndex[0], triIndex[1])
                                prev_index = triIndex[1]
                        true_face_indices[geometry_map == geometry_key] = geometry_face_idx_source_mesh_idx_correspondence[points_faces[geometry_map == geometry_key]]
                else:
                    true_face_indices = np.arange(len(source.faces))

                print("Downsampling the points...")

                sampled_pcd = o3d.geometry.PointCloud()
                sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)
                sampled_pcd.normals = o3d.utility.Vector3dVector(sampled_normals)

                coord_index_map = {}
                for index in range(len(sampled_points)):
                    coord_index_map[tuple(sampled_points[index])] = index

                pcd = sampled_pcd.farthest_point_down_sample(20000)
                downsampled_points = np.asarray(pcd.points)
                downsampled_normals = np.asarray(pcd.normals)

                downsample_idx = np.array(
                    [coord_index_map[tuple(downsampled_points[index])] for index in range(len(downsampled_points))]
                ).astype(dtype=int).tolist()

                fps_mask = np.zeros(np.shape(sampled_points)[0], dtype=int)
                fps_mask[downsample_idx] = 1

                # Process
                downsampled_pcd = o3d.geometry.PointCloud()
                downsampled_pcd.points = o3d.utility.Vector3dVector(downsampled_points)
                downsampled_pcd.normals = o3d.utility.Vector3dVector(downsampled_normals)
                downsampled_pcd.scale(2 / (((np.asarray(downsampled_points).max(0) - np.asarray(downsampled_points).min(0)) ** 2).sum()) ** 0.5 , center=tuple([2 * c for c in downsampled_points.mean(0)]))

                downsampled_points = np.asarray(downsampled_pcd.points)
                downsampled_normals = np.asarray(downsampled_pcd.normals)
                downsampled_normals = downsampled_normals / np.linalg.norm(downsampled_normals, axis=1).reshape(-1, 1)

                os.makedirs(f"{args.input_dir}/precomputed_inference_connectivity", exist_ok=True)
                np.savez(f"{args.input_dir}/precomputed_inference_connectivity/{model_id}", sampled_points=sampled_points, sampled_normals=sampled_normals, points_faces=points_faces, true_face_indices=true_face_indices, fps_mask=fps_mask)

            # Prepare the data dict
            data_dict = {"point_xyz": torch.from_numpy(downsampled_points).float().unsqueeze(0).to(device),
                         "point_normal": torch.from_numpy(downsampled_normals).float().unsqueeze(0).to(device),
                         "vert_batch_ids": torch.zeros(downsampled_points.shape[0], dtype=torch.uint8).to(device),
                         "scan_ids": [model_name]}

            print("Running the model...")
            # Run the model
            model = PointGroup(cfg)
            model.load_state_dict(torch.load(args.ckpt_path, map_location=device)["state_dict"])
            model.to(device)
            model.eval()
            with torch.no_grad():
                model.test_step(data_dict, 0)
                output = model.val_test_step_outputs

            semantic_labels = []
            masks = []
            confs = []
            masks_dict = {}
            base_pred_idx = -1
            for idx, pred in enumerate(output[0][2]):
                semantic_labels.append(pred["label_id"])
                masks.append(rle_decode(pred["pred_mask"]).astype(bool))
                confs.append(pred["conf"])
                masks_dict[idx] = {"conf": pred["conf"], "semantic": pred["label_id"] - 1}
                if pred["label_id"] - 1 == 3:
                    base_pred_idx = idx

            # debug_vis(downsampled_points, masks)

            print("Mapping the predictions to the full point cloud...")
            all_masks = postporcess_masks(masks, masks_dict, sampled_points, downsampled_points, fps_mask)

            print("Mapping the predictions to the mesh...")
            # map_single_mesh(mesh, masks_dict, face_indices, vertex_ids, base_pred_idx, output_dir, model_id):
            map_single_mesh_connectivity(source, masks_dict, true_face_indices, base_pred_idx, args.output_path, model_id, all_masks, precomputed_segmentation_map)
            # Clear cache
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error processing the mesh: {input_mesh_path}: {e}")
            continue

    os.system(f"python ../motion_inference.py --pred_path {os.path.abspath(args.output_path)} --output_path {os.path.abspath(args.output_path)} --export")
