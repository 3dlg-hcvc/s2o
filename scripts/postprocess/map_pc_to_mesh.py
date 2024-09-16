import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import open3d as o3d
import seaborn as sns
import trimesh
from plyfile import PlyData, PlyElement
from tqdm import tqdm

sys.path.append("../../../../proj-opmotion")

import pygltftoolkit as pygltk

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

def sample_and_export_points(trimesh_mesh, triangle_instance_mask, pred_inst_triangle_map, args):
    for instance_id, instance_dict in pred_inst_triangle_map.items():
        if not instance_id in triangle_instance_mask:
            continue
        temp_mesh = copy.deepcopy(trimesh_mesh)
        visual_copy = copy.deepcopy(temp_mesh.visual)
        temp_mesh.update_faces(triangle_instance_mask == int(instance_id))
        temp_mesh.remove_unreferenced_vertices()
        temp_mesh.visual = visual_copy.face_subset(triangle_instance_mask == int(instance_id))

        if hasattr(temp_mesh.visual, "material"):
            if isinstance(temp_mesh.visual.material, trimesh.visual.material.PBRMaterial):
                temp_mesh.visual.material = temp_mesh.visual.material.to_simple()
            elif isinstance(temp_mesh.visual, trimesh.visual.color.ColorVisuals):
                temp_mesh.visual = temp_mesh.visual.to_simple()

        if hasattr(temp_mesh.visual, "material"):
            if temp_mesh.visual.uv is None or temp_mesh.visual.material.image is None:
                result = trimesh.sample.sample_surface(temp_mesh, 100000, sample_color=False)
                points = result[0]
                colors = np.array([temp_mesh.visual.material.main_color[:3] / 255] * 100000)
            else:
                result = trimesh.sample.sample_surface(temp_mesh, 100000, sample_color=True)
                points = result[0]
                colors = result[2][:, :3] / 255
        else:
            result = trimesh.sample.sample_surface(temp_mesh, 100000, sample_color=True)
            points = result[0]
            colors = result[2][:, :3] / 255

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        downpcd = pcd.voxel_down_sample(voxel_size=0.005)
        # o3d.visualization.draw_geometries([downpcd])

        os.makedirs(f"{args.output_dir}/pcd", exist_ok=True)
        with open(f"{args.output_dir}/pcd/{args.id}-{instance_id}.npz", "wb+") as outfile:
            np.savez(outfile, points=np.array(downpcd.points), colors=np.array(downpcd.colors), instance=np.asarray(instance_id), semantic=np.asarray(instance_dict["semantic"]))


def save_nonindexed_geometry(mesh: trimesh.Trimesh, save_path: str, export_type: str = 'obj'):
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError('Input must be a trimesh.Trimesh object')

    vertices = mesh.vertices
    faces = mesh.faces
    face_colors = mesh.visual.face_colors

    nonindexed_vertices = []
    nonindexed_faces = []

    # Create truly non-indexed vertices and faces
    for face_idx, face in enumerate(faces):
        face_vertices = vertices[face]
        start_idx = len(nonindexed_vertices)
        nonindexed_vertices.extend(face_vertices)
        nonindexed_faces.append([start_idx, start_idx + 1, start_idx + 2])

    nonindexed_mesh = trimesh.Trimesh(vertices=nonindexed_vertices, faces=nonindexed_faces, process=False)
    nonindexed_mesh.visual.face_colors = face_colors

    if export_type == 'obj':
        nonindexed_mesh.export(save_path)
    elif export_type == 'ply':
        vertex_data = np.array([(v[0], v[1], v[2]) for v in nonindexed_vertices],
                            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

        face_data = np.array([(f, c[0], c[1], c[2], c[3]) 
                            for f, c in zip(nonindexed_faces, face_colors)],
                            dtype=[('vertex_indices', 'i4', (3,)),
                                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('alpha', 'u1')])

        vertex_element = PlyElement.describe(vertex_data, 'vertex')
        face_element = PlyElement.describe(face_data, 'face')

        ply_data = PlyData([vertex_element, face_element], text=True)
        ply_data.write(save_path)
    return nonindexed_mesh


def generate_gt(args):
    os.makedirs(f"{args.output_dir}/gt")
    os.makedirs(f"{args.output_dir}/obj")
    os.makedirs(f"{args.output_dir}/parts")
    os.makedirs(f"{args.output_dir}/pcd")
    with open(args.data_json, 'r') as f:
        data = json.load(f)

    for model_id in tqdm(data[args.split].keys()):
        gltf = pygltk.load(f"{args.data_path}/{model_id}/{model_id}.glb")
        gltf.load_stk_segmentation_openable(f"{args.data_path}/{model_id}/{model_id}.art-stk.json")
        face_colors = []
        semantic_map = []
        if args.mode == 'instance':
            num_instances = np.unique(gltf.segmentation_map).shape[0]
            # Generate palette with sns
            palette = sns.color_palette("husl", n_colors=num_instances)
            instance_color_map = {instance: palette[instance] for instance in np.unique(gltf.segmentation_map)}
        elif args.mode == 'semantic':
            # Use PARTNETSIM_COLOR_MAP as base color and generate subcolors based on number of instances of each semantic class
            palette = [None, None, None, None]
            sem_inst_counters = [0, 0, 0]
            for part in gltf.segmentation_parts.values():
                if part.label != "base":
                    sem_inst_counters[S2O_SEM_REF[part.label]] += 1
            for idx, sem_inst_count in enumerate(sem_inst_counters):
                if sem_inst_count > 0:
                    palette[idx] = sns.light_palette(S2O_COLOR_MAP_HEX[idx], n_colors=sem_inst_count + 1)[1:]
            palette[3] = sns.light_palette(S2O_COLOR_MAP_HEX[3], n_colors=2)[1:]
            instance_color_map = {}
            for instance in np.unique(gltf.segmentation_map):
                sem_label = gltf.segmentation_parts[instance].label
                if sem_label == "base":
                    instance_color_map[instance] = palette[3][0]
                else:
                    sem_inst_counters[S2O_SEM_REF[sem_label]] -= 1
                    instance_color_map[instance] = palette[S2O_SEM_REF[sem_label]][sem_inst_counters[S2O_SEM_REF[sem_label]]]
        for face_part in gltf.segmentation_map:
            sem_face = gltf.segmentation_parts[face_part].label
            face_colors.append(instance_color_map[face_part])
            semantic_map.append(S2O_SEM_REF[sem_face])
        mesh = gltf.create_colored_trimesh(np.array(face_colors))
        nonindexed_mesh = save_nonindexed_geometry(mesh, f"{args.output_dir}/obj/{model_id}.obj")

        np.savez(f"{args.output_dir}/gt/{model_id}", semantic=semantic_map, instance=gltf.segmentation_map)

        # Export parts
        for part_idx in np.unique(gltf.segmentation_map):
            part_mesh = copy.deepcopy(nonindexed_mesh)
            part_mesh.update_faces(gltf.segmentation_map == part_idx)
            part_mesh.remove_unreferenced_vertices()
            part_mesh.visual = nonindexed_mesh.visual.face_subset(gltf.segmentation_map == part_idx)
            os.makedirs(f"{args.output_dir}/parts/{model_id}", exist_ok=True)
            _ = save_nonindexed_geometry(part_mesh, f"{args.output_dir}/parts/{model_id}/{part_idx}.obj")

        # Sample points from parts
        segmentation_dict = {}
        for part_idx in np.unique(gltf.segmentation_map):
            segmentation_dict[part_idx] = {'semantic': S2O_SEM_REF[gltf.segmentation_parts[part_idx].label], 'conf': 1.0}
        sample_and_export_points(nonindexed_mesh, gltf.segmentation_map, segmentation_dict, args)



def map_single_mesh(args, face_indices, vertex_ids):
    with open(f"{args.predict_dir}/{args.id}.txt", 'r') as f:
        pred = f.readlines()
    pred = [line.strip().split(' ') for line in pred]
    masks = {}
    all_masks = None
    base_pred_idx = -1
    for idx, pred_line in enumerate(pred):
        mask_path = pred_line[0]
        mask_semantic = int(pred_line[1])
        if mask_semantic == 3 and base_pred_idx == -1:
            base_pred_idx = idx
        mask_conf = float(pred_line[2])
        mask = np.loadtxt(f"{args.predict_dir}/{mask_path}")
        mask = mask.astype(np.int32)
        masks[idx] = {'conf': mask_conf, 'semantic': mask_semantic}
        if all_masks is None:
            all_masks = mask * idx
        else:
            all_masks[mask == 1] = idx

    mapped_faces = -np.ones(len(gltf.faces), dtype=np.int32)
    for face_idx in range(len(gltf.faces)):
        face_preds = all_masks[face_indices == face_idx]
        vertex_preds = []
        for vertex_id in gltf.faces[face_idx]:
            vertex_preds.extend(all_masks[vertex_ids == vertex_id])
        preds = np.concatenate([face_preds, vertex_preds])
        most_common_count = np.bincount(preds).max()
        most_common_preds = np.where(np.bincount(preds) == most_common_count)[0]
        if len(most_common_preds) == 1:
            most_common_pred = most_common_preds[0]
        else:
            most_common_pred = most_common_preds[0]
            for pred in most_common_preds:
                if masks[pred]['conf'] > masks[most_common_pred]['conf']:
                    most_common_pred = pred
        mapped_faces[face_idx] = most_common_pred
    mapped_faces[mapped_faces == -1] = base_pred_idx

    face_colors = []
    if args.mode == 'instance':
        num_instances = np.unique(mapped_faces).shape[0]
        # Generate palette with sns
        palette = sns.color_palette("husl", n_colors=num_instances)
        instance_color_map = {instance: palette[instance] for instance in np.unique(mapped_faces)}
    elif args.mode == 'semantic':
        # Use PARTNETSIM_COLOR_MAP as base color and generate subcolors based on number of instances of each semantic class
        palette = [None, None, None, None]
        sem_inst_counters = [0, 0, 0]
        for inst in np.unique(mapped_faces):
            sem_label = masks[inst]['semantic']
            if sem_label != 3:
                sem_inst_counters[sem_label] += 1
        for idx, sem_inst_count in enumerate(sem_inst_counters):
            if sem_inst_count > 0:
                palette[idx] = sns.light_palette(S2O_COLOR_MAP_HEX[idx], n_colors=sem_inst_count + 1)[1:]
        palette[3] = sns.light_palette(S2O_COLOR_MAP_HEX[3], n_colors=2)[1:]
        instance_color_map = {}
        for instance in np.unique(mapped_faces):
            sem_label = masks[instance]['semantic']
            if sem_label == 3:
                instance_color_map[instance] = palette[3][0]
            else:
                sem_inst_counters[sem_label] -= 1
                instance_color_map[instance] = palette[sem_label][sem_inst_counters[sem_label]]
    semantic_map = []
    sem_inst_counters = [0, 0, 0]
    for face_idx in range(len(gltf.faces)):
        semantic_label = masks[mapped_faces[face_idx]]['semantic']
        semantic_map.append(semantic_label)
        face_colors.append(instance_color_map[mapped_faces[face_idx]])

    mesh = gltf.create_colored_trimesh(np.array(face_colors))
    nonindexed_mesh = save_nonindexed_geometry(mesh, f"{args.output_dir}/obj/{args.id}.obj")

    np.savez(f"{args.output_dir}/pred/{args.id}", semantic=semantic_map, instance=mapped_faces)

    # Export parts
    for part_idx in np.unique(mapped_faces):
        part_mesh = copy.deepcopy(nonindexed_mesh)
        part_mesh.update_faces(mapped_faces == part_idx)
        part_mesh.remove_unreferenced_vertices()
        part_mesh.visual = nonindexed_mesh.visual.face_subset(mapped_faces == part_idx)
        os.makedirs(f"{args.output_dir}/parts/{args.id}", exist_ok=True)
        _ = save_nonindexed_geometry(part_mesh, f"{args.output_dir}/parts/{args.id}/{part_idx}.obj")

    # Sample points from parts
    sample_and_export_points(nonindexed_mesh, mapped_faces, masks, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, required=True)
    parser.add_argument('--split', type=str, default='val', choices=['test', 'val'],
                        help='specify the split of data: val | test')
    parser.add_argument('--mode', type=str, default='semantic', choices=['semantic', 'instance'],
                        help='specify instance or semantic mode: semantic | instance')
    parser.add_argument('--output_dir', type=str, default='./mapped_meshes',
                        help='the directory of the output')
    parser.add_argument('--data_path', type=str, help='Path to GLBs')
    parser.add_argument('--data_json', type=str, help='Path to data.json')
    parser.add_argument('--sampled_data', type=str, help='Path to dataset (h5)')

    parser.add_argument('-i', '--id', type=str, default=None, help='specify one scene id')
    parser.add_argument('-g', '--gt', action="store_true", help='generate gt data')

    args = parser.parse_args()
    args.predict_dir = args.exp_dir

    if args.gt:
        generate_gt(args)
    elif args.id:
        os.makedirs(f"{args.output_dir}/pred")
        os.makedirs(f"{args.output_dir}/obj")
        # Loading h5
        with h5py.File(args.sampled_data, 'r') as f:
            model_ids = np.asarray(f['model_ids'][:])
            model_id_idx_map = {model_id.decode('utf-8'): idx for idx, model_id in enumerate(model_ids)}
            idx = model_id_idx_map[args.id]
            face_indices = np.asarray(f['face_indexes'][idx])
            vertex_ids = np.asarray(f['vertex_ids'][idx])
            gltf = pygltk.load(f"{args.data_path}/{args.id}/{args.id}.glb")
        map_single_mesh(args, face_indices, vertex_ids, gltf)
    else:
        os.makedirs(f"{args.output_dir}/pred")
        os.makedirs(f"{args.output_dir}/obj")
        # Generating for all ids
        with open(args.data_json, 'r') as f:
            data = json.load(f)
        with h5py.File(args.sampled_data, 'r') as f:
            model_ids = np.asarray(f['model_ids'][:])
            model_id_idx_map = {model_id.decode('utf-8'): idx for idx, model_id in enumerate(model_ids)}
            for model_id in tqdm(data[args.split].keys()):
                gltf = pygltk.load(f"{args.data_path}/{model_id}/{model_id}.glb")
                gltf.load_stk_segmentation_openable(f"{args.data_path}/{model_id}/{model_id}.art-stk.json")
                idx = model_id_idx_map[model_id]
                face_indices = np.asarray(f['pygltk_face_indexes'][idx])
                vertex_ids = np.asarray(f['vertex_ids'][idx])
                args.id = model_id
                map_single_mesh(args, face_indices, vertex_ids)
