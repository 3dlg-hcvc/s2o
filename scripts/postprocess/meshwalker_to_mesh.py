import argparse
import copy
import json
import os
import sys

import numpy as np
import open3d as o3d
import trimesh
from plyfile import PlyData, PlyElement
from tqdm import tqdm

sys.path.append("../..")
import pygltftoolkit as pygltk

S2O_COLOR_MAP = {
    0: (0.0, 107.0 / 255.0, 164.0 / 255.0),
    1: (255.0 / 255.0, 128.0 / 255.0, 14.0 / 255.0),
    2: (44.0 / 255.0, 160.0 / 255.0, 44.0 / 255.0),
    3: (171.0 / 255.0, 171.0 / 255.0, 171.0 / 255.0),
}


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', type=str, required=True, help='Path to predictions (models.npz)')
    parser.add_argument('--glb_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(f"./{args.output_dir}/obj")
    os.makedirs(f"./{args.output_dir}/parts")
    os.makedirs(f"./{args.output_dir}/pcd")
    os.makedirs(f"./{args.output_dir}/pred")

    models = np.load(f"{args.pred_path}/models.npz", allow_pickle=True)

    for model_path in tqdm(models.keys()):
        args.id = model_path.split("/")[-1].split(".")[0]
        os.makedirs(f"{args.output_dir}/parts/{args.id}", exist_ok=True)
        model_data = models[model_path]
        gltf_path = f"{args.glb_path}/{args.id}/{args.id}.glb"

        gltf = pygltk.load(gltf_path)

        if type(model_data) is not dict:
            model_data = model_data.tolist()

        predictions = model_data["pred"].argmax(axis=1)
        semantic_labels = []
        face_colors = []
        for face in gltf.faces:
            face_predictions = predictions[face]
            common_face_prediction = np.bincount(face_predictions).argmax()
            face_color = S2O_COLOR_MAP[common_face_prediction]
            face_colors.append(np.asarray(face_color))
            semantic_labels.append(common_face_prediction)
        semantic_labels = np.array(semantic_labels)
        mesh = gltf.create_colored_trimesh(np.array(face_colors))
        instance_labels = np.zeros_like(semantic_labels)
        instance_dict = {}
        for idx, sem_label in enumerate(np.unique(semantic_labels)):
            instance_labels[semantic_labels == sem_label] = idx
            instance_dict[idx] = {"semantic": sem_label}

        nonindexed_mesh = save_nonindexed_geometry(mesh, f"{args.output_dir}/obj/{args.id}.obj")
        np.savez(f"{args.output_dir}/pred/{args.id}", semantic=semantic_labels, instance=instance_labels)
        sample_and_export_points(mesh, instance_labels, instance_dict, args)

        for part_id in np.unique(instance_labels):
            part_mesh = copy.deepcopy(nonindexed_mesh)
            part_mesh.update_faces(instance_labels == part_id)
            part_mesh.remove_unreferenced_vertices()
            part_mesh.visual = nonindexed_mesh.visual.face_subset(instance_labels == part_id)
            _ = save_nonindexed_geometry(part_mesh, f"{args.output_dir}/parts/{args.id}/{part_id}.obj")
