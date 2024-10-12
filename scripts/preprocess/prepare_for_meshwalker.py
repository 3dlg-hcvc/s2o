import json
import os
import sys
import argparse

import numpy as np
from tqdm import tqdm

sys.path.append("../..")
import pygltftoolkit as pygltk

PARTNETSIM_CATEGORY_TO_ID = {"drawer": 0, "door": 1, "lid": 2, "base": 3}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb_path", type=str, required=True)
    parser.add_argument("--data_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True, help="{MeshWalkerPath}/data/{DataName}")
    args = parser.parse_args()

    os.makedirs(f"{args.output_dir}", exist_ok=True)

    with open(f"{args.data_json}") as f:
        splits_data = json.load(f)

    splits = ["train", "val"]

    model_ids = {"train": [], "val": []}
    id_category_mapping = {}

    for split in splits:
        for model_id, model_info in splits_data[split].items():
            model_ids[split].append(model_id)
            id_category_mapping[model_id] = model_info["category"]

    for split in splits:
        os.makedirs(f"{args.output_dir}/{split}", exist_ok=True)

        for model_id in tqdm(model_ids[split]):

            glb_path = f"{args.glb_path}/{model_id}/{model_id}.glb"
            artpre_path = f"{args.glb_path}/{model_id}/{model_id}.art-stk.json"

            gltf = pygltk.load(glb_path)
            gltf.load_stk_segmentation_openable(artpre_path)

            vertices = gltf.vertices
            faces = gltf.faces
            labels = -np.ones(len(gltf.vertices), dtype=np.int_)

            for pid, part in gltf.segmentation_parts.items():
                part_mask = gltf.segmentation_map == pid
                part_semantic_label = PARTNETSIM_CATEGORY_TO_ID[part.label]
                part_face_indices = faces[part_mask]
                part_vertex_indices = np.unique(part_face_indices)
                labels[part_vertex_indices] = part_semantic_label

            """mask = np.zeros(len(gltf.segmentation_map), dtype=bool)
            for pid, part in gltf.segmentation_parts.items():
                if part.label in PARTNETSIM_CATEGORY_TO_ID.keys():
                    mask[gltf.segmentation_map == pid] = True

            vertex_mask = np.zeros(len(vertices), dtype=bool)
            for face in faces[mask]:
                vertex_mask[face] = True

            old_to_new_vertex_index = np.zeros(len(vertices), dtype=int)
            old_to_new_vertex_index[vertex_mask] = np.arange(len(np.where(vertex_mask)[0]))
            vertices = vertices[vertex_mask]
            faces = faces[mask]
            faces = old_to_new_vertex_index[faces]
            labels = labels[vertex_mask]
            
            new_to_old_vertex_map = np.zeros(len(vertices), dtype=int)"""


            label = id_category_mapping[model_id]

            mesh_data = {'vertices': vertices,
                         'faces': faces,
                         'label': label,
                         'labels': labels}

            filename = f"{args.output_dir}/{split}/{model_id}"
            np.savez(f"{filename}.npz", **mesh_data)
