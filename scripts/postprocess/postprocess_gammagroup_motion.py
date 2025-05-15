import argparse
import json
import os
from glob import glob

import numpy as np
import trimesh
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_dir', type=str, required=True)
    parser.add_argument('--motion_type', type=str, default="motion")
    parser.add_argument('--output_motion_type', type=str, default="motion_post")

    args = parser.parse_args()

    os.makedirs(f"{args.predict_dir}/{args.output_motion_type}", exist_ok=True)

    model_ids = [path.split("/")[-1] for path in glob(f"{args.predict_dir}/parts/*")]

    for model_id in tqdm(model_ids):
        json_part_ids = [path.split("/")[-1].split("-")[-1].split(".")[0]
                         for path in glob(f"{args.predict_dir}/{args.motion_type}/{model_id}-*.json")]

        mesh_part_ids = [os.path.splitext(os.path.basename(path))[0]
                         for path in glob(f"{args.predict_dir}/parts/{model_id}/*.obj")]

        common_ids = sorted(set(json_part_ids) & set(mesh_part_ids))

        if len(common_ids) == 0:
            continue

        part_meshes = {part_idx: trimesh.load_mesh(f"{args.predict_dir}/parts/{model_id}/{part_idx}.obj")
                       for part_idx in common_ids}
        motion_annos = {part_idx: json.load(open(f"{args.predict_dir}/{args.motion_type}/{model_id}-{part_idx}.json"))
                        for part_idx in common_ids}

        for part_idx, part_mesh in part_meshes.items():
            motion_anno = motion_annos[part_idx]

            # "Straighten" the axis
            maxis = np.array(motion_anno["maxis"])
            maxis = maxis / np.linalg.norm(maxis)
            maxis_abs = np.abs(maxis)
            dominant_dim = np.argmax(maxis_abs)
            dominant_sign = np.sign(maxis[dominant_dim])
            maxis = np.zeros(3)
            maxis[dominant_dim] = dominant_sign
            motion_anno["maxis"] = maxis.tolist()

            # MOTION_CODES = {0: "revolute", 1: "prismatic", 2: "fixed"}
            if motion_anno["mtype"] == 0:
                # Push the origin to the closest corner of the bounding box
                bbox = part_mesh.bounding_box_oriented.vertices
                morigin = np.array(motion_anno["morigin"])
                # Compute the distance from the origin to each corner
                corner_dists = np.linalg.norm(bbox - morigin, axis=1)
                closest_corner_idx = np.argmin(corner_dists)
                closest_corner = bbox[closest_corner_idx]
                motion_anno["morigin"] = closest_corner.tolist()

            # Save the motion annotation
            with open(f"{args.predict_dir}/{args.output_motion_type}/{model_id}-{part_idx}.json", "w") as f:
                json.dump(motion_anno, f)
