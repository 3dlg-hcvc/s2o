import argparse
import copy
import glob
import json
import os

import numpy as np
import open3d as o3d
from Helper3D import getArrowMesh, getURDF

from opmotion import (
    CatBox,
    Evaluator,
    HierarchyEngine,
    MotionEngine,
    PartnetsimParser,
    RuleHierarchyPredictor,
    RuleMotionPredictor,
)

np.random.seed(6)

SEMANTIC_MAPPING = {0: "drawer", 1: "door", 2: "lid", 3: "base"}
FRONT = np.array([-1, 0, 0])
UP = np.array([0, 0, 1])


def parseIns(ins_paths):
    global SEMANTIC_MAPPING, FRONT, UP
    parts = {}
    for ins_path in ins_paths:
        # Load the instances and process them into the catbox format
        ins = np.load(ins_path)
        points = np.array(ins["points"])
        # colors = np.array(ins["colors"])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        semantic = SEMANTIC_MAPPING[int(ins["semantic"])]
        if semantic == "base":
            pcd.paint_uniform_color([0.5, 0.5, 0.5])
        elif semantic == "drawer":
            pcd.paint_uniform_color([0, 0, 1])
        elif semantic == "door":
            pcd.paint_uniform_color([0, 1, 0])
        elif semantic == "lid":
            pcd.paint_uniform_color([1, 0, 0])

        colored_pcd = o3d.geometry.PointCloud()
        colored_pcd.points = o3d.utility.Vector3dVector(points)
        # colored_pcd.colors = o3d.utility.Vector3dVector(colors)
        ins_id = int(ins["instance"])

        # Create the CatBox for each part
        # Get the axis aligned bounding box for the parts
        aabb = pcd.get_axis_aligned_bounding_box()
        min_bound = aabb.get_min_bound()
        max_bound = aabb.get_max_bound()
        center = (max_bound + min_bound) / 2
        dim = max_bound - min_bound

        catbox = CatBox(
                    center,
                    dim,
                    cat=semantic,
                    front=FRONT,
                    up=UP,
                    id=ins_id,
                    mesh=pcd,
                    colored_mesh=colored_pcd,
                    is_pcd=True
                )

        parts[ins_id] = catbox
        # o3d.visualization.draw_geometries(catbox.get_mesh(front=True, up=True, triangleMesh=True))

    return parts


def motion_inference(input_parts):
    # Use the hierarchy engine to get the hierarchy
    hierarchy_engine = HierarchyEngine(predictors=[RuleHierarchyPredictor()])
    hier_parts = hierarchy_engine.process(input_parts)

    # Use the motion engine to get the motions
    motion_engine = MotionEngine(predictors=[RuleMotionPredictor()])
    motion_parts = motion_engine.process(hier_parts)

    for part_id, catbox in motion_parts.items():
        print(f"{part_id}: {catbox.cat} {catbox.parent} {catbox.motionType} {catbox.motionAxis} {catbox.motionOrigin}")

    motion_mesh = []
    for catbox in motion_parts.values():
        motion_mesh += catbox.get_mesh(triangleMesh=True, motion=True)

    """coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries([motion_mesh, coordinate])"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_path",
        default=None,
        required=True,
        metavar="FILE",
        help="Path to the directory with predictions (contains pcd/ folder)",
    )
    parser.add_argument(
        "--output_path", 
        default=None,
        metavar="FILE",
        help="Path to output motion inference results",
    )
    parser.add_argument(
        "--export",
        default=False,
        action="store_true",
        help="Export flag"
    )
    args = parser.parse_args()
    model_ids = np.unique(["-".join(item.split("/")[-1].split(".")[0].split("-")[:-1]) for item in glob.glob(f"{args.pred_path}/pcd/*.npz")])
    print(len(model_ids))
    print(args.pred_path)
    if args.export:
        if os.path.exists(f"{args.output_path}/opmotion"):
            os.system(f"rm -r {args.output_path}/opmotion")
        os.makedirs(f"{args.output_path}/opmotion", exist_ok=True)
    for model_id in model_ids:
        print(model_id)
        ins_paths = glob.glob(f"{args.pred_path}/pcd/{model_id}-*.npz")
        parts = parseIns(ins_paths)
        hierarchy_engine = HierarchyEngine(predictors=[RuleHierarchyPredictor()])
        hier_parts = hierarchy_engine.process(parts)

        # Use the motion engine to get the motions
        motion_engine = MotionEngine(predictors=[RuleMotionPredictor()])
        motion_parts = motion_engine.process(hier_parts)
        for motion_part in motion_parts.values():
            if motion_part.motionType != "fixed":
                mtype = 0 if motion_part.motionType == "revolute" else 1
                motion_dict = {"mtype": mtype, "morigin": motion_part.motionOrigin.tolist(), "maxis": motion_part.motionAxis.tolist()}
                if args.export:
                    with open(f"{args.output_path}/opmotion/{model_id}-{motion_part.id}.json", "w+") as json_file:
                        json.dump(motion_dict, json_file)
