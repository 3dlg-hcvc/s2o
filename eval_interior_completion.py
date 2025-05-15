import argparse
import json
import os
from glob import glob

import chamferdist
import numpy as np
import torch
import trimesh
from tqdm import tqdm


def compute_bbox_iou(aabb1, aabb2):
    # https://pbr-book.org/3ed-2018/Geometry_and_Transformations/Bounding_Boxes#:~:text=The%20intersection%20of%20two%20bounding,Intersection%20of%20Two%20Bounding%20Boxes.
    (center1, dim1, cat1) = aabb1
    (center2, dim2, cat2) = aabb2
    if int(cat1) != int(cat2):
        return -np.inf

    aabb1_min, aabb1_max = center1 - dim1 / 2, center1 + dim1 / 2
    aabb2_min, aabb2_max = center2 - dim2 / 2, center2 + dim2 / 2
    max_min = np.maximum(aabb1_min, aabb2_min)
    min_max = np.minimum(aabb1_max, aabb2_max)

    intersection_dims = np.maximum(0, min_max - max_min)
    intersection_volume = np.prod(intersection_dims)

    gt_volume = np.prod(aabb1_max - aabb1_min)
    pred_volume = np.prod(aabb2_max - aabb2_min)
    union_volume = gt_volume + pred_volume - intersection_volume

    return intersection_volume / union_volume


def greedy_matching(list1, list2, iou_threshold):
    num_boxes_list1 = len(list1)
    num_boxes_list2 = len(list2)
    max_dim = max(num_boxes_list1, num_boxes_list2)
    distance_matrix = -np.ones((max_dim, max_dim))
    row_indices, col_indices = -np.ones(max_dim, dtype=int), -np.ones(max_dim, dtype=int)

    for i, bbox1 in enumerate(list1):
        current_matching_iou = -np.inf
        current_matching_index = -1
        for j, bbox2 in enumerate(list2):
            iou = compute_bbox_iou(bbox1, bbox2)
            if iou < iou_threshold:
                continue
            if iou > current_matching_iou and j not in col_indices:
                current_matching_index = j
                current_matching_iou = iou
            distance_matrix[i][j] = iou
        row_indices[i] = i
        col_indices[i] = current_matching_index
    return row_indices, col_indices, distance_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_dir", type=str, required=True)
    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--data_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--rescale", action="store_true")
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()

    with open(args.data_json, "r") as f:
        data = json.load(f)

    model_ids = list(data["val"].keys())

    metric = chamferdist.ChamferDistance()

    os.makedirs(f"{args.output_dir}/cd_pcd", exist_ok=True)

    chamfer_dists = []
    chamfer_dists_nb = []
    chamfer_dists_d = []
    chamfer_dists_b = []

    chamfer_dist = 0
    chamfer_dist_nb = 0
    chamfer_dist_d = 0
    chamfer_dist_b = 0

    drawer_tp = 0
    drawer_fp = 0
    drawer_fn = 0
    
    drawer_macro_precisions = []
    drawer_macro_recalls = []
    drawer_macro_f1s = []
    
    has_drawer_models = []

    for model_id in tqdm(model_ids):
        # Load gt parts
        gt_parts_path = f"{args.gt_path}/{model_id}/parts"
        gt_parts_mesh_dict = {}
        gt_path = f"{args.gt_path}/gt/{model_id}.npz"

        gt_map = np.load(gt_path, allow_pickle=True)
        gt_instance_semantic_map = {}
        semantic_labels = []
        for instance_id in np.unique(gt_map["instance"]):
            gt_instance_semantic_map[instance_id] = gt_map["semantic"][gt_map["instance"] == instance_id][0]
            semantic_labels.append(gt_map["semantic"][gt_map["instance"] == instance_id][0])
        # if 0 not in semantic_labels:
        #     continue

        for gt_part_path in glob(f"{gt_parts_path}/*.obj"):
            part_idx = int(gt_part_path.split("/")[-1].split(".")[0])
            gt_parts_mesh_dict[part_idx] = trimesh.load(gt_part_path)

        gt_whole_shape = trimesh.load(f"{args.gt_path}/obj/{model_id}.obj")

        # Load predicted parts
        pred_parts_path = f"{args.predict_dir}/{model_id}"
        pred_parts_mesh_dict = {}
        pred_path = f"{args.predict_dir}/pred/{model_id}.npz"

        pred_map = np.load(pred_path, allow_pickle=True)
        pred_instance_semantic_map = {}
        for instance_id in np.unique(pred_map["instance_map"]):
            pred_instance_semantic_map[instance_id] = pred_map["semantic_map"][pred_map["instance_map"] == instance_id][0]

        for predict_part_path in glob(f"{pred_parts_path}/*.obj"):
            part_idx = int(predict_part_path.split("/")[-1].split(".")[0])
            pred_parts_mesh_dict[part_idx] = trimesh.load(predict_part_path)

        pred_whole_shape = trimesh.load(f"{args.predict_dir}/obj/{model_id}.obj")

        if args.rescale:
            gt_bbox_min, gt_bbox_max = gt_whole_shape.bounds
            gt_diag = np.linalg.norm(gt_bbox_max - gt_bbox_min)
            pred_bbox_min, pred_bbox_max = pred_whole_shape.bounds
            pred_diag = np.linalg.norm(pred_bbox_max - pred_bbox_min)

            scale = gt_diag / pred_diag

            for part_idx in pred_parts_mesh_dict:
                pred_parts_mesh_dict[part_idx].vertices *= scale

            pred_whole_shape.vertices *= scale

            gt_centroid = (gt_bbox_min + gt_bbox_max) / 2
            pred_centroid = (pred_bbox_min + pred_bbox_max) / 2

            translation = gt_centroid - pred_centroid

            for part_idx in pred_parts_mesh_dict:
                pred_parts_mesh_dict[part_idx].vertices += translation

            pred_whole_shape.vertices += translation

        if not os.path.exists(f"{args.gt_path}/cd_pcd/{model_id}.npz") or args.recompute:
            gt_points, gt_faces_sampled = trimesh.sample.sample_surface(gt_whole_shape, 20000)

            # Determine and export the instance and semantic labels for the sampled points
            gt_instance_labels = gt_map["instance"][gt_faces_sampled]
            gt_semantic_labels = gt_map["semantic"][gt_faces_sampled]

            os.makedirs(f"{args.gt_path}/cd_pcd", exist_ok=True)
            np.savez(
                f"{args.gt_path}/cd_pcd/{model_id}.npz",
                gt_points=gt_points,
                gt_instance_labels=gt_instance_labels,
                gt_semantic_labels=gt_semantic_labels,
            )
        if not os.path.exists(f"{args.output_dir}/cd_pcd/{model_id}.npz") or args.recompute:
            pred_points, pred_faces_sampled = trimesh.sample.sample_surface(pred_whole_shape, 20000)

            pred_instance_labels = pred_map["instance_map"][pred_faces_sampled]
            pred_semantic_labels = pred_map["semantic_map"][pred_faces_sampled]

            os.makedirs(f"{args.output_dir}/cd_pcd", exist_ok=True)
            np.savez(
                f"{args.output_dir}/cd_pcd/{model_id}.npz",
                pred_points=pred_points,
                pred_instance_labels=pred_instance_labels,
                pred_semantic_labels=pred_semantic_labels,
            )

        gt_points_data = np.load(f"{args.gt_path}/cd_pcd/{model_id}.npz", allow_pickle=True)
        gt_points_data = {key: gt_points_data[key] for key in gt_points_data.files}
        pred_points_data = np.load(f"{args.output_dir}/cd_pcd/{model_id}.npz", allow_pickle=True)
        pred_points_data = {key: pred_points_data[key] for key in pred_points_data.files}

        # Chamfer distance - between the whole shapes

        gt_points = torch.from_numpy(gt_points_data["gt_points"]).float().cuda()
        pred_points = torch.from_numpy(pred_points_data["pred_points"]).float().cuda()

        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(gt_points_data["gt_points"])
        pcd.paint_uniform_color([0, 1, 0])
        pred_pcd = o3d.geometry.PointCloud()
        pred_pcd.points = o3d.utility.Vector3dVector(pred_points_data["pred_points"])
        pred_pcd.paint_uniform_color([1, 0, 0])
        # o3d.visualization.draw_geometries([pcd, pred_pcd])

        chamfer_dist = metric(gt_points.unsqueeze(0), pred_points.unsqueeze(0), bidirectional=True).detach().cpu().item() / (len(gt_points) + len(pred_points))
        chamfer_dists.append(chamfer_dist)

        # Chamfer distance no base, exclude semantic label 3

        gt_points = gt_points_data["gt_points"][gt_points_data["gt_semantic_labels"] != 3]
        pred_points = pred_points_data["pred_points"][pred_points_data["pred_semantic_labels"] != 3]

        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(gt_points)
        pcd.paint_uniform_color([0, 1, 0])
        pred_pcd = o3d.geometry.PointCloud()
        pred_pcd.points = o3d.utility.Vector3dVector(pred_points)
        pred_pcd.paint_uniform_color([1, 0, 0])
        # o3d.visualization.draw_geometries([pcd, pred_pcd])

        gt_points = torch.from_numpy(gt_points).float().cuda()
        pred_points = torch.from_numpy(pred_points).float().cuda()

        chamfer_dist_nb = metric(gt_points.unsqueeze(0), pred_points.unsqueeze(0), bidirectional=True).detach().cpu().item() / (len(gt_points) + len(pred_points))
        chamfer_dists_nb.append(chamfer_dist_nb)

        # Chamfer distance drawers, consider only semantic label 0

        gt_points = gt_points_data["gt_points"][gt_points_data["gt_semantic_labels"] == 0]
        pred_points = pred_points_data["pred_points"][pred_points_data["pred_semantic_labels"] == 0]

        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(gt_points)
        pcd.paint_uniform_color([0, 1, 0])
        pred_pcd = o3d.geometry.PointCloud()
        pred_pcd.points = o3d.utility.Vector3dVector(pred_points)
        pred_pcd.paint_uniform_color([1, 0, 0])
        # o3d.visualization.draw_geometries([pcd, pred_pcd])

        gt_points = torch.from_numpy(gt_points).float().cuda()
        pred_points = torch.from_numpy(pred_points).float().cuda()

        if (len(gt_points) + len(pred_points)) != 0:
            chamfer_dist_d = metric(gt_points.unsqueeze(0), pred_points.unsqueeze(0), bidirectional=True).detach().cpu().item() / (len(gt_points) + len(pred_points))
            chamfer_dists_d.append(chamfer_dist_d)
        
        gt_points = gt_points_data["gt_points"][gt_points_data["gt_semantic_labels"] == 3]
        pred_points = pred_points_data["pred_points"][pred_points_data["pred_semantic_labels"] == 3]

        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(gt_points)
        pcd.paint_uniform_color([0, 1, 0])
        pred_pcd = o3d.geometry.PointCloud()
        pred_pcd.points = o3d.utility.Vector3dVector(pred_points)
        pred_pcd.paint_uniform_color([1, 0, 0])
        # o3d.visualization.draw_geometries([pcd, pred_pcd])

        gt_points = torch.from_numpy(gt_points).float().cuda()
        pred_points = torch.from_numpy(pred_points).float().cuda()

        if (len(gt_points) + len(pred_points)) != 0:
            chamfer_dist_b = metric(gt_points.unsqueeze(0), pred_points.unsqueeze(0), bidirectional=True).detach().cpu().item() / (len(gt_points) + len(pred_points))
            chamfer_dists_b.append(chamfer_dist_b)
        
        # Update print statement to include base CD
        print(f"CD: {chamfer_dist}, CD NB: {chamfer_dist_nb}, CD D: {chamfer_dist_d}, CD B: {chamfer_dist_b}")
        
        gt_drawer_parts = []
        for instance_id in np.unique(gt_map["instance"]):
            sem_label = gt_map["semantic"][gt_map["instance"] == instance_id][0]
            if int(sem_label) == 0:  # Drawer semantic label
                # Find faces for this instance
                instance_mask = gt_map["instance"] == instance_id
                # Get vertices from the faces
                triangles = gt_whole_shape.vertices[gt_whole_shape.faces[instance_mask]]
                vertices = triangles.reshape(-1, 3)
                
                if len(vertices) > 0:
                    bbox_min = np.min(vertices, axis=0)
                    bbox_max = np.max(vertices, axis=0)
                    center = (bbox_max + bbox_min) / 2
                    dim = bbox_max - bbox_min
                    gt_drawer_parts.append((center, dim, 0))
        
        # Collect predicted drawer parts (semantic label = 0)
        pred_drawer_parts = []
        for instance_id in np.unique(pred_map["instance_map"]):
            sem_label = pred_map["semantic_map"][pred_map["instance_map"] == instance_id][0]
            if int(sem_label) == 0:  # Drawer semantic label
                # Find faces for this instance
                instance_mask = pred_map["instance_map"] == instance_id
                # Get vertices from the faces
                triangles = pred_whole_shape.vertices[pred_whole_shape.faces[instance_mask]]
                vertices = triangles.reshape(-1, 3)
                
                if len(vertices) > 0:
                    bbox_min = np.min(vertices, axis=0)
                    bbox_max = np.max(vertices, axis=0)
                    center = (bbox_max + bbox_min) / 2
                    dim = bbox_max - bbox_min
                    pred_drawer_parts.append((center, dim, 0))
        
        # Only compute metrics if there are ground truth drawers
        if len(gt_drawer_parts) > 0:
            has_drawer_models.append(model_id)
            
            # Match drawer parts using greedy matching with IoU threshold of 0.5
            matched_gt_indices, matching_pred_indices, _ = greedy_matching(gt_drawer_parts, pred_drawer_parts, 0.8)
            
            # Count TP, FP, FN for this model
            current_tp = np.sum(matching_pred_indices >= 0)
            current_fp = len(pred_drawer_parts) - current_tp
            current_fn = len(gt_drawer_parts) - current_tp
            
            # Accumulate for micro averages
            drawer_tp += current_tp
            drawer_fp += current_fp
            drawer_fn += current_fn
            
            # Calculate per-object metrics
            per_obj_precision = current_tp / (current_tp + current_fp) if (current_tp + current_fp) > 0 else 0
            per_obj_recall = current_tp / (current_tp + current_fn) if (current_tp + current_fn) > 0 else 0
            per_obj_f1 = 2 * (per_obj_precision * per_obj_recall) / (per_obj_precision + per_obj_recall) if (per_obj_precision + per_obj_recall) > 0 else 0
            
            # Store for macro averages
            drawer_macro_precisions.append(per_obj_precision)
            drawer_macro_recalls.append(per_obj_recall)
            drawer_macro_f1s.append(per_obj_f1)
    # Average and export json with results
    chamfer_dist = np.mean(chamfer_dists)
    chamfer_dist_nb = np.mean(chamfer_dists_nb)
    chamfer_dist_d = np.mean(chamfer_dists_d)
    chamfer_dist_b = np.mean(chamfer_dists_b)

    results = {
        "cd": float(chamfer_dist) * 1000,
        "cd_nb": float(chamfer_dist_nb) * 1000,
        "cd_d": float(chamfer_dist_d) * 1000,
        "cd_b": float(chamfer_dist_b) * 1000,
    }

    # Calculate drawer detection metrics
    drawer_micro_precision = drawer_tp / (drawer_tp + drawer_fp) if (drawer_tp + drawer_fp) > 0 else 0
    drawer_micro_recall = drawer_tp / (drawer_tp + drawer_fn) if (drawer_tp + drawer_fn) > 0 else 0
    drawer_micro_f1 = 2 * (drawer_micro_precision * drawer_micro_recall) / (drawer_micro_precision + drawer_micro_recall) if (drawer_micro_precision + drawer_micro_recall) > 0 else 0

    drawer_macro_precision = np.mean(drawer_macro_precisions) if drawer_macro_precisions else 0
    drawer_macro_recall = np.mean(drawer_macro_recalls) if drawer_macro_recalls else 0
    drawer_macro_f1 = np.mean(drawer_macro_f1s) if drawer_macro_f1s else 0

    # Add to results dictionary
    results.update({
        "drawer_micro_precision": float(drawer_micro_precision),
        "drawer_micro_recall": float(drawer_micro_recall),
        "drawer_micro_f1": float(drawer_micro_f1),
        "drawer_macro_precision": float(drawer_macro_precision),
        "drawer_macro_recall": float(drawer_macro_recall),
        "drawer_macro_f1": float(drawer_macro_f1),
        "drawer_models_count": len(has_drawer_models),
        "total_models_count": len(model_ids)
    })

    os.makedirs(args.output_dir, exist_ok=True)

    with open(f"{args.output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=4)
