import copy

import numpy as np
import open3d as o3d
from opmotion.utils import distance_edges, distance_point_line

DEBUG = False
VOXEL = False  # whether to use mesh to voxel
POINTNUM = 200000  # the default choice is 200000


class RuleMotionPredictor:
    def __init__(self):
        self.name = "rule_motion_predictor"
        # Below threshold is for the distance comparison
        self.DISTHRESHOLD = 0.01
        # Below threshold is for the voxelizing PC
        self.SIZETHRESHOLD = 0.005
        # Below threshold is for the ratio of the handle
        self.RATIOTHRESHOLD = 0.1

    def predict(self, catboxes):
        catboxes = copy.deepcopy(catboxes)
        # Rule-based motion prediction based on the part category
        for catbox in catboxes.values():
            parent_catbox = None
            if catbox.parent in catboxes:
                parent_catbox = catboxes[catbox.parent]
            self.process(catbox, parent_catbox)
        return catboxes

    def process(self, catbox, parent_catbox):
        if catbox.cat == "base":
            catbox.motionType = "fixed"
        elif catbox.cat == "drawer":
            catbox.motionType = "prismatic"
            catbox.motionAxis = catbox.box.front
            # The origin is just for visualization of the prismatic axis
            catbox.motionOrigin = catbox.box.center
        elif catbox.cat == "door":
            self.processDoor(catbox, parent_catbox)
        elif catbox.cat == "lid":
            self.processLid(catbox, parent_catbox)
        else:
            raise ValueError(f"Unsupported category {catbox.cat}")

    def processDoor(self, catbox, parent_catbox):
        catbox.motionType = "revolute"
        # Infer the motion axis of the door
        box_info = catbox.get_box_info()
        parent_box_info = parent_catbox.get_box_info()

        face = self.judge_front_back(box_info, parent_box_info)

        handle_position = self.get_handle_position(catbox, box_info)
        potentials = ["right", "left", "up", "down"]
        edges = {}
        distances = {}
        for potential in potentials:
            edges[potential] = catbox.box.get_edge(face, potential)
            distances[potential] = distance_point_line(
                handle_position, edges[potential]
            )

        # The top priority is on the left or on the right
        if np.abs(distances["left"] - distances["right"]) > self.DISTHRESHOLD:
            if distances["left"] < distances["right"]:
                final_selection = "right"
            else:
                final_selection = "left"
        else:
            # Judge if the handle is on the top or on the bottom
            if distances["up"] < distances["down"]:
                final_selection = "down"
            else:
                final_selection = "up"
        catbox.motionOrigin, catbox.motionAxis = self.get_final_motion(
            final_selection,
            edges[final_selection],
            catbox.box.up,
            catbox.box.right,
        )

    def processLid(self, catbox, parent_catbox):
        catbox.motionType = "revolute"
        # Infer the motion axis of the door
        box_info = catbox.get_box_info()
        parent_box_info = parent_catbox.get_box_info()

        face = self.judge_up_down(box_info, parent_box_info)
        # Pick the corresponding edge
        edge = catbox.box.get_edge(face, "back")
        # Set the motion origin and axis based on the openess
        axis = edge[0] - edge[1]
        axis = axis / np.linalg.norm(axis)
        if np.dot(axis, catbox.box.right) > 0:
            catbox.motionOrigin = edge[0]
            catbox.motionAxis = -axis
        else:
            catbox.motionOrigin = edge[1]
            catbox.motionAxis = axis


    def judge_up_down(self, box_info, parent_box_info):
        # Judge if the axis is on the up or in the down based on the distance between the point of the parent part up face (because up and down are parallel)
        parent_up_face_center = np.mean(
            parent_box_info["up"]["faceVertices"], axis=0
        )
        box_up_face_center = np.mean(box_info["up"]["faceVertices"], axis=0)
        box_down_face_center = np.mean(box_info["down"]["faceVertices"], axis=0)
        if np.linalg.norm(
            parent_up_face_center - box_up_face_center
        ) < np.linalg.norm(parent_up_face_center - box_down_face_center):
            face = "up"
        else:
            face = "down"
        return face
    
    def judge_front_back(self, box_info, parent_box_info):
        # Judge if the axis is on the back or in the front based on the distance between the point of the parent part front face (because front and back are parallel)
        parent_front_face_center = np.mean(
            parent_box_info["front"]["faceVertices"], axis=0
        )
        box_front_face_center = np.mean(box_info["front"]["faceVertices"], axis=0)
        box_back_face_center = np.mean(box_info["back"]["faceVertices"], axis=0)
        if np.linalg.norm(
            parent_front_face_center - box_front_face_center
        ) < np.linalg.norm(parent_front_face_center - box_back_face_center):
            face = "front"
        else:
            face = "back"
        return face

    def get_handle_position(self, catbox, box_info):
        # Indicate the position of the handle based on the mesh
        front_froniter = np.dot(catbox.box.front, box_info["front"]["faceVertices"][0])
        back_frontier = np.dot(catbox.box.front, box_info["back"]["faceVertices"][0])
        middle = (front_froniter + back_frontier) / 2

        part_mesh = catbox.mesh
        if catbox.is_pcd:
            pcd = part_mesh.voxel_down_sample(self.SIZETHRESHOLD)
            points = np.asarray(pcd.points)
        else:
            if VOXEL:
                voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
                    part_mesh,
                    voxel_size=self.SIZETHRESHOLD,
                    min_bound=part_mesh.get_min_bound(),
                    max_bound=part_mesh.get_max_bound() + self.SIZETHRESHOLD,
                )
                voxels = voxel_grid.get_voxels()
                points = np.array(
                    [
                        voxel_grid.get_voxel_center_coordinate(voxel.grid_index)
                        for voxel in voxels
                    ]
                )
            else:
                pcd = part_mesh.sample_points_uniformly(number_of_points=POINTNUM)
                pcd = pcd.voxel_down_sample(self.SIZETHRESHOLD)
                points = np.asarray(pcd.points)

        relax_ratios = [1 / 3, 2 / 3, 1]
        for relax_ratio in relax_ratios:
            filtered_points = points[
                np.dot(catbox.box.front, points.T)
                > front_froniter - (front_froniter - back_frontier) * relax_ratio
            ]
            if len(filtered_points) > 0:
                break

        if len(filtered_points) / len(points) > self.RATIOTHRESHOLD:
            # Use enough points to fit a plane and filter the plane to get other points as the hints for the handle
            if relax_ratio < 2 / 3:
                relax_ratio = 2 / 3
                filtered_points = points[
                    np.dot(catbox.box.front, points.T)
                    > front_froniter - (front_froniter - back_frontier) * relax_ratio
                ]
            # Judge the handle position from the filtered points based on the normal of the points
            filtered_pcd = o3d.geometry.PointCloud()
            filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

            if len(filtered_points) >= 3:
                plane_model, inliers = filtered_pcd.segment_plane(
                    distance_threshold=0.01, ransac_n=3, num_iterations=1000
                )
            else:
                inliers = np.arange(len(filtered_points))

            mask = np.ones(len(filtered_points), dtype=np.bool_)
            mask[inliers] = 0
            if np.sum(mask) != 0:
                filtered_points = filtered_points[mask]

        # Calculate the handle position
        handle_position = np.mean(filtered_points, axis=0)

        if DEBUG:
            # Visualize the original point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.paint_uniform_color(np.array([0, 1, 0]))
            # Visualize the point cloud used to calculate the handle position
            filtered_pcd = o3d.geometry.PointCloud()
            filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
            filtered_pcd.paint_uniform_color(np.array([1, 0, 0]))
            # Visualize the handle position
            handle_pcd = o3d.geometry.PointCloud()
            handle_pcd.points = o3d.utility.Vector3dVector([handle_position])
            handle_pcd.paint_uniform_color(np.array([0, 0, 1]))
            world = o3d.geometry.TriangleMesh.create_coordinate_frame()
            o3d.visualization.draw_geometries([pcd, filtered_pcd, handle_pcd, world])

        return handle_position

    def get_final_motion(self, final_selection, edge, up, right):
        # Give the direction based on the openness, the axis direction need to consider the openness
        axis = edge[0] - edge[1]
        axis = axis / np.linalg.norm(axis)

        if final_selection == "left":
            if np.dot(axis, up) > 0:
                return edge[0], -axis
            else:
                return edge[1], axis
        elif final_selection == "right":
            if np.dot(axis, up) > 0:
                return edge[1], axis
            else:
                return edge[0], -axis
        elif final_selection == "up":
            if np.dot(axis, right) > 0:
                return edge[0], -axis
            else:
                return edge[1], axis
        elif final_selection == "down":
            if np.dot(axis, right) > 0:
                return edge[1], axis
            else:
                return edge[0], -axis
        else:
            raise ValueError(f"Unsupported selection {final_selection}")
