import copy
import json
import xml.etree.ElementTree as ET

import numpy as np
from Helper3D import (
    getArrowMesh,
    getMotionMesh,
    getOpen3DFromTrimeshScene,
    getSphereMesh,
    getURDF,
)

from opmotion.model import CatBox

COLORMAP = {
    "base": np.array([171.0 / 255.0, 171.0 / 255.0, 171.0 / 255.0]),
    "drawer": np.array([0.0, 107.0 / 255.0, 164.0 / 255.0]),
    "door": np.array([255.0 / 255.0, 128.0 / 255.0, 14.0 / 255.0]),
    "lid": np.array([200.0 / 255.0, 82.0 / 255.0, 0.0]),
}


class PartnetsimParser:
    def __init__(self, model_path, specified_parts=None):
        self.specified_parts = specified_parts
        self.model_path = model_path
        self.urdf, self.controller = getURDF(f"{model_path}/mobility.urdf")
        # Update the parts infor based on the urdf, only fetch the part name from the json file
        self.parts = self._init_info(f"{model_path}/mobility_v2.json")
        self.parts_catbox = None
        # Record current parts_catbox are in merging mode or not
        self.merge_base = True

    def get_parts(self):
        return self.parts

    def get_parts_catbox(self, merge_base=True):
        self._update_parts_catbox(merge_base)
        return self.parts_catbox

    def _update_parts_catbox(self, merge_base=True):
        if self.parts_catbox is None or self.merge_base != merge_base:
            # If call this function the first time or change the requirement for merge_base, then update the parts catbox based on the parts
            self.merge_base = merge_base
            # Merge all base part
            if self.merge_base:
                parts_process = {}
                parts_process["base"] = None
                for part_id, part in self.parts.items():
                    if part["category"] != "base":
                        parts_process[part_id] = part
                    else:
                        if parts_process["base"] is None:
                            parts_process["base"] = copy.deepcopy(part)
                        else:
                            parts_process["base"]["mesh"] += part["mesh"]
            else:
                parts_process = self.parts

            self.parts_catbox = {}
            for part_id, part in parts_process.items():
                # Get the axis aligned bounding box for the parts
                #import pdb; pdb.set_trace()
                aabb = part["mesh"].get_axis_aligned_bounding_box()
                min_bound = aabb.get_min_bound()
                max_bound = aabb.get_max_bound()
                center = (max_bound + min_bound) / 2
                dim = max_bound - min_bound
                catbox = CatBox(
                    center,
                    dim,
                    cat=part["category"],
                    front=part["front"],
                    up=part["up"],
                    id=part_id,
                    mesh=part["mesh"],
                    colored_mesh=part["colored_mesh"],
                    parent=part["parent"],
                    motionType=part["motionType"],
                    motionAxis=part["motionAxis"],
                    motionOrigin=part["motionOrigin"],
                )

                self.parts_catbox[part_id] = catbox

    def _init_info(self, path):
        file = open(path, "r")
        anno = json.load(file)
        id_part_map = {}
        junk_counter = 0 
        for part in anno:
            part_id = int(part["id"])
            if part["joint"] == "junk":
                junk_counter += 1
                continue
            else:
                if self.specified_parts is None:
                    id_part_map[part_id - junk_counter] = self.processCategory(part["name"])
                else:
                    if str(part_id) in self.specified_parts:
                        id_part_map[part_id - junk_counter] = self.specified_parts[str(part_id)]
                    else:
                        id_part_map[part_id - junk_counter] = "base"
        parts = {}

        # Process the URDF to get the motion parameters in the world coordinate
        self.urdf.updateMotionWorld()
        for part_name, node in self.controller.items():
            if part_name == "base":
                print(part_name)
                # Ignore the virtural "base" link
                continue
            parts[part_name] = {}
            part_id = int(part_name.split("_")[1])
            if part_id not in id_part_map:
                raise ValueError("The json file doesn't have this joint")
            parts[part_name]["category"] = id_part_map[part_id]
            parts[part_name]["motionType"] = node.joint.joint_type
            parts[part_name]["motionAxis"] = node.axis_world
            parts[part_name]["motionOrigin"] = node.origin_world
            parts[part_name]["parent"] = node.parent.name
            # Here we use the predefined knowledge in partnetsim, the front and the up direction
            parts[part_name]["front"] = np.array([-1, 0, 0])
            parts[part_name]["up"] = np.array([0, 0, 1])
            # Get the mesh for this node
            color = COLORMAP[parts[part_name]["category"]]
            parts[part_name]["mesh"] = getOpen3DFromTrimeshScene(
                node.getControllerNodeMesh(), random_color=False, color=color
            )
            parts[part_name]["colored_mesh"] = node.getControllerNodeMesh()
        return parts

    def processCategory(self, name):
        # TODO: Need to update this function to support more complicated part mapping
        if "door" in name:
            return "door"
        elif "drawer" in name:
            return "drawer"
        elif "lid" in name:
            return "lid"
        else:
            return "base"

    def get_model_mesh(self, motion=False):
        mesh = []
        for part in self.parts.values():
            mesh.append(part["mesh"])
            if motion and part["motionType"] != "fixed":
                mesh += getMotionMesh(
                    part["motionType"], part["motionAxis"], part["motionOrigin"]
                )
        return mesh

    def get_catbox_mesh(
        self, normal=False, front=True, up=True, motion=False, triangleMesh=False
    ):
        self._update_parts_catbox()
        mesh = []
        for part in self.parts_catbox.values():
            mesh += part.get_mesh(normal, front, up, motion, triangleMesh)
        return mesh


class ImprovedPartnetsimParser:
    def __init__(self, model_path, specified_parts=None):
        self.specified_parts = specified_parts
        self.model_path = model_path
        self.urdf, self.controller = getURDF(f"{model_path}/mobility.urdf")
        # Update the parts infor based on the urdf, only fetch the part name from the json file
        self.parts = self._init_info(f"{model_path}/mobility_v2.json")
        self.parts_catbox = None
        # Record current parts_catbox are in merging mode or not
        self.merge_base = True

    def get_parts(self):
        return self.parts

    def get_parts_catbox(self, merge_base=True):
        self._update_parts_catbox(merge_base)
        return self.parts_catbox

    def _update_parts_catbox(self, merge_base=True):
        if self.parts_catbox is None or self.merge_base != merge_base:
            # If call this function the first time or change the requirement for merge_base, then update the parts catbox based on the parts
            self.merge_base = merge_base
            # Merge all base part
            if self.merge_base:
                parts_process = {}
                parts_process["base"] = None
                for part_id, part in self.parts.items():
                    if part["category"] != "base":
                        parts_process[part_id] = part
                    else:
                        if parts_process["base"] is None:
                            parts_process["base"] = copy.deepcopy(part)
                        else:
                            parts_process["base"]["mesh"] += part["mesh"]
            else:
                parts_process = self.parts
            
            self.parts_catbox = {}
            for part_id, part in parts_process.items():
                # Get the axis aligned bounding box for the parts
                aabb = part["mesh"].get_axis_aligned_bounding_box()
                min_bound = aabb.get_min_bound()
                max_bound = aabb.get_max_bound()
                center = (max_bound + min_bound) / 2
                dim = max_bound - min_bound
                catbox = CatBox(
                    center,
                    dim,
                    cat=part["category"],
                    front=part["front"],
                    up=part["up"],
                    id=part_id,
                    mesh=part["mesh"],
                    colored_mesh=part["colored_mesh"],
                    parent=part["parent"],
                    motionType=part["motionType"],
                    motionAxis=part["motionAxis"],
                    motionOrigin=part["motionOrigin"],
                )

                self.parts_catbox[part_id] = catbox

    def _init_info(self, path):
        file = open(path, "r")
        anno = json.load(file)
        id_part_map = {}
        junk_counter = 0 
        for part in anno:
            part_id = int(part["id"])
            if part["joint"] == "junk":
                junk_counter += 1
                continue
            else:
                if self.specified_parts is None:
                    id_part_map[part_id - junk_counter] = self.processCategory(part["name"])
                else:
                    if str(part_id) in self.specified_parts:
                        id_part_map[part_id - junk_counter] = self.specified_parts[str(part_id)]
                    else:
                        id_part_map[part_id - junk_counter] = "base"
        parts = {}

        # Process the URDF to get the motion parameters in the world coordinate
        self.urdf.updateMotionWorld()
        for part_name, node in self.controller.items():
            if part_name == "base":
                print(part_name)
                # Ignore the virtural "base" link
                continue
            parts[part_name] = {}
            part_id = int(part_name.split("_")[1])
            if part_id not in id_part_map:
                raise ValueError("The json file doesn't have this joint")
            parts[part_name]["category"] = id_part_map[part_id]
            parts[part_name]["motionType"] = node.joint.joint_type
            parts[part_name]["motionAxis"] = node.axis_world
            parts[part_name]["motionOrigin"] = node.origin_world
            parts[part_name]["parent"] = node.parent.name
            # Here we use the predefined knowledge in partnetsim, the front and the up direction
            parts[part_name]["front"] = np.array([-1, 0, 0])
            parts[part_name]["up"] = np.array([0, 0, 1])
            # Get the mesh for this node
            color = COLORMAP[parts[part_name]["category"]]
            parts[part_name]["mesh"] = getOpen3DFromTrimeshScene(
                node.getControllerNodeMesh(), random_color=False, color=color
            )
            parts[part_name]["colored_mesh"] = node.getControllerNodeMesh()
        return parts

    def processCategory(self, name):
        # TODO: Need to update this function to support more complicated part mapping
        if "door" in name:
            return "door"
        elif "drawer" in name:
            return "drawer"
        elif "lid" in name:
            return "lid"
        else:
            return "base"

    def get_model_mesh(self, motion=False):
        mesh = []
        for part in self.parts.values():
            mesh.append(part["mesh"])
            if motion and part["motionType"] != "fixed":
                mesh += getMotionMesh(
                    part["motionType"], part["motionAxis"], part["motionOrigin"]
                )
        return mesh

    def get_catbox_mesh(
        self, normal=False, front=True, up=True, motion=False, triangleMesh=False
    ):
        self._update_parts_catbox()
        mesh = []
        for part in self.parts_catbox.values():
            mesh += part.get_mesh(normal, front, up, motion, triangleMesh)
        return mesh