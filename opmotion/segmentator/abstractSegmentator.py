import open3d as o3d
import trimesh
import pytorch_lightning as pl
import hydra
import os
from abc import ABC, abstractmethod
import sys
sys.path.append("../..")
from Helper3D import SampleSurfaceFromTrimeshScene

# AbstractSegmentator class for mesh and point segmentation. 
# --- Input:
#       -- ckpt_path: path to checkpoint
#       -- (optional) output_path: directory to save output in. Nothing is saved if not specified. 
# --- Methods:
#       -- load_mesh_get_points(file_path, FPSNUM)
#           - file_path: relative or absolute path to the mesh. Only .obj and .glb files are accepted.
#           - FPSNUM: number of points to be downsampled to.
#

class AbstractSegmentator(ABC):
    def __init__(self, ckpt_path, output_path=None): 
        self.ckpt_path = ckpt_path
        self.output_path = output_path

    @abstractmethod
    def load_mesh_get_points(self, file_path, FPSNUM):
        with open(file_path) as f:
            if ".obj" in file_path:
                obj_dict = trimesh.exchange.obj.load_obj(f)
            elif ".glb" in file_path:
                obj_dict = trimesh.exchange.gltf.load_glb(f)
            else:
                raise Exception("ERROR: Unsopported file format. Please specify path to .obj or .glb file")
            self.trimesh = trimesh.Trimesh(vertices=obj_dict["vertices"], faces=obj_dict["faces"], vertex_normals=obj_dict["vertex_normals"], visual=obj_dict["visual"])
        self.sampled_points, self.point_colors, self.point_normals, self.scene_face_indexes, self.point_barycentric_coordinates, self.point_geometry_map = SampleSurfaceFromTrimeshScene(self.trimesh, self.FPSNUM)

     
