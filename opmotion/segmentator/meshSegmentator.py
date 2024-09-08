import open3d 
import trimesh
from abstractSegmentator import AbstractSegmentator
import pytorch_lightning as pl
import os
import hydra
from importlib import import_module

# TODO: Add more details about cfg structure
# MeshSegmentator class for mesh segmentation. 
# --- Input:
#       -- cfg_directory_path: absolute path to hydra configs directory.
#       -- ckpt_path: path to model checkpoint. 
#       -- (optional) output_path: directory to save output in. Nothing is saved if not specified. 
# --- Methods:
#       -- load(input_path, FPSNUM=20000)
#           - input_path: relative or absolute path to the mesh. Only .obj and .glb files are accepted.
#           - FPSNUM: number of points to be downsampled to, defaults to 20000,
#

class MeshSegmentator(AbstractSegmentator):
    def __init__(self, cfg_directory_path, ckpt_path, output_path=None):
        super().init(ckpt_path, output_path=output_path)
        trainer = pl.Trainer(accelerator="gpu", devices=1, logger=False)
        hydra.initialize_config_dir(config_dir=cfg_directory_path, job_name="inference_on_single_mesh")
        self.cfg = hydra.compose(config_name="config", overrides=[f"ckpt_path={ckpt_path}"])
        self.model = getattr(import_module("minsu3d.model"), self.cfg.model.network.module)(self.cfg)

    def load(self, input_path, FPSNUM=20000):
        super().load_mesh_get_points(input_path, FPSNUM=FPSNUM)
    

