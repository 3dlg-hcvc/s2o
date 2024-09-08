import numpy as np
import json
import copy
import trimesh

class HSSDParser:
    def __init__(self, model_path):
        self.model_path = model_path
        with open(model_path, "r") as file_obj:
            resolver = trimesh.visual.resolvers.FilePathResolver("/".join(model_path.split("/")[:-1]) + "/material.mtl")
            kwargs = trimesh.exchange.obj.load_obj(file_obj, maintain_order=True, resolver=resolver, group_material=False)
            self.trimesh_scene = trimesh.exchange.load.load_kwargs(kwargs)