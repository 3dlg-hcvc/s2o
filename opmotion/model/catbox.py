import numpy as np
import open3d as o3d
from opmotion.model.base import _Box
from Helper3D import getMotionMesh


class CatBox:
    def __init__(
        self,
        center,
        dim,
        rotMat=np.eye(4),
        cat=None,
        front=None,
        up=None,
        id=None,
        mesh=None,
        colored_mesh=None,
        parent=None,
        motionType=None,
        motionAxis=None,
        motionOrigin=None,
        is_pcd=False,
    ):
        """Initialize a CatBox. Initialize a box with the dim, and rotate then translate

        :param center: The center of the 3D bounding box
        :type center: np.ndarray(3)
        :param dim: The dimension of the 3D bounding box
        :type dim: np.ndarray(3)
        :param rotMat: The rotation of the 3D bounding box
        :type rotMat: ndarray(4, 4)
        :param cat: category of the catbox
        :type cat: string, optional
        :param front: The front direction (can be not accurate, this will be aligned with the box three main direction)
        :type front: np.ndarray(3)
        :param up: The up direction (can be not accurate, this will be aligned with the box three main direction)
        :type up: np.ndarray(3)
        :param mesh: mesh in the catbox
        :type mesh: open3d.geometry.TriangleMesh
        :param mesh: colored mesh in the catbox, the colored version, this is in the trimesh.Scene format
        :type mesh: trimesh.Scene
        :param id: unique for the catbox
        :type id: int or string
        """
        if not isinstance(center, np.ndarray):
            raise ValueError("center must be numpy array")
        if not isinstance(dim, np.ndarray):
            raise ValueError("dim must be numpy array")
        if not isinstance(rotMat, np.ndarray):
            raise ValueError("rotMat must be numpy array")
        if not isinstance(front, np.ndarray):
            raise ValueError("front must be numpy array")
        if not isinstance(up, np.ndarray):
            raise ValueError("up must be numpy array")

        self.box = _Box(center, dim, rotMat)
        if not front is None and not up is None:
            self.box.update_front_up(front, up)
        self.cat = cat
        self.id = id
        self.mesh = mesh
        self.colored_mesh = colored_mesh
        # For the hierarchy structure
        self.parent = parent
        # Set the motion parameters
        self.motionType = motionType
        self.motionAxis = motionAxis
        self.motionOrigin = motionOrigin
        self.is_pcd = is_pcd

    def resetParentMotion(self):
        # self.mesh = None
        self.parent = None
        self.motionType = None
        self.motionAxis = None
        self.motionOrigin = None

    def setPraentMotion(self, anno):
        self.parent = anno["parent"]
        self.motionType = anno["motionType"]
        self.motionAxis = np.array(anno["motionAxis"])
        self.motionOrigin = np.array(anno["motionOrigin"])

    def getInfo(self):
        result = {
            "id": self.id,
            "cat": self.cat,
            "diagonal": (self.box.dim**2).sum() ** 0.5,
            "parent": self.parent,
            "motionType": self.motionType,
        }
        if self.motionAxis is not None:
            result["motionAxis"] = list(self.motionAxis)
        else:
            result["motionAxis"] = None
        if self.motionOrigin is not None:
            result["motionOrigin"] = list(self.motionOrigin)
        else:
            result["motionOrigin"] = None

        return result

    def get_edge(self, face1, face2):
        """Get the edge between two faces

        :param face1: the first face
        :type face1: str ["front", "back", "left", "right", "top", "bottom"]
        :param face2: the second face
        :type face2: str ["front", "back", "left", "right", "top", "bottom"]
        :return: return the two vertices on the edge between two faces
        """
        return self.box.get_edge(face1, face2)

    def get_box_info(self):
        """Get the info  of all faces of the box"""
        return self.box.get_box_info()

    def get_bbx_color(self):
        if self.cat == "base":
            return np.array([171.0 / 255.0, 171.0 / 255.0, 171.0 / 255.0])
        elif self.cat == "door":
            return np.array([255.0 / 255.0, 128.0 / 255.0, 14.0 / 255.0])
        elif self.cat == "drawer":
            return np.array([0.0, 107.0 / 255.0, 164.0 / 255.0])
        elif self.cat == "lid":
            return np.array([200.0 / 255.0, 82.0 / 255.0, 0.0])

    def get_mesh(
        self, normal=False, front=False, up=False, motion=False, triangleMesh=False
    ):
        """Get the mesh of the catbox to visualize

        :param normal: Choose whether to visualize the normal of each face of the box, defaults to False
        :type normal: bool, optional
        :param front: Choose whether to visualize the front direction, defaults to False
        :type front: bool, optional
        :param up: Choose whether to visualize the up direction, defaults to False
        :type up: bool, optional
        :param motion: Choose whether to visualize the motion axis/origin, defaults to False
        :type motion: bool, optional
        :param triangleMesh: Choose whether to visualize the triangleMesh, defaults to False
        :type triangleMesh: bool, optional
        :return: return the box line mesh with additional stuffs based on the parameters
        :rtype: open3d.geometry.TriangleMesh
        """
        mesh = []
        mesh += self.box.get_mesh(normal, front, up, bbxColor=self.get_bbx_color())
        if motion and self.cat != "base":
            # Ignore parts that are not in our consideration
            if self.motionType is None:
                raise ValueError("The motion is undefined")
            if self.motionType != "fixed":
                mesh += getMotionMesh(
                    self.motionType, self.motionAxis, self.motionOrigin
                )
        if triangleMesh:
            mesh.append(self.mesh)
        return mesh

    def _repr__(self):
        self.__str__()

    def __str__(self):
        return f"CatBox: id={self.id}, cat={self.cat}"
