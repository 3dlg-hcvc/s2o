import numpy as np
import open3d as o3d
from opmotion.utils.transform import _trasnfrom_vector
from Helper3D import getBoxMesh, getArrowMesh

DEBUG_MODE = True


class _Box:
    def __init__(self, center, dim, rotMat):
        self.center = center
        self.dim = dim
        self.rotMat = rotMat
        self.vertices, self.faces, self.normals = self._init_with_cdr()
        # Initialize the face parameters
        self.faceInfo = None
        self.front, self.frontFace = None, None
        self.back, self.backFace = None, None
        self.up, self.upFace = None, None
        self.down, self.downFace = None, None
        self.right, self.rightFace = None, None
        self.left, self.leftFace = None, None

    def get_edge(self, face1, face2):
        """Get the edge between two faces

        :param face1: the first face
        :type face1: str ["front", "back", "left", "right", "top", "bottom"]
        :param face2: the second face
        :type face2: str ["front", "back", "left", "right", "top", "bottom"]
        :return: return the two vertices on the edge between two faces
        """
        if self.faceInfo is None:
            raise ValueError("Face info is not initialized")
        # Get the vertices of the two faces
        face1Vertices = self.faceInfo[face1]["faceVerticesIndex"]
        face2Vertices = self.faceInfo[face2]["faceVerticesIndex"]

        # Get the vertices of the edge
        edgeVerticesIndex = np.intersect1d(face1Vertices, face2Vertices)
        edgeVertices = self.vertices[edgeVerticesIndex]
        return edgeVertices

    def update_front_up(self, front, up):
        # Process to get the front normal and front face
        self.front, self.frontFace = self._process_direction(front, label="front")
        # Get the back face and back normal based on the front
        self.backFace = (self.frontFace + 3) % 6
        self.back = self.normals[self.backFace]
        # Process to get the up normal and up face
        self.up, self.upFace = self._process_direction(up, label="up")
        if self.upFace == self.frontFace or self.upFace == self.backFace:
            raise ValueError("Front and up cannot be in the same axis")
        # Get the down face and down normal based on the up
        self.downFace = (self.upFace + 3) % 6
        self.down = self.normals[self.downFace]
        # Based on the front and up, we can get the right, left faces
        self.right, self.rightFace = self._process_direction(
            np.cross(self.up, self.front), label="right"
        )
        # Get the left face and left normal based on the right
        self.leftFace = (self.rightFace + 3) % 6
        self.left = self.normals[self.leftFace]

        """Get the information about all the faces, vertices and normals"""
        self.faceInfo = {
            "front": {
                "faceIndex": self.frontFace,
                "faceNormal": self.front,
                "faceVerticesIndex": self.faces[self.frontFace],
                "faceVertices": self.vertices[self.faces[self.frontFace]],
            },
            "back": {
                "faceIndex": self.backFace,
                "faceNormal": self.back,
                "faceVerticesIndex": self.faces[self.backFace],
                "faceVertices": self.vertices[self.faces[self.backFace]],
            },
            "up": {
                "faceIndex": self.upFace,
                "faceNormal": self.up,
                "faceVerticesIndex": self.faces[self.upFace],
                "faceVertices": self.vertices[self.faces[self.upFace]],
            },
            "down": {
                "faceIndex": self.downFace,
                "faceNormal": self.down,
                "faceVerticesIndex": self.faces[self.downFace],
                "faceVertices": self.vertices[self.faces[self.downFace]],
            },
            "right": {
                "faceIndex": self.rightFace,
                "faceNormal": self.right,
                "faceVerticesIndex": self.faces[self.rightFace],
                "faceVertices": self.vertices[self.faces[self.rightFace]],
            },
            "left": {
                "faceIndex": self.leftFace,
                "faceNormal": self.left,
                "faceVerticesIndex": self.faces[self.leftFace],
                "faceVertices": self.vertices[self.faces[self.leftFace]],
            },
        }

    def get_box_info(self):
        return self.faceInfo

    def _init_with_cdr(self):
        vertices = np.array(
            [
                np.array([1, -1, -1]),
                np.array([1, 1, -1]),
                np.array([-1, 1, -1]),
                np.array([-1, -1, -1]),
                np.array([1, -1, 1]),
                np.array([1, 1, 1]),
                np.array([-1, 1, 1]),
                np.array([-1, -1, 1]),
            ]
        ).astype(np.float64)
        faces = (
            np.array(
                [
                    np.array([1, 2, 6, 5]),
                    np.array([2, 3, 7, 6]),
                    np.array([5, 6, 7, 8]),
                    np.array([3, 4, 8, 7]),
                    np.array([1, 5, 8, 4]),
                    np.array([4, 3, 2, 1]),
                ]
            )
            - 1
        )
        normals = np.array(
            [
                np.array([1, 0, 0]),
                np.array([0, 1, 0]),
                np.array([0, 0, 1]),
                np.array([-1, 0, 0]),
                np.array([0, -1, 0]),
                np.array([0, 0, -1]),
            ]
        ).astype(np.float64)

        # Do scaling based on the dim (only for the vertices)
        vertices[:, 0] *= self.dim[0] / 2 
        vertices[:, 1] *= self.dim[1] / 2
        vertices[:, 2] *= self.dim[2] / 2
        # Do rotation based on rotMat
        vertices = _trasnfrom_vector(vertices, self.rotMat)
        normals = _trasnfrom_vector(normals, self.rotMat)
        # Do translation based on the center
        vertices += self.center

        return vertices, faces, normals

    # Process the direction to make it aligned with the existed normals
    def _process_direction(self, direction, label="direction"):
        oldDirection = direction
        cosValues = [
            np.clip(
                np.dot(normal, direction)
                / (np.linalg.norm(normal) * np.linalg.norm(direction)),
                -1,
                1,
            )
            for normal in self.normals
        ]
        angleErrors = np.arccos(cosValues) / np.pi * 180
        # Use the most similar one as the front
        directionFace = np.argmin(angleErrors)
        direction = self.normals[directionFace]
        if angleErrors[directionFace] != 0:
            print(f"The {label} {oldDirection} is modified to {direction}")
        return direction, directionFace

    def get_mesh(self, normal=False, front=False, up=False, bbxColor=np.array([0, 0, 0])):
        mesh = []
        mesh.append(getBoxMesh(self.vertices, color=bbxColor))
        if normal:
            # Draw the normal arrows for the box
            for i in range(np.shape(self.normals)[0]):
                mesh.append(getArrowMesh(self.center, self.center + self.normals[i]))
        if front:
            # Draw the front arrow for the box
            if self.front is None and DEBUG_MODE:
                print("There is no front defined")
            if not self.front is None:
                mesh.append(
                    getArrowMesh(
                        self.center, self.center + self.front, color=[1, 0.5, 0]
                    )
                )
        if up:
            # Draw the up arrow for the box
            if self.up is None and DEBUG_MODE:
                print("There is no up defined")
            if not self.up is None:
                mesh.append(
                    getArrowMesh(self.center, self.center + self.up, color=[1, 0, 0.5])
                )

        return mesh

    def get_center(self):
        return self.center
