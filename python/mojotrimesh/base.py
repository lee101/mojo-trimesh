"""A compact triangle-mesh container for the covered API."""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, i64, lib


class Trimesh:
    """Triangle mesh compatible with the subset consumed by this package."""

    def __init__(
        self,
        vertices=None,
        faces=None,
        process: bool = True,
        validate: bool = False,
        **kwargs,
    ):
        if vertices is None:
            vertices = np.empty((0, 3), dtype=np.float64)
        if faces is None:
            faces = np.empty((0, 3), dtype=np.int64)
        self.vertices = f64(vertices, copy=True).reshape((-1, 3))
        self.faces = i64(faces, copy=True).reshape((-1, 3))
        self.visual = kwargs.get("visual")
        self._validate(require_finite=validate)
        if process and len(self.faces):
            self._remove_unreferenced()

    def _validate(self, *, require_finite: bool = True) -> None:
        if require_finite and not np.isfinite(self.vertices).all():
            raise ValueError("vertices must be finite")
        if len(self.faces) and (
            self.faces.min() < 0 or self.faces.max() >= len(self.vertices)
        ):
            raise ValueError("faces reference vertices outside the vertex array")

    def _remove_unreferenced(self) -> None:
        used = np.unique(self.faces)
        if len(used) == len(self.vertices) and np.array_equal(
            used, np.arange(len(self.vertices))
        ):
            return
        inverse = np.full(len(self.vertices), -1, dtype=np.int64)
        inverse[used] = np.arange(len(used), dtype=np.int64)
        self.vertices = np.ascontiguousarray(self.vertices[used])
        self.faces = np.ascontiguousarray(inverse[self.faces])

    @property
    def triangles(self) -> np.ndarray:
        return np.ascontiguousarray(self.vertices[self.faces])

    @property
    def area_faces(self) -> np.ndarray:
        self._validate(require_finite=True)
        result = np.empty(len(self.faces), dtype=np.float64)
        if len(result):
            lib().mt_face_areas(
                addr(self.vertices, np.float64),
                addr(self.faces, np.int64),
                addr(result, np.float64),
                len(result),
            )
        return result

    @property
    def area(self) -> float:
        return float(self.area_faces.sum())

    @property
    def face_normals(self) -> np.ndarray:
        triangles = self.triangles
        normals = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 0.0
        normals[valid] /= lengths[valid, None]
        normals[~valid] = 0.0
        return normals

    @property
    def volume(self) -> float:
        triangles = self.triangles
        if len(triangles) == 0:
            return 0.0
        signed = np.einsum(
            "ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])
        )
        return float(signed.sum() / 6.0)

    @property
    def bounds(self) -> np.ndarray:
        if not len(self.vertices):
            return np.empty((2, 3), dtype=np.float64)
        return np.vstack((self.vertices.min(axis=0), self.vertices.max(axis=0)))

    @property
    def extents(self) -> np.ndarray:
        bounds = self.bounds
        return bounds[1] - bounds[0]

    @property
    def is_watertight(self) -> bool:
        if not len(self.faces):
            return False
        edges = np.sort(
            self.faces[:, [[0, 1], [1, 2], [2, 0]]].reshape((-1, 2)), axis=1
        )
        _, counts = np.unique(edges, axis=0, return_counts=True)
        return bool(np.all(counts == 2))

    def copy(self) -> "Trimesh":
        return Trimesh(self.vertices.copy(), self.faces.copy(), process=False)

    def contains(self, points) -> np.ndarray:
        from .ray.ray_triangle import RayMeshIntersector

        return RayMeshIntersector(self).contains_points(points)

    @property
    def ray(self):
        from .ray.ray_triangle import RayMeshIntersector

        return RayMeshIntersector(self)

    @property
    def triangles_tree(self):
        return None

    def __len__(self) -> int:
        return len(self.faces)
