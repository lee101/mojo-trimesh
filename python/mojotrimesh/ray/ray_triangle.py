"""Brute-force ray-triangle queries executed by Mojo."""

from __future__ import annotations

import numpy as np

from .._lib import addr, f64, lib


def _vectors(value, name: str) -> np.ndarray:
    array = f64(value)
    if array.ndim == 1:
        array = array.reshape((1, 3))
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def ray_triangle_id(
    triangles,
    ray_origins,
    ray_directions,
    triangles_normal=None,
    tree=None,
    multiple_hits=True,
):
    """Find intersections between triangles and forward rays."""
    del triangles_normal, tree
    triangles = f64(triangles)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("triangles must have shape (n, 3, 3)")
    if not np.isfinite(triangles).all():
        raise ValueError("triangles must contain only finite values")
    origins = _vectors(ray_origins, "ray_origins")
    directions = _vectors(ray_directions, "ray_directions")
    if len(origins) != len(directions):
        raise ValueError("ray origins and directions must have the same length")
    if np.any(np.einsum("ij,ij->i", directions, directions) == 0.0):
        raise ValueError("ray directions must be nonzero")
    if not len(triangles) or not len(origins):
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
        )

    maximum = len(triangles) * len(origins)
    capacity = min(max(16, len(origins) * (1 if not multiple_hits else 4)), maximum)

    def execute(size: int):
        index_triangle = np.empty(max(size, 1), dtype=np.int64)
        index_ray = np.empty(max(size, 1), dtype=np.int64)
        distance = np.empty(max(size, 1), dtype=np.float64)
        total = lib().mt_ray_hits(
            addr(triangles, np.float64),
            addr(origins, np.float64),
            addr(directions, np.float64),
            addr(index_triangle, np.int64),
            addr(index_ray, np.int64),
            addr(distance, np.float64),
            size,
            len(triangles),
            len(origins),
            int(bool(multiple_hits)),
        )
        total = int(total)
        if total < 0 or total > maximum:
            raise RuntimeError(f"Mojo ray kernel returned invalid hit count {total}")
        return total, index_triangle, index_ray, distance

    total, index_triangle, index_ray, distance = execute(capacity)
    if total > capacity:
        total, index_triangle, index_ray, distance = execute(total)
    index_triangle = index_triangle[:total]
    index_ray = index_ray[:total]
    distance = distance[:total]
    locations = origins[index_ray] + directions[index_ray] * distance[:, None]
    return index_triangle, index_ray, locations


def _unique_locations(index_triangle, index_ray, locations):
    if not len(locations):
        return index_triangle, index_ray, locations
    seen = set()
    selected = []
    rounded = np.round(locations, decimals=12)
    for i, (ray, point) in enumerate(zip(index_ray, rounded)):
        key = (int(ray), float(point[0]), float(point[1]), float(point[2]))
        if key not in seen:
            seen.add(key)
            selected.append(i)
    selected = np.asarray(selected, dtype=np.int64)
    return index_triangle[selected], index_ray[selected], locations[selected]


class RayMeshIntersector:
    def __init__(self, mesh):
        self.mesh = mesh

    def intersects_id(
        self,
        ray_origins,
        ray_directions,
        return_locations=False,
        multiple_hits=True,
        **kwargs,
    ):
        del kwargs
        result = ray_triangle_id(
            self.mesh.triangles,
            ray_origins,
            ray_directions,
            multiple_hits=multiple_hits,
        )
        if return_locations:
            return _unique_locations(*result)
        return result[0], result[1]

    def intersects_location(self, ray_origins, ray_directions, **kwargs):
        index_triangle, index_ray, locations = self.intersects_id(
            ray_origins,
            ray_directions,
            return_locations=True,
            **kwargs,
        )
        return locations, index_ray, index_triangle

    def intersects_first(self, ray_origins, ray_directions, **kwargs):
        origins = _vectors(ray_origins, "ray_origins")
        index_triangle, index_ray = self.intersects_id(
            origins,
            ray_directions,
            multiple_hits=False,
            **kwargs,
        )
        result = np.full(len(origins), -1, dtype=np.int64)
        result[index_ray] = index_triangle
        return result

    def intersects_any(self, ray_origins, ray_directions, **kwargs):
        origins = _vectors(ray_origins, "ray_origins")
        _, index_ray = self.intersects_id(origins, ray_directions, **kwargs)
        result = np.zeros(len(origins), dtype=bool)
        result[np.unique(index_ray)] = True
        return result

    def contains_points(self, points):
        points = _vectors(points, "points")
        direction = np.array([0.43950645, 0.61759863, 0.65123472])
        directions = np.tile(direction, (len(points), 1))
        _, index_ray, locations = self.intersects_id(
            points, directions, return_locations=True, multiple_hits=True
        )
        del locations
        counts = np.bincount(index_ray, minlength=len(points))
        return (counts % 2) == 1
