"""Surface and volume sampling with trimesh-compatible entry points."""

from __future__ import annotations

import logging

import numpy as np

from ._lib import addr, f64, i64, lib

log = logging.getLogger(__name__)
_TREE_THRESHOLD = 512


def _mesh_arrays(mesh):
    vertices = f64(mesh.vertices)
    faces = i64(mesh.faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("mesh vertices must have shape (n, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("mesh faces must have shape (m, 3)")
    if not np.isfinite(vertices).all():
        raise ValueError("mesh vertices must be finite")
    if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("mesh faces reference vertices outside the vertex array")
    return vertices, faces


def _face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    areas = np.empty(len(faces), dtype=np.float64)
    if len(faces):
        lib().mt_face_areas(
            addr(vertices, np.float64),
            addr(faces, np.int64),
            addr(areas, np.float64),
            len(faces),
        )
    return areas


def sample_surface(
    mesh,
    count,
    face_weight=None,
    sample_color=False,
    seed=None,
):
    """Sample points uniformly from triangle surfaces."""
    count = int(count)
    if count < 0:
        raise ValueError("count must be non-negative")
    vertices, faces = _mesh_arrays(mesh)
    if count == 0:
        empty_points = np.empty((0, 3), dtype=np.float64)
        empty_faces = np.empty(0, dtype=np.int64)
        if sample_color:
            return empty_points, empty_faces, np.empty((0, 4), dtype=np.uint8)
        return empty_points, empty_faces
    if len(faces) == 0:
        raise ValueError("cannot sample a mesh with no faces")

    weights = (
        _face_areas(vertices, faces)
        if face_weight is None
        else f64(face_weight).reshape(-1)
    )
    if len(weights) != len(faces):
        raise ValueError("face_weight must contain one value per face")
    if np.any(weights < 0.0) or not np.isfinite(weights).all():
        raise ValueError("face_weight must be finite and non-negative")
    cumulative = np.ascontiguousarray(np.cumsum(weights), dtype=np.float64)
    if cumulative[-1] <= 0.0:
        raise ValueError("face weights must have a positive sum")

    random = np.random.random if seed is None else np.random.default_rng(seed).random
    picks = np.ascontiguousarray(random(count) * cumulative[-1])
    barycentric = np.ascontiguousarray(random((count, 2)), dtype=np.float64)
    points = np.empty((count, 3), dtype=np.float64)
    face_index = np.empty(count, dtype=np.int64)
    lib().mt_sample_surface(
        addr(vertices, np.float64),
        addr(faces, np.int64),
        addr(cumulative, np.float64),
        addr(picks, np.float64),
        addr(barycentric, np.float64),
        addr(points, np.float64),
        addr(face_index, np.int64),
        len(faces),
        count,
    )

    if not sample_color:
        return points, face_index
    visual = getattr(mesh, "visual", None)
    colors = getattr(visual, "face_colors", None)
    if colors is None:
        raise ValueError("sample_color requires mesh.visual.face_colors")
    return points, face_index, np.asarray(colors)[face_index]


def sample_surface_even(mesh, count, radius=None, seed=None):
    """Approximately even surface samples using trimesh's rejection rule."""
    count = int(count)
    if count <= 0:
        return np.empty((0, 3)), np.empty(0, dtype=np.int64)
    if radius is None:
        vertices, faces = _mesh_arrays(mesh)
        radius = np.sqrt(_face_areas(vertices, faces).sum() / (3.0 * count))
    points, face_index = sample_surface(mesh, count * 3, seed=seed)
    if len(points) >= _TREE_THRESHOLD:
        from scipy.spatial import cKDTree

        pairs = cKDTree(points, copy_data=False).query_pairs(
            float(radius), output_type="ndarray"
        )
        degree = np.bincount(pairs.ravel(), minlength=len(points))
        column = degree[pairs].argmax(axis=1)
        highest = pairs.ravel()[column + 2 * np.arange(len(column))]
        mask = np.ones(len(points), dtype=bool)
        mask[highest] = False
        selected = np.flatnonzero(mask)
    else:
        mask = np.empty(len(points), dtype=np.int64)
        degree = np.empty(len(points), dtype=np.int64)
        lib().mt_remove_close(
            addr(points, np.float64),
            addr(mask, np.int64),
            addr(degree, np.int64),
            len(points),
            float(radius),
        )
        selected = np.flatnonzero(mask)
    if len(selected) < count:
        log.warning("only got %d/%d samples!", len(selected), count)
    selected = selected[:count]
    return points[selected], face_index[selected]


def sample_surface_sphere(count: int) -> np.ndarray:
    u, v = np.random.random((2, int(count)))
    theta = np.pi * 2.0 * u
    z = 2.0 * v - 1.0
    radial = np.sqrt(1.0 - z * z)
    return np.column_stack((radial * np.cos(theta), radial * np.sin(theta), z))


def volume_rectangular(extents, count, transform=None) -> np.ndarray:
    samples = (np.random.random((int(count), 3)) - 0.5) * np.asarray(extents)
    if transform is not None:
        matrix = np.asarray(transform, dtype=np.float64)
        samples = samples @ matrix[:3, :3].T + matrix[:3, 3]
    return samples


def volume_mesh(mesh, count) -> np.ndarray:
    count = int(count)
    points = np.random.random((count, 3)) * np.asarray(mesh.extents)
    points += np.asarray(mesh.bounds)[0]
    return points[np.asarray(mesh.contains(points), dtype=bool)][:count]
