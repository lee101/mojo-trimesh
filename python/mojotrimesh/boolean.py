"""BSP mesh booleans with signatures matching :mod:`trimesh.boolean`.

The implementation operates on closed, consistently wound triangle meshes and
does not require Blender or an external boolean engine.
"""

from __future__ import annotations

import math

import numpy as np

from .base import Trimesh

_EPSILON = 1.0e-8
_COPLANAR = 0
_FRONT = 1
_BACK = 2
_SPANNING = 3


def _dot3(first, second):
    return (
        first[0] * second[0]
        + first[1] * second[1]
        + first[2] * second[2]
    )


def _cross3(first, second):
    return np.array(
        (
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        ),
        dtype=np.float64,
    )


class _Vertex:
    __slots__ = ("position",)

    def __init__(self, position):
        self.position = np.asarray(position, dtype=np.float64)

    def clone(self):
        return _Vertex(self.position.copy())

    def interpolate(self, other, amount):
        return _Vertex(self.position + (other.position - self.position) * amount)


class _Plane:
    __slots__ = ("normal", "offset")

    def __init__(self, normal, offset):
        self.normal = np.asarray(normal, dtype=np.float64)
        self.offset = float(offset)

    @classmethod
    def from_vertices(cls, vertices):
        origin = vertices[0].position
        for index in range(2, len(vertices)):
            normal = _cross3(
                vertices[index - 1].position - origin,
                vertices[index].position - origin,
            )
            length2 = _dot3(normal, normal)
            if length2 > _EPSILON * _EPSILON:
                normal /= math.sqrt(length2)
                return cls(normal, _dot3(normal, origin))
        return None

    def clone(self):
        return _Plane(self.normal.copy(), self.offset)

    def flip(self):
        self.normal *= -1.0
        self.offset *= -1.0

    def split_polygon(
        self,
        polygon,
        coplanar_front,
        coplanar_back,
        front,
        back,
    ):
        types = []
        polygon_type = _COPLANAR
        for vertex in polygon.vertices:
            distance = _dot3(self.normal, vertex.position) - self.offset
            kind = (
                _BACK
                if distance < -_EPSILON
                else _FRONT
                if distance > _EPSILON
                else _COPLANAR
            )
            polygon_type |= kind
            types.append(kind)

        if polygon_type == _COPLANAR:
            target = (
                coplanar_front
                if _dot3(self.normal, polygon.plane.normal) > 0.0
                else coplanar_back
            )
            target.append(polygon)
        elif polygon_type == _FRONT:
            front.append(polygon)
        elif polygon_type == _BACK:
            back.append(polygon)
        else:
            front_vertices = []
            back_vertices = []
            count = len(polygon.vertices)
            for index, first in enumerate(polygon.vertices):
                second = polygon.vertices[(index + 1) % count]
                first_type = types[index]
                second_type = types[(index + 1) % count]
                if first_type != _BACK:
                    front_vertices.append(first)
                if first_type != _FRONT:
                    back_vertices.append(
                        first.clone() if first_type != _BACK else first
                    )
                if (first_type | second_type) == _SPANNING:
                    direction = second.position - first.position
                    amount = (
                        self.offset - _dot3(self.normal, first.position)
                    ) / _dot3(self.normal, direction)
                    vertex = first.interpolate(second, amount)
                    front_vertices.append(vertex)
                    back_vertices.append(vertex.clone())
            if len(front_vertices) >= 3:
                split = _Polygon(front_vertices)
                if split.plane is not None:
                    front.append(split)
            if len(back_vertices) >= 3:
                split = _Polygon(back_vertices)
                if split.plane is not None:
                    back.append(split)


class _Polygon:
    __slots__ = ("vertices", "plane")

    def __init__(self, vertices):
        self.vertices = vertices
        self.plane = _Plane.from_vertices(vertices)

    def clone(self):
        return _Polygon([vertex.clone() for vertex in self.vertices])

    def flip(self):
        self.vertices.reverse()
        if self.plane is not None:
            self.plane.flip()


class _Node:
    __slots__ = ("plane", "front", "back", "polygons")

    def __init__(self, polygons=None):
        self.plane = None
        self.front = None
        self.back = None
        self.polygons = []
        if polygons:
            self.build(polygons)

    def clone(self):
        node = _Node()
        node.plane = self.plane.clone() if self.plane is not None else None
        node.front = self.front.clone() if self.front is not None else None
        node.back = self.back.clone() if self.back is not None else None
        node.polygons = [polygon.clone() for polygon in self.polygons]
        return node

    def invert(self):
        for polygon in self.polygons:
            polygon.flip()
        if self.plane is not None:
            self.plane.flip()
        if self.front is not None:
            self.front.invert()
        if self.back is not None:
            self.back.invert()
        self.front, self.back = self.back, self.front

    def clip_polygons(self, polygons):
        if self.plane is None:
            return list(polygons)
        front = []
        back = []
        for polygon in polygons:
            self.plane.split_polygon(polygon, front, back, front, back)
        if self.front is not None:
            front = self.front.clip_polygons(front)
        if self.back is not None:
            back = self.back.clip_polygons(back)
        else:
            back = []
        return front + back

    def clip_to(self, other):
        self.polygons = other.clip_polygons(self.polygons)
        if self.front is not None:
            self.front.clip_to(other)
        if self.back is not None:
            self.back.clip_to(other)

    def all_polygons(self):
        polygons = list(self.polygons)
        if self.front is not None:
            polygons.extend(self.front.all_polygons())
        if self.back is not None:
            polygons.extend(self.back.all_polygons())
        return polygons

    def build(self, polygons):
        if not polygons:
            return
        if self.plane is None:
            self.plane = polygons[0].plane.clone()
        front = []
        back = []
        for polygon in polygons:
            self.plane.split_polygon(
                polygon, self.polygons, self.polygons, front, back
            )
        if front:
            if self.front is None:
                self.front = _Node()
            self.front.build(front)
        if back:
            if self.back is None:
                self.back = _Node()
            self.back.build(back)


def _polygons(mesh):
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    result = []
    for face in faces:
        polygon = _Polygon([_Vertex(vertices[index].copy()) for index in face])
        if polygon.plane is not None:
            result.append(polygon)
    return result


def _to_mesh(polygons):
    vertices = []
    faces = []
    lookup = {}

    def vertex_index(position):
        key = (
            round(position[0] / _EPSILON),
            round(position[1] / _EPSILON),
            round(position[2] / _EPSILON),
        )
        existing = lookup.get(key)
        if existing is not None:
            return existing
        index = len(vertices)
        lookup[key] = index
        vertices.append(np.asarray(position, dtype=np.float64))
        return index

    for polygon in polygons:
        if len(polygon.vertices) < 3:
            continue
        for vertex in polygon.vertices:
            vertex_index(vertex.position)
    candidates = np.asarray(vertices, dtype=np.float64)

    for polygon in polygons:
        if len(polygon.vertices) < 3:
            continue
        boundary = []
        boundary_indices = []
        for offset, first_vertex in enumerate(polygon.vertices):
            first = first_vertex.position
            second = polygon.vertices[(offset + 1) % len(polygon.vertices)].position
            edge = second - first
            length2 = _dot3(edge, edge)
            if length2 <= _EPSILON * _EPSILON:
                continue
            delta = candidates - first
            amounts = (delta @ edge) / length2
            valid = (amounts >= -_EPSILON) & (amounts < 1.0 - _EPSILON)
            projected_delta = delta[valid] - amounts[valid, None] * edge
            on_edge = np.einsum(
                "ij,ij->i", projected_delta, projected_delta
            ) <= (2.0 * _EPSILON) ** 2
            edge_indices = np.flatnonzero(valid)[on_edge]
            order = np.argsort(amounts[edge_indices], kind="stable")
            for point_index in edge_indices[order]:
                point = candidates[point_index]
                if (
                    not boundary
                    or _dot3(point - boundary[-1], point - boundary[-1])
                    > _EPSILON * _EPSILON
                ):
                    boundary.append(point)
                    boundary_indices.append(point_index)
        if len(boundary) < 3:
            continue
        center = np.mean(boundary, axis=0)
        center_index = vertex_index(center)
        for offset, first in enumerate(boundary_indices):
            second = boundary_indices[(offset + 1) % len(boundary_indices)]
            if center_index == first or first == second or second == center_index:
                continue
            a, b, c = vertices[center_index], vertices[first], vertices[second]
            cross = _cross3(b - a, c - a)
            if _dot3(cross, cross) <= _EPSILON * _EPSILON:
                continue
            faces.append((center_index, first, second))
    return Trimesh(
        np.asarray(vertices, dtype=np.float64).reshape((-1, 3)),
        np.asarray(faces, dtype=np.int64).reshape((-1, 3)),
        process=True,
    )


def _pair(first, second, operation):
    a = _Node(_polygons(first))
    b = _Node(_polygons(second))
    if operation == "union":
        a.clip_to(b)
        b.clip_to(a)
        b.invert()
        b.clip_to(a)
        b.invert()
        a.build(b.all_polygons())
    elif operation == "intersection":
        a.invert()
        b.clip_to(a)
        b.invert()
        a.clip_to(b)
        b.clip_to(a)
        a.build(b.all_polygons())
        a.invert()
    elif operation == "difference":
        a.invert()
        a.clip_to(b)
        b.clip_to(a)
        b.invert()
        b.clip_to(a)
        b.invert()
        a.build(b.all_polygons())
        a.invert()
    else:
        raise ValueError(f"unknown boolean operation {operation!r}")
    return _to_mesh(a.all_polygons())


def _boolean(meshes, operation, engine=None, check_volume=True, **kwargs):
    del kwargs
    if engine not in (None, "mojo", "bsp"):
        raise ValueError("engine must be None, 'mojo', or 'bsp'")
    meshes = list(meshes)
    if not meshes:
        raise ValueError("at least one mesh is required")
    if check_volume:
        for mesh in meshes:
            if not bool(mesh.is_watertight) or float(mesh.volume) <= 0.0:
                raise ValueError("Not all meshes are volumes!")
    result = Trimesh(meshes[0].vertices, meshes[0].faces, process=False)
    for mesh in meshes[1:]:
        result = _pair(result, mesh, operation)
    return result


def union(meshes, engine=None, check_volume=True, **kwargs):
    """Compute the boolean union of a sequence of meshes."""
    return _boolean(meshes, "union", engine, check_volume, **kwargs)


def intersection(meshes, engine=None, check_volume=True, **kwargs):
    """Compute the boolean intersection of a sequence of meshes."""
    return _boolean(meshes, "intersection", engine, check_volume, **kwargs)


def difference(meshes, engine=None, check_volume=True, **kwargs):
    """Compute ``meshes[0] - meshes[1:]``."""
    return _boolean(meshes, "difference", engine, check_volume, **kwargs)
