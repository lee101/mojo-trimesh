import numpy as np
import pytest
import trimesh

import mojotrimesh as mt


@pytest.fixture(scope="module")
def rays():
    mesh = trimesh.creation.icosphere(subdivisions=2)
    random = np.random.default_rng(4)
    origins = random.normal(size=(200, 3))
    origins *= (2.0 / np.linalg.norm(origins, axis=1))[:, None]
    directions = -origins + random.normal(scale=0.1, size=origins.shape)
    return mesh, origins, directions


def _hit_map(result):
    triangles, rays, locations = result
    return {
        (int(ray), int(triangle)): location
        for triangle, ray, location in zip(triangles, rays, locations)
    }


@pytest.mark.parametrize("multiple_hits", [True, False])
def test_ray_triangle_id_parity(rays, multiple_hits):
    mesh, origins, directions = rays
    ours = mt.ray.ray_triangle_id(
        mesh.triangles, origins, directions, multiple_hits=multiple_hits
    )
    theirs = trimesh.ray.ray_triangle.ray_triangle_id(
        mesh.triangles, origins, directions, multiple_hits=multiple_hits
    )
    ours_map = _hit_map(ours)
    theirs_map = _hit_map(theirs)
    assert ours_map.keys() == theirs_map.keys()
    for key in ours_map:
        assert np.allclose(ours_map[key], theirs_map[key], atol=2e-14)


def test_ray_triangle_no_hits_and_validation():
    triangles = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=float)
    result = mt.ray.ray_triangle_id(
        triangles, [[0, 0, 1]], [[0, 0, 1]]
    )
    assert result[0].shape == (0,)
    assert result[1].shape == (0,)
    assert result[2].shape == (0, 3)
    with pytest.raises(ValueError):
        mt.ray.ray_triangle_id(triangles, [[0, 0, 1]], [[0, 0, 0]])
    with pytest.raises(ValueError, match="finite"):
        mt.ray.ray_triangle_id(triangles, [[0, 0, np.nan]], [[0, 0, 1]])
    with pytest.raises(ValueError, match="finite"):
        mt.ray.ray_triangle_id(
            np.full((1, 3, 3), np.inf), [[0, 0, 1]], [[0, 0, -1]]
        )


def test_intersects_location_deduplicates_edges():
    box = trimesh.creation.box()
    mesh = mt.Trimesh(box.vertices, box.faces)
    origins = np.array([[-2, 0, 0], [0, 0, 0], [2, 2, 2]], dtype=float)
    directions = np.array([[1, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=float)
    ours = mt.ray.RayMeshIntersector(mesh).intersects_location(origins, directions)
    theirs = trimesh.ray.ray_triangle.RayMeshIntersector(box).intersects_location(
        origins, directions
    )
    assert np.allclose(ours[0], theirs[0])
    assert np.array_equal(ours[1], theirs[1])
    assert np.array_equal(ours[2], theirs[2])


def test_intersects_first_and_any(rays):
    source, origins, directions = rays
    mesh = mt.Trimesh(source.vertices, source.faces)
    ours = mt.ray.RayMeshIntersector(mesh)
    theirs = trimesh.ray.ray_triangle.RayMeshIntersector(source)
    assert np.array_equal(
        ours.intersects_first(origins, directions),
        theirs.intersects_first(origins, directions),
    )
    assert np.array_equal(
        ours.intersects_any(origins, directions),
        theirs.intersects_any(origins, directions),
    )


def test_contains_points_parity():
    box = trimesh.creation.box()
    mesh = mt.Trimesh(box.vertices, box.faces)
    points = np.random.default_rng(2).uniform(-1.0, 1.0, (1000, 3))
    assert np.array_equal(mesh.contains(points), box.contains(points))
