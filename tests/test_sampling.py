import numpy as np
import pytest
import trimesh

import mojotrimesh as mt


@pytest.fixture(scope="module")
def sphere():
    return trimesh.creation.icosphere(subdivisions=2, radius=1.7)


def test_mesh_geometry_matches_trimesh(sphere):
    mesh = mt.Trimesh(sphere.vertices, sphere.faces)
    assert np.array_equal(mesh.triangles, sphere.triangles)
    assert np.allclose(mesh.area_faces, sphere.area_faces)
    assert mesh.area == pytest.approx(sphere.area)
    assert mesh.volume == pytest.approx(sphere.volume)
    assert np.array_equal(mesh.bounds, sphere.bounds)
    assert np.array_equal(mesh.extents, sphere.extents)
    assert mesh.is_watertight
    assert np.allclose(mesh.face_normals, sphere.face_normals)
    copied = mesh.copy()
    copied.vertices[0] += 1.0
    assert not np.array_equal(copied.vertices, mesh.vertices)
    assert mesh.triangles_tree is None
    assert len(mesh) == len(sphere.faces)


def test_mesh_rejects_unsafe_face_indices_and_narrowing():
    vertices = np.eye(3)
    with pytest.raises(ValueError, match="outside"):
        mt.Trimesh(vertices, [[0, 1, 3]], process=False)
    with pytest.raises(ValueError, match="integer"):
        mt.Trimesh(vertices, [[0.0, 1.5, 2.0]], process=False)
    with pytest.raises(OverflowError, match="int64"):
        mt.Trimesh(vertices, np.array([[0, 1, 2**63]], dtype=np.uint64))


def test_external_mesh_is_validated_before_ffi():
    class Mesh:
        vertices = np.eye(3)
        faces = np.array([[0, 1, 99]])

    with pytest.raises(ValueError, match="outside"):
        mt.sample_surface(Mesh(), 1, seed=1)

    mesh = mt.Trimesh(np.eye(3), [[0, 1, 2]], process=False)
    mesh.faces[0, 2] = 99
    with pytest.raises(ValueError, match="outside"):
        _ = mesh.area_faces


@pytest.mark.parametrize("count", [1, 1000])
def test_sample_surface_seeded_exact(sphere, count):
    ours = mt.sample.sample_surface(sphere, count, seed=42)
    theirs = trimesh.sample.sample_surface(sphere, count, seed=42)
    assert np.array_equal(ours[1], theirs[1])
    assert np.allclose(ours[0], theirs[0], atol=2e-15)


def test_sample_surface_weighted_exact(sphere):
    weights = np.linspace(0.01, 2.0, len(sphere.faces))
    ours = mt.sample_surface(sphere, 2000, face_weight=weights, seed=8)
    theirs = trimesh.sample.sample_surface(
        sphere, 2000, face_weight=weights, seed=8
    )
    assert np.array_equal(ours[1], theirs[1])
    assert np.allclose(ours[0], theirs[0], atol=2e-15)


def test_sample_surface_colors(sphere):
    sphere.visual.face_colors = np.arange(len(sphere.faces) * 4, dtype=np.uint8).reshape(
        (-1, 4)
    )
    ours = mt.sample_surface(sphere, 200, sample_color=True, seed=1)
    theirs = trimesh.sample.sample_surface(
        sphere, 200, sample_color=True, seed=1
    )
    assert np.allclose(ours[0], theirs[0])
    assert np.array_equal(ours[1], theirs[1])
    assert np.array_equal(ours[2], theirs[2])


def test_sample_surface_validation(sphere):
    points, faces = mt.sample_surface(sphere, 0, seed=1)
    assert points.shape == (0, 3)
    assert faces.shape == (0,)
    with pytest.raises(ValueError):
        mt.sample_surface(sphere, 2, face_weight=np.ones(3), seed=1)
    with pytest.raises(ValueError):
        mt.sample_surface(sphere, -1, seed=1)


@pytest.mark.parametrize("radius", [None, 0.1, 0.2])
def test_sample_surface_even_exact(sphere, radius):
    ours = mt.sample_surface_even(sphere, 100, radius=radius, seed=4)
    theirs = trimesh.sample.sample_surface_even(
        sphere, 100, radius=radius, seed=4
    )
    assert np.array_equal(ours[1], theirs[1])
    assert np.allclose(ours[0], theirs[0], atol=2e-15)


@pytest.mark.parametrize("count", [1, 2, 3, 169, 171])
def test_sample_surface_even_simd_tail_and_tree_threshold(sphere, count):
    ours = mt.sample_surface_even(sphere, count, radius=0.12, seed=17)
    theirs = trimesh.sample.sample_surface_even(
        sphere, count, radius=0.12, seed=17
    )
    assert np.array_equal(ours[1], theirs[1])
    assert np.allclose(ours[0], theirs[0], atol=2e-15)


def test_sample_surface_sphere_exact():
    np.random.seed(3)
    ours = mt.sample_surface_sphere(500)
    np.random.seed(3)
    theirs = trimesh.sample.sample_surface_sphere(500)
    assert np.allclose(ours, theirs)
    assert np.allclose(np.linalg.norm(ours, axis=1), 1.0)


def test_volume_rectangular_exact():
    transform = trimesh.transformations.rotation_matrix(0.3, [1, 2, 3])
    transform[:3, 3] = [4.0, -2.0, 1.0]
    np.random.seed(7)
    ours = mt.volume_rectangular([2.0, 3.0, 4.0], 200, transform)
    np.random.seed(7)
    theirs = trimesh.sample.volume_rectangular(
        [2.0, 3.0, 4.0], 200, transform
    )
    assert np.allclose(ours, theirs)


def test_volume_mesh_exact_for_box():
    box = trimesh.creation.box(extents=[2.0, 3.0, 4.0])
    mesh = mt.Trimesh(box.vertices, box.faces)
    np.random.seed(11)
    ours = mt.volume_mesh(mesh, 500)
    np.random.seed(11)
    theirs = trimesh.sample.volume_mesh(box, 500)
    assert np.array_equal(ours, theirs)
