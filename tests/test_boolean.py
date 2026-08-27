import numpy as np
import pytest
import trimesh

import mojotrimesh as mt
from mojotrimesh.boolean import _points_on_segment


def overlapping_boxes():
    first = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    transform = trimesh.transformations.rotation_matrix(0.4, [0, 0, 1])
    transform[:3, 3] = [0.5, 0.2, 0.0]
    second = trimesh.creation.box(extents=[2.0, 1.5, 2.0], transform=transform)
    return first, second


@pytest.mark.parametrize("operation", ["union", "intersection", "difference"])
def test_boolean_box_parity(operation):
    meshes = overlapping_boxes()
    ours = getattr(mt.boolean, operation)(meshes)
    theirs = getattr(trimesh.boolean, operation)(meshes)
    assert ours.is_watertight
    assert ours.volume == pytest.approx(theirs.volume, abs=7e-8)
    query = np.random.default_rng(5).uniform([-1.5, -1.5, -1.1], [2, 1.5, 1.1], (500, 3))
    assert np.mean(ours.contains(query) == theirs.contains(query)) > 0.995


@pytest.mark.parametrize("operation", ["intersection", "difference"])
def test_boolean_curved_mesh_parity(operation):
    sphere = trimesh.creation.icosphere(subdivisions=1)
    box = trimesh.creation.box(extents=[1.5, 1.5, 1.5])
    box.apply_translation([0.4, 0.0, 0.0])
    ours = getattr(mt.boolean, operation)([sphere, box])
    theirs = getattr(trimesh.boolean, operation)([sphere, box])
    assert ours.is_watertight
    assert ours.volume == pytest.approx(theirs.volume, abs=2e-8)


def test_boolean_disjoint_union():
    first = trimesh.creation.box()
    second = trimesh.creation.box()
    second.apply_translation([3.0, 0.0, 0.0])
    ours = mt.union([first, second])
    theirs = trimesh.boolean.union([first, second])
    assert ours.is_watertight
    assert ours.volume == pytest.approx(theirs.volume)


def test_boolean_three_mesh_union():
    meshes = []
    for offset in [0.0, 0.5, 1.0]:
        mesh = trimesh.creation.box()
        mesh.apply_translation([offset, 0.0, 0.0])
        meshes.append(mesh)
    ours = mt.union(meshes)
    theirs = trimesh.boolean.union(meshes)
    assert ours.is_watertight
    assert ours.volume == pytest.approx(theirs.volume, abs=1e-8)


def test_boolean_volume_validation():
    triangle = mt.Trimesh(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        [[0, 1, 2]],
        process=False,
    )
    with pytest.raises(ValueError, match="volumes"):
        mt.union([triangle, triangle])
    result = mt.union([triangle], check_volume=False)
    assert len(result.faces) == 1


@pytest.mark.parametrize("count", [1, 3, 4, 5, 9])
def test_points_on_segment_simd_tail(count):
    candidates = np.column_stack(
        (
            np.linspace(1.0, 0.0, count),
            np.zeros(count),
            np.zeros(count),
        )
    )
    candidates = np.ascontiguousarray(candidates, dtype=np.float64)
    amounts = np.empty(count, dtype=np.float64)
    indices = np.empty(count, dtype=np.int64)
    found = _points_on_segment(
        candidates,
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        amounts,
        indices,
    )
    valid = np.flatnonzero(candidates[:, 0] < 1.0)
    expected = valid[np.argsort(candidates[valid, 0], kind="stable")]
    assert found == len(expected)
    assert np.array_equal(indices[:found], expected)
