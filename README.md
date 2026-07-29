# mojo-trimesh

A focused port of the compute-heavy parts of
[trimesh](https://github.com/mikedh/trimesh) to Mojo. It provides surface
sampling, batched ray queries, point containment, and dependency-free mesh
booleans behind a small NumPy API.

The covered names mirror upstream, but this is not a drop-in replacement for
the full package. Code using only the subset below can use
`import mojotrimesh as trimesh`. Functions also accept upstream
`trimesh.Trimesh` instances and other mesh-like objects with `vertices` and
`faces` arrays.

## Covered API

- `Trimesh`: triangle storage plus `triangles`, `area_faces`, `area`,
  `face_normals`, `volume`, `bounds`, `extents`, `is_watertight`, `contains`,
  and `ray`.
- `sample.sample_surface`, `sample_surface_even`, `sample_surface_sphere`,
  `volume_rectangular`, and `volume_mesh`.
- `ray.ray_triangle.ray_triangle_id` and `ray.RayMeshIntersector`, including
  `intersects_id`, `intersects_location`, `intersects_first`, `intersects_any`,
  and `contains_points`.
- `boolean.union`, `intersection`, and `difference` for closed,
  consistently-wound triangle meshes. Operations accept one or more meshes
  and return a watertight `mojotrimesh.Trimesh`.

Seeded surface sampling reproduces trimesh's selected faces and points.
Ray-hit tests reproduce upstream triangle/ray IDs and locations, including
first-hit and shared-edge behavior. Boolean tests compare watertightness,
volume, and point occupancy against trimesh's manifold3d backend.

## Install and build

```bash
pixi install
pixi run build
pixi run test
```

The build task compiles one Mojo unit into
`dist/libmojo-trimesh.so`. Pixi activates `python/` on `PYTHONPATH`; no wheel
installation is required inside the checkout.

## Usage

```python
import numpy as np
import mojotrimesh as trimesh

vertices = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])
faces = np.array([
    [0, 2, 1],
    [0, 1, 3],
    [0, 3, 2],
    [1, 2, 3],
])
mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

points, face_index = trimesh.sample.sample_surface(mesh, 10_000, seed=7)
first = mesh.ray.intersects_first(
    ray_origins=np.array([[2.0, 0.2, 0.2]]),
    ray_directions=np.array([[-1.0, 0.0, 0.0]]),
)

assert points.shape == (10_000, 3)
assert face_index.shape == (10_000,)
assert first.shape == (1,)
```

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux 6.8.0-136-generic. Each library receives identical arrays; one warm-up
excludes shared-library and spatial-index initialization, and the table reports
the best repeated wall time. Relative is `trimesh / mojo-trimesh`.

| case | mojo-trimesh | trimesh | relative |
| --- | ---: | ---: | ---: |
| sample_surface, 1M points / 5,120 faces | 161.70 ms | 529.14 ms | 3.27x faster |
| sample_surface weighted, 1M / 5,120 faces | 159.74 ms | 511.64 ms | 3.20x faster |
| sample_surface_even, 1,000 points | 2.81 ms | 4.04 ms | 1.43x faster |
| intersects_first, 20k rays / 320 faces | 85.43 ms | 2899.47 ms | 33.94x faster |
| intersects_location, 20k rays / 320 faces | 143.42 ms | 3130.61 ms | 21.83x faster |
| contains, 20k points / 320 faces | 91.84 ms | 1713.55 ms | 18.66x faster |
| difference, two rotated boxes | 9.34 ms | 0.47 ms | 0.05x slower |

The remaining slower row is an intentional limitation, not a benchmark
omission. BSP boolean topology is useful without an external engine, but
manifold3d is highly optimized C++ and remains much faster on small meshes.

No GPU path is provided. These kernels operate on caller-owned host arrays;
moving them to a device would add transfer and launch overhead that this
benchmark does not measure.

## How it works

Python passes C-contiguous `float64` vertices and coordinates and `int64` face
indices to a C ABI as integer addresses. Mojo reconstructs mutable pointers,
operates directly on caller-owned row-major buffers, and returns into NumPy
output arrays. There are no cross-language allocations or per-element ctypes
calls.

Surface sampling fuses cumulative-weight search, barycentric reflection, face
gathering, and interpolation in one kernel. Ray queries use a fused
Möller-Trumbore scan across a ray batch; output buffers grow and retry only when
the initial hit estimate is insufficient. Even sampling uses a SIMD close-pair
scan with a scalar remainder for small candidate sets and switches to a
zero-copy cKDTree pair search once index construction pays for itself. This
brute-force ray layout is effective for dense ray batches and moderate meshes,
but its cost remains `rays * triangles`.

Booleans use a BSP constructive-solid-geometry implementation. Polygons are
split against planes, clipped according to the requested set operation, and
triangulated after resolving T-junctions along split edges. Variable-size
topology stays in Python/NumPy; fixed-width area and query work stays in Mojo.

## Not covered

This is not a port of trimesh's scene graph, loaders/exporters, repair,
simplification, registration, remeshing, paths, visuals, or proximity modules.
The ray API accepts the upstream `tree` argument for compatibility but currently
uses no BVH or r-tree, so very large meshes with few rays favor upstream.
Booleans do not preserve material, color, UV, or other face attributes and use
floating-point BSP predicates rather than manifold3d's robust topology. Texture
UV interpolation for `sample_color=True` is not implemented; face colors are.

MIT licensed, copyright Lee Penkman.
