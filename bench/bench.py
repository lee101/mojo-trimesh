"""Benchmarks against trimesh on identical geometry and random inputs."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojotrimesh as mt  # noqa: E402


def timeit(function, repeat=5):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def machine_name():
    cpu = platform.processor()
    if cpu.lower() in ("", "x86_64", "amd64") and os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo", encoding="utf8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    return f"{cpu or platform.machine()}, {platform.system()} {platform.release()}"


def surface_case():
    mesh = trimesh.creation.icosphere(subdivisions=4)
    return (
        lambda: mt.sample_surface(mesh, 1_000_000, seed=7),
        lambda: trimesh.sample.sample_surface(mesh, 1_000_000, seed=7),
    )


def weighted_surface_case():
    mesh = trimesh.creation.icosphere(subdivisions=4)
    weights = np.linspace(0.1, 2.0, len(mesh.faces))
    return (
        lambda: mt.sample_surface(mesh, 1_000_000, face_weight=weights, seed=7),
        lambda: trimesh.sample.sample_surface(
            mesh, 1_000_000, face_weight=weights, seed=7
        ),
    )


def even_surface_case():
    mesh = trimesh.creation.icosphere(subdivisions=3)
    return (
        lambda: mt.sample_surface_even(mesh, 1000, seed=7),
        lambda: trimesh.sample.sample_surface_even(mesh, 1000, seed=7),
    )


def ray_case(method):
    source = trimesh.creation.icosphere(subdivisions=2)
    mesh = mt.Trimesh(source.vertices, source.faces)
    random = np.random.default_rng(8)
    origins = random.normal(size=(20_000, 3))
    origins *= (2.0 / np.linalg.norm(origins, axis=1))[:, None]
    directions = -origins + random.normal(scale=0.15, size=origins.shape)
    ours = mt.ray.RayMeshIntersector(mesh)
    theirs = trimesh.ray.ray_triangle.RayMeshIntersector(source)
    ours_function = lambda: getattr(ours, method)(origins, directions)
    theirs_function = lambda: getattr(theirs, method)(origins, directions)
    ours_function()
    theirs_function()
    return ours_function, theirs_function


def contains_case():
    source = trimesh.creation.icosphere(subdivisions=2)
    mesh = mt.Trimesh(source.vertices, source.faces)
    points = np.random.default_rng(9).uniform(-1.2, 1.2, (20_000, 3))
    return lambda: mesh.contains(points), lambda: source.contains(points)


def boolean_case():
    first = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    transform = trimesh.transformations.rotation_matrix(0.4, [0, 0, 1])
    transform[:3, 3] = [0.5, 0.2, 0.0]
    second = trimesh.creation.box(extents=[2.0, 1.5, 2.0], transform=transform)
    return (
        lambda: mt.boolean.difference([first, second]),
        lambda: trimesh.boolean.difference([first, second]),
    )


CASES = [
    ("sample_surface, 1M points / 5,120 faces", surface_case, 5),
    ("sample_surface weighted, 1M / 5,120 faces", weighted_surface_case, 5),
    ("sample_surface_even, 1,000 points", even_surface_case, 3),
    ("intersects_first, 20k rays / 320 faces", lambda: ray_case("intersects_first"), 3),
    (
        "intersects_location, 20k rays / 320 faces",
        lambda: ray_case("intersects_location"),
        3,
    ),
    ("contains, 20k points / 320 faces", contains_case, 3),
    ("difference, two rotated boxes", boolean_case, 5),
]


def main():
    print(f"Machine: {machine_name()}")
    print()
    print("| case | mojo-trimesh | trimesh | relative |")
    print("| --- | ---: | ---: | ---: |")
    for name, setup, repeat in CASES:
        ours, theirs = setup()
        ours()
        theirs()
        mojo_seconds = timeit(ours, repeat)
        upstream_seconds = timeit(theirs, repeat)
        relative = upstream_seconds / mojo_seconds
        label = "faster" if relative >= 1.0 else "slower"
        print(
            f"| {name} | {mojo_seconds * 1000:.2f} ms | "
            f"{upstream_seconds * 1000:.2f} ms | {relative:.2f}x {label} |"
        )


if __name__ == "__main__":
    main()
