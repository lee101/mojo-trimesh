"""Mojo kernels for the compute-heavy core of triangle-mesh queries."""

__version__ = "0.1.0"

from . import boolean, ray, sample
from .base import Trimesh
from .boolean import difference, intersection, union
from .sample import (
    sample_surface,
    sample_surface_even,
    sample_surface_sphere,
    volume_mesh,
    volume_rectangular,
)

__all__ = [
    "Trimesh",
    "boolean",
    "difference",
    "intersection",
    "ray",
    "sample",
    "sample_surface",
    "sample_surface_even",
    "sample_surface_sphere",
    "union",
    "volume_mesh",
    "volume_rectangular",
]
