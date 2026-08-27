"""Load the compiled Mojo kernels and declare their C signatures."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJOTRIMESH_LIB") or os.path.join(
    ROOT, "dist", "libmojo-trimesh.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mt_face_areas": ([I, I, I, I], None),
    "mt_points_on_segment": ([I, I, I, I] + [F] * 7, I),
    "mt_sample_surface": ([I] * 9, None),
    "mt_remove_close": ([I, I, I, I, F], I),
    "mt_ray_hits": ([I] * 10, I),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if os.environ.get("MOJOTRIMESH_LIB") and os.path.exists(LIB) and not force:
        return LIB
    sources = [
        os.path.join(path, name)
        for path, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        if os.path.getmtime(LIB) >= max(map(os.path.getmtime, sources)):
            return LIB
    mojo = shutil.which("mojo")
    if mojo is None:
        raise BuildError("mojo not found; run inside `pixi run` or set MOJOTRIMESH_LIB")
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    command = [
        mojo,
        "build",
        "--emit",
        "shared-lib",
        os.path.join(SRC, "kernels.mojo"),
        "-o",
        LIB,
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        raise BuildError((result.stderr or result.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def f64(value, *, copy: bool = False) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype.kind == "c":
        raise TypeError("complex values cannot be represented as float64")
    if copy:
        return np.array(value, dtype=np.float64, order="C", copy=True)
    return np.ascontiguousarray(value, dtype=np.float64)


def i64(value, *, copy: bool = False) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype.kind not in "iub":
        if source.dtype.kind != "f":
            raise TypeError("indices must contain real integer values")
        if not np.isfinite(source).all() or not np.equal(source, np.trunc(source)).all():
            raise ValueError("indices must contain finite integer values")
        limit = np.iinfo(np.int64)
        if np.any(source < limit.min) or np.any(source >= float(2**63)):
            raise OverflowError("indices do not fit in int64")
    elif source.dtype.kind == "u" and source.size:
        if source.max() > np.iinfo(np.int64).max:
            raise OverflowError("indices do not fit in int64")
    if copy:
        return np.array(value, dtype=np.int64, order="C", copy=True)
    return np.ascontiguousarray(value, dtype=np.int64)


def addr(array: np.ndarray, dtype) -> int:
    if not isinstance(array, np.ndarray) or not array.flags.c_contiguous:
        raise TypeError("FFI buffers must be C-contiguous NumPy arrays")
    expected = np.dtype(dtype)
    if array.dtype != expected:
        raise TypeError(f"FFI buffer must have dtype {expected}, got {array.dtype}")
    address = int(array.ctypes.data)
    if array.size and address == 0:
        raise RuntimeError("NumPy returned a null address for a non-empty buffer")
    return address
