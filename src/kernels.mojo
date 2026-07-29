"""Compute kernels exported through a C ABI.

All arrays are contiguous row-major buffers owned by Python. Addresses cross
the ABI as `Int` so exported functions remain non-parametric.
"""

from std.math import sqrt
from std.sys.info import simd_width_of

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


@export("mt_face_areas")
def mt_face_areas(
    vertices_addr: Int,
    faces_addr: Int,
    areas_addr: Int,
    face_count: Int,
) abi("C"):
    var vertices = fp(vertices_addr)
    var faces = ip(faces_addr)
    var areas = fp(areas_addr)
    for f in range(face_count):
        var ia = Int(faces[f * 3]) * 3
        var ib = Int(faces[f * 3 + 1]) * 3
        var ic = Int(faces[f * 3 + 2]) * 3
        var abx = vertices[ib] - vertices[ia]
        var aby = vertices[ib + 1] - vertices[ia + 1]
        var abz = vertices[ib + 2] - vertices[ia + 2]
        var acx = vertices[ic] - vertices[ia]
        var acy = vertices[ic + 1] - vertices[ia + 1]
        var acz = vertices[ic + 2] - vertices[ia + 2]
        var cx = aby * acz - abz * acy
        var cy = abz * acx - abx * acz
        var cz = abx * acy - aby * acx
        areas[f] = 0.5 * sqrt(cx * cx + cy * cy + cz * cz)


@export("mt_sample_surface")
def mt_sample_surface(
    vertices_addr: Int,
    faces_addr: Int,
    cumulative_addr: Int,
    picks_addr: Int,
    barycentric_addr: Int,
    points_addr: Int,
    indices_addr: Int,
    face_count: Int,
    sample_count: Int,
) abi("C"):
    var vertices = fp(vertices_addr)
    var faces = ip(faces_addr)
    var cumulative = fp(cumulative_addr)
    var picks = fp(picks_addr)
    var barycentric = fp(barycentric_addr)
    var points = fp(points_addr)
    var indices = ip(indices_addr)

    for s in range(sample_count):
        var lo = 0
        var hi = face_count
        while lo < hi:
            var mid = (lo + hi) // 2
            if cumulative[mid] <= picks[s]:
                lo = mid + 1
            else:
                hi = mid
        var f = lo
        if f >= face_count:
            f = face_count - 1
        indices[s] = Int64(f)

        var u = barycentric[s * 2]
        var v = barycentric[s * 2 + 1]
        if u + v > 1.0:
            u = 1.0 - u
            v = 1.0 - v

        var ia = Int(faces[f * 3]) * 3
        var ib = Int(faces[f * 3 + 1]) * 3
        var ic = Int(faces[f * 3 + 2]) * 3
        for axis in range(3):
            var a = vertices[ia + axis]
            points[s * 3 + axis] = (
                a
                + u * (vertices[ib + axis] - a)
                + v * (vertices[ic + axis] - a)
            )


@export("mt_remove_close")
def mt_remove_close(
    points_addr: Int,
    mask_addr: Int,
    degree_addr: Int,
    point_count: Int,
    radius: Float64,
) abi("C") -> Int:
    var points = fp(points_addr)
    var mask = ip(mask_addr)
    var degree = ip(degree_addr)
    var radius2 = radius * radius
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    var ones = SIMD[DType.int64, W](1)
    var zeros = SIMD[DType.int64, W](0)
    while i + W <= point_count:
        degree.store(i, zeros)
        mask.store(i, ones)
        i += W
    while i < point_count:
        degree[i] = 0
        mask[i] = 1
        i += 1

    for i in range(point_count):
        var x0 = points[i * 3]
        var y0 = points[i * 3 + 1]
        var z0 = points[i * 3 + 2]
        var degree_i: Int64 = 0
        var j = i + 1
        while j + W <= point_count:
            var x = (points + j * 3).strided_load[width=W](3)
            var y = (points + j * 3 + 1).strided_load[width=W](3)
            var z = (points + j * 3 + 2).strided_load[width=W](3)
            var dx = x - x0
            var dy = y - y0
            var dz = z - z0
            var close = (dx * dx + dy * dy + dz * dz).le(radius2)
            var increments = close.cast[DType.int64]()
            degree.store(j, degree.load[width=W](j) + increments)
            degree_i += increments.reduce_add()
            j += W
        while j < point_count:
            var dx = points[j * 3] - x0
            var dy = points[j * 3 + 1] - y0
            var dz = points[j * 3 + 2] - z0
            if dx * dx + dy * dy + dz * dz <= radius2:
                degree_i += 1
                degree[j] += 1
            j += 1
        degree[i] += degree_i

    for i in range(point_count):
        var x0 = points[i * 3]
        var y0 = points[i * 3 + 1]
        var z0 = points[i * 3 + 2]
        var degree_i = degree[i]
        var degree_i_vec = SIMD[DType.int64, W](degree_i)
        var j = i + 1
        while j + W <= point_count:
            var x = (points + j * 3).strided_load[width=W](3)
            var y = (points + j * 3 + 1).strided_load[width=W](3)
            var z = (points + j * 3 + 2).strided_load[width=W](3)
            var dx = x - x0
            var dy = y - y0
            var dz = z - z0
            var close = (dx * dx + dy * dy + dz * dz).le(radius2)
            var remove_i = close & degree_i_vec.ge(degree.load[width=W](j))
            if remove_i.reduce_bit_count() != 0:
                mask[i] = 0
            var remove_j = close & ~remove_i
            mask.store(j, remove_j.select(zeros, mask.load[width=W](j)))
            j += W
        while j < point_count:
            var dx = points[j * 3] - x0
            var dy = points[j * 3 + 1] - y0
            var dz = points[j * 3 + 2] - z0
            if dx * dx + dy * dy + dz * dz <= radius2:
                if degree_i >= degree[j]:
                    mask[i] = 0
                else:
                    mask[j] = 0
            j += 1

    var kept = 0
    i = 0
    while i + W <= point_count:
        kept += Int(mask.load[width=W](i).reduce_add())
        i += W
    while i < point_count:
        kept += Int(mask[i])
        i += 1
    return kept


@export("mt_ray_hits")
def mt_ray_hits(
    triangles_addr: Int,
    origins_addr: Int,
    directions_addr: Int,
    triangle_index_addr: Int,
    ray_index_addr: Int,
    distance_addr: Int,
    capacity: Int,
    triangle_count: Int,
    ray_count: Int,
    multiple_hits: Int,
) abi("C") -> Int:
    var triangles = fp(triangles_addr)
    var origins = fp(origins_addr)
    var directions = fp(directions_addr)
    var triangle_index = ip(triangle_index_addr)
    var ray_index = ip(ray_index_addr)
    var distance = fp(distance_addr)
    var hit_count = 0

    for r in range(ray_count):
        var ox = origins[r * 3]
        var oy = origins[r * 3 + 1]
        var oz = origins[r * 3 + 2]
        var dx = directions[r * 3]
        var dy = directions[r * 3 + 1]
        var dz = directions[r * 3 + 2]
        var direction2 = dx * dx + dy * dy + dz * dz
        var best_t = 1.7976931348623157e308
        var best_triangle = -1

        for t in range(triangle_count):
            var base = t * 9
            var ax = triangles[base]
            var ay = triangles[base + 1]
            var az = triangles[base + 2]
            var e1x = triangles[base + 3] - ax
            var e1y = triangles[base + 4] - ay
            var e1z = triangles[base + 5] - az
            var e2x = triangles[base + 6] - ax
            var e2y = triangles[base + 7] - ay
            var e2z = triangles[base + 8] - az

            var px = dy * e2z - dz * e2y
            var py = dz * e2x - dx * e2z
            var pz = dx * e2y - dy * e2x
            var det = e1x * px + e1y * py + e1z * pz
            if abs(det) <= 1.0e-15:
                continue
            var inv_det = 1.0 / det
            var sx = ox - ax
            var sy = oy - ay
            var sz = oz - az
            var u = (sx * px + sy * py + sz * pz) * inv_det
            if u < -1.0e-8 or u > 1.00000001:
                continue
            var qx = sy * e1z - sz * e1y
            var qy = sz * e1x - sx * e1z
            var qz = sx * e1y - sy * e1x
            var v = (dx * qx + dy * qy + dz * qz) * inv_det
            if v < -1.0e-8 or u + v > 1.00000001:
                continue
            var hit_t = (e2x * qx + e2y * qy + e2z * qz) * inv_det
            if hit_t * direction2 <= -1.0e-6:
                continue

            if multiple_hits != 0:
                if hit_count < capacity:
                    triangle_index[hit_count] = Int64(t)
                    ray_index[hit_count] = Int64(r)
                    distance[hit_count] = hit_t
                hit_count += 1
            elif hit_t < best_t:
                best_t = hit_t
                best_triangle = t

        if multiple_hits == 0 and best_triangle >= 0:
            if hit_count < capacity:
                triangle_index[hit_count] = Int64(best_triangle)
                ray_index[hit_count] = Int64(r)
                distance[hit_count] = best_t
            hit_count += 1
    return hit_count
