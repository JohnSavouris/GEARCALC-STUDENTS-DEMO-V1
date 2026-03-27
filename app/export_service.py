from __future__ import annotations

import math
from typing import Iterable


def _dedupe(points: list[tuple[float, float]], tol: float = 1e-9) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for x, y in points:
        if not out:
            out.append((float(x), float(y)))
            continue
        px, py = out[-1]
        if abs(px - x) > tol or abs(py - y) > tol:
            out.append((float(x), float(y)))
    return out


def _polygon_area(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return 0.5 * area


def _polygon_centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    area = _polygon_area(points)
    if abs(area) < 1e-12:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (sum(xs) / max(1, len(xs)), sum(ys) / max(1, len(ys)))

    cx = 0.0
    cy = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        cross = x1 * y2 - x2 * y1
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    f = 1.0 / (6.0 * area)
    return cx * f, cy * f


def _normalize_outline(x_mm: Iterable[float], y_mm: Iterable[float]) -> list[tuple[float, float]]:
    pts = [(float(x), float(y)) for x, y in zip(x_mm, y_mm)]
    if len(pts) < 3:
        raise ValueError("Outline must contain at least 3 points.")

    pts = _dedupe(pts)
    if len(pts) >= 2:
        x0, y0 = pts[0]
        x1, y1 = pts[-1]
        if abs(x0 - x1) < 1e-9 and abs(y0 - y1) < 1e-9:
            pts = pts[:-1]

    if len(pts) < 3:
        raise ValueError("Outline became degenerate after cleanup.")

    cx, cy = _polygon_centroid(pts)
    pts = [(x - cx, y - cy) for x, y in pts]

    if _polygon_area(pts) < 0.0:
        pts.reverse()
    return pts


def _safe_name(name: str) -> str:
    raw = (name or "gearcalc_export").strip().replace(" ", "_")
    return "".join(ch for ch in raw if ch.isalnum() or ch in ("_", "-", ".")) or "gearcalc_export"


def _cross_2d(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _point_in_tri(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> bool:
    c1 = _cross_2d(a, b, p)
    c2 = _cross_2d(b, c, p)
    c3 = _cross_2d(c, a, p)
    has_neg = (c1 < -1e-10) or (c2 < -1e-10) or (c3 < -1e-10)
    has_pos = (c1 > 1e-10) or (c2 > 1e-10) or (c3 > 1e-10)
    return not (has_neg and has_pos)


def _triangulate_polygon(points: list[tuple[float, float]]) -> list[tuple[int, int, int]]:
    if len(points) < 3:
        raise ValueError("Need at least 3 points to triangulate polygon.")

    idx = list(range(len(points)))
    tris: list[tuple[int, int, int]] = []
    guard = 0
    while len(idx) > 3:
        ear_found = False
        n = len(idx)
        for k in range(n):
            i_prev = idx[(k - 1) % n]
            i_curr = idx[k]
            i_next = idx[(k + 1) % n]
            a, b, c = points[i_prev], points[i_curr], points[i_next]

            if _cross_2d(a, b, c) <= 1e-12:
                continue

            contains = False
            for j in idx:
                if j in (i_prev, i_curr, i_next):
                    continue
                if _point_in_tri(points[j], a, b, c):
                    contains = True
                    break
            if contains:
                continue

            tris.append((i_prev, i_curr, i_next))
            del idx[k]
            ear_found = True
            break

        if not ear_found:
            raise ValueError("Polygon triangulation failed; outline may be self-intersecting.")
        guard += 1
        if guard > 100000:
            raise ValueError("Polygon triangulation exceeded safety limit.")

    tris.append((idx[0], idx[1], idx[2]))
    return tris


def build_dxf_bytes(name: str, x_mm: Iterable[float], y_mm: Iterable[float]) -> bytes:
    pts = _normalize_outline(x_mm, y_mm)
    safe = _safe_name(name)

    lines = [
        "0", "SECTION",
        "2", "HEADER",
        "9", "$INSUNITS",
        "70", "4",
        "0", "ENDSEC",
        "0", "SECTION",
        "2", "ENTITIES",
        "0", "LWPOLYLINE",
        "8", "GEARCALC",
        "90", str(len(pts)),
        "70", "1",
    ]
    for x, y in pts:
        lines.extend(["10", f"{x:.9f}", "20", f"{y:.9f}"])

    lines.extend([
        "0", "ENDSEC",
        "0", "SECTION",
        "2", "OBJECTS",
        "0", "DICTIONARY",
        "5", "C",
        "330", "0",
        "100", "AcDbDictionary",
        "281", "1",
        "3", safe,
        "350", "0",
        "0", "ENDSEC",
        "0", "EOF",
    ])
    return ("\n".join(lines) + "\n").encode("utf-8")


def _normal(a: tuple[float, float, float], b: tuple[float, float, float], c: tuple[float, float, float]) -> tuple[float, float, float]:
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if norm < 1e-12:
        return 0.0, 0.0, 0.0
    return nx / norm, ny / norm, nz / norm


def _facet_to_lines(a: tuple[float, float, float], b: tuple[float, float, float], c: tuple[float, float, float]) -> list[str]:
    nx, ny, nz = _normal(a, b, c)
    return [
        f"  facet normal {nx:.9e} {ny:.9e} {nz:.9e}",
        "    outer loop",
        f"      vertex {a[0]:.9e} {a[1]:.9e} {a[2]:.9e}",
        f"      vertex {b[0]:.9e} {b[1]:.9e} {b[2]:.9e}",
        f"      vertex {c[0]:.9e} {c[1]:.9e} {c[2]:.9e}",
        "    endloop",
        "  endfacet",
    ]


def build_stl_bytes(name: str, x_mm: Iterable[float], y_mm: Iterable[float], thickness_mm: float) -> bytes:
    pts2d = _normalize_outline(x_mm, y_mm)
    thickness = float(thickness_mm)
    if not math.isfinite(thickness) or thickness <= 0.0:
        raise ValueError("STL thickness must be > 0.")

    z_top = 0.5 * thickness
    z_bot = -0.5 * thickness
    top = [(x, y, z_top) for x, y in pts2d]
    bot = [(x, y, z_bot) for x, y in pts2d]
    top_tris = _triangulate_polygon(pts2d)

    facets: list[str] = [f"solid {_safe_name(name)}"]
    for ia, ib, ic in top_tris:
        facets.extend(_facet_to_lines(top[ia], top[ib], top[ic]))
        facets.extend(_facet_to_lines(bot[ic], bot[ib], bot[ia]))

    n = len(pts2d)
    for i in range(n):
        j = (i + 1) % n
        facets.extend(_facet_to_lines(top[i], bot[i], bot[j]))
        facets.extend(_facet_to_lines(top[i], bot[j], top[j]))

    facets.append(f"endsolid {_safe_name(name)}")
    return ("\n".join(facets) + "\n").encode("utf-8")
