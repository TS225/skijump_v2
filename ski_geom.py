"""Small mesh-building helpers shared by the scenery and the skier (Panda3D).

World axes: x = lateral (right +), y = along the hill (jump direction +), z = up.
"""

from __future__ import annotations

import math

from panda3d.core import (
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    NodePath,
)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def smoothstep(x: float) -> float:
    x = clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def make_box(half: float = 0.5, name: str = "box") -> NodePath:
    """Lit cube of edge 2*half (use set_scale for other proportions)."""
    vdata = GeomVertexData(name, GeomVertexFormat.get_v3n3(), Geom.UH_static)
    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    tris = GeomTriangles(Geom.UH_static)
    axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    index = 0
    for a in range(3):
        n, u, v = axes[a], axes[(a + 1) % 3], axes[(a + 2) % 3]
        for sign in (-1.0, 1.0):
            nn = tuple(c * sign for c in n)
            corners = [tuple(nn[i] * half + su * u[i] * half + sv * v[i] * half for i in range(3))
                       for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            e1 = tuple(corners[1][i] - corners[0][i] for i in range(3))
            e2 = tuple(corners[2][i] - corners[0][i] for i in range(3))
            cross = (e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2],
                     e1[0] * e2[1] - e1[1] * e2[0])
            if sum(cross[i] * nn[i] for i in range(3)) < 0:       # counter-clockwise from outside
                corners.reverse()
            for c in corners:
                vertex.add_data3(*c)
                normal.add_data3(*nn)
            tris.add_vertices(index, index + 1, index + 2)
            tris.add_vertices(index, index + 2, index + 3)
            index += 4
    geom = Geom(vdata)
    geom.add_primitive(tris)
    node = GeomNode(name)
    node.add_geom(geom)
    return NodePath(node)


class MeshBuilder:
    """Collects coloured, textured, lit triangles into one GeomNode."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.vdata = GeomVertexData(name, GeomVertexFormat.get_v3n3c4t2(), Geom.UH_static)
        self.vw = GeomVertexWriter(self.vdata, "vertex")
        self.nw = GeomVertexWriter(self.vdata, "normal")
        self.cw = GeomVertexWriter(self.vdata, "color")
        self.tw = GeomVertexWriter(self.vdata, "texcoord")
        self.tris = GeomTriangles(Geom.UH_static)
        self.count = 0

    def vertex(self, p, n, color, uv=(0.0, 0.0)) -> int:
        self.vw.add_data3(*p)
        self.nw.add_data3(*n)
        self.cw.add_data4(*color)
        self.tw.add_data2(*uv)
        self.count += 1
        return self.count - 1

    def triangle(self, a: int, b: int, c: int) -> None:
        self.tris.add_vertices(a, b, c)

    def quad(self, p0, p1, p2, p3, n, color, uvs=((0, 0), (1, 0), (1, 1), (0, 1))) -> None:
        i = [self.vertex(p, n, color, uv) for p, uv in zip((p0, p1, p2, p3), uvs)]
        self.triangle(i[0], i[1], i[2])
        self.triangle(i[0], i[2], i[3])

    def ribbon(self, pts, half_width: float, color, l_center: float = 0.0, z_off: float = 0.0,
               uv_scale: float = 6.0) -> None:
        """A strip along a path of (d, h) points, half_width to each side of lateral l_center."""
        if len(pts) < 2:
            return
        idx = []
        run = 0.0
        for i, (d, h) in enumerate(pts):
            if i:
                run += math.hypot(d - pts[i - 1][0], h - pts[i - 1][1])
            a = pts[max(i - 1, 0)]
            b = pts[min(i + 1, len(pts) - 1)]
            td, th = b[0] - a[0], b[1] - a[1]
            ln = math.hypot(td, th) or 1.0
            n = (0.0, -th / ln, td / ln)
            left = self.vertex((l_center - half_width, d, h + z_off), n, color,
                               ((l_center - half_width) / uv_scale, run / uv_scale))
            right = self.vertex((l_center + half_width, d, h + z_off), n, color,
                                ((l_center + half_width) / uv_scale, run / uv_scale))
            idx.append((left, right))
        for (l0, r0), (l1, r1) in zip(idx, idx[1:]):
            self.triangle(l0, r0, l1)
            self.triangle(r0, r1, l1)

    def wall(self, pts, l: float, height: float, color, side: float = 1.0, uv_scale: float = 6.0,
             z_off: float = 0.0) -> None:
        """A vertical panel along a path of (d, h) points at lateral l; side = facing direction (+1/-1 in x)."""
        run = 0.0
        prev = None
        for i, (d, h) in enumerate(pts):
            if i:
                run += math.hypot(d - pts[i - 1][0], h - pts[i - 1][1])
            n = (side, 0.0, 0.0)
            bot = self.vertex((l, d, h + z_off), n, color, (run / uv_scale, 0.0))
            top = self.vertex((l, d, h + z_off + height), n, color, (run / uv_scale, 1.0))
            if prev is not None:
                self.triangle(prev[0], bot, prev[1])
                self.triangle(bot, top, prev[1])
            prev = (bot, top)

    def box(self, center, size, color) -> None:
        cx, cy, cz = center
        hx, hy, hz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
        faces = (
            ((1, 0, 0), ((hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz), (hx, -hy, hz))),
            ((-1, 0, 0), ((-hx, hy, -hz), (-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz))),
            ((0, 1, 0), ((hx, hy, -hz), (-hx, hy, -hz), (-hx, hy, hz), (hx, hy, hz))),
            ((0, -1, 0), ((-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz))),
            ((0, 0, 1), ((-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz))),
            ((0, 0, -1), ((-hx, hy, -hz), (hx, hy, -hz), (hx, -hy, -hz), (-hx, -hy, -hz))),
        )
        for n, corners in faces:
            self.quad(*[(cx + c[0], cy + c[1], cz + c[2]) for c in corners], n, color)

    def cone(self, x: float, y: float, z: float, radius: float, height: float, color_base,
             color_top=None, sides: int = 8) -> None:
        color_top = color_top or color_base
        apex = self.vertex((x, y, z + height), (0.0, 0.0, 1.0), color_top, (0.5, 1.0))
        ring = []
        for i in range(sides):
            a = 2 * math.pi * i / sides
            nx, ny = math.cos(a), math.sin(a)
            ring.append(self.vertex((x + radius * nx, y + radius * ny, z), (nx * 0.866, ny * 0.866, 0.5),
                                    color_base, (i / sides, 0.0)))
        for i in range(sides):
            self.triangle(apex, ring[i], ring[(i + 1) % sides])

    def grid(self, d_nodes, l_nodes, height, color_at, uv_scale: float = 14.0) -> None:
        """Terrain: a vertex at every (d, l); normals from finite differences."""
        rows = []
        for d in d_nodes:
            row = []
            for l in l_nodes:
                z = height(d, l)
                dz_dd = (height(d + 1.0, l) - height(d - 1.0, l)) / 2.0
                dz_dl = (height(d, l + 1.0) - height(d, l - 1.0)) / 2.0
                ln = math.sqrt(dz_dd * dz_dd + dz_dl * dz_dl + 1.0)
                row.append(self.vertex((l, d, z), (-dz_dl / ln, -dz_dd / ln, 1.0 / ln),
                                       color_at(d, l, z), (l / uv_scale, d / uv_scale)))
            rows.append(row)
        for r0, r1 in zip(rows, rows[1:]):
            for i in range(len(l_nodes) - 1):
                self.triangle(r0[i], r0[i + 1], r1[i])
                self.triangle(r0[i + 1], r1[i + 1], r1[i])

    def build(self) -> NodePath:
        geom = Geom(self.vdata)
        geom.add_primitive(self.tris)
        node = GeomNode(self.name)
        node.add_geom(geom)
        return NodePath(node)


if __name__ == "__main__":
    print("This file is a library module of the ski jump game.")
    print("Start the game with:   python ski_jump.py")
