"""The jumper model.

The human figure is loaded from the file ski_jumper_very_basic.obj that lives in the
game's root folder (next to the Python files), the way Panda3D loads any model:

    self.ski_jumper = self.loader.loadModel("ski_jumper_very_basic.obj")
    self.ski_jumper.reparentTo(self.render)

Here the model is parented to the skier's root node instead of render, and that root
node is moved to wherever the jumper is (track, air, landing hill) every frame.

The file is an upright figure made of 8 separate objects (head, torso, two arms, two
legs, two skis; Blender's Y-up export, skis along Z, about 1.32 m tall).  The parts
are re-parented under joints so the figure can crouch on the inrun, stand up for the
jump, lean into the V-style flight position and land with a telemark.

Loading strategy (so the game always starts):
  1. Panda3D's own loader (loader.load_model) - uses the p3assimp plugin of the pip wheel.
     Its result is only trusted if the 8 parts can be recognised.
  2. otherwise this module's own OBJ parser builds the 8 parts from the same file.
  3. if the file is missing, a simple box figure is used.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

from panda3d.core import (
    Filename,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    NodePath,
)

from ski_geom import clamp, make_box

MODEL_FILE = "ski_jumper_very_basic.obj"
MODEL_SCALE = 1.30          # the file's figure is 1.32 m tall -> a 1.72 m athlete, 2.6 m skis
HELMET_COLOR = (0.96, 0.96, 0.98)
SKI_COLOR = (0.98, 0.78, 0.08)

# pose keyframes: p = 1 crouch (inrun), 0.5 upright, 0 flight (V style)
# angles in degrees about the model's X axis, positive = lean forward
POSE_CROUCH = (38.0, 62.0)      # (body lean about the boots, extra bend at the hips)
POSE_UPRIGHT = (3.0, 0.0)
POSE_FLIGHT = (70.0, 8.0)


def pose_angles(p: float) -> tuple[float, float]:
    """(lean, hip bend) for the pose parameter p (1 crouch, 0.5 upright, 0 flight)."""
    p = clamp(p, 0.0, 1.0)
    if p >= 0.5:
        t = (p - 0.5) / 0.5
        a, b = POSE_UPRIGHT, POSE_CROUCH
    else:
        t = p / 0.5
        a, b = POSE_FLIGHT, POSE_UPRIGHT
    return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t


# --------------------------------------------------------------------------
#  OBJ parsing (pure Python)
# --------------------------------------------------------------------------
@dataclass
class ObjMesh:
    name: str
    positions: list[tuple[float, float, float]] = field(default_factory=list)   # one per face corner
    normals: list[tuple[float, float, float]] = field(default_factory=list)
    triangles: list[tuple[int, int, int]] = field(default_factory=list)

    def bounds(self):
        xs = [p[0] for p in self.positions]
        ys = [p[1] for p in self.positions]
        zs = [p[2] for p in self.positions]
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def parse_obj(path: str) -> dict[str, ObjMesh]:
    """Read the objects ('o' / 'g' lines) of a Wavefront OBJ file; faces are triangulated."""
    verts: list[tuple[float, float, float]] = []
    norms: list[tuple[float, float, float]] = []
    meshes: dict[str, ObjMesh] = {}
    current: ObjMesh | None = None

    def pick(idx: int, seq: list) -> int:
        return idx - 1 if idx > 0 else len(seq) + idx

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            tag = parts[0]
            if tag == "v":
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif tag == "vn":
                norms.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif tag in ("o", "g"):
                name = " ".join(parts[1:]) or f"part{len(meshes)}"
                if name not in meshes:
                    meshes[name] = ObjMesh(name)
                current = meshes[name]
            elif tag == "f":
                if current is None:
                    current = meshes.setdefault("mesh", ObjMesh("mesh"))
                corners = []
                for tok in parts[1:]:
                    f = tok.split("/")
                    vi = pick(int(f[0]), verts)
                    ni = pick(int(f[2]), norms) if len(f) > 2 and f[2] else None
                    corners.append((verts[vi], norms[ni] if ni is not None else None))
                base = len(current.positions)
                # fan triangulation; a flat face normal is used where the file has none
                p0, p1, p2 = corners[0][0], corners[1][0], corners[2][0]
                e1 = tuple(p1[i] - p0[i] for i in range(3))
                e2 = tuple(p2[i] - p0[i] for i in range(3))
                fn = (e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2],
                      e1[0] * e2[1] - e1[1] * e2[0])
                ln = math.sqrt(sum(c * c for c in fn)) or 1.0
                fn = tuple(c / ln for c in fn)
                for pos, nrm in corners:
                    current.positions.append(pos)
                    current.normals.append(nrm if nrm is not None else fn)
                for k in range(1, len(corners) - 1):
                    current.triangles.append((base, base + k, base + k + 1))
    return {k: m for k, m in meshes.items() if m.positions}


def classify_parts(bounds: dict) -> dict | None:
    """Recognise the 8 parts of the jumper from their bounding boxes.

    bounds: key -> ((minx, miny, minz), (maxx, maxy, maxz)) in the model's own Y-up frame.
    Returns {'head', 'torso', 'arms': [k, k], 'legs': [k, k], 'skis': [k, k]}; each pair is
    sorted by x (first = smaller x = the jumper's right side when facing +Z).  None if the
    model does not look like the expected figure.
    """
    if len(bounds) != 8:
        return None
    lo = [min(b[0][i] for b in bounds.values()) for i in range(3)]
    hi = [max(b[1][i] for b in bounds.values()) for i in range(3)]
    height, length = hi[1] - lo[1], hi[2] - lo[2]
    if height <= 0 or length <= 0:
        return None

    def ext(k, axis):
        return bounds[k][1][axis] - bounds[k][0][axis]

    def ctr(k, axis):
        return (bounds[k][1][axis] + bounds[k][0][axis]) / 2.0

    keys = list(bounds)
    skis = [k for k in keys if ext(k, 2) >= 0.6 * length and bounds[k][1][1] <= lo[1] + 0.15 * height]
    if len(skis) != 2:
        return None
    rest = [k for k in keys if k not in skis]
    head = max(rest, key=lambda k: bounds[k][1][1])
    if bounds[head][1][1] < lo[1] + 0.9 * height or max(ext(head, i) for i in range(3)) > 0.35 * height:
        return None
    rest.remove(head)
    legs = [k for k in rest if bounds[k][1][1] <= lo[1] + 0.65 * height]
    if len(legs) != 2:
        return None
    rest = [k for k in rest if k not in legs]
    if len(rest) != 3:
        return None
    mid_x = (lo[0] + hi[0]) / 2.0
    torso = min(rest, key=lambda k: abs(ctr(k, 0) - mid_x))
    arms = [k for k in rest if k != torso]
    return {"head": head, "torso": torso,
            "arms": sorted(arms, key=lambda k: ctr(k, 0)),
            "legs": sorted(legs, key=lambda k: ctr(k, 0)),
            "skis": sorted(skis, key=lambda k: ctr(k, 0))}


def mesh_to_nodepath(mesh: ObjMesh) -> NodePath:
    vdata = GeomVertexData(mesh.name, GeomVertexFormat.get_v3n3(), Geom.UH_static)
    vw = GeomVertexWriter(vdata, "vertex")
    nw = GeomVertexWriter(vdata, "normal")
    for p, n in zip(mesh.positions, mesh.normals):
        vw.add_data3(*p)
        nw.add_data3(*n)
    tris = GeomTriangles(Geom.UH_static)
    for a, b, c in mesh.triangles:
        tris.add_vertices(a, b, c)
    geom = Geom(vdata)
    geom.add_primitive(tris)
    node = GeomNode(mesh.name)
    node.add_geom(geom)
    return NodePath(node)


# --------------------------------------------------------------------------
#  Skier classes
# --------------------------------------------------------------------------
class SkierBase:
    """Common pose animation: the pose parameter eases toward its target."""

    kind = "base"

    def __init__(self, parent: NodePath) -> None:
        self.root = parent.attach_new_node("skier")
        self.p = 1.0
        self._target = (1.0, 0.0, False)

    def set_pose(self, p: float, v_angle: float, telemark: bool) -> None:
        """p: 1 crouch on the inrun, 0.5 upright, 0 flight.  v_angle: V of the skis in flight, deg."""
        self._target = (p, v_angle, telemark)

    def snap(self) -> None:
        self.p = self._target[0]
        self._apply(self.p, self._target[1], self._target[2])

    def update(self, dt: float) -> None:
        k = 1.0 - math.exp(-dt * 7.0)
        self.p += (self._target[0] - self.p) * k
        self._apply(self.p, self._target[1], self._target[2])

    def set_suit_color(self, rgb) -> None:
        raise NotImplementedError

    def _apply(self, p: float, v_angle: float, telemark: bool) -> None:
        raise NotImplementedError


class ModelSkier(SkierBase):
    """The jumper from ski_jumper_very_basic.obj, rigged with joints."""

    kind = "model"
    TELEMARK_SHIFT = 0.35      # model units (x MODEL_SCALE metres)

    def __init__(self, loader, parent: NodePath, model_path: str) -> None:
        super().__init__(parent)
        self.source = ""
        self.raw = self.root.attach_new_node("to_panda")
        # the file is Y-up with the skis along +Z; Panda3D is Z-up and the jump runs along +Y:
        # a pitch of 90 degrees maps up (+Y) to +Z and +Z to -Y, a heading of 180 flips it to +Y
        self.raw.set_hpr(180.0, 90.0, 0.0)
        self.raw.set_scale(MODEL_SCALE)
        self.rig = self.raw.attach_new_node("rig")
        parts = self._load_with_loader(loader, model_path)
        roles = None
        if parts is not None:
            roles = classify_parts({k: v[1] for k, v in parts.items()})
            if roles is not None:
                self.source = "loader.load_model"
            else:
                self._clear_rig()
                parts = None
        if roles is None:
            parts = self._load_with_parser(model_path)
            roles = classify_parts({k: v[1] for k, v in parts.items()}) if parts else None
            if roles is not None:
                self.source = "OBJ parser"
        if roles is None:
            raise RuntimeError("the jumper model could not be recognised")
        self._assemble(parts, roles)

    # ---- loading ---------------------------------------------------------------
    def _clear_rig(self) -> None:
        for child in list(self.rig.get_children()):
            child.remove_node()

    def _load_with_loader(self, loader, path: str):
        """Panda3D's loader; returns {key: (NodePath, bounds)} or None."""
        try:
            model = loader.load_model(Filename.from_os_specific(path).get_fullpath())
            if model is None or model.is_empty():
                return None
            model.reparent_to(self.rig)
            box = model.get_tight_bounds(self.rig)
            if box is None:
                return None
            if (box[1].y - box[0].y) > (box[1].z - box[0].z) * 1.05:
                # the loader already converted the file's Y-up to Z-up: undo it, so that this
                # class always works in the file's own frame
                model.set_hpr(0.0, -90.0, 0.0)
            geoms = model.find_all_matches("**/+GeomNode")
            parts = {}
            for i in range(geoms.get_num_paths()):
                np = geoms.get_path(i)
                b = np.get_tight_bounds(self.rig)
                if b is None:
                    continue
                parts[f"{i}:{np.get_name()}"] = (np, ((b[0].x, b[0].y, b[0].z), (b[1].x, b[1].y, b[1].z)))
            return parts or None
        except Exception:
            return None

    def _load_with_parser(self, path: str):
        try:
            meshes = parse_obj(path)
        except OSError:
            return None
        parts = {}
        for name, mesh in meshes.items():
            np = mesh_to_nodepath(mesh)
            np.reparent_to(self.rig)
            parts[name] = (np, mesh.bounds())
        return parts

    # ---- rig ---------------------------------------------------------------------
    def _assemble(self, parts: dict, roles: dict) -> None:
        def b(k):
            return parts[k][1]

        def c(k, axis):
            return (b(k)[0][axis] + b(k)[1][axis]) / 2.0

        legs = roles["legs"]
        lo = [min(b(k)[0][i] for k in parts) for i in range(3)]
        hi = [max(b(k)[1][i] for k in parts) for i in range(3)]
        x0 = (lo[0] + hi[0]) / 2.0                          # centre line of the figure
        foot_y = min(b(k)[0][1] for k in legs)
        hip_y = max(b(k)[1][1] for k in legs)
        bind_z = sum(c(k, 2) for k in legs) / 2.0           # boots
        ski_bottom = min(b(k)[0][1] for k in roles["skis"])
        # origin of the skier = the bindings on the bottom of the skis
        self.rig.set_pos(-x0, -ski_bottom, -bind_z)

        self.lean = self.rig.attach_new_node("lean")
        self.lean.set_pos(x0, foot_y, bind_z)
        self.hip = self.lean.attach_new_node("hip")
        self.hip.set_pos(0.0, hip_y - foot_y, 0.0)
        self.leg_pivots = []
        for k in legs:
            pv = self.lean.attach_new_node("leg_pivot")
            pv.set_pos(c(k, 0) - x0, 0.0, 0.0)
            self.leg_pivots.append(pv)
        self.ski_pivots = []
        self.ski_base = []
        for k in roles["skis"]:
            pv = self.rig.attach_new_node("ski_pivot")
            zmin, zmax = b(k)[0][2], b(k)[1][2]
            pos = (c(k, 0), c(k, 1), zmin + 0.25 * (zmax - zmin))   # V opens around the tail
            pv.set_pos(*pos)
            self.ski_pivots.append(pv)
            self.ski_base.append(pos)

        for k in (roles["head"], roles["torso"], *roles["arms"]):
            parts[k][0].wrt_reparent_to(self.hip)
        for k, pv in zip(legs, self.leg_pivots):
            parts[k][0].wrt_reparent_to(pv)
        for k, pv in zip(roles["skis"], self.ski_pivots):
            parts[k][0].wrt_reparent_to(pv)

        self.suit_parts = [parts[k][0] for k in (roles["torso"], *roles["arms"], *legs)]
        self.head_part = parts[roles["head"]][0]
        self.ski_parts = [parts[k][0] for k in roles["skis"]]
        self._color(self.head_part, HELMET_COLOR)
        for s in self.ski_parts:
            self._color(s, SKI_COLOR)
        self.set_suit_color((0.90, 0.15, 0.15))
        self.snap()

    @staticmethod
    def _color(np: NodePath, rgb) -> None:
        np.set_material_off(1)                  # the .mtl colours (if any) are replaced by ours
        np.set_color(rgb[0], rgb[1], rgb[2], 1.0, 1)

    def set_suit_color(self, rgb) -> None:
        for part in self.suit_parts:
            self._color(part, rgb)

    # ---- pose --------------------------------------------------------------------------
    def _apply(self, p: float, v_angle: float, telemark: bool) -> None:
        lean, hip = pose_angles(p)
        flight = clamp(1.0 - 2.0 * p, 0.0, 1.0)
        shift = self.TELEMARK_SHIFT if telemark else 0.0
        leg_extra = (0.0, 0.0)
        if telemark:                                    # landing: upright, knees bent, one ski forward
            lean, hip = 8.0, -4.0
            leg_extra = (-12.0, 20.0)                   # right leg back, left leg forward
        self.lean.set_p(lean)
        self.hip.set_p(hip)
        for i, pv in enumerate(self.leg_pivots):
            sign = -1.0 if i == 0 else 1.0
            pv.set_p(leg_extra[i])
            pv.set_z(sign * shift)
        for i, pv in enumerate(self.ski_pivots):
            sign = -1.0 if i == 0 else 1.0              # index 0 = right ski, 1 = left ski
            pv.set_r(sign * v_angle * flight)           # tips open: left ski turns +, right ski -
            pv.set_z(self.ski_base[i][2] + sign * shift)


class ProceduralSkier(SkierBase):
    """Box figure used only if the model file is missing."""

    kind = "procedural"

    def __init__(self, parent: NodePath) -> None:
        super().__init__(parent)
        self.skis = []
        for side in (-1, 1):
            pivot = self.root.attach_new_node(f"ski_pivot_{side}")
            pivot.set_pos(side * 0.16, 0.0, 0.0)
            ski = make_box(0.5, "ski")
            ski.set_scale(0.11, 2.4, 0.04)
            ski.set_pos(0.0, 0.55, 0.0)
            ski.set_color(*SKI_COLOR, 1.0)
            ski.reparent_to(pivot)
            self.skis.append(pivot)
        self.legs = self._part((0.30, 0.22, 0.60), (0.9, 0.15, 0.15))
        self.torso = self._part((0.36, 0.85, 0.26), (0.9, 0.15, 0.15))
        self.head = self._part((0.22, 0.24, 0.22), HELMET_COLOR)
        self.snap()

    def _part(self, scale, color) -> NodePath:
        part = make_box(0.5, "part")
        part.set_scale(*scale)
        part.set_color(*color, 1.0)
        part.reparent_to(self.root)
        return part

    def set_suit_color(self, rgb) -> None:
        self.legs.set_color(*rgb, 1.0)
        self.torso.set_color(*rgb, 1.0)

    def _apply(self, p: float, v_angle: float, telemark: bool) -> None:
        t = clamp(p, 0.0, 1.0)

        def lerp(a: float, b: float) -> float:
            return a + (b - a) * t

        self.legs.set_pos(0.0, lerp(-0.05, -0.10), lerp(0.30, 0.34))
        self.legs.set_p(lerp(10.0, 55.0))
        self.torso.set_pos(0.0, lerp(0.42, 0.18), lerp(0.55, 0.70))
        self.torso.set_p(lerp(0.0, -28.0))
        self.head.set_pos(0.0, lerp(0.95, 0.52), lerp(0.58, 0.78))
        va = v_angle * clamp(1.0 - 2.0 * p, 0.0, 1.0)
        self.skis[0].set_h(va)
        self.skis[1].set_h(-va)
        shift = 0.30 if telemark else 0.0
        self.skis[0].set_y(shift)
        self.skis[1].set_y(-shift)
        if telemark:
            self.torso.set_p(35.0)
            self.torso.set_pos(0.0, 0.0, 0.85)
            self.head.set_pos(0.0, 0.1, 1.25)
            self.legs.set_p(10.0)


def default_model_path() -> str:
    """The model lives in the game's root folder, beside the Python files."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILE)


def create_skier(loader, parent: NodePath, model_path: str | None = None) -> SkierBase:
    path = model_path or default_model_path()
    if os.path.exists(path):
        try:
            return ModelSkier(loader, parent, path)
        except Exception as exc:                       # never stop the game for the model
            print(f"[skier] could not use {path}: {exc}; using the box figure")
    else:
        print(f"[skier] {path} not found; using the box figure")
    return ProceduralSkier(parent)


if __name__ == "__main__":
    print("This file is a library module of the ski jump game.")
    print("Start the game with:   python ski_jump.py")
