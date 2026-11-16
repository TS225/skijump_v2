"""Tests for ski_skier: OBJ parsing, part recognition and the rig (works without Panda3D)."""
import math, os, sys, types, unittest
from unittest.mock import MagicMock

try:
    import panda3d.core  # noqa: F401
    REAL_PANDA = True
except ImportError:                                  # stand-ins so the pure logic can be tested
    REAL_PANDA = False
    class NP:
        def __init__(self, name=""):
            self.name=name; self.children=[]; self.calls=[]; self.parent=None
        def attach_new_node(self, node=None, *a):
            n=NP(node if isinstance(node,str) else "node"); n.parent=self; self.children.append(n); return n
        def reparent_to(self, p): self.parent=p; p.children.append(self)
        def wrt_reparent_to(self, p): self.parent=p; p.children.append(self)
        def get_children(self): return list(self.children)
        def remove_node(self): pass
        def __getattr__(self, item):
            if item.startswith("set_"):
                def f(*args): self.calls.append((item,args)); setattr(self,"v_"+item,args)
                return f
            raise AttributeError(item)
    core=types.ModuleType("panda3d.core")
    for n in ("Geom","GeomNode","GeomTriangles","GeomVertexData","GeomVertexFormat","GeomVertexWriter","Filename"):
        setattr(core,n,MagicMock(name=n))
    core.NodePath=NP
    sys.modules.update({"panda3d":types.ModuleType("panda3d"),"panda3d.core":core})

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ski_skier as sk

OBJ = sk.default_model_path()


class ParserTests(unittest.TestCase):
    def test_model_file_is_in_the_game_folder(self):
        self.assertTrue(os.path.exists(OBJ), OBJ)
        self.assertEqual(os.path.dirname(OBJ), os.path.dirname(os.path.abspath(sk.__file__)))

    def test_parse_real_model(self):
        meshes = sk.parse_obj(OBJ)
        self.assertEqual(list(meshes), ["Sphere", "Cylinder", "Cylinder.001", "Cylinder.002",
                                        "Cylinder.003", "Cylinder.004", "Cube", "Cube.001"])
        for m in meshes.values():
            self.assertEqual(len(m.positions), len(m.normals))
            self.assertTrue(m.triangles)
            self.assertTrue(all(max(t) < len(m.positions) for t in m.triangles))
            for n in m.normals:
                self.assertAlmostEqual(math.sqrt(sum(c * c for c in n)), 1.0, delta=0.02)
        # the 6 faces of a ski box are quads -> 12 triangles
        self.assertEqual(len(meshes["Cube"].triangles), 12)
        lo, hi = meshes["Cube"].bounds()
        self.assertAlmostEqual(hi[2] - lo[2], 2.0, places=2)           # skis are 2 units long

    def test_overall_size(self):
        meshes = sk.parse_obj(OBJ)
        lo = [min(m.bounds()[0][i] for m in meshes.values()) for i in range(3)]
        hi = [max(m.bounds()[1][i] for m in meshes.values()) for i in range(3)]
        self.assertAlmostEqual(hi[1] - lo[1], 1.3246, places=3)         # height
        self.assertAlmostEqual((hi[1] - lo[1]) * sk.MODEL_SCALE, 1.72, delta=0.02)   # a real athlete

    def test_classification(self):
        meshes = sk.parse_obj(OBJ)
        roles = sk.classify_parts({k: m.bounds() for k, m in meshes.items()})
        self.assertEqual(roles["head"], "Sphere")
        self.assertEqual(roles["torso"], "Cylinder")
        self.assertEqual(set(roles["arms"]), {"Cylinder.001", "Cylinder.002"})
        self.assertEqual(set(roles["legs"]), {"Cylinder.003", "Cylinder.004"})
        self.assertEqual(set(roles["skis"]), {"Cube", "Cube.001"})
        for pair in (roles["arms"], roles["legs"], roles["skis"]):    # sorted by x
            cx = [sum(sk_ for sk_ in (meshes[k].bounds()[0][0], meshes[k].bounds()[1][0])) for k in pair]
            self.assertLess(cx[0], cx[1])

    def test_classification_rejects_other_shapes(self):
        meshes = sk.parse_obj(OBJ)
        b = {k: m.bounds() for k, m in meshes.items()}
        self.assertIsNone(sk.classify_parts({k: v for k, v in list(b.items())[:-1]}))   # 7 parts
        flat = {k: ((0, 0, 0), (1, 1, 1)) for k in b}
        self.assertIsNone(sk.classify_parts(flat))

    def test_pose_keyframes(self):
        self.assertEqual(sk.pose_angles(1.0), sk.POSE_CROUCH)
        self.assertEqual(sk.pose_angles(0.5), sk.POSE_UPRIGHT)
        self.assertEqual(sk.pose_angles(0.0), sk.POSE_FLIGHT)
        lean_mid, _ = sk.pose_angles(0.25)
        self.assertTrue(sk.POSE_UPRIGHT[0] < lean_mid < sk.POSE_FLIGHT[0])
        self.assertEqual(sk.pose_angles(5.0), sk.POSE_CROUCH)           # clamped


@unittest.skipIf(REAL_PANDA, "rig test uses stand-in Panda3D classes")
class RigTests(unittest.TestCase):
    def make(self):
        class Loader:                                  # the loader result is not usable -> OBJ parser path
            def load_model(self, path): return None
        root = sk.NodePath("render")
        return sk.create_skier(Loader(), root)

    def test_model_skier_is_built_from_the_obj(self):
        s = self.make()
        self.assertEqual(s.kind, "model")
        self.assertEqual(s.source, "OBJ parser")
        # parts hang under joints
        self.assertEqual(len(s.leg_pivots), 2)
        self.assertEqual(len(s.ski_pivots), 2)
        self.assertEqual(len(s.suit_parts), 5)
        # origin at the bindings on the bottom of the skis (x centre, ski bottom y, boot z)
        pos = s.rig.v_set_pos
        self.assertAlmostEqual(pos[0], 0.0167, delta=0.005)
        self.assertAlmostEqual(pos[1], -0.0532, delta=0.002)
        self.assertAlmostEqual(pos[2], -0.07, delta=0.03)
        # raw frame: heading 180 + pitch 90 turns the file's Y-up/+Z-forward into Z-up/+Y-forward
        self.assertEqual(s.raw.v_set_hpr, (180.0, 90.0, 0.0))
        self.assertEqual(s.raw.v_set_scale, (sk.MODEL_SCALE,))

    def test_pose_is_applied_to_the_joints(self):
        s = self.make()
        s.set_pose(1.0, 24.0, False); s.snap()
        self.assertEqual(s.lean.v_set_p, (sk.POSE_CROUCH[0],))
        self.assertEqual(s.hip.v_set_p, (sk.POSE_CROUCH[1],))
        self.assertEqual(s.ski_pivots[0].v_set_r, (0.0,))           # skis parallel on the inrun
        s.set_pose(0.0, 24.0, False); s.snap()
        self.assertEqual(s.lean.v_set_p, (sk.POSE_FLIGHT[0],))
        self.assertEqual(s.ski_pivots[1].v_set_r, (24.0,))          # V style: left ski +, right ski -
        self.assertEqual(s.ski_pivots[0].v_set_r, (-24.0,))
        s.set_pose(0.5, 0.0, True); s.snap()                        # telemark
        self.assertEqual(s.ski_pivots[1].v_set_z[0] - s.ski_pivots[0].v_set_z[0] > 0.5, True)
        self.assertEqual(s.leg_pivots[1].v_set_p, (20.0,))

    def test_pose_eases(self):
        s = self.make()
        s.set_pose(0.0, 0.0, False); s.snap()
        s.set_pose(1.0, 0.0, False)
        s.update(0.05)
        self.assertTrue(0.0 < s.p < 1.0)
        for _ in range(100):
            s.update(0.05)
        self.assertAlmostEqual(s.p, 1.0, places=3)

    def test_missing_file_falls_back(self):
        s = sk.create_skier(object(), sk.NodePath("r"), "/no/such/file.obj")
        self.assertEqual(s.kind, "procedural")
        s.set_pose(0.0, 20.0, False); s.snap(); s.update(0.1)

    def test_suit_colour(self):
        s = self.make()
        s.set_suit_color((0.1, 0.2, 0.3))
        self.assertEqual(s.suit_parts[0].calls[-1][0], "set_color")
        self.assertEqual(s.suit_parts[0].calls[-1][1][:3], (0.1, 0.2, 0.3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
