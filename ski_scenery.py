"""Scenery for the ski jump game (Panda3D).  A library module - run ski_jump.py to play.

Everything is built in code (no image files needed):

    Textures      procedural snow, ice, crowd, advertising banner, cloud and sun textures
    Environment   gradient sky dome, sun, clouds, a ring of snowy mountains, far ground
    HillScene     the hill with textured terrain, valley slopes, inrun on a steel trestle,
                  ice track with grooves, landing hill with distance lines, advertising
                  fences, grandstands with crowd, floodlight masts, judges tower, start hut,
                  scoreboard, fir forest, distance numbers and wind flags that follow the
                  real (simulated) wind
"""

from __future__ import annotations

import bisect
import math
import random

from panda3d.core import (
    CardMaker,
    CompassEffect,
    NodePath,
    PNMImage,
    Texture,
    TextNode,
    TransparencyAttrib,
)

import ski_physics as sp
from ski_geom import MeshBuilder, clamp, lerp, smoothstep

SUN_DIR = (-0.45, 0.55, 0.70)           # direction towards the sun (x right, y jump direction, z up)


def _norm(v):
    n = math.sqrt(sum(c * c for c in v)) or 1.0
    return tuple(c / n for c in v)


SUN_DIR = _norm(SUN_DIR)


# --------------------------------------------------------------------------
#  Procedural pixel generators (pure Python, testable)
# --------------------------------------------------------------------------
def _smooth(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def value_noise(w: int, h: int, cells: int, rng: random.Random) -> list[list[float]]:
    """Tileable value noise in 0..1 with `cells` lattice cells across."""
    grid = [[rng.random() for _ in range(cells)] for _ in range(cells)]
    out = []
    for y in range(h):
        fy = y / h * cells
        y0 = int(fy)
        ty = _smooth(fy - y0)
        y1 = (y0 + 1) % cells
        y0 %= cells
        row = []
        for x in range(w):
            fx = x / w * cells
            x0 = int(fx)
            tx = _smooth(fx - x0)
            x1 = (x0 + 1) % cells
            x0 %= cells
            a = grid[y0][x0] + (grid[y0][x1] - grid[y0][x0]) * tx
            b = grid[y1][x0] + (grid[y1][x1] - grid[y1][x0]) * tx
            row.append(a + (b - a) * ty)
        out.append(row)
    return out


def fractal_noise(w: int, h: int, seed: int, cells: int = 4, octaves: int = 4) -> list[list[float]]:
    rng = random.Random(seed)
    total = [[0.0] * w for _ in range(h)]
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        layer = value_noise(w, h, cells * (2 ** o), rng)
        for y in range(h):
            row, lrow = total[y], layer[y]
            for x in range(w):
                row[x] += lrow[x] * amp
        norm += amp
        amp *= 0.5
    return [[v / norm for v in row] for row in total]


def snow_pixels(size: int = 128, seed: int = 1) -> list[list[tuple[float, float, float]]]:
    """Packed snow: soft bluish dimples, a few glittering crystals."""
    n = fractal_noise(size, size, seed, 4, 4)
    rng = random.Random(seed + 100)
    out = []
    for y in range(size):
        row = []
        for x in range(size):
            v = n[y][x]
            r = 0.90 + 0.10 * v
            g = 0.92 + 0.08 * v
            b = 0.97 + 0.03 * v
            if v < 0.40:                                   # shaded dimples
                k = (0.40 - v) * 0.30
                r, g, b = r - k * 1.2, g - k, b - k * 0.4
            if rng.random() < 0.012:                       # sparkle
                r = g = b = 1.0
            row.append((clamp(r, 0, 1), clamp(g, 0, 1), clamp(b, 0, 1)))
        out.append(row)
    return out


def ice_pixels(w: int = 32, h: int = 128, seed: int = 2) -> list[list[tuple[float, float, float]]]:
    """Inrun ice track: smooth blue-white with fine streaks along the track."""
    rng = random.Random(seed)
    streak = [rng.random() for _ in range(w)]
    n = fractal_noise(w, h, seed, 2, 3)
    out = []
    for y in range(h):
        row = []
        for x in range(w):
            s = 0.5 + 0.5 * streak[x]
            v = 0.80 + 0.12 * n[y][x] + 0.06 * s
            row.append((clamp(v - 0.10, 0, 1), clamp(v - 0.02, 0, 1), clamp(v + 0.08, 0, 1)))
        out.append(row)
    return out


CROWD_COLORS = ((0.85, 0.10, 0.10), (0.10, 0.25, 0.85), (0.95, 0.95, 0.95), (0.95, 0.80, 0.10),
                (0.10, 0.60, 0.25), (0.12, 0.12, 0.15), (0.95, 0.45, 0.10), (0.60, 0.20, 0.70))


def crowd_pixels(w: int = 128, h: int = 64, seed: int = 3) -> list[list[tuple[float, float, float]]]:
    """Rows of spectators in winter clothes with a skin-coloured head."""
    rng = random.Random(seed)
    img = [[(0.10, 0.11, 0.15)] * w for _ in range(h)]
    skin = (0.92, 0.72, 0.58)
    for top in range(0, h - 8, 10):
        for left in range(0, w - 5, 6):
            if rng.random() < 0.10:
                continue
            color = rng.choice(CROWD_COLORS)
            for dy in range(2, 9):                        # body
                for dx in range(0, 5):
                    img[top + dy][left + dx] = color
            for dy in range(0, 2):                        # head
                for dx in (1, 2, 3):
                    img[top + dy][left + dx] = skin
    return img


def banner_pixels(w: int = 256, h: int = 32, seed: int = 4) -> list[list[tuple[float, float, float]]]:
    """Advertising boards: coloured panels with white lettering-like strokes."""
    rng = random.Random(seed)
    palette = ((0.85, 0.12, 0.12), (0.10, 0.30, 0.75), (0.95, 0.75, 0.05), (0.10, 0.55, 0.25),
               (0.90, 0.90, 0.92), (0.15, 0.15, 0.20), (0.90, 0.40, 0.05))
    img = [[(1, 1, 1)] * w for _ in range(h)]
    for left in range(0, w, 32):
        bg = rng.choice(palette)
        ink = (1, 1, 1) if sum(bg) < 1.9 else (0.1, 0.1, 0.15)
        for y in range(h):
            for x in range(left, left + 32):
                img[y][x] = bg if (x - left) not in (0, 31) else (0.95, 0.95, 0.95)
        for stroke in range(5):                           # pseudo letters
            sx = left + 3 + stroke * 5
            sh = rng.randint(8, 18)
            for y in range(16 - sh // 2, 16 + sh // 2):
                for x in (sx, sx + 1):
                    img[y][x] = ink
            if rng.random() < 0.6:
                for x in range(sx, sx + 4):
                    img[16 - sh // 2][x] = ink
    return img


def cloud_pixels(w: int = 128, h: int = 64, seed: int = 5) -> list[list[tuple[float, float, float, float]]]:
    """A soft cumulus cloud: a sum of gaussian puffs, flatter and greyer at the bottom."""
    rng = random.Random(seed)
    puffs = []
    for _ in range(14):
        cx = rng.uniform(0.22, 0.78) * w
        cy = rng.uniform(0.42, 0.62) * h
        r = rng.uniform(0.07, 0.17) * w
        puffs.append((cx, cy, r))
    out = []
    for y in range(h):
        row = []
        for x in range(w):
            a = 0.0
            for cx, cy, r in puffs:
                d2 = ((x - cx) / r) ** 2 + ((y - cy) / (r * 0.8)) ** 2
                a += math.exp(-d2 * 1.6)
            alpha = clamp(a, 0.0, 1.0) ** 1.4
            shade = 0.80 + 0.20 * (1.0 - y / h)
            row.append((shade, shade, min(1.0, shade + 0.03), alpha))
        out.append(row)
    return out


def sun_pixels(size: int = 64) -> list[list[tuple[float, float, float, float]]]:
    out = []
    c = (size - 1) / 2.0
    for y in range(size):
        row = []
        for x in range(size):
            r = math.hypot(x - c, y - c) / c
            a = clamp(math.exp(-(r / 0.22) ** 2) + 0.30 * math.exp(-(r / 0.75) ** 2), 0.0, 1.0)
            row.append((1.0, 0.96, 0.82, a * (1.0 if r < 1 else 0.0)))
        out.append(row)
    return out


TEXTURE_BUILDERS = {
    "snow": lambda: (snow_pixels(128, 1), False),
    "snow2": lambda: (snow_pixels(128, 7), False),
    "ice": lambda: (ice_pixels(), False),
    "crowd": lambda: (crowd_pixels(), False),
    "banner": lambda: (banner_pixels(), False),
    "cloud": lambda: (cloud_pixels(), True),
    "sun": lambda: (sun_pixels(), True),
}


def make_texture(pixels, name: str, alpha: bool = False, repeat: bool = True) -> Texture:
    h, w = len(pixels), len(pixels[0])
    img = PNMImage(w, h, 4 if alpha else 3)
    for y in range(h):
        row = pixels[y]
        for x in range(w):
            px = row[x]
            img.set_xel(x, y, px[0], px[1], px[2])
            if alpha:
                img.set_alpha(x, y, px[3])
    tex = Texture(name)
    tex.load(img)
    if repeat:
        tex.set_wrap_u(Texture.WM_repeat)
        tex.set_wrap_v(Texture.WM_repeat)
    else:
        tex.set_wrap_u(Texture.WM_clamp)
        tex.set_wrap_v(Texture.WM_clamp)
    tex.set_minfilter(Texture.FT_linear_mipmap_linear)
    tex.set_magfilter(Texture.FT_linear)
    return tex


class Textures:
    """Lazily built, cached procedural textures."""

    def __init__(self) -> None:
        self._cache: dict[str, Texture] = {}

    def get(self, name: str) -> Texture:
        if name not in self._cache:
            pixels, alpha = TEXTURE_BUILDERS[name]()
            self._cache[name] = make_texture(pixels, name, alpha, repeat=name not in ("cloud", "sun"))
        return self._cache[name]


# --------------------------------------------------------------------------
#  Weather -> sky
# --------------------------------------------------------------------------
def cloudiness_for(weather: sp.Weather) -> float:
    """0 = clear blue sky ... 1 = fully overcast, from the weather settings."""
    c = 0.12
    if weather.snow in ("wet", "slush"):
        c += 0.40
    if weather.temperature_c > -1.0:
        c += 0.15
    c += min(0.35, weather.gustiness * 0.8)
    c += min(0.15, weather.wind_speed * 0.02)
    return clamp(c, 0.0, 1.0)


def sky_palette(cloudiness: float):
    """(horizon, zenith) colours of the sky."""
    clear_h, clear_z = (0.80, 0.89, 0.98), (0.20, 0.45, 0.86)
    over_h, over_z = (0.74, 0.77, 0.81), (0.50, 0.55, 0.63)
    k = clamp(cloudiness, 0.0, 1.0)
    mix = lambda a, b: tuple(lerp(x, y, k) for x, y in zip(a, b))
    return mix(clear_h, over_h), mix(clear_z, over_z)


# --------------------------------------------------------------------------
#  Environment: sky, sun, clouds, mountains, far ground
# --------------------------------------------------------------------------
class Environment:
    def __init__(self, render: NodePath, camera: NodePath, textures: Textures) -> None:
        self.render, self.camera, self.textures = render, camera, textures
        self.root = render.attach_new_node("environment")
        self.sky_np: NodePath | None = None
        self.fog_color = sky_palette(0.2)[0]
        self.cloudiness = 0.2

    @property
    def sun_light_direction(self):
        return tuple(-c for c in SUN_DIR)

    def rebuild(self, center_d: float, valley: float, cloudiness: float) -> None:
        """Create the sky and the distant landscape around the hill (valley = floor height)."""
        self.destroy()
        self.root = self.render.attach_new_node("environment")
        self.cloudiness = cloudiness
        horizon, zenith = sky_palette(cloudiness)
        self.fog_color = horizon
        self._sky(horizon, zenith, cloudiness)
        self._sun(center_d, cloudiness)
        self._clouds(center_d, cloudiness)
        self._mountains(center_d, valley, cloudiness)
        self._far_ground(center_d, valley)

    def destroy(self) -> None:
        if self.sky_np is not None:
            self.sky_np.remove_node()
            self.sky_np = None
        if self.root is not None:
            self.root.remove_node()

    # ---- sky dome follows the camera position but not its rotation -----------------
    def _sky(self, horizon, zenith, cloudiness: float) -> None:
        mb = MeshBuilder("sky")
        elevs = (-20, -6, 0, 4, 9, 16, 25, 36, 50, 68, 90)
        segs = 36
        rings = []
        for e in elevs:
            er = math.radians(e)
            t = clamp(e / 62.0, 0.0, 1.0) ** 0.65
            ring = []
            for i in range(segs):
                a = 2 * math.pi * i / segs
                d = (math.cos(er) * math.sin(a), math.cos(er) * math.cos(a), math.sin(er))
                col = [lerp(horizon[j], zenith[j], t) for j in range(3)]
                if e < 0:                                   # below the horizon: haze
                    col = [horizon[j] * 0.92 for j in range(3)]
                glow = max(0.0, sum(d[j] * SUN_DIR[j] for j in range(3))) ** 10 * (1.0 - 0.8 * cloudiness)
                col = [clamp(col[0] + glow * 0.30, 0, 1), clamp(col[1] + glow * 0.22, 0, 1),
                       clamp(col[2] + glow * 0.05, 0, 1)]
                ring.append(mb.vertex(tuple(c * 1900.0 for c in d), tuple(-c for c in d), (*col, 1.0)))
            rings.append(ring)
        for r0, r1 in zip(rings, rings[1:]):
            for i in range(segs):
                j = (i + 1) % segs
                mb.triangle(r0[i], r0[j], r1[i])
                mb.triangle(r0[j], r1[j], r1[i])
        sky = mb.build()
        sky.reparent_to(self.camera)
        sky.set_light_off()
        sky.set_fog_off()
        sky.set_two_sided(True)
        sky.set_depth_write(False)
        sky.set_depth_test(False)
        sky.set_bin("background", 0)
        sky.set_effect(CompassEffect.make(self.render))
        self.sky_np = sky

    def _card(self, parent: NodePath, tex: Texture, w: float, h: float) -> NodePath:
        cm = CardMaker("card")
        cm.set_frame(-w, w, -h, h)
        card = parent.attach_new_node(cm.generate())
        card.set_texture(tex)
        card.set_transparency(TransparencyAttrib.M_alpha)
        card.set_light_off()
        card.set_depth_write(False)
        card.set_two_sided(True)
        card.set_billboard_point_eye()
        return card

    def _sun(self, center_d: float, cloudiness: float) -> None:
        if cloudiness > 0.75:
            return
        sun = self._card(self.root, self.textures.get("sun"), 190.0, 190.0)
        sun.set_pos(0.0 + SUN_DIR[0] * 1500.0, center_d + SUN_DIR[1] * 1500.0, SUN_DIR[2] * 1500.0)
        sun.set_fog_off()
        sun.set_color_scale(1, 1, 1, 1.0 - cloudiness)
        sun.set_bin("background", 5)

    def _clouds(self, center_d: float, cloudiness: float) -> None:
        rng = random.Random(11)
        tex = self.textures.get("cloud")
        count = 7 + int(16 * cloudiness)
        grey = 1.0 - 0.38 * cloudiness
        for _ in range(count):
            a = rng.uniform(0, 2 * math.pi)
            dist = rng.uniform(500, 1500)
            z = rng.uniform(260, 520) - 120 * cloudiness
            size = rng.uniform(170, 330)
            card = self._card(self.root, tex, size, size * 0.5)
            card.set_pos(math.sin(a) * dist, center_d + math.cos(a) * dist, z)
            card.set_color_scale(grey, grey, grey, 0.5 + 0.5 * rng.random())

    def _mountains(self, center_d: float, valley: float, cloudiness: float) -> None:
        """A ring of ridged mountains: forest, rock and snow, washed out by the distance fog."""
        rng = random.Random(21)
        segs = 144
        comps = [(f, rng.uniform(0, 2 * math.pi)) for f in (2, 3, 5, 8, 13, 21)]
        weights = (0.55, 0.40, 0.32, 0.20, 0.12, 0.07)

        def height(theta: float) -> float:
            total = 0.0
            for (f, ph), wgt in zip(comps, weights):
                total += wgt * (1.0 - abs(math.sin(f * theta + ph)))
            return 90.0 + 300.0 * total / sum(weights)

        rows = ((720.0, 0.0), (800.0, 0.62), (890.0, 1.0), (1010.0, 0.78), (1150.0, 0.30))
        mb = MeshBuilder("mountains")
        dim = 1.0 - 0.18 * cloudiness
        grid = []
        for i in range(segs):
            th = 2 * math.pi * i / segs
            h0 = height(th) * (0.85 + 0.3 * rng.random())
            col_row = []
            for r, f in rows:
                z = valley - 25.0 + h0 * f
                frac = f
                if frac < 0.30:
                    col = (0.15, 0.27, 0.21)                      # conifer forest
                elif frac < 0.62:
                    col = (0.40, 0.42, 0.46)                      # rock
                else:
                    col = (0.94, 0.96, 1.00)                      # snow
                jitter = 0.92 + 0.12 * rng.random()
                col = tuple(clamp(c * dim * jitter, 0, 1) for c in col)
                nrm = _norm((-math.sin(th) * 0.5, -math.cos(th) * 0.5, 0.75))
                col_row.append(mb.vertex((math.sin(th) * r, center_d + math.cos(th) * r, z), nrm, (*col, 1.0)))
            grid.append(col_row)
        for i in range(segs):
            j = (i + 1) % segs
            for k in range(len(rows) - 1):
                mb.triangle(grid[i][k], grid[j][k], grid[i][k + 1])
                mb.triangle(grid[j][k], grid[j][k + 1], grid[i][k + 1])
        mnt = mb.build()
        mnt.reparent_to(self.root)
        mnt.set_two_sided(True)

    def _far_ground(self, center_d: float, valley: float) -> None:
        mb = MeshBuilder("far_ground")
        radii = (60.0, 150.0, 300.0, 520.0, 800.0, 1250.0)
        segs = 40
        rings = []
        for r in radii:
            ring = []
            for i in range(segs):
                a = 2 * math.pi * i / segs
                z = valley - 0.3 + 14.0 * (r / 1250.0) ** 2
                ring.append(mb.vertex((math.sin(a) * r, center_d + math.cos(a) * r, z), (0, 0, 1),
                                      (0.86, 0.90, 0.95, 1.0), (math.sin(a) * r / 50.0, math.cos(a) * r / 50.0)))
            rings.append(ring)
        for r0, r1 in zip(rings, rings[1:]):
            for i in range(segs):
                j = (i + 1) % segs
                mb.triangle(r0[i], r1[i], r0[j])
                mb.triangle(r0[j], r1[i], r1[j])
        ground = mb.build()
        ground.set_texture(self.textures.get("snow"))
        ground.set_two_sided(True)
        ground.reparent_to(self.root)


# --------------------------------------------------------------------------
#  The hill
# --------------------------------------------------------------------------
class HillScene:
    """Visual model of a hill built from the same Inrun / LandingProfile the physics uses."""

    def __init__(self, parent: NodePath, spec: sp.HillSpec, gate: int, textures: Textures) -> None:
        self.spec, self.gate, self.textures = spec, gate, textures
        self.inrun = sp.Inrun(spec, gate)
        self.profile = sp.LandingProfile(spec)
        self.valley = self.profile.y[-1]
        self.root = parent.attach_new_node(f"hill_{spec.key}")
        self._inrun_d: list[float] = []
        self._inrun_h: list[float] = []
        for i in range(int(self.inrun.length) + 2):
            d, h = self.inrun.position(min(float(i), self.inrun.length))
            self._inrun_d.append(d)
            self._inrun_h.append(h)
        self.start_d, self.start_h = self._inrun_d[0], self._inrun_h[0]
        self.center_d = (self.start_d + self.profile.x_end) / 2.0
        self.rng = random.Random(spec.key)
        self.flags: list[tuple[NodePath, float]] = []
        self._build()

    # ---- terrain model (also used by the camera and the shadow) ------------------------
    def _flat_half_width(self, d: float) -> float:
        t = smoothstep((d + 10.0) / 25.0)
        return lerp(3.5, self.spec.half_width + 1.0, t)

    def _track_height(self, d: float) -> float:
        xs = self._inrun_d
        if d <= xs[0]:
            return self._inrun_h[0]
        i = min(bisect.bisect_right(xs, d) - 1, len(xs) - 2)
        f = (d - xs[i]) / max(xs[i + 1] - xs[i], 1e-9)
        return self._inrun_h[i] + f * (self._inrun_h[i + 1] - self._inrun_h[i])

    def clearance(self, d: float) -> float:
        """Height of the inrun above the hillside: a steel trestle in the middle."""
        u = clamp((d - self.start_d) / max(-self.start_d, 1.0), 0.0, 1.0)
        return self.spec.table_height + 7.0 * math.sin(math.pi * u)

    def base_height(self, d: float) -> float:
        if d >= 0.0:
            return self.profile.y[-1] if d >= self.profile.x_end else self.profile.ground(d)
        return self._track_height(d) - self.clearance(d)

    def terrain_height(self, d: float, l: float) -> float:
        fw = self._flat_half_width(d)
        z = self.base_height(d)
        off = abs(l) - fw
        if off > 0.0:
            z += min(off, 22.0) * 0.5                                  # snow bank beside the hill
            if off > 22.0:
                z -= (off - 22.0) * 0.55                               # slope down into the valley
            z += 0.7 * math.sin(0.11 * d + 0.23 * l) * min(1.0, off / 8.0)
            z = max(z, self.valley - 0.5)
        return z

    def _terrain_color(self, d: float, l: float, z: float):
        fw = self._flat_half_width(d)
        far = clamp((abs(l) - fw - 14.0) / 60.0, 0.0, 1.0)
        shade = 0.5 + 0.5 * math.sin(0.07 * d + 0.4 * l) * math.sin(0.05 * l - 0.03 * d)
        return (0.95 - 0.30 * far - 0.03 * shade, 0.96 - 0.22 * far - 0.02 * shade, 1.0 - 0.28 * far, 1.0)

    # ---- construction -----------------------------------------------------------------
    def _add(self, mb: MeshBuilder, texture: str | None = None, light_off: bool = False) -> NodePath:
        np = mb.build()
        np.reparent_to(self.root)
        np.set_two_sided(True)
        if texture:
            np.set_texture(self.textures.get(texture))
        if light_off:
            np.set_light_off()
        return np

    def _build(self) -> None:
        self._terrain()
        self._surfaces()
        self._lines()
        self._fences()
        self._trestle()
        self._buildings()
        self._stands()
        self._masts()
        self._trees()
        self._flags()
        self._markers()

    def _terrain(self) -> None:
        spec, prof = self.spec, self.profile
        hw = spec.half_width
        d_nodes = []
        d = self.start_d - 30.0
        while d < prof.x_end + 90.0:
            d_nodes.append(d)
            d += 3.0
        l_half = [0.0, 3.5, hw + 1.0, hw + 3.0, hw + 7.0, hw + 14.0, hw + 22.0, hw + 34.0,
                  hw + 50.0, hw + 75.0, hw + 110.0, hw + 160.0]
        l_nodes = sorted({-v for v in l_half} | set(l_half))
        mb = MeshBuilder("terrain")
        mb.grid(d_nodes, l_nodes, self.terrain_height, self._terrain_color, uv_scale=12.0)
        self._add(mb, "snow")

    def _surfaces(self) -> None:
        spec, prof, inrun = self.spec, self.profile, self.inrun
        hw = spec.half_width
        track = MeshBuilder("track")
        pts = [inrun.position(min(s, inrun.length)) for s in _frange(0.0, inrun.length, 1.0)]
        track.ribbon(pts, 1.0, (0.78, 0.88, 0.97, 1.0), z_off=0.03, uv_scale=8.0)
        for side in (-0.45, 0.45):                                   # the two ski grooves
            track.ribbon(pts, 0.07, (0.30, 0.42, 0.60, 1.0), l_center=side, z_off=0.06)
        table = [inrun.position(s) for s in _frange(inrun.s_table, inrun.length, 0.5)]
        track.ribbon(table, 1.0, (0.92, 0.94, 0.98, 1.0), z_off=0.08, uv_scale=8.0)
        self._add(track, "ice")

        land = MeshBuilder("landing")
        step = 4
        pts = [(prof.x[i], prof.y[i]) for i in range(0, len(prof.x), step)]
        if pts[-1][0] < prof.x[-1]:
            pts.append((prof.x[-1], prof.y[-1]))
        land.ribbon(pts, hw, (1.0, 1.0, 1.0, 1.0), z_off=0.05, uv_scale=8.0)
        self._add(land, "snow2")
        edge = MeshBuilder("edge")
        for side in (-hw, hw):
            edge.ribbon(pts, 0.30, (1.0, 0.50, 0.08, 1.0), l_center=side, z_off=0.09)
        self._add(edge)

    def _lines(self) -> None:
        spec, prof = self.spec, self.profile
        mb = MeshBuilder("lines")
        s = 5.0
        while s <= spec.hill_size * 1.12:
            dark = int(s) % 10 == 0
            self._cross_line(mb, s, 0.16, (0.30, 0.38, 0.52, 1.0) if dark else (0.58, 0.64, 0.76, 1.0))
            s += 5.0
        self._cross_line(mb, prof.s_p, 0.5, (0.10, 0.70, 0.20, 1.0))
        self._cross_line(mb, spec.k_point, 0.8, (0.95, 0.08, 0.08, 1.0))
        self._cross_line(mb, spec.hill_size, 0.55, (0.12, 0.30, 0.95, 1.0))
        self._cross_line(mb, prof.s_l, 0.5, (0.95, 0.85, 0.08, 1.0))
        self._add(mb)

    def _cross_line(self, mb: MeshBuilder, s: float, width: float, color) -> None:
        p = self.profile
        d0 = p._interp(p.s, p.x, s)
        d1 = p._interp(p.s, p.x, s + width)
        mb.ribbon([(d0, p.ground(d0)), (d1, p.ground(d1))], self.spec.half_width, color, z_off=0.11)

    def _fences(self) -> None:
        """Advertising boards along the landing hill and outrun, rails along the inrun."""
        spec, prof, inrun = self.spec, self.profile, self.inrun
        hw = spec.half_width
        ads = MeshBuilder("ads")
        pts = [(prof.x[i], prof.y[i]) for i in range(0, len(prof.x), 8)]
        for side, facing in ((-1.0, 1.0), (1.0, -1.0)):
            ads.wall(pts, side * (hw + 0.6), 1.15, (1.0, 1.0, 1.0, 1.0), side=facing, uv_scale=8.0)
        self._add(ads, "banner")
        rails = MeshBuilder("rails")
        track = [inrun.position(min(s, inrun.length)) for s in _frange(0.0, inrun.length, 2.0)]
        for side, facing in ((-1.0, 1.0), (1.0, -1.0)):
            rails.wall(track, side * 1.25, 0.65, (0.42, 0.46, 0.52, 1.0), side=facing, z_off=0.0)
        self._add(rails)

    def _trestle(self) -> None:
        """Steel columns and cross beams under the inrun."""
        inrun = self.inrun
        mb = MeshBuilder("trestle")
        steel = (0.45, 0.50, 0.56, 1.0)
        s = 5.0
        while s < inrun.length - 2.0:
            d, h = inrun.position(s)
            ground = self.base_height(d)
            top = h - 0.35
            if top - ground > 0.8:
                for side in (-1.4, 1.4):
                    mb.box((side, d, (top + ground) / 2.0), (0.38, 0.38, top - ground), steel)
                mb.box((0.0, d, top - 0.15), (3.2, 0.32, 0.30), steel)
                if int(s) % 12 < 6:                                   # diagonal-ish brace blocks
                    mb.box((0.0, d, (top + ground) / 2.0), (2.8, 0.22, 0.22), steel)
            s += 6.0
        self._add(mb)

    def _buildings(self) -> None:
        spec, prof = self.spec, self.profile
        mb = MeshBuilder("buildings")
        glass = (0.20, 0.30, 0.42, 1.0)
        # start hut on top of the inrun with a roof
        d0, h0 = self.start_d - 2.0, self.start_h
        mb.box((3.6, d0, h0 + 1.3), (3.6, 4.2, 2.6), (0.62, 0.42, 0.30, 1.0))
        mb.box((3.6, d0, h0 + 2.75), (4.2, 4.8, 0.3), (0.35, 0.18, 0.12, 1.0))
        mb.box((1.7, d0, h0 + 1.2), (0.12, 2.0, 1.1), glass)
        # start bar across the track
        mb.box((0.0, self.start_d + 0.6, self.start_h + 0.55), (2.6, 0.12, 0.12), (0.9, 0.1, 0.1, 1.0))
        # take-off table edge: a bright bar so the nose of the hill is easy to see
        mb.box((0.0, 0.0, 0.05), (2.3, 0.18, 0.18), (1.0, 0.25, 0.05, 1.0))
        # judges tower beside the K point
        dk = prof.x_k
        gz = prof.ground(dk)
        lt = spec.half_width + 9.0
        mb.box((lt, dk, gz + 5.0), (5.0, 8.0, 10.0), (0.46, 0.50, 0.58, 1.0))
        mb.box((lt - 2.55, dk, gz + 8.2), (0.12, 7.0, 2.2), glass)
        mb.box((lt, dk, gz + 10.4), (5.6, 8.6, 0.4), (0.25, 0.28, 0.33, 1.0))
        # scoreboard at the end of the outrun
        de = prof.x_end - 14.0
        for side in (-1.0, 1.0):
            mb.box((side * (spec.half_width + 5.0), de, prof.y[-1] + 3.0), (0.5, 0.5, 6.0), (0.3, 0.3, 0.34, 1.0))
        mb.box((0.0, de, prof.y[-1] + 6.2), (spec.half_width * 2.0 + 10.5, 0.5, 2.8), (0.10, 0.12, 0.16, 1.0))
        self._add(mb)
        label = TextNode("scoreboard")
        label.set_text(f"{spec.name}")
        label.set_align(TextNode.A_center)
        label.set_text_color(1.0, 0.85, 0.2, 1.0)
        np = self.root.attach_new_node(label)
        np.set_scale(2.0)
        np.set_pos(0.0, de - 0.4, prof.y[-1] + 5.7)
        np.set_h(180.0)
        np.set_light_off()

    def _stands(self) -> None:
        spec, prof = self.spec, self.profile
        hw = spec.half_width
        concrete = MeshBuilder("stand_steps")
        crowd = MeshBuilder("stand_crowd")
        zones = [(prof.x_flat - 10.0, prof.x_flat + 55.0, 5), (prof.x_k - 18.0, prof.x_k + 18.0, 3)]
        for d0, d1, tiers in zones:
            for side in (-1.0, 1.0):
                self._stand(concrete, crowd, d0, d1, side, tiers)
        self._add(concrete)
        self._add(crowd, "crowd")

    def _stand(self, steps: MeshBuilder, crowd: MeshBuilder, d0: float, d1: float, side: float, tiers: int) -> None:
        hw = self.spec.half_width
        dc = (d0 + d1) / 2.0
        length = d1 - d0
        base0 = self.terrain_height(dc, side * (hw + 3.6))
        for k in range(tiers):
            lk = side * (hw + 3.6 + k * 1.4)
            ground = self.terrain_height(dc, lk)
            top = base0 + 1.0 + k * 1.0
            bottom = min(ground - 0.5, top - 1.0)
            steps.box((lk, dc, (top + bottom) / 2.0), (1.4, length, top - bottom), (0.62, 0.62, 0.66, 1.0))
            x = lk - side * 0.7
            crowd.quad((x, d0, top), (x, d1, top), (x, d1, top + 1.15), (x, d0, top + 1.15),
                       (-side, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0),
                       ((0, 0), (length / 12.0, 0), (length / 12.0, 1), (0, 1)))

    def _masts(self) -> None:
        spec, prof = self.spec, self.profile
        hw = spec.half_width
        poles = MeshBuilder("masts")
        lamps = MeshBuilder("lamps")
        d = 25.0
        while d < prof.x_end - 20.0:
            for side in (-1.0, 1.0):
                l = side * (hw + 11.0)
                z = self.terrain_height(d, l)
                poles.box((l, d, z + 8.0), (0.35, 0.35, 16.0), (0.40, 0.42, 0.46, 1.0))
                lamps.box((l - side * 1.0, d, z + 16.2), (3.0, 0.7, 0.9), (1.0, 0.98, 0.85, 1.0))
            d += 45.0
        self._add(poles)
        self._add(lamps, light_off=True)

    def _trees(self) -> None:
        """A forest of fir trees: trunk, three tiers of branches with snow on top."""
        spec, prof = self.spec, self.profile
        hw = spec.half_width
        mb = MeshBuilder("trees")
        count = 520 if spec.k_point < 150 else 700
        placed = 0
        tries = 0
        while placed < count and tries < count * 4:
            tries += 1
            side = self.rng.choice((-1.0, 1.0))
            td = self.start_d - 20.0 + self.rng.random() * (prof.x_end + abs(self.start_d) + 120.0)
            fw = self._flat_half_width(td)
            off = 12.0 + self.rng.random() ** 1.5 * 110.0
            tl = side * (fw + off)
            if td > 0.0 and off < 14.0 + 0.0:
                continue
            z = self.terrain_height(td, tl)
            tall = 7.0 + self.rng.random() * 10.0
            shade = 0.70 + 0.30 * self.rng.random()
            green = (0.08 * shade, 0.30 * shade, 0.14 * shade, 1.0)
            snow = (0.80, 0.86, 0.88, 1.0)
            mb.box((tl, td, z + tall * 0.1), (0.5, 0.5, tall * 0.2), (0.28, 0.18, 0.10, 1.0))
            for t in range(3):
                zt = z + tall * (0.12 + 0.26 * t)
                rad = (1.0 - 0.28 * t) * (1.5 + tall * 0.13)
                mb.cone(tl, td, zt, rad, tall * 0.42, green, snow if self.rng.random() < 0.55 else green, 8)
            placed += 1
        self._add(mb)

    def _flags(self) -> None:
        """Wind flags on poles; they are turned by update() to show the real wind."""
        spec, prof = self.spec, self.profile
        hw = spec.half_width
        poles = MeshBuilder("flagpoles")
        spots = [(-12.0, 4.5), (prof.x_k * 0.45, hw + 2.5), (prof.x_k * 0.45, -(hw + 2.5)),
                 (prof.x_k, -(hw + 2.5)), (prof.x_l, hw + 2.5)]
        for d, l in spots:
            z = self.terrain_height(d, l) if d > 0 else self._track_height(d) - 1.0
            poles.box((l, d, z + 4.5), (0.18, 0.18, 9.0), (0.85, 0.85, 0.88, 1.0))
            cloth = MeshBuilder("flag")
            cloth.quad((0, 0, 0), (0, 1.0, 0), (0, 1.0, 0.45), (0, 0, 0.45), (1, 0, 0), (1.0, 0.45, 0.05, 1.0))
            cloth.quad((0, 1.0, 0), (0, 2.0, 0), (0, 2.0, 0.45), (0, 1.0, 0.45), (1, 0, 0), (1.0, 1.0, 1.0, 1.0))
            np = cloth.build()
            np.reparent_to(self.root)
            np.set_two_sided(True)
            np.set_pos(l, d, z + 8.6)
            self.flags.append((np, d * 0.37))
        self._add(poles)

    def _markers(self) -> None:
        spec, prof = self.spec, self.profile
        hw = spec.half_width
        for s in range(10, int(spec.hill_size * 1.1) + 1, 10):
            for side in (-1.0, 1.0):
                self._label(f"{s}", s, side * (hw + 1.9), 1.8, (0.08, 0.08, 0.25, 1.0))
        self._label(f"K {spec.k_point:.0f}", spec.k_point, hw + 5.0, 3.4, (0.9, 0.08, 0.08, 1.0))
        self._label(f"HS {spec.hill_size:.0f}", spec.hill_size, hw + 5.0, 3.4, (0.10, 0.25, 0.95, 1.0))

    def _label(self, text: str, s: float, l: float, scale: float, color) -> None:
        p = self.profile
        d = p._interp(p.s, p.x, s)
        node = TextNode(f"lbl{text}")
        node.set_text(text)
        node.set_align(TextNode.A_center)
        node.set_text_color(*color)
        np = self.root.attach_new_node(node)
        np.set_scale(scale)
        np.set_pos(l, d, p.ground(d) + 1.5)
        np.set_billboard_axis()
        np.set_light_off()

    # ---- per frame -------------------------------------------------------------------------
    def update(self, t: float, wind: tuple[float, float, float]) -> None:
        """Turn the flags with the wind (wind = air velocity (d, l, h))."""
        wd, wl, _ = wind
        speed = math.hypot(wd, wl)
        for np, phase in self.flags:
            if speed > 0.05:
                np.set_h(math.degrees(math.atan2(-wl, wd)))          # +Y of the flag -> direction of the air
            droop = 1.0 - min(1.0, speed / 4.5)
            np.set_p(-70.0 * droop + 5.0 * math.sin(t * 7.0 + phase))
            np.set_r(7.0 * math.sin(t * 9.0 + phase) * min(1.0, 0.3 + speed / 5.0))
            np.set_sy(clamp(0.45 + speed / 7.0, 0.45, 1.5))

    def destroy(self) -> None:
        self.root.remove_node()


def _frange(start: float, stop: float, step: float):
    v = start
    while v < stop:
        yield v
        v += step
    yield stop


if __name__ == "__main__":
    print("This file is a library module of the ski jump game.")
    print("Start the game with:   python ski_jump.py")
