"""SKI JUMP  -  Python 3.13 + Panda3D

    >>>  START THE GAME WITH:      python ski_jump.py      <<<      (this is the main file)

Files that must lie in the same folder (they are libraries, you do not run them):
    ski_jump.py                 this file: window, menus, camera, HUD, game flow
    ski_physics.py              hills, weather, friction, projectile motion, scoring
    ski_competition.py          up to 10 players: training jump + 1st + 2nd round, standings
    ski_skier.py                the jumper (loads ski_jumper_very_basic.obj)
    ski_scenery.py              sky, mountains, textured hill, stands, trees, wind flags
    ski_sound.py                sound effects (generated into the folder "sounds")
    ski_geom.py                 mesh helpers
    ski_jumper_very_basic.obj   the jumper model (must be in this folder)

    pip install "panda3d>=1.10.15"

Controls: press H in the game, or see the "Controls" button in the settings menu.
"""

from __future__ import annotations

import math
import os
import random
import sys
from dataclasses import dataclass, field, replace

from direct.gui import DirectGuiGlobals as DGG
from direct.gui.DirectGui import DirectButton, DirectEntry, DirectFrame
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    AmbientLight,
    AntialiasAttrib,
    CardMaker,
    ClockObject,
    DirectionalLight,
    Fog,
    LineSegs,
    TextNode,
    TransparencyAttrib,
    Vec4,
    loadPrcFileData,
)

import ski_physics as sp
from ski_competition import MAX_PLAYERS, PLAYER_COLORS, ROUND_TITLES, Competition, JumpRecord
from ski_geom import clamp, make_box
from ski_scenery import Environment, HillScene, Textures, cloudiness_for
from ski_skier import MODEL_FILE, create_skier
from ski_sound import SoundManager

ROOT = os.path.dirname(os.path.abspath(__file__))
WHITE = (1, 1, 1, 1)
YELLOW = (1.0, 0.92, 0.35, 1)
LIGHTBLUE = (0.70, 0.85, 1.0, 1)
GREEN = (0.45, 1.0, 0.55, 1)
RED = (1.0, 0.50, 0.45, 1)
GREY = (0.75, 0.80, 0.90, 1)
BTN = (0.16, 0.24, 0.40, 1)
BTN_GREEN = (0.10, 0.50, 0.25, 1)
BTN_RED = (0.45, 0.15, 0.15, 1)

CAMERA_PRESETS: dict[str, tuple[float, float, float]] = {
    # name: (yaw deg, pitch deg, distance m); yaw 0 = behind the jumper, 90 = right side, 180 = in front
    "Chase": (0.0, 10.0, 13.0),
    "Side": (90.0, 4.0, 17.0),
    "Front": (180.0, 6.0, 15.0),
    "Top": (0.0, 75.0, 26.0),
    "Wide": (38.0, 14.0, 60.0),
    "Cockpit": (0.0, 0.0, 0.0),
}
CAMERA_ORDER = tuple(CAMERA_PRESETS)

HELP_SECTIONS = (
    ("THE JUMP", (
        ("SPACE", "start the attempt at the gate  /  telemark in flight  /  continue"),
        ("UP arrow", "JUMP: press it just BEFORE the table edge (watch the timing bar)"),
        ("", "in flight: UP = more angle of attack (more lift, more drag)"),
        ("DOWN arrow", "lean forward: less drag and lift (a tighter tuck on the inrun)"),
        ("LEFT / RIGHT", "lateral balance: steer against the cross wind"),
        ("SPACE (air)", "prepare the landing (telemark) 0.15 - 0.9 s before touch-down"),
    )),
    ("CAMERA (works all the time, also during the jump)", (
        ("mouse drag", "rotate freely around the jumper     wheel: zoom"),
        ("W A S D", "rotate the camera          Q / E: zoom in / out"),
        ("C or F1 - F6", "chase, side, front, top, wide, cockpit        R: reset this view"),
    )),
    ("OTHER", (
        ("H", "show / hide key help        M: sound on / off        T: slow motion"),
        ("ESC", "back to the settings menu (from the menu: quit)"),
        ("ENTER", "same as SPACE in the menus"),
    )),
)
HELP_COMPACT = (
    "SPACE start / telemark     UP jump at the nose, more lift     DOWN lean forward",
    "LEFT/RIGHT balance     C camera     drag mouse = rotate     H help     M sound     ESC menu",
)

NAME_POOL = ("Anna", "Matti", "Liisa", "Jukka", "Sofia", "Eino", "Helmi", "Onni", "Aino", "Veeti")


def find_model() -> str | None:
    """The jumper model must be in the game's root folder (next to this file)."""
    for folder in (ROOT, os.getcwd(), os.path.join(ROOT, "models"), os.path.join(ROOT, "_files")):
        path = os.path.join(folder, MODEL_FILE)
        if os.path.exists(path):
            return path
    return None


def describe_wind(deg: float) -> str:
    names = ("head wind", "head, from right", "from the right", "tail, from right",
             "tail wind", "tail, from left", "from the left", "head, from left")
    return names[int(round(deg / 45.0)) % 8]


def angle_diff(target: float, current: float) -> float:
    return (target - current + 180.0) % 360.0 - 180.0


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# --------------------------------------------------------------------------
#  Settings
# --------------------------------------------------------------------------
@dataclass
class Settings:
    hill: str = sp.DEFAULT_HILL
    preset: str = "Calm winter day"
    wind_speed: float = 0.0
    wind_from: float = 0.0
    gustiness: float = 0.0
    temperature: float = -4.0
    snow: str = "packed"
    body_mass: float = 65.0
    gate: int = sp.HILLS[sp.DEFAULT_HILL].gate_ref
    camera: str = "Chase"
    wind_changes: bool = True          # the wind changes a little from jump to jump
    sound: bool = True
    mode: str = "competition"          # "competition" or "free"
    n_players: int = 2
    names: list[str] = field(default_factory=lambda: [f"Player {i + 1}" for i in range(MAX_PLAYERS)])

    def apply_preset(self, name: str) -> None:
        w = sp.WEATHER_PRESETS[name]
        self.preset = name
        self.wind_speed, self.wind_from = w.wind_speed, w.wind_from_deg
        self.gustiness, self.temperature, self.snow = w.gustiness, w.temperature_c, w.snow

    def weather(self) -> sp.Weather:
        return sp.Weather(self.wind_speed, self.wind_from, self.gustiness,
                          self.temperature, self.snow, self.preset)


def estimate_edge_speed(settings: Settings) -> float:
    spec = sp.HILLS[settings.hill]
    sim = sp.JumpSimulation(spec, settings.weather(), sp.Jumper(body_mass=settings.body_mass),
                            settings.gate, seed=1)
    sim.start()
    while sim.phase is sp.Phase.INRUN:
        sim.advance(0.02)
    return sim.edge_speed


# --------------------------------------------------------------------------
#  The application
# --------------------------------------------------------------------------
class SkiApp(ShowBase):
    BAR_T0, BAR_T1 = 0.6, -0.10        # timing bar: 0.6 s before the edge ... 0.1 s after

    def __init__(self) -> None:
        loadPrcFileData("", "window-title Ski Jump\n"
                            "win-size 1280 800\n"
                            "framebuffer-multisample 1\nmultisamples 4\n")
        super().__init__()
        self.disable_mouse()
        self.set_background_color(0.62, 0.78, 0.96)
        self.render.set_antialias(AntialiasAttrib.M_multisample)

        self.rng = random.Random()
        self.settings = Settings()
        self.scene: HillScene | None = None
        self.sim: sp.JumpSimulation | None = None
        self.comp: Competition | None = None
        self.cur_player = None
        self.cur_weather: sp.Weather | None = None
        self.state = "menu"
        self.time_scale = 1.0
        self.typing = False
        self.help_mode = 1                    # 0 hidden, 1 compact, 2 full overlay
        self.keys = {k: False for k in ("up", "down", "left", "right", "w", "a", "s", "d", "q", "e")}
        self.page: list = []
        self.menu_values: dict[str, OnscreenText] = {}
        self.entries: list[DirectEntry] = []
        self.help_widgets: list = []
        self._last_phase: sp.Phase | None = None
        self._landed_time = 0.0
        self._result_recorded = False
        self.best_total = 0.0
        self.jump_no = 0
        self.landing_marker = None
        self.wind_np = None
        self._wind_timer = 0.0

        self.textures = Textures()
        self._setup_lights()
        self.env = Environment(self.render, self.camera, self.textures)
        self._setup_skier()
        self._setup_shadow()
        self.sound = SoundManager(self.loader, os.path.join(ROOT, "sounds"), self.settings.sound)
        self._setup_camera()
        self._setup_hud()
        self._setup_input()
        self._rebuild_scene()
        print("[game]", self.skier_info)
        print("[game]", self.sound.describe())
        self.show_menu()
        self.sound.play("beep", 0.5)                 # a short beep: you hear at once that sound works
        self.task_mgr.add(self._update, "update")

    # ================================================================= setup
    def _setup_lights(self) -> None:
        ambient = AmbientLight("ambient")
        ambient.set_color(Vec4(0.50, 0.53, 0.62, 1))
        self.ambient_np = self.render.attach_new_node(ambient)
        self.render.set_light(self.ambient_np)
        self.sun = DirectionalLight("sun")
        self.sun.set_color(Vec4(0.90, 0.86, 0.76, 1))
        self.sun_np = self.render.attach_new_node(self.sun)
        self.render.set_light(self.sun_np)
        self.fog = Fog("fog")
        self.fog.set_exp_density(0.0011)
        self.render.set_fog(self.fog)

    def _apply_look(self) -> None:
        """Sun direction and fog colour follow the environment (weather)."""
        vx, vy, vz = self.env.sun_light_direction
        self.sun_np.set_pos(0, 0, 0)
        self.sun_np.look_at(vx, vy, vz)
        k = 1.0 - 0.55 * self.env.cloudiness
        self.sun.set_color(Vec4(0.90 * k, 0.86 * k, 0.76 * k, 1))
        r, g, b = self.env.fog_color
        self.fog.set_color(r, g, b)
        self.set_background_color(r, g, b)

    def _setup_skier(self) -> None:
        path = find_model()
        if path is None:
            self.skier = create_skier(self.loader, self.render, os.path.join(ROOT, MODEL_FILE))
            self.skier_info = (f"WARNING: {MODEL_FILE} not found in {ROOT} - using a simple box figure. "
                               f"Put the .obj file in the game folder.")
        else:
            self.skier = create_skier(self.loader, self.render, path)
            if self.skier.kind == "model":
                self.skier_info = f"jumper model: {path}  (loaded via {self.skier.source})"
            else:
                self.skier_info = f"WARNING: {path} could not be used - using a simple box figure"
        self.skier.root.hide()

    def _setup_shadow(self) -> None:
        cm = CardMaker("shadow")
        cm.set_frame(-0.9, 0.9, -1.7, 1.7)
        self.shadow = self.render.attach_new_node(cm.generate())
        self.shadow.set_texture(self.textures.get("sun"))          # soft round alpha blob
        self.shadow.set_transparency(TransparencyAttrib.M_alpha)
        self.shadow.set_color(0.0, 0.0, 0.05, 0.6)
        self.shadow.set_light_off()
        self.shadow.set_depth_write(False)
        self.shadow.set_two_sided(True)
        self.shadow.hide()

    def _setup_camera(self) -> None:
        self.camLens.set_fov(65.0)
        self.camLens.set_near_far(0.2, 4000.0)
        self.cam_mode = self.settings.camera
        yaw, pitch, dist = CAMERA_PRESETS[self.cam_mode]
        self.yaw = self.t_yaw = yaw
        self.pitch = self.t_pitch = pitch
        self.dist = self.t_dist = dist
        self.cp_yaw = self.cp_pitch = 0.0
        self.cam_target = [0.0, 0.0, 0.0]
        self.cam_dir = [0.0, 1.0, 0.0]
        self.dragging = False
        self.last_mouse = (0.0, 0.0)

    def _setup_hud(self) -> None:
        def text(parent, pos, scale, fg=WHITE, align=TextNode.A_center, shadow=(0, 0, 0, 0.9)):
            return OnscreenText(text="", parent=parent, pos=pos, scale=scale, fg=fg, shadow=shadow,
                                align=align, mayChange=True)

        self.hud = text(self.a2dTopLeft, (0.05, -0.09), 0.052, align=TextNode.A_left)
        self.big = text(self.aspect2d, (0, 0.72), 0.10, YELLOW, shadow=(0, 0, 0, 1))
        self.sub = text(self.aspect2d, (0, 0.61), 0.05, shadow=(0, 0, 0, 1))
        self.cue = text(self.aspect2d, (0, -0.42), 0.06, shadow=(0, 0, 0, 1))
        self.comp_text = text(self.a2dTopCenter, (0, -0.07), 0.052, YELLOW)
        self.cam_text = text(self.a2dBottomRight, (-0.05, 0.05), 0.038, align=TextNode.A_right)
        self.keys_text = text(self.a2dBottomLeft, (0.05, 0.11), 0.036, GREY, align=TextNode.A_left)
        self.wind_text = text(self.a2dTopRight, (-0.05, -0.08), 0.045, align=TextNode.A_right)
        # take-off timing bar
        self.bar = DirectFrame(parent=self.aspect2d, frameColor=(0.05, 0.07, 0.12, 0.75),
                               frameSize=(-0.62, 0.62, -0.03, 0.03), pos=(0, 0, -0.55))
        self.bar_good = DirectFrame(parent=self.bar, frameColor=(0.85, 0.75, 0.15, 0.9),
                                    frameSize=(0, 1, -0.03, 0.03))
        self.bar_perfect = DirectFrame(parent=self.bar, frameColor=(0.2, 0.9, 0.3, 0.95),
                                       frameSize=(0, 1, -0.03, 0.03))
        self.bar_marker = DirectFrame(parent=self.bar, frameColor=WHITE,
                                      frameSize=(-0.008, 0.008, -0.05, 0.05))
        OnscreenText(text="press UP in the green zone - just before the edge", parent=self.bar,
                     pos=(0, 0.06), scale=0.04, fg=WHITE, shadow=(0, 0, 0, 1))
        # tau = seconds before the edge: good = timing quality >= 0.7, perfect >= 0.95
        for frame, (a, b) in ((self.bar_good, (0.089, 0.005)), (self.bar_perfect, (0.055, 0.024))):
            frame["frameSize"] = (self._bar_x(a), self._bar_x(b), -0.03, 0.03)
        self.bar.hide()

    def _bar_x(self, tau: float) -> float:
        return -0.62 + (self.BAR_T0 - tau) / (self.BAR_T0 - self.BAR_T1) * 1.24

    def _setup_input(self) -> None:
        for ev, key in (("arrow_up", "up"), ("arrow_down", "down"), ("arrow_left", "left"),
                        ("arrow_right", "right"), ("w", "w"), ("a", "a"), ("s", "s"), ("d", "d"),
                        ("q", "q"), ("e", "e")):
            self.accept(ev, self._key_down, [key])
            self.accept(ev + "-up", self._key_up, [key])
        for key in ("space", "enter"):
            self.accept(key, self._on_space)
        self.accept("escape", self._on_escape)
        self.accept("c", self._cycle_camera)
        self.accept("r", self._reset_camera)
        self.accept("t", self._toggle_slow)
        self.accept("h", self._cycle_help)
        self.accept("m", self._toggle_sound)
        for i, name in enumerate(CAMERA_ORDER):
            self.accept(f"f{i + 1}", self._set_camera, [name])
        self.accept("wheel_up", self._zoom, [-1])
        self.accept("wheel_down", self._zoom, [1])
        self.accept("mouse1", self._mouse_down)
        self.accept("mouse1-up", self._mouse_up)

    # ================================================================= scene
    def _rebuild_scene(self) -> None:
        if self.scene is not None:
            self.scene.destroy()
        self.scene = HillScene(self.render, sp.HILLS[self.settings.hill], self.settings.gate, self.textures)
        self._rebuild_environment()
        self._remove_landing_marker()
        self._place_skier_at_gate()

    def _rebuild_environment(self) -> None:
        c = cloudiness_for(self.settings.weather())
        self.env.rebuild(self.scene.center_d, self.scene.valley, c)
        self._apply_look()

    def _weather_changed(self) -> None:
        c = cloudiness_for(self.settings.weather())
        if abs(c - self.env.cloudiness) > 0.04:
            self._rebuild_environment()

    def _place_skier_at_gate(self) -> None:
        d, h = self.scene.inrun.position(0.0)
        self.skier.root.set_pos(0.0, d, h)
        self.skier.root.set_hpr(0.0, -math.degrees(self.scene.inrun.gamma), 0.0)
        self.skier.set_pose(1.0, 0.0, False)
        self.skier.snap()
        self.cam_target = [0.0, d, h + 0.9]

    def _remove_landing_marker(self) -> None:
        if self.landing_marker is not None:
            self.landing_marker.remove_node()
            self.landing_marker = None

    # ================================================================= input
    def _key_down(self, key: str) -> None:
        if self.typing:
            return
        self.keys[key] = True
        if key == "up" and self.state == "run" and self.sim is not None:
            self.sim.press_up()

    def _key_up(self, key: str) -> None:
        self.keys[key] = False

    def _set_typing(self, value: bool) -> None:
        self.typing = value
        if value:
            for k in self.keys:
                self.keys[k] = False

    def _on_space(self) -> None:
        if self.typing:
            return
        if self.state == "announce":
            self.prepare_jump()
        elif self.state == "ready":
            self.sim.start()
            self.state = "run"
            self.big.setText("")
            self.sub.setText("")
            self.sound.play("beep", 0.7)
        elif self.state == "run" and self.sim is not None:
            self.sim.press_telemark()
        elif self.state == "result":
            self._after_result()
        elif self.state == "table":
            self.comp.start_next_round()
            self.show_announce()
        elif self.state == "final":
            self.show_menu()

    def _on_escape(self) -> None:
        if self.typing:
            return
        if self.help_widgets:
            self._close_help()
        elif self.state == "menu":
            self.userExit()
        elif self.state == "players":
            self.show_menu()
        else:
            self.show_menu()

    def _toggle_slow(self) -> None:
        if not self.typing:
            self.time_scale = 0.3 if self.time_scale == 1.0 else 1.0

    def _toggle_sound(self) -> None:
        if not self.typing:
            self.settings.sound = self.sound.toggle()
            if self.state == "menu":
                self._refresh_menu()

    def _cycle_help(self) -> None:
        if self.typing:
            return
        if self.help_widgets:
            self._close_help()
            return
        self.help_mode = (self.help_mode + 1) % 3
        if self.help_mode == 2:
            self._open_help()
            self.help_mode = 1

    def _mouse_pos(self):
        mw = self.mouseWatcherNode
        if mw.has_mouse():
            return mw.get_mouse_x(), mw.get_mouse_y()
        return None

    def _mouse_down(self) -> None:
        pos = self._mouse_pos()
        if pos is not None and not self.typing:
            self.dragging = True
            self.last_mouse = pos

    def _mouse_up(self) -> None:
        self.dragging = False

    def _zoom(self, direction: int) -> None:
        self.t_dist = clamp(self.t_dist * (1.1 ** direction), 3.0, 160.0)
        self.dist = self.t_dist

    def _set_camera(self, name: str) -> None:
        if self.typing:
            return
        self.cam_mode = name
        self.settings.camera = name
        yaw, pitch, dist = CAMERA_PRESETS[name]
        self.t_yaw, self.t_pitch, self.t_dist = yaw, pitch, dist
        self.cp_yaw = self.cp_pitch = 0.0

    def _cycle_camera(self) -> None:
        self._set_camera(CAMERA_ORDER[(CAMERA_ORDER.index(self.cam_mode) + 1) % len(CAMERA_ORDER)])

    def _reset_camera(self) -> None:
        self._set_camera(self.cam_mode)

    def _rotate_camera(self, dyaw: float, dpitch: float) -> None:
        if self.cam_mode == "Cockpit":
            self.cp_yaw += dyaw
            self.cp_pitch = clamp(self.cp_pitch + dpitch, -85.0, 85.0)
        else:
            self.yaw += dyaw
            self.t_yaw = self.yaw
            self.pitch = clamp(self.pitch + dpitch, -10.0, 89.0)
            self.t_pitch = self.pitch

    # ================================================================= widgets
    def _page_frame(self, half_w: float, half_h: float, z: float = 0.0, color=(0.04, 0.07, 0.14, 0.9)):
        frame = DirectFrame(parent=self.aspect2d, frameColor=color,
                            frameSize=(-half_w, half_w, -half_h, half_h), pos=(0, 0, z))
        self.page.append(frame)
        return frame

    def _text(self, frame, text, x, z, scale=0.05, fg=WHITE, align=TextNode.A_left, bucket=None):
        w = OnscreenText(text=text, parent=frame, pos=(x, z), scale=scale, fg=fg, align=align,
                         mayChange=True)
        (self.page if bucket is None else bucket).append(w)
        return w

    def _button(self, frame, text, x, z, cmd, extra=(), width=0.07, scale=0.05, color=BTN, bucket=None):
        b = DirectButton(parent=frame, text=text, text_scale=scale, text_fg=WHITE,
                         frameSize=(-width, width, -0.035, 0.07), frameColor=color, relief=DGG.FLAT,
                         pos=(x, 0, z), command=cmd, extraArgs=list(extra))
        (self.page if bucket is None else bucket).append(b)
        return b

    @staticmethod
    def _destroy_all(widgets: list) -> None:
        for w in reversed(widgets):            # children were added after their frame
            try:
                w.destroy()
            except Exception:                  # already gone together with its parent
                pass

    def _clear_page(self) -> None:
        self._destroy_all(self.page)
        self.page = []
        self.menu_values = {}
        self.entries = []
        self.typing = False

    def _table(self, frame, headers, rows, xs, aligns, y0, step=0.072, scale=0.044,
               colors=None, header_color=LIGHTBLUE) -> None:
        for h, x, a in zip(headers, xs, aligns):
            self._text(frame, h, x, y0, scale, header_color, a)
        for r, row in enumerate(rows):
            y = y0 - (r + 1) * step
            color = colors[r] if colors else WHITE
            for cell, x, a in zip(row, xs, aligns):
                self._text(frame, cell, x, y, scale, color, a)

    # ================================================================= help
    def _open_help(self) -> None:
        self._close_help()
        frame = DirectFrame(parent=self.aspect2d, frameColor=(0.02, 0.05, 0.10, 0.95),
                            frameSize=(-1.25, 1.25, -0.85, 0.85), pos=(0, 0, 0))
        self.help_widgets.append(frame)

        def label(text, x, z, scale, fg, align=TextNode.A_left):
            w = OnscreenText(text=text, parent=frame, pos=(x, z), scale=scale, fg=fg, align=align)
            self.help_widgets.append(w)

        label("KEYBOARD CONTROLS", 0.0, 0.74, 0.085, YELLOW, TextNode.A_center)
        z = 0.62
        for title, lines in HELP_SECTIONS:
            label(title, -1.15, z, 0.052, LIGHTBLUE)
            z -= 0.075
            for key, desc in lines:
                label(key, -1.12, z, 0.044, YELLOW)
                label(desc, -0.55, z, 0.040, WHITE)
                z -= 0.065
            z -= 0.04
        label(self.skier_info, -1.15, -0.70, 0.032, GREY)
        label(self.sound.describe(), -1.15, -0.745, 0.032, GREY)
        label("press H or ESC to close", 0.0, -0.81, 0.045, YELLOW, TextNode.A_center)

    def _close_help(self) -> None:
        self._destroy_all(self.help_widgets)
        self.help_widgets = []

    # ================================================================= settings menu
    ROWS = ("hill", "preset", "wind_speed", "wind_from", "gustiness", "temperature", "snow",
            "body_mass", "gate", "camera", "wind_changes", "sound")
    NAMES = {"hill": "Hill", "preset": "Weather", "wind_speed": "Wind speed", "wind_from": "Wind from",
             "gustiness": "Gusts", "temperature": "Temperature", "snow": "Snow / track",
             "body_mass": "Jumper weight", "gate": "Start gate", "camera": "Camera",
             "wind_changes": "Wind varies per jump", "sound": "Sound"}

    def _enter_menu_state(self) -> None:
        self._clear_page()
        self._close_help()
        self.state = "menu"
        self.sim = None
        self.cur_player = None
        self.big.setText("")
        self.sub.setText("")
        self.comp_text.setText("")
        self.cue.setText("")
        self.skier.root.hide()
        self.shadow.hide()
        self.sound.stop_all()

    def show_menu(self) -> None:
        self._enter_menu_state()
        s = self.settings
        frame = self._page_frame(1.12, 0.93)
        self._text(frame, "SKI JUMP", 0.0, 0.80, 0.10, YELLOW, TextNode.A_center)
        z = 0.67
        for key in self.ROWS:
            self._text(frame, self.NAMES[key], -1.05, z)
            self._button(frame, "<", 0.02, z, self._change, [key, -1])
            self.menu_values[key] = self._text(frame, "", 0.12, z, 0.047, LIGHTBLUE)
            self._button(frame, ">", 0.98, z, self._change, [key, 1])
            z -= 0.085
        self.menu_values["info"] = self._text(frame, "", 0.0, -0.40, 0.037, LIGHTBLUE, TextNode.A_center)
        self.menu_values["info2"] = self._text(frame, "", 0.0, -0.46, 0.037, LIGHTBLUE, TextNode.A_center)
        self.menu_values["players"] = self._text(frame, "", 0.0, -0.55, 0.042, YELLOW, TextNode.A_center)
        self._button(frame, "Players & competition", -0.70, -0.70, self.show_players, width=0.40, scale=0.045)
        self._button(frame, "Controls (H)", -0.05, -0.70, self._open_help, width=0.30, scale=0.045)
        self._button(frame, "START", 0.45, -0.72, self.start_game, width=0.20, scale=0.06, color=BTN_GREEN)
        self._button(frame, "Quit", 0.85, -0.70, self.userExit, width=0.17, scale=0.045, color=BTN_RED)
        self._text(frame, self.skier_info, -1.05, -0.84, 0.028, GREY)
        self._refresh_menu()

    def _change(self, key: str, step: int) -> None:
        s = self.settings
        if key == "hill":
            keys = list(sp.HILLS)
            s.hill = keys[(keys.index(s.hill) + step) % len(keys)]
            s.gate = sp.HILLS[s.hill].gate_ref
            self._rebuild_scene()
        elif key == "preset":
            names = list(sp.WEATHER_PRESETS)
            i = names.index(s.preset) if s.preset in names else -1
            s.apply_preset(names[(i + step) % len(names)])
        elif key == "wind_speed":
            s.wind_speed = clamp(round(s.wind_speed + 0.5 * step, 1), 0.0, 12.0)
        elif key == "wind_from":
            s.wind_from = (s.wind_from + 15.0 * step) % 360.0
        elif key == "gustiness":
            s.gustiness = clamp(round(s.gustiness + 0.05 * step, 2), 0.0, 0.6)
        elif key == "temperature":
            s.temperature = clamp(s.temperature + step, -25.0, 8.0)
        elif key == "snow":
            s.snow = sp.SNOW_TYPES[(sp.SNOW_TYPES.index(s.snow) + step) % len(sp.SNOW_TYPES)]
        elif key == "body_mass":
            s.body_mass = clamp(s.body_mass + step, 45.0, 95.0)
        elif key == "gate":
            s.gate = int(clamp(s.gate + step, 1, sp.HILLS[s.hill].gate_max))
            self._rebuild_scene()
        elif key == "camera":
            self._set_camera(CAMERA_ORDER[(CAMERA_ORDER.index(s.camera) + step) % len(CAMERA_ORDER)])
        elif key == "wind_changes":
            s.wind_changes = not s.wind_changes
        elif key == "sound":
            s.sound = self.sound.toggle()
            if s.sound:
                self.sound.play("beep", 0.5)
        if key in ("wind_speed", "wind_from", "gustiness", "temperature", "snow"):
            s.preset = "Custom"
        if key in ("preset", "temperature", "snow", "gustiness", "wind_speed"):
            self._weather_changed()
        self._refresh_menu()

    def _refresh_menu(self) -> None:
        if "hill" not in self.menu_values:
            return
        s = self.settings
        spec = sp.HILLS[s.hill]
        v = self.menu_values
        v["hill"].setText(spec.name)
        v["preset"].setText(s.preset)
        v["wind_speed"].setText(f"{s.wind_speed:.1f} m/s")
        v["wind_from"].setText(f"{s.wind_from:.0f} deg  ({describe_wind(s.wind_from)})")
        v["gustiness"].setText(f"{s.gustiness * 100:.0f} %")
        v["temperature"].setText(f"{s.temperature:+.0f} C")
        v["snow"].setText(s.snow)
        v["body_mass"].setText(f"{s.body_mass:.0f} kg  (+{sp.Jumper().equipment_mass:.1f} kg equipment)")
        v["gate"].setText(f"{s.gate}  of {spec.gate_max}")
        v["camera"].setText(s.camera)
        v["wind_changes"].setText("yes" if s.wind_changes else "no")
        v["sound"].setText("on" if s.sound else "off (M)")
        w = s.weather()
        rho = sp.air_density(s.temperature, spec.altitude)
        v["info"].setText(f"friction mu {w.friction():.3f}   air density {rho:.3f} kg/m3   "
                          f"altitude {spec.altitude:.0f} m   g {sp.G:.3f} m/s2")
        v["info2"].setText(f"inrun {self.scene.inrun.length:.0f} m   expected edge speed "
                           f"{estimate_edge_speed(s) * 3.6:.1f} km/h   K {spec.k_point:.0f}   HS {spec.hill_size:.0f}")
        if s.mode == "competition":
            v["players"].setText(f"Competition: {s.n_players} player{'s' if s.n_players > 1 else ''} - "
                                 f"training jump + 2 rounds")
        else:
            v["players"].setText("Free jumping (single jumps, no competition)")

    # ================================================================= players page
    def show_players(self) -> None:
        self._clear_page()
        s = self.settings
        frame = self._page_frame(1.12, 0.93)
        self._text(frame, "PLAYERS & COMPETITION", 0.0, 0.80, 0.08, YELLOW, TextNode.A_center)
        self._text(frame, "Mode", -1.0, 0.66)
        self._button(frame, "<", -0.20, 0.66, self._players_change, ["mode", -1])
        self.menu_values["mode"] = self._text(frame, "", -0.10, 0.66, 0.047, LIGHTBLUE)
        self._button(frame, ">", 0.78, 0.66, self._players_change, ["mode", 1])
        self._text(frame, "Players (max 10)", -1.0, 0.56)
        self._button(frame, "-", -0.20, 0.56, self._players_change, ["n", -1])
        self.menu_values["n"] = self._text(frame, "", -0.10, 0.56, 0.047, LIGHTBLUE)
        self._button(frame, "+", 0.78, 0.56, self._players_change, ["n", 1])
        self._text(frame, "Type a name for each jumper (click a field).", -1.0, 0.45, 0.04, GREY)
        self.entries = []
        for i in range(MAX_PLAYERS):
            col, row = divmod(i, 5)
            x = -1.0 + col * 1.05
            z = 0.33 - row * 0.13
            swatch = DirectFrame(parent=frame, frameColor=(*PLAYER_COLORS[i], 1),
                                 frameSize=(-0.025, 0.025, -0.025, 0.025), pos=(x + 0.03, 0, z + 0.016))
            self.page.append(swatch)
            self._text(frame, f"{i + 1}", x + 0.09, z, 0.045, YELLOW)
            entry = DirectEntry(parent=frame, text="", scale=0.05, width=11, numLines=1,
                                pos=(x + 0.17, 0, z), initialText=s.names[i], focus=0,
                                frameColor=(0.10, 0.16, 0.28, 1), text_fg=WHITE,
                                focusInCommand=self._set_typing, focusInExtraArgs=[True],
                                focusOutCommand=self._set_typing, focusOutExtraArgs=[False])
            self.page.append(entry)
            self.entries.append(entry)
        self._text(frame, "Order: training jump and 1st round in the order above.  2nd round: the leader "
                          "jumps last.", -1.0, -0.40, 0.034, GREY)
        self._text(frame, "Points: distance + style (5 judges) + wind and gate compensation, as in real "
                          "ski jumping.", -1.0, -0.455, 0.034, GREY)
        self._button(frame, "Back", -0.55, -0.70, self._players_back, width=0.25, scale=0.05)
        self._button(frame, "START", 0.30, -0.72, self._players_start, width=0.25, scale=0.06,
                     color=BTN_GREEN)
        self._players_refresh()

    def _read_names(self) -> None:
        for i, e in enumerate(self.entries):
            try:
                name = e.get().strip()
            except Exception:
                name = ""
            self.settings.names[i] = name[:14] or f"Player {i + 1}"

    def _players_back(self) -> None:
        self._read_names()
        self.show_menu()

    def _players_start(self) -> None:
        self._read_names()
        self.start_game()

    def _players_change(self, what: str, step: int) -> None:
        s = self.settings
        if what == "mode":
            s.mode = "free" if s.mode == "competition" else "competition"
        else:
            s.n_players = int(clamp(s.n_players + step, 1, MAX_PLAYERS))
        self._players_refresh()

    def _players_refresh(self) -> None:
        s = self.settings
        if "mode" not in self.menu_values:
            return
        comp = s.mode == "competition"
        self.menu_values["mode"].setText("Competition (training + 2 rounds)" if comp else "Free jumping")
        self.menu_values["n"].setText(str(s.n_players))
        for i, e in enumerate(self.entries):
            if comp and i < s.n_players:
                e.show()
            else:
                e.hide()

    # ================================================================= game flow
    def start_game(self) -> None:
        s = self.settings
        self._clear_page()
        self.sound.play("beep", 0.6)
        if s.mode == "competition":
            self.comp = Competition(s.names[:s.n_players])
            self.best_total = 0.0
            self.show_announce()
        else:
            self.comp = None
            self.prepare_jump()

    def _player_label(self) -> str:
        if self.comp is None or self.cur_player is None:
            return ""
        return f"{self.comp.round_title}  -  {self.cur_player.name}  ({self.comp.turn_number}/{self.comp.round_size})"

    def show_announce(self) -> None:
        """Announces the next jumper (and where he/she is in the start list)."""
        self._clear_page()
        self.state = "announce"
        self.sim = None
        self.sound.stop_all()
        comp = self.comp
        player = comp.current_player
        self.cur_player = player
        self.skier.set_suit_color(player.color)
        self._place_skier_at_gate()
        self.skier.root.show()
        self.shadow.hide()
        self.comp_text.setText(comp.round_title)
        frame = self._page_frame(0.95, 0.62, z=-0.05)
        self._text(frame, comp.round_title.upper(), 0.0, 0.50, 0.08, YELLOW, TextNode.A_center)
        self._text(frame, f"Jump {comp.turn_number} of {comp.round_size}", 0.0, 0.40, 0.05, LIGHTBLUE,
                   TextNode.A_center)
        self._text(frame, "Next jumper:", 0.0, 0.27, 0.05, WHITE, TextNode.A_center)
        self._text(frame, player.name, 0.0, 0.14, 0.11, (*player.color, 1), TextNode.A_center)
        nxt = comp.next_player
        self._text(frame, f"then: {nxt.name}" if nxt else "last jumper of this round", 0.0, 0.03, 0.045,
                   GREY, TextNode.A_center)
        if comp.round_key == "training":
            self._text(frame, "Training jump - the points are not counted", 0.0, -0.08, 0.045, GREY,
                       TextNode.A_center)
        else:
            order = ", ".join(p.name for p in comp.start_list)
            self._text(frame, f"Start list: {order}", 0.0, -0.08, 0.036, GREY, TextNode.A_center)
        self._text(frame, "Pass the keyboard to the jumper, then press SPACE", 0.0, -0.30, 0.05, YELLOW,
                   TextNode.A_center)
        self._text(frame, "SPACE = go to the start gate", 0.0, -0.42, 0.042, WHITE, TextNode.A_center)

    def _jump_weather(self) -> sp.Weather:
        w = self.settings.weather()
        if not self.settings.wind_changes:
            return w
        base = w.wind_speed if w.wind_speed > 0 else self.rng.uniform(0.0, 1.2)
        return replace(w, wind_speed=round(clamp(base * self.rng.uniform(0.6, 1.4), 0.0, 12.0), 1),
                       wind_from_deg=(w.wind_from_deg + self.rng.uniform(-30.0, 30.0)) % 360.0)

    def prepare_jump(self) -> None:
        self._clear_page()
        s = self.settings
        spec = sp.HILLS[s.hill]
        if self.scene is None or self.scene.spec is not spec or self.scene.gate != s.gate:
            self._rebuild_scene()
        self._remove_landing_marker()
        self.cur_weather = self._jump_weather()
        self.sim = sp.JumpSimulation(spec, self.cur_weather, sp.Jumper(body_mass=s.body_mass), s.gate,
                                     seed=self.rng.randrange(1 << 30))
        if self.comp is not None:
            self.cur_player = self.comp.current_player
            self.skier.set_suit_color(self.cur_player.color)
        else:
            self.cur_player = None
            self.skier.set_suit_color(PLAYER_COLORS[0])
        self.state = "ready"
        self._last_phase = sp.Phase.READY
        self._landed_time = 0.0
        self._result_recorded = False
        self.jump_no += 1
        self.skier.root.show()
        self._place_skier_at_gate()
        self._set_camera(s.camera)
        self.sound.stop_all()
        self.big.setText("Press SPACE to start")
        who = f"{self.cur_player.name}   " if self.cur_player else ""
        self.sub.setText(f"{who}{spec.name}   gate {s.gate}   wind {self.cur_weather.wind_speed:.1f} m/s")
        self.comp_text.setText(self._player_label())

    def _after_step(self) -> None:
        sim = self.sim
        if sim.phase is not self._last_phase:
            if sim.phase is sp.Phase.FLIGHT:
                self.big.setText(sim.takeoff_note)
                self.sub.setText("")
                self.sound.play("takeoff", 0.9)
            elif sim.phase is sp.Phase.OUTRUN:
                land = sim.landing
                self.big.setText("Crash!" if land.crashed else f"{land.distance:.1f} m")
                self.sub.setText(f"telemark: {land.telemark}   fall height {land.fall_height:.1f} m")
                self._add_landing_marker()
                if land.crashed:
                    self.sound.play("crash", 1.0)
                else:
                    self.sound.play("landing", 0.9)
                    k = sim.spec.k_point
                    self.sound.play("cheer", clamp(0.25 + (land.distance - 0.8 * k) / (0.45 * k), 0.25, 1.0))
            self._last_phase = sim.phase
        if sim.phase is sp.Phase.FINISHED or (sim.phase is sp.Phase.OUTRUN and self._landed_time > 4.5):
            if self.state == "run":
                self._show_result()

    def _add_landing_marker(self) -> None:
        sim = self.sim
        marker = make_box(0.5, "flag")
        marker.set_scale(0.15, 0.15, 3.0)
        marker.set_pos(sim.l, sim.landing.d, self.scene.profile.ground(sim.landing.d) + 1.5)
        marker.set_color(1.0, 0.15, 0.15, 1.0)
        marker.reparent_to(self.render)
        self.landing_marker = marker

    # ================================================================= per frame
    def _update(self, task):
        dt = min(ClockObject.get_global_clock().get_dt(), 0.05)
        if self.state == "run" and self.sim is not None:
            controls = sp.Controls(up=self.keys["up"], down=self.keys["down"],
                                   left=self.keys["left"], right=self.keys["right"])
            self.sim.advance(dt * self.time_scale, controls)
            if self.sim.phase in (sp.Phase.OUTRUN, sp.Phase.FINISHED):
                self._landed_time += dt
            self._after_step()
        self._update_sound()
        self._update_skier(dt)
        self._update_camera(dt)
        self._update_hud(dt)
        wind = self.sim.wind.vector() if self.sim is not None else self.settings.weather().mean_wind()
        self.scene.update(task.time, wind)
        return task.cont

    def _update_sound(self) -> None:
        sim = self.sim
        if sim is None or self.state != "run":
            self.sound.update("none", 0.0)
            return
        phase = {sp.Phase.INRUN: "inrun", sp.Phase.FLIGHT: "flight", sp.Phase.OUTRUN: "outrun"}.get(sim.phase, "none")
        self.sound.update(phase, sim.speed())

    def _skier_state(self):
        """(l, d, h, pitch, roll, pose, v_angle, telemark) of the jumper for drawing."""
        sim = self.sim
        if sim is None:
            return None
        prof = self.scene.profile
        phase = sim.phase
        if phase in (sp.Phase.READY, sp.Phase.INRUN):
            d, h = sim.inrun.position(sim.s)
            pose = 1.0
            if phase is sp.Phase.INRUN and sim.up_presses and sim.s > sim.inrun.s_table - 8.0:
                pose = 0.5                                   # standing up for the jump
            return 0.0, d, h, -math.degrees(sim.inrun.angle(sim.s)), 0.0, pose, 0.0, False
        if phase is sp.Phase.FLIGHT:
            roll = clamp(-sim.vl * 3.0, -25.0, 25.0)
            return sim.l, sim.d, sim.h, sim.pitch_deg, roll, 0.0, 24.0, sim.t_tele is not None
        roll, pose = 0.0, 0.5
        if sim.landing is not None and sim.landing.crashed:
            roll = (self._landed_time * 420.0) % 360.0 if sim.v > 0.5 else 90.0
            pose = 0.9
        return sim.l, sim.d, prof.ground(sim.d), -math.degrees(prof.slope(sim.d)), roll, pose, 0.0, False

    def _update_skier(self, dt: float) -> None:
        st = self._skier_state()
        if st is None:
            return
        l, d, h, pitch, roll, pose, vang, tele = st
        self.skier.root.set_pos(l, d, h)
        self.skier.root.set_hpr(0.0, pitch, roll)
        self.skier.set_pose(pose, vang, tele)
        self.skier.update(dt)
        cockpit = self.cam_mode == "Cockpit" and self.state in ("run", "ready")
        if cockpit:
            self.skier.root.hide()
        else:
            self.skier.root.show()
        # soft shadow on the hill below the flying jumper
        if self.sim is not None and self.sim.phase is sp.Phase.FLIGHT and not cockpit:
            prof = self.scene.profile
            g = prof.ground(d)
            hag = max(0.0, h - g)
            self.shadow.show()
            self.shadow.set_pos(l, d, g + 0.12)
            self.shadow.set_p(-90.0 - math.degrees(prof.slope(d)))
            self.shadow.set_scale(1.0 + 0.06 * hag)
            self.shadow.set_alpha_scale(clamp(1.0 / (1.0 + 0.12 * hag), 0.15, 1.0))
        else:
            self.shadow.hide()

    def _skier_velocity(self):
        sim = self.sim
        if sim is None:
            return (0.0, 1.0, 0.0)
        if sim.phase is sp.Phase.FLIGHT:
            return (sim.vl, sim.vd, sim.vh)
        if sim.phase in (sp.Phase.READY, sp.Phase.INRUN):
            a = sim.inrun.angle(sim.s)
            return (0.0, math.cos(a), -math.sin(a))
        b = self.scene.profile.slope(sim.d)
        return (0.0, math.cos(b), -math.sin(b))

    def _update_camera(self, dt: float) -> None:
        k = self.keys
        spin = 90.0 * dt
        self._rotate_camera((-spin if k["a"] else 0.0) + (spin if k["d"] else 0.0),
                            (-spin * 0.7 if k["w"] else 0.0) + (spin * 0.7 if k["s"] else 0.0))
        if k["q"]:
            self.t_dist = self.dist = clamp(self.dist * (1.0 - 0.9 * dt), 3.0, 160.0)
        if k["e"]:
            self.t_dist = self.dist = clamp(self.dist * (1.0 + 0.9 * dt), 3.0, 160.0)
        if self.dragging:
            pos = self._mouse_pos()
            if pos is not None:
                dx, dy = pos[0] - self.last_mouse[0], pos[1] - self.last_mouse[1]
                self.last_mouse = pos
                self._rotate_camera(-dx * 140.0, -dy * 100.0)
        ease = 1.0 - math.exp(-dt * 5.0)
        self.yaw += angle_diff(self.t_yaw, self.yaw) * ease
        self.pitch += (self.t_pitch - self.pitch) * ease
        self.dist += (self.t_dist - self.dist) * ease

        scene = self.scene
        if self.state in ("menu", "players", "help") or self.sim is None and self.state != "announce":
            d0 = scene.profile.x_k * 0.45
            tgt = [0.0, d0, scene.profile.ground(d0) + 25.0]
        elif self.sim is None:                                    # announce: look at the jumper at the gate
            d, h = scene.inrun.position(0.0)
            tgt = [0.0, d, h + 0.9]
        else:
            st = self._skier_state()
            tgt = [st[0], st[1], st[2] + 0.9]
        follow = 1.0 - math.exp(-dt / 0.05)
        for i in range(3):
            self.cam_target[i] += (tgt[i] - self.cam_target[i]) * follow

        x, y, z = self.cam_target
        in_game = self.state in ("run", "ready") and self.sim is not None
        if self.cam_mode == "Cockpit" and in_game:
            vx, vy, vz = self._skier_velocity()
            n = math.sqrt(vx * vx + vy * vy + vz * vz) or 1.0
            sm = 1.0 - math.exp(-dt * 6.0)
            for i, v in enumerate((vx / n, vy / n, vz / n)):
                self.cam_dir[i] += (v - self.cam_dir[i]) * sm
            fx, fy, fz = self.cam_dir
            ln = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
            heading = math.degrees(math.atan2(-fx, fy)) + self.cp_yaw
            pitch = math.degrees(math.asin(clamp(fz / ln, -1.0, 1.0)))
            self.camera.set_pos(x, y, z + 0.2)
            self.camera.set_hpr(heading, pitch + self.cp_pitch, 0.0)
            return
        yaw, pit = math.radians(self.yaw), math.radians(self.pitch)
        dist = self.dist if self.dist > 1.0 else 20.0
        cx = x + dist * math.sin(yaw) * math.cos(pit)
        cy = y - dist * math.cos(yaw) * math.cos(pit)
        cz = z + dist * math.sin(pit)
        gz = scene.terrain_height(cy, cx) + 1.2              # keep the camera above the snow
        self.camera.set_pos(cx, cy, max(cz, gz))
        self.camera.look_at(x, y, z)

    # ================================================================= HUD
    def _update_hud(self, dt: float) -> None:
        sim = self.sim
        self.cam_text.setText(f"View: {self.cam_mode}   (C / F1-F6, drag = rotate, wheel = zoom)"
                              + ("   SLOW" if self.time_scale < 1.0 else "")
                              + ("" if self.sound.enabled else "   MUTED"))
        if self.help_mode == 1 and self.state in ("ready", "run", "announce"):
            self.keys_text.setText("\n".join(HELP_COMPACT))
        else:
            self.keys_text.setText("")
        if sim is None or self.state not in ("ready", "run", "result"):
            self.hud.setText("")
            self.bar.hide()
            self.cue.setText("")
            self.wind_text.setText("")
            self._draw_wind(0.0, 0.0, 0.0)
            return
        lines = [f"Speed    {sim.speed() * 3.6:6.1f} km/h"]
        if sim.phase in (sp.Phase.READY, sp.Phase.INRUN):
            lines.append(f"To edge  {sim.distance_to_edge():6.1f} m   (gate {sim.gate})")
        elif sim.phase is sp.Phase.FLIGHT:
            lines.append(f"Height   {sim.height_above_ground():6.1f} m")
            lines.append(f"Distance {sim.profile.surface_distance(sim.d):6.1f} m")
            lines.append(f"Attack angle {sim.alpha:4.1f} deg  (cmd {sim.alpha_cmd:4.1f})")
            side = "right" if sim.l > 0.05 else "left" if sim.l < -0.05 else ""
            lines.append(f"Line     {abs(sim.l):4.1f} m {side}")
        lines.append(f"Time     {sim.t:6.2f} s")
        self.hud.setText("\n".join(lines))

        self.cue.setText("")
        if sim.phase is sp.Phase.INRUN:
            tau = sim.time_to_edge()
            if tau < self.BAR_T0 + 0.02:
                self.bar.show()
                self.bar_marker.set_x(clamp(self._bar_x(tau), -0.62, 0.62))
            else:
                self.bar.hide()
        else:
            self.bar.hide()
        if sim.phase is sp.Phase.FLIGHT:
            self._flight_cues()

        wd, wl, _ = sim.wind.vector()
        self._wind_timer += dt
        if self._wind_timer > 0.05:
            self._wind_timer = 0.0
            self._draw_wind(wd, wl, math.hypot(wd, wl))
        head = -wd
        self.wind_text.setText(
            f"Wind {math.hypot(wd, wl):3.1f} m/s   {'head' if head > 0 else 'tail'} {abs(head):3.1f}"
            f"   cross {'<-' if wl < 0 else '->'} {abs(wl):3.1f}\n"
            f"{sim.weather.temperature_c:+.0f} C   {sim.weather.snow}   mu {sim.mu:.3f}")

    def _flight_cues(self) -> None:
        sim = self.sim
        hag = sim.height_above_ground()
        beta = sim.profile.slope(sim.d)
        v_n = -(sim.vd * math.sin(beta) + sim.vh * math.cos(beta))
        eta = (hag * math.cos(beta)) / max(v_n, 0.5)
        if sim.t_tele is not None:
            self.cue.setText("telemark ready")
            self.cue.setFg(GREEN)
        elif hag < 8.0 and v_n > 0.0:
            if 0.15 <= eta <= 0.9:
                self.cue.setText("SPACE: TELEMARK NOW!")
                self.cue.setFg(GREEN)
            elif eta > 0.9:
                self.cue.setText(f"landing in {eta:.1f} s - get ready (SPACE = telemark)")
                self.cue.setFg(YELLOW)
            else:
                self.cue.setText("too late for the telemark")
                self.cue.setFg(RED)

    def _draw_wind(self, wd: float, wl: float, speed: float) -> None:
        """Compass: up = jump direction. The arrow shows where the air is moving."""
        if self.wind_np is not None:
            self.wind_np.remove_node()
            self.wind_np = None
        if speed <= 0.0 and self.state not in ("ready", "run", "result"):
            return
        ls = LineSegs("wind")
        ls.set_thickness(2.0)
        ls.set_color(0.8, 0.85, 0.95, 0.8)
        r = 0.11
        for i in range(25):
            a = 2 * math.pi * i / 24
            (ls.move_to if i == 0 else ls.draw_to)(r * math.sin(a), 0, r * math.cos(a))
        ls.move_to(0, 0, r)
        ls.draw_to(0, 0, r * 1.25)
        if speed > 0.05:
            length = min(0.10, 0.015 * speed + 0.02)
            ux, uz = wl / speed, wd / speed
            ls.set_thickness(4.0)
            if wd < 0:
                ls.set_color(0.35, 1.0, 0.45, 1)
            else:
                ls.set_color(1.0, 0.45, 0.35, 1)
            hx, hz = ux * length, uz * length
            ls.move_to(-hx, 0, -hz)
            ls.draw_to(hx, 0, hz)
            px, pz = -uz, ux
            ls.draw_to(hx - ux * 0.04 + px * 0.025, 0, hz - uz * 0.04 + pz * 0.025)
            ls.move_to(hx, 0, hz)
            ls.draw_to(hx - ux * 0.04 - px * 0.025, 0, hz - uz * 0.04 - pz * 0.025)
        self.wind_np = self.a2dTopRight.attach_new_node(ls.create())
        self.wind_np.set_pos(-0.2, 0, -0.34)

    # ================================================================= results
    def _show_result(self) -> None:
        if self._result_recorded:
            return
        self._result_recorded = True
        sim = self.sim
        sc, land = sim.score, sim.landing
        self.state = "result"
        self.big.setText("")
        self.sub.setText("")
        self._clear_page()
        self.best_total = max(self.best_total, sc.total)
        player = self.cur_player
        counted = True
        rank_line = ""
        if self.comp is not None and player is not None:
            counted = self.comp.round_key != "training"
            self.comp.record(JumpRecord.from_score(sc, land))
            if counted:
                row = next(r for r in self.comp.standings() if r.player is player)
                rank_line = f"Standing now: {ordinal(row.rank)}   ({row.total:.1f} points)"
            else:
                rank_line = "Training jump - not counted"
        frame = self._page_frame(0.72, 0.66, z=-0.04, color=(0.04, 0.07, 0.14, 0.86))
        head = f"{player.name}:  " if player else f"Jump {self.jump_no}:  "
        title = "CRASH" if land.crashed else f"{sc.distance:.1f} m"
        self._text(frame, head + title, 0.0, 0.54, 0.075, (*player.color, 1) if player else YELLOW,
                   TextNode.A_center)
        if self.comp is not None:
            self._text(frame, self.comp.round_title, 0.0, 0.46, 0.042, LIGHTBLUE, TextNode.A_center)
        rows = [("Distance points", f"{sc.distance_points:7.1f}"),
                ("Style points", f"{sc.style_points:7.1f}"),
                ("Wind compensation", f"{sc.wind_points:+7.1f}"),
                ("Gate compensation", f"{sc.gate_points:+7.1f}")]
        z = 0.36
        for label, value in rows:
            self._text(frame, label, -0.62, z, 0.05)
            self._text(frame, value, 0.62, z, 0.05, WHITE, TextNode.A_right)
            z -= 0.075
        self._text(frame, "Judges  " + "  ".join(f"{j:4.1f}" for j in sc.judges), -0.62, z + 0.01, 0.038, GREY)
        z -= 0.085
        self._text(frame, "TOTAL", -0.62, z, 0.062, GREEN)
        self._text(frame, f"{sc.total:7.1f}", 0.62, z, 0.062, GREEN, TextNode.A_right)
        z -= 0.085
        if rank_line:
            self._text(frame, rank_line, 0.0, z, 0.048, YELLOW, TextNode.A_center)
            z -= 0.07
        self._text(frame, f"telemark: {land.telemark}   fall height {land.fall_height:.1f} m   "
                          f"airtime {land.airtime:.1f} s   edge {sim.edge_speed * 3.6:.0f} km/h",
                   0.0, z, 0.034, LIGHTBLUE, TextNode.A_center)
        z -= 0.06
        notes = ", ".join(sc.notes) if sc.notes else "clean jump"
        self._text(frame, f"Notes: {notes}", 0.0, z, 0.036, (1.0, 0.8, 0.5, 1), TextNode.A_center)
        self._text(frame, "SPACE: continue        ESC: settings", 0.0, -0.58, 0.048, WHITE, TextNode.A_center)

    def _after_result(self) -> None:
        """SPACE on the result page."""
        if self.comp is None:
            self.prepare_jump()
        elif self.comp.round_complete:
            if self.comp.finished:
                self.show_final()
            else:
                self.show_round_table()
        else:
            self.show_announce()

    def show_round_table(self) -> None:
        """After the training jump and after the 1st round."""
        self._clear_page()
        self.state = "table"
        self.sim = None
        self.skier.root.hide()
        self.shadow.hide()
        self.sound.stop_all()
        comp = self.comp
        frame = self._page_frame(1.0, 0.8)
        if comp.round_key == "training":
            self._text(frame, "TRAINING JUMP - DISTANCES", 0.0, 0.68, 0.075, YELLOW, TextNode.A_center)
            table = comp.training_table()
            rows = [[str(r), p.name, f"{d:.1f} m"] for r, p, d in table]
            colors = [(*p.color, 1) for _, p, _ in table]
            self._table(frame, ["#", "Jumper", "Distance"], rows, [-0.85, -0.65, 0.75],
                        [TextNode.A_left, TextNode.A_left, TextNode.A_right], 0.52, colors=colors)
            nxt = "Next: 1st round"
        else:
            self._text(frame, "STANDING AFTER THE 1ST ROUND", 0.0, 0.68, 0.075, YELLOW, TextNode.A_center)
            standing = comp.standings()
            rows = [[str(r.rank), r.player.name, f"{r.total:.1f}", f"{r.best_distance:.1f} m"] for r in standing]
            colors = [(*r.player.color, 1) for r in standing]
            self._table(frame, ["#", "Jumper", "Points", "Distance"], rows, [-0.85, -0.65, 0.35, 0.85],
                        [TextNode.A_left, TextNode.A_left, TextNode.A_right, TextNode.A_right], 0.52,
                        colors=colors)
            nxt = "Next: 2nd round - the leader jumps last"
        self._text(frame, nxt, 0.0, -0.68, 0.048, LIGHTBLUE, TextNode.A_center)
        self._text(frame, "SPACE: continue", 0.0, -0.75, 0.045, YELLOW, TextNode.A_center)

    def show_final(self) -> None:
        self._clear_page()
        self.state = "final"
        self.sim = None
        self.skier.root.hide()
        self.shadow.hide()
        self.sound.stop_all()
        self.sound.play("cheer", 1.0)
        comp = self.comp
        standing = comp.standings()
        frame = self._page_frame(1.12, 0.86)
        self._text(frame, "FINAL STANDINGS", 0.0, 0.74, 0.09, YELLOW, TextNode.A_center)
        winner = standing[0].player
        self._text(frame, f"Winner: {winner.name}", 0.0, 0.63, 0.06, (*winner.color, 1), TextNode.A_center)
        rows, colors = [], []
        for r in standing:
            fmt = lambda v: "-" if v is None else f"{v:.1f}"
            rows.append([str(r.rank), r.player.name, fmt(r.round1), fmt(r.round2), f"{r.total:.1f}",
                         f"{r.best_distance:.1f} m"])
            colors.append((*r.player.color, 1))
        self._table(frame, ["#", "Jumper", "1st round", "2nd round", "TOTAL", "Best"], rows,
                    [-1.0, -0.85, 0.15, 0.50, 0.80, 1.05],
                    [TextNode.A_left, TextNode.A_left, TextNode.A_right, TextNode.A_right,
                     TextNode.A_right, TextNode.A_right], 0.52, step=0.066, colors=colors)
        self._text(frame, "SPACE: back to the settings menu", 0.0, -0.78, 0.048, YELLOW, TextNode.A_center)


def main() -> None:
    if sys.version_info < (3, 10):
        sys.exit("Python 3.10 or newer is required (3.13 recommended).")
    SkiApp().run()


if __name__ == "__main__":
    main()
