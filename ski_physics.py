"""Ski jumping physics and scoring (pure Python 3.13, no Panda3D).

Coordinates (metres, SI units)
    d  horizontal distance from the take-off edge, positive in the jump direction
    l  lateral offset, positive to the jumper's right
    h  height relative to the take-off edge, positive up
The wind is the velocity of the *air* in (d, l, h).  Wind direction is given as
the direction the wind blows FROM, clockwise from the jump direction:
0 deg = head wind, 90 deg = from the right, 180 deg = tail wind, 270 = from the left.

Model
    inrun   rolling on a (gate dependent) track of straight / circular / table
            sections: gravity, snow friction (mu * normal force, including the
            centripetal part in the curve) and air drag relative to the wind.
    flight  point mass with weight m*g, aerodynamic drag and lift from the
            relative air velocity (density from temperature and altitude),
            lift/drag coefficients that depend on the angle of attack (with
            stall), the take-off leg force as an additional upward force, a
            lateral balance force and a wind with gusts.
    landing impact speed normal to the slope ("equivalent fall height"),
            telemark timing, lateral drift and a slide through the outrun.
    scoring FIS style: K-point = 60 points, points per metre depend on the hill,
            five judges (best and worst dropped), wind and gate compensation.
The numbers (hill geometry, aerodynamic coefficients, friction) are realistic
approximations, tuned so that a K100 hill gives ~25.4 m/s (91 km/h) at the edge
and ~100-110 m jumps; they are not official FIS data.
"""

from __future__ import annotations

import bisect
import math
import random
from dataclasses import dataclass, field
from enum import Enum

G = 9.80665          # standard gravity, m/s^2
R_AIR = 287.05       # specific gas constant of dry air, J/(kg K)

# --------------------------------------------------------------------------
#  Weather
# --------------------------------------------------------------------------
SNOW_TYPES = ("ice", "packed", "wet", "slush")
# kinetic friction coefficient of waxed skis in the track at about -6 C
_SNOW_MU = {"ice": 0.016, "packed": 0.022, "wet": 0.034, "slush": 0.050}


def snow_friction(snow: str, temp_c: float) -> float:
    """Ski/track friction coefficient: best glide near -6 C, worse when colder or warmer."""
    return _SNOW_MU[snow] * (1.0 + 0.018 * abs(temp_c + 6.0))


def air_density(temp_c: float, altitude_m: float) -> float:
    """Dry air density from the barometric formula and the ideal gas law (kg/m^3)."""
    pressure = 101325.0 * (1.0 - 2.25577e-5 * altitude_m) ** 5.25588
    return pressure / (R_AIR * (temp_c + 273.15))


@dataclass
class Weather:
    wind_speed: float = 0.0        # mean wind speed, m/s
    wind_from_deg: float = 0.0     # 0 head, 90 from right, 180 tail, 270 from left
    gustiness: float = 0.15        # gust standard deviation as a fraction of the mean wind
    temperature_c: float = -4.0
    snow: str = "packed"
    label: str = "Custom"

    def friction(self) -> float:
        return snow_friction(self.snow, self.temperature_c)

    def mean_wind(self) -> tuple[float, float, float]:
        """Mean air velocity (d, l, h).  A head wind over a slope gives some updraft."""
        a = math.radians(self.wind_from_deg)
        wd = -self.wind_speed * math.cos(a)
        wl = -self.wind_speed * math.sin(a)
        wh = 0.10 * max(0.0, -wd)
        return wd, wl, wh


WEATHER_PRESETS: dict[str, Weather] = {
    "Calm winter day": Weather(0.0, 0.0, 0.0, -4.0, "packed", "Calm winter day"),
    "Light head wind": Weather(2.0, 0.0, 0.15, -5.0, "packed", "Light head wind"),
    "Strong head wind": Weather(5.5, 10.0, 0.20, -8.0, "packed", "Strong head wind"),
    "Tail wind": Weather(2.5, 180.0, 0.20, -3.0, "packed", "Tail wind"),
    "Cross wind from right": Weather(4.0, 90.0, 0.25, -5.0, "packed", "Cross wind from right"),
    "Gusty cross wind": Weather(5.0, 300.0, 0.45, -2.0, "packed", "Gusty cross wind"),
    "Cold, hard ice track": Weather(1.0, 20.0, 0.10, -16.0, "ice", "Cold, hard ice track"),
    "Warm, wet snow": Weather(1.5, 40.0, 0.20, 1.5, "wet", "Warm, wet snow"),
    "Thaw, slush": Weather(2.0, 160.0, 0.25, 3.0, "slush", "Thaw, slush"),
}


class Wind:
    """Mean wind plus gusts (Ornstein-Uhlenbeck noise in speed and direction)."""

    TAU = 1.6   # gust correlation time, s

    def __init__(self, weather: Weather, rng: random.Random) -> None:
        self.weather = weather
        self.rng = rng
        self.speed_dev = 0.0
        self.dir_dev = 0.0

    def step(self, dt: float) -> None:
        w = self.weather
        decay = math.exp(-dt / self.TAU)
        noise = math.sqrt(1.0 - decay * decay)
        sigma_v = w.gustiness * max(w.wind_speed, 1.0)
        sigma_a = math.radians(18.0) * min(1.0, w.gustiness * 3.0)
        self.speed_dev = self.speed_dev * decay + sigma_v * noise * self.rng.gauss(0, 1)
        self.dir_dev = self.dir_dev * decay + sigma_a * noise * self.rng.gauss(0, 1)

    def vector(self) -> tuple[float, float, float]:
        w = self.weather
        speed = max(0.0, w.wind_speed + self.speed_dev)
        a = math.radians(w.wind_from_deg) + self.dir_dev
        wd = -speed * math.cos(a)
        wl = -speed * math.sin(a)
        wh = 0.10 * max(0.0, -wd)
        return wd, wl, wh


# --------------------------------------------------------------------------
#  Hills
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class HillSpec:
    key: str
    name: str
    k_point: float            # K: distance of the construction point along the slope, m
    hill_size: float          # HS, m
    inrun_angle: float        # gamma, deg
    takeoff_angle: float      # alpha: table inclination below horizontal, deg
    table_length: float       # m
    inrun_radius: float       # r1: transition curve radius, m
    inrun_straight: float     # straight inrun length from the reference gate, m
    gate_step: float          # extra track length per gate, m
    gate_ref: int             # gate the nominal length belongs to
    gate_max: int
    table_height: float       # table edge above the slope below it, m
    slope_start: float        # landing profile slope angle at the table, deg
    slope_p: float            # at P (start of the landing area), deg
    slope_k: float            # at K, deg
    slope_l: float            # at L (end of the landing area), deg
    p_ratio: float            # P / K
    l_ratio: float            # L / HS
    landing_radius: float     # r2: transition to the outrun, m
    outrun_length: float      # m
    altitude: float           # m above sea level
    meter_value: float        # points per metre
    head_wind_pts: float      # points per m/s head wind (deducted)
    tail_wind_pts: float      # points per m/s tail wind (added)
    gate_pts: float           # points per gate
    half_width: float = 10.0  # half width of the landing hill, m
    typical_speed: float = 25.0   # documentation only: edge speed, m/s


HILLS: dict[str, HillSpec] = {
    "K90": HillSpec(
        "K90", "K90 normal hill (HS 100)", 90, 100, 35.0, 10.5, 6.0, 90.0, 47.0, 0.9, 12, 25,
        2.9, 26.0, 28.0, 35.0, 35.5, 0.74, 1.10, 130.0, 80.0, 700.0, 2.0, 5.0, 6.5, 2.0,
        typical_speed=24.5),
    "K100": HillSpec(
        "K100", "K100 hill (HS 112)", 100, 112, 35.0, 11.0, 6.5, 100.0, 51.0, 0.9, 12, 25,
        3.0, 26.0, 30.0, 36.0, 36.5, 0.75, 1.10, 160.0, 90.0, 800.0, 1.8, 5.5, 7.0, 3.8,
        typical_speed=25.4),
    "K120": HillSpec(
        "K120", "K120 large hill (HS 134)", 120, 134, 35.0, 11.0, 6.5, 105.0, 55.0, 0.9, 12, 25,
        3.1, 28.0, 32.0, 36.0, 36.5, 0.76, 1.10, 200.0, 100.0, 900.0, 1.8, 6.0, 7.5, 4.0,
        typical_speed=26.0),
    "K185": HillSpec(
        "K185", "K185 flying hill (HS 225)", 185, 225, 36.0, 10.5, 7.0, 150.0, 63.0, 1.0, 12, 25,
        3.0, 32.0, 34.0, 36.0, 36.5, 0.72, 1.10, 300.0, 200.0, 1000.0, 1.2, 4.0, 5.0, 3.6,
        typical_speed=28.3),
}
DEFAULT_HILL = "K100"


class Inrun:
    """The track from the start gate to the table edge: straight, circular arc, table."""

    def __init__(self, spec: HillSpec, gate: int) -> None:
        self.spec = spec
        self.gate = gate
        self.gamma = math.radians(spec.inrun_angle)
        self.alpha = math.radians(spec.takeoff_angle)
        self.r1 = spec.inrun_radius
        self.l_straight = spec.inrun_straight + (gate - spec.gate_ref) * spec.gate_step
        self.l_arc = self.r1 * (self.gamma - self.alpha)
        self.l_table = spec.table_length
        self.s_table = self.l_straight + self.l_arc      # table starts here
        self.length = self.s_table + self.l_table        # table edge
        # position of the edge when the track starts at (0, 0); used to shift to edge = (0, 0)
        self._end = self._raw_position(self.length)

    def angle(self, s: float) -> float:
        """Track angle below horizontal (rad) at track position s."""
        if s <= self.l_straight:
            return self.gamma
        if s <= self.s_table:
            return self.gamma - (s - self.l_straight) / self.r1
        return self.alpha

    def curvature(self, s: float) -> float:
        """1/r in the concave transition curve, else 0."""
        return 1.0 / self.r1 if self.l_straight < s <= self.s_table else 0.0

    def _raw_position(self, s: float) -> tuple[float, float]:
        g, r1 = self.gamma, self.r1
        if s <= self.l_straight:
            return s * math.cos(g), -s * math.sin(g)
        x0, h0 = self.l_straight * math.cos(g), -self.l_straight * math.sin(g)
        if s <= self.s_table:
            th = g - (s - self.l_straight) / r1
            return x0 + r1 * (math.sin(g) - math.sin(th)), h0 - r1 * (math.cos(th) - math.cos(g))
        x1, h1 = self._raw_position(self.s_table)
        u = s - self.s_table
        return x1 + u * math.cos(self.alpha), h1 - u * math.sin(self.alpha)

    def position(self, s: float) -> tuple[float, float]:
        """(d, h) of the track at s, with the table edge at (0, 0)."""
        x, h = self._raw_position(min(s, self.length))
        return x - self._end[0], h - self._end[1]

    @property
    def start_height(self) -> float:
        return self.position(0.0)[1]


class LandingProfile:
    """The landing hill and outrun below the table edge.

    The slope angle is piecewise linear along the slope (start, P, K, L), then a
    circular transition of radius r2 flattens it into the outrun.  Arrays are
    sampled every 0.25 m of slope length.
    """

    DS = 0.25

    def __init__(self, spec: HillSpec) -> None:
        self.spec = spec
        s_p = spec.p_ratio * spec.k_point
        s_k = spec.k_point
        s_l = spec.l_ratio * spec.hill_size
        pts = [(0.0, spec.slope_start), (s_p, spec.slope_p), (s_k, spec.slope_k), (s_l, spec.slope_l)]
        self.s_p, self.s_k, self.s_l = s_p, s_k, s_l
        l_trans = spec.landing_radius * math.radians(spec.slope_l)
        s_flat = s_l + l_trans
        s_end = s_flat + spec.outrun_length

        def theta(s: float) -> float:
            if s <= s_l:
                for (s0, a0), (s1, a1) in zip(pts, pts[1:]):
                    if s <= s1:
                        return math.radians(a0 + (a1 - a0) * (s - s0) / (s1 - s0))
                return math.radians(spec.slope_l)
            if s <= s_flat:
                return math.radians(spec.slope_l) * (1.0 - (s - s_l) / l_trans)
            return 0.0

        self.s: list[float] = []
        self.x: list[float] = []
        self.y: list[float] = []          # surface height relative to the table edge
        self.th: list[float] = []         # slope angle below horizontal (rad)
        x, y, s = 0.0, -spec.table_height, 0.0
        while s <= s_end + 1e-9:
            t = theta(s)
            self.s.append(s)
            self.x.append(x)
            self.y.append(y)
            self.th.append(t)
            tm = theta(s + self.DS / 2)
            x += self.DS * math.cos(tm)
            y -= self.DS * math.sin(tm)
            s += self.DS
        self.x_p = self._interp(self.s, self.x, s_p)
        self.x_k = self._interp(self.s, self.x, s_k)
        self.x_l = self._interp(self.s, self.x, s_l)
        self.x_flat = self._interp(self.s, self.x, s_flat)
        self.x_end = self.x[-1]

    @staticmethod
    def _interp(xs: list[float], ys: list[float], x: float) -> float:
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        i = bisect.bisect_right(xs, x) - 1
        f = (x - xs[i]) / (xs[i + 1] - xs[i])
        return ys[i] + f * (ys[i + 1] - ys[i])

    def ground(self, d: float) -> float:
        """Surface height (relative to the table edge) at horizontal distance d."""
        if d <= 0.0:
            return -self.spec.table_height
        return self._interp(self.x, self.y, d)

    def slope(self, d: float) -> float:
        """Slope angle below horizontal (rad) at distance d."""
        return self._interp(self.x, self.th, max(d, 0.0))

    def surface_distance(self, d: float) -> float:
        """Distance along the slope from the edge to horizontal distance d (how jumps are measured)."""
        return self._interp(self.x, self.s, max(d, 0.0))

    @property
    def k_height(self) -> float:
        return self._interp(self.s, self.y, self.s_k)


# --------------------------------------------------------------------------
#  Jumper and aerodynamics
# --------------------------------------------------------------------------
@dataclass
class Jumper:
    body_mass: float = 65.0        # kg, athlete without equipment
    equipment_mass: float = 4.5    # kg, skis, boots, suit, helmet
    leg_strength: float = 1.6      # peak take-off force as a multiple of body weight at 70 kg

    @property
    def mass(self) -> float:
        return self.body_mass + self.equipment_mass

    def scale(self) -> float:
        """Geometric scale of areas with body mass (area ~ mass^(2/3))."""
        return (self.body_mass / 70.0) ** (2.0 / 3.0)

    def inrun_cda(self) -> float:
        """Drag area C_D*A in the crouched inrun position, m^2."""
        return 0.32 * self.scale()

    def flight_area(self) -> float:
        """Reference area in flight: body (scales with mass) plus the two skis, m^2."""
        return 0.60 * self.scale() + 0.45

    def push_force(self) -> float:
        """Peak optimal take-off force, N.  Strength grows slower than mass (mass^0.85)."""
        return self.leg_strength * G * self.body_mass ** 0.85 * 70.0 ** 0.15


CN1 = 1.30          # normal-force slope of the jumper + skis (flat plate like)
CD0 = 0.10          # parasitic drag coefficient
FLIGHT_RAMP = 1.0   # s: after take-off the flight position builds up (lift and drag area 45% -> 100%)
SIDE_CDA = 1.60     # side drag area C*A of the jumper seen from the side, m^2
ALPHA_STALL = 38.0  # angle of attack where the flow starts to separate, deg
ALPHA_MIN, ALPHA_MAX = 5.0, 50.0
ALPHA_OPT_LOW, ALPHA_OPT_HIGH = 18.0, 32.0      # band the judges like


def _smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def aero_coefficients(alpha_deg: float) -> tuple[float, float]:
    """Lift and drag coefficient of the jumper-ski system at angle of attack alpha."""
    a = math.radians(max(0.0, min(90.0, alpha_deg)))
    stall = _smoothstep((alpha_deg - (ALPHA_STALL - 3.0)) / 10.0)
    cn = CN1 * math.sin(a) * (1.0 - 0.38 * stall)
    return cn * math.cos(a), CD0 + cn * math.sin(a) + 0.20 * stall


def timing_quality(e: float) -> float:
    """Bell shaped quality (0..1) of the take-off timing.

    e = time of the 'up' press relative to passing the table edge, s
    (negative = before the edge).  Best a few hundredths of a second before the
    edge, so the window is only ~0.1 s wide, narrower on the late side."""
    e0 = -0.035
    width = 0.09 if e < e0 else 0.05
    return math.exp(-(((e - e0) / width) ** 2))


# --------------------------------------------------------------------------
#  Simulation
# --------------------------------------------------------------------------
class Phase(Enum):
    READY = "ready"
    INRUN = "inrun"
    FLIGHT = "flight"
    OUTRUN = "outrun"
    FINISHED = "finished"


@dataclass
class Controls:
    up: bool = False      # raise the angle of attack (lean back): more lift, more drag
    down: bool = False    # lean forward: less drag, less lift
    left: bool = False    # lateral balance
    right: bool = False


@dataclass
class Landing:
    distance: float            # measured along the slope, m (0.5 m resolution)
    d: float                   # horizontal distance, m
    lateral: float             # lateral offset at touchdown, m
    normal_speed: float        # speed into the slope, m/s
    fall_height: float         # equivalent fall height v_n^2 / (2 g), m
    lateral_speed: float
    alpha: float               # angle of attack at touchdown, deg
    telemark: str              # 'good', 'early', 'late' or 'none'
    crashed: bool
    airtime: float
    flight_speed: float        # speed at touchdown, m/s


@dataclass
class Score:
    distance: float
    distance_points: float
    judges: list[float]
    style_points: float
    wind_points: float
    gate_points: float
    total: float
    notes: list[str] = field(default_factory=list)


class JumpSimulation:
    """One jump from the start gate to the end of the outrun."""

    DT = 1.0 / 240.0
    PUSH_TIME = 0.22          # duration of the take-off leg force, s
    K_LATERAL = 0.12          # lateral balance force, as a fraction of weight
    ALPHA_RATE = 14.0         # how fast the player changes the commanded angle, deg/s
    ALPHA_TAU = 0.28          # body reaction time constant, s
    ALPHA_START = 24.0        # angle of attack just after take-off, deg
    MISS_QUALITY = 0.30       # leg force fraction when 'up' is not pressed in time
    STAND_DRAG = 1.35         # extra drag while standing up on the table

    def __init__(self, spec: HillSpec | None = None, weather: Weather | None = None,
                 jumper: Jumper | None = None, gate: int | None = None,
                 seed: int | None = None) -> None:
        self.spec = spec or HILLS[DEFAULT_HILL]
        self.weather = weather or Weather()
        self.jumper = jumper or Jumper()
        self.gate = self.spec.gate_ref if gate is None else gate
        if not 1 <= self.gate <= self.spec.gate_max:
            raise ValueError("gate out of range")
        self.rng = random.Random(seed)
        self.inrun = Inrun(self.spec, self.gate)
        self.profile = LandingProfile(self.spec)
        self.wind = Wind(self.weather, self.rng)
        self.rho = air_density(self.weather.temperature_c, self.spec.altitude)
        self.mu = self.weather.friction()
        self.mu_land = self.mu * 2.4        # sliding on the landing hill / outrun
        self.mass = self.jumper.mass

        self.phase = Phase.READY
        self.t = 0.0
        self._accum = 0.0
        # track state
        self.s = 0.0
        self.v = 0.0
        # flight / outrun state: position of the skis and velocity, m and m/s
        self.d, self.l, self.h = self.inrun.position(0.0)[0], 0.0, self.inrun.position(0.0)[1]
        self.vd = self.vl = self.vh = 0.0
        self.alpha = self.alpha_cmd = self.ALPHA_START
        self.pitch_deg = -spec_angle(self.inrun.gamma)   # visual: skis pitch, deg
        # events
        self.up_presses: list[float] = []
        self.t_edge = 0.0
        self.timing: float | None = None          # e of the evaluated press
        self.timing_q = 0.0
        self.jump_quality = 0.0
        self.t_flight = 0.0
        self.t_tele: float | None = None
        self.landing: Landing | None = None
        self.score: Score | None = None
        self.edge_speed = 0.0
        self.max_speed = 0.0
        # statistics for scoring
        self._wind_head_sum = 0.0
        self._wind_n = 0
        self._alpha_dev = 0.0
        self._alpha_activity = 0.0
        self.message = ""
        self.takeoff_note = ""        # feedback about the take-off timing, kept for the HUD

    # ---- player events ------------------------------------------------
    def start(self) -> None:
        if self.phase is Phase.READY:
            self.phase = Phase.INRUN
            self.t = 0.0

    def press_up(self) -> None:
        if self.phase is Phase.INRUN:
            self.up_presses.append(self.t)

    def press_telemark(self) -> None:
        if self.phase is Phase.FLIGHT and self.t_tele is None:
            self.t_tele = self.t

    # ---- time stepping --------------------------------------------------
    def advance(self, dt: float, controls: Controls | None = None) -> None:
        """Advance the simulation by dt seconds using fixed internal steps."""
        controls = controls or Controls()
        self._accum += dt
        while self._accum >= self.DT and self.phase not in (Phase.READY, Phase.FINISHED):
            self._accum -= self.DT
            self._step(self.DT, controls)

    def _step(self, dt: float, c: Controls) -> None:
        self.t += dt
        self.wind.step(dt)
        if self.phase is Phase.INRUN:
            self._step_inrun(dt, c)
        elif self.phase is Phase.FLIGHT:
            self._step_flight(dt, c)
        elif self.phase is Phase.OUTRUN:
            self._step_outrun(dt)

    # ---- inrun ------------------------------------------------------------
    def _inrun_accel(self, s: float, v: float, cda: float, wind: tuple[float, float, float]) -> float:
        th = self.inrun.angle(s)
        k = self.inrun.curvature(s)
        w_along = wind[0] * math.cos(th) - wind[2] * math.sin(th)
        v_rel = v - w_along
        drag = 0.5 * self.rho * cda * v_rel * abs(v_rel) / self.mass
        normal = G * math.cos(th) + k * v * v          # per unit mass
        return G * math.sin(th) - self.mu * normal - drag

    def _step_inrun(self, dt: float, c: Controls) -> None:
        cda = self.jumper.inrun_cda()
        if c.down:
            cda *= 0.96                                  # tuck a little deeper
        if self.up_presses and self.s > self.inrun.s_table - 12.0:
            cda *= self.STAND_DRAG                       # standing up for the jump
        wind = self.wind.vector()
        s, v = self.s, self.v
        k1v = self._inrun_accel(s, v, cda, wind)
        k1s = v
        k2v = self._inrun_accel(s + 0.5 * dt * k1s, v + 0.5 * dt * k1v, cda, wind)
        k2s = v + 0.5 * dt * k1v
        k3v = self._inrun_accel(s + 0.5 * dt * k2s, v + 0.5 * dt * k2v, cda, wind)
        k3s = v + 0.5 * dt * k2v
        k4v = self._inrun_accel(s + dt * k3s, v + dt * k3v, cda, wind)
        k4s = v + dt * k3v
        v_new = max(0.0, v + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v))
        s_new = s + dt / 6.0 * (k1s + 2 * k2s + 2 * k3s + k4s)
        self.max_speed = max(self.max_speed, v_new)
        if s_new >= self.inrun.length:
            frac = (self.inrun.length - s) / max(s_new - s, 1e-9)
            self.t_edge = self.t - dt + frac * dt
            self.v = v + frac * (v_new - v)
            self.s = self.inrun.length
            self._take_off()
            return
        self.s, self.v = s_new, v_new
        self.d, self.h = self.inrun.position(self.s)
        self.pitch_deg = -math.degrees(self.inrun.angle(self.s))

    def _take_off(self) -> None:
        """Pass the table edge: evaluate the timing of the 'up' press and start the flight."""
        alpha = self.inrun.alpha
        self.edge_speed = self.v
        self.vd = self.v * math.cos(alpha)
        self.vh = -self.v * math.sin(alpha)
        self.vl = 0.0
        self.d, self.l, self.h = 0.0, 0.0, 0.0
        best_q, best_e = 0.0, None
        for tp in self.up_presses:
            e = tp - self.t_edge
            if e < -0.8 or e > 0.0:
                continue
            q = timing_quality(e)
            if best_e is None or q > best_q:
                best_q, best_e = q, e
        self.timing = best_e
        self.timing_q = best_q
        # 'up' omitted (or far too early): the leg force is less than optimal
        self.jump_quality = self.MISS_QUALITY + (1.0 - self.MISS_QUALITY) * best_q
        self.phase = Phase.FLIGHT
        self.t_flight = 0.0
        self.alpha = self.alpha_cmd = self.ALPHA_START
        if best_e is None:
            self.takeoff_note = "No jump! (press UP just before the edge)"
        elif best_q > 0.8:
            self.takeoff_note = f"Perfect take-off ({best_e * 1000:+.0f} ms)"
        elif best_e < 0:
            self.takeoff_note = f"Take-off too early ({best_e * 1000:+.0f} ms)"
        else:
            self.takeoff_note = f"Take-off too late ({best_e * 1000:+.0f} ms)"
        self.message = self.takeoff_note

    # ---- flight -------------------------------------------------------------
    def _flight_force(self, vd: float, vl: float, vh: float, alpha: float,
                      wind: tuple[float, float, float], t_f: float, lateral_u: float,
                      area_mult: float, lift_mult: float = 1.0,
                      drag_mult: float = 1.0) -> tuple[float, float, float]:
        """Acceleration (d, l, h) of the jumper."""
        m = self.mass
        rd, rl, rh = vd - wind[0], vl - wind[1], vh - wind[2]       # velocity relative to the air
        v_rel = math.sqrt(rd * rd + rl * rl + rh * rh)
        ad = al = 0.0
        ah = -G
        if v_rel > 1e-3:
            ud, ul, uh = rd / v_rel, rl / v_rel, rh / v_rel
            cl, cd = aero_coefficients(alpha)
            qs = 0.5 * self.rho * v_rel * v_rel * self.jumper.flight_area() * area_mult
            # lift direction: perpendicular to the flow, in the vertical plane through it
            ld, ll, lh = -uh * ud, -uh * ul, 1.0 - uh * uh
            norm = math.sqrt(ld * ld + ll * ll + lh * lh)
            if norm > 1e-6:
                ld, ll, lh = ld / norm, ll / norm, lh / norm
            else:
                ld = ll = lh = 0.0
            cl *= lift_mult
            cd *= drag_mult
            ad += (qs * (-cd * ud + cl * ld)) / m
            al += (qs * (-cd * ul + cl * ll)) / m
            ah += (qs * (-cd * uh + cl * lh)) / m
        # side force: the body and the V-skis seen from the side (cross wind and drift)
        al -= 0.5 * self.rho * SIDE_CDA * abs(rl) * rl * area_mult / m
        # additional upward force from the take-off leg extension (normal to the table)
        if 0.0 <= t_f < self.PUSH_TIME:
            f_up = (self.jumper.push_force() * self.jump_quality
                    * math.sin(math.pi * t_f / self.PUSH_TIME))
            a = self.inrun.alpha
            ad += f_up * math.sin(a) / m
            ah += f_up * math.cos(a) / m
        al += lateral_u * self.K_LATERAL * G
        return ad, al, ah

    def _step_flight(self, dt: float, c: Controls) -> None:
        # pilot input: commanded angle of attack and lateral balance
        old_cmd = self.alpha_cmd
        self.alpha_cmd += ((1 if c.up else 0) - (1 if c.down else 0)) * self.ALPHA_RATE * dt
        self.alpha_cmd = max(ALPHA_MIN, min(ALPHA_MAX, self.alpha_cmd))
        self._alpha_activity += abs(self.alpha_cmd - old_cmd)
        lateral_u = (1 if c.right else 0) - (1 if c.left else 0)
        # the landing posture (telemark) costs lift and adds drag; the flight position builds up
        lift_mult, drag_mult = (0.85, 1.25) if self.t_tele is not None else (1.0, 1.0)
        area_mult = self.flight_position(self.t_flight)

        wind = self.wind.vector()
        t_f = self.t_flight
        state = (self.d, self.l, self.h, self.vd, self.vl, self.vh, self.alpha)

        def deriv(st, tf):
            d, l, h, vd, vl, vh, al = st
            ad, alat, ah = self._flight_force(vd, vl, vh, al, wind, tf, lateral_u,
                                              area_mult, lift_mult, drag_mult)
            dal = (self.alpha_cmd - al) / self.ALPHA_TAU
            return (vd, vl, vh, ad, alat, ah, dal)

        def add(st, k, f):
            return tuple(a + f * b for a, b in zip(st, k))

        k1 = deriv(state, t_f)
        k2 = deriv(add(state, k1, dt / 2), t_f + dt / 2)
        k3 = deriv(add(state, k2, dt / 2), t_f + dt / 2)
        k4 = deriv(add(state, k3, dt), t_f + dt)
        new = tuple(s + dt / 6.0 * (a + 2 * b + 2 * cc + dd)
                    for s, a, b, cc, dd in zip(state, k1, k2, k3, k4))

        # ground contact between the old and the new position?
        gap_old = self.h - self.profile.ground(self.d)
        gap_new = new[2] - self.profile.ground(new[0])
        self.t_flight += dt
        self._wind_head_sum += -wind[0]
        self._wind_n += 1
        self._alpha_dev += max(0.0, abs(self.alpha - 0.5 * (ALPHA_OPT_LOW + ALPHA_OPT_HIGH))
                               - 0.5 * (ALPHA_OPT_HIGH - ALPHA_OPT_LOW)) * dt
        if gap_new <= 0.0 and self.t_flight > 0.05:
            f = gap_old / max(gap_old - gap_new, 1e-9)
            f = max(0.0, min(1.0, f))
            contact = tuple(a + f * (b - a) for a, b in zip(state, new))
            self.d, self.l, self.h, self.vd, self.vl, self.vh, self.alpha = contact
            self.h = self.profile.ground(self.d)
            self.t -= dt * (1.0 - f)
            self.t_flight -= dt * (1.0 - f)
            self._touch_down()
            return
        self.d, self.l, self.h, self.vd, self.vl, self.vh, self.alpha = new
        self.max_speed = max(self.max_speed, math.sqrt(self.vd ** 2 + self.vl ** 2 + self.vh ** 2))
        speed_h = math.hypot(self.vd, self.vl)
        self.pitch_deg = math.degrees(math.atan2(self.vh, max(speed_h, 1e-6))) + self.alpha

    @staticmethod
    def flight_position(t_flight: float) -> float:
        """Effective area factor: right after take-off the body is not yet in the flight position."""
        return 0.45 + 0.55 * _smoothstep(t_flight / FLIGHT_RAMP)

    # ---- landing --------------------------------------------------------------
    def height_above_ground(self) -> float:
        return self.h - self.profile.ground(self.d)

    def _touch_down(self) -> None:
        beta = self.profile.slope(self.d)
        nd, nh = math.sin(beta), math.cos(beta)          # slope normal (d, h)
        td, th = math.cos(beta), -math.sin(beta)         # slope tangent (d, h)
        v_n = -(self.vd * nd + self.vh * nh)             # speed into the slope
        v_t = self.vd * td + self.vh * th
        v_n = max(v_n, 0.0)
        fall_h = v_n * v_n / (2.0 * G)
        # telemark classification
        if self.t_tele is None:
            tele = "none"
        else:
            tau = self.t - self.t_tele
            tele = "good" if 0.12 <= tau <= 0.9 else ("early" if tau > 0.9 else "late")
        crashed = (fall_h > 3.2 or abs(self.vl) > 4.0 or abs(self.l) > self.spec.half_width - 0.5
                   or self.alpha > 46.0)
        dist = round(self.profile.surface_distance(self.d) * 2.0) / 2.0
        flight_speed = math.sqrt(self.vd ** 2 + self.vl ** 2 + self.vh ** 2)
        self.landing = Landing(dist, self.d, self.l, v_n, fall_h, self.vl, self.alpha,
                               tele, crashed, self.t_flight, flight_speed)
        # impact: the speed along the slope is kept, minus a loss that depends on the impact
        loss = 0.05 * v_n if tele == "good" else 0.15 * v_n
        self.v = max(0.0, v_t - loss)
        self.vd, self.vh = self.v * td, self.v * th
        self.pitch_deg = -math.degrees(beta)
        self.phase = Phase.OUTRUN
        self.score = self.compute_score()
        if crashed:
            self.message = "Crash!"
        else:
            self.message = f"{dist:.1f} m"

    def _step_outrun(self, dt: float) -> None:
        land = self.landing
        beta = self.profile.slope(self.d)
        a = G * math.sin(beta) - self.mu_land * G * math.cos(beta)
        a -= 0.5 * self.rho * 0.45 * self.v * abs(self.v) / self.mass       # upright body
        if land and land.crashed:
            a -= 9.0
        elif self.d > self.profile.x_l:
            a -= 4.5                                                       # braking in the outrun
        self.v = max(0.0, self.v + a * dt)
        self.d += self.v * math.cos(beta) * dt
        self.h = self.profile.ground(self.d)
        self.vl *= math.exp(-3.0 * dt)
        self.l += self.vl * dt
        self.pitch_deg = -math.degrees(self.profile.slope(self.d))
        if self.v < 0.4 or self.d >= self.profile.x_end - 1.0:
            self.v = 0.0
            self.phase = Phase.FINISHED

    # ---- scoring ----------------------------------------------------------------
    def average_head_wind(self) -> float:
        return self._wind_head_sum / self._wind_n if self._wind_n else 0.0

    def compute_score(self) -> Score:
        land = self.landing
        assert land is not None
        spec = self.spec
        dist_pts = 60.0 + (land.distance - spec.k_point) * spec.meter_value
        notes: list[str] = []

        ded = 0.0
        if land.telemark == "none":
            ded += 3.0
            notes.append("no telemark")
        elif land.telemark == "early":
            ded += 1.2
            notes.append("telemark too early")
        elif land.telemark == "late":
            ded += 1.5
            notes.append("telemark too late")
        if land.fall_height > 1.5:
            fall_ded = min(6.0, (land.fall_height - 1.5) * 1.6)
            ded += fall_ded
            notes.append(f"hard landing ({land.fall_height:.1f} m fall height)")
        if abs(land.lateral) > 1.5:
            ded += min(2.5, (abs(land.lateral) - 1.5) * 0.6)
            notes.append("off the centre line")
        flight_ded = min(4.0, self._alpha_dev * 0.12 + max(0.0, self._alpha_activity - 70.0) * 0.03)
        if flight_ded > 0.3:
            notes.append("unstable flight position")
        ded += flight_ded
        ded += (1.0 - self.timing_q) * 2.0
        if self.timing is None:
            notes.append("no take-off jump")
        base = 19.0
        if land.crashed:
            base = 8.0
            notes.append("fall")
        judges: list[float] = []
        for _ in range(5):
            mark = base - ded + self.rng.choice((-0.5, -0.5, 0.0, 0.0, 0.0, 0.5, 0.5))
            judges.append(max(0.0, min(20.0, round(mark * 2.0) / 2.0)))
        style = sum(sorted(judges)[1:4])

        head = self.average_head_wind()
        if head >= 0:
            wind_pts = -head * spec.head_wind_pts
        else:
            wind_pts = -head * spec.tail_wind_pts
        gate_pts = (spec.gate_ref - self.gate) * spec.gate_pts
        total = dist_pts + style + wind_pts + gate_pts
        return Score(land.distance, dist_pts, judges, style, wind_pts, gate_pts, total, notes)

    # ---- helpers for the user interface ---------------------------------------
    def speed(self) -> float:
        if self.phase is Phase.INRUN or self.phase is Phase.READY:
            return self.v
        if self.phase is Phase.FLIGHT:
            return math.sqrt(self.vd ** 2 + self.vl ** 2 + self.vh ** 2)
        return self.v

    def distance_to_edge(self) -> float:
        """Track length still to go to the table edge (inrun only)."""
        return max(0.0, self.inrun.length - self.s) if self.phase in (Phase.READY, Phase.INRUN) else 0.0

    def time_to_edge(self) -> float:
        v = max(self.v, 1.0)
        return self.distance_to_edge() / v


def spec_angle(rad: float) -> float:
    return math.degrees(rad)


def run_jump(spec: HillSpec | None = None, weather: Weather | None = None,
             jumper: Jumper | None = None, gate: int | None = None, seed: int = 1,
             press_offset: float | None = -0.035, alpha_cmd: float | None = None,
             telemark_height: float = 3.0, lateral_hold: bool = True,
             max_time: float = 60.0) -> JumpSimulation:
    """Run a complete scripted jump (used by the tests and for calibration).

    press_offset  when to press 'up' relative to the edge in seconds (None = never)
    alpha_cmd     angle of attack to hold in flight (None = leave the default)
    lateral_hold  steer against lateral drift automatically
    """
    sim = JumpSimulation(spec, weather, jumper, gate, seed)
    sim.start()
    dt = JumpSimulation.DT
    pressed = False
    tele_done = False
    c = Controls()
    while sim.phase is not Phase.FINISHED and sim.t < max_time:
        if sim.phase is Phase.INRUN and press_offset is not None and not pressed:
            if sim.time_to_edge() <= -press_offset:
                sim.press_up()
                pressed = True
        if sim.phase is Phase.FLIGHT:
            c = Controls()
            if alpha_cmd is not None:
                if sim.alpha_cmd < alpha_cmd - 0.2:
                    c.up = True
                elif sim.alpha_cmd > alpha_cmd + 0.2:
                    c.down = True
            if lateral_hold:
                if sim.l + 0.8 * sim.vl > 0.3:
                    c.left = True
                elif sim.l + 0.8 * sim.vl < -0.3:
                    c.right = True
            if (not tele_done and sim.t_flight > 1.0
                    and sim.height_above_ground() < telemark_height):
                sim.press_telemark()
                tele_done = True
        else:
            c = Controls()
        sim.advance(dt, c)
    return sim


if __name__ == "__main__":
    print("This file is a library module of the ski jump game.")
    print("Start the game with:   python ski_jump.py")
