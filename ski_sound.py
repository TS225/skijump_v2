"""Simple sound effects for the ski jump game.

The sounds are synthesised with plain Python the first time the game starts and
written as 16-bit mono WAV files into the folder "sounds" next to the game, so no
audio assets are needed (delete the folder to regenerate them).

    ski_slide   skis hissing on snow / ice track (loop)  - only while sliding on the ground
    air_rush    wind noise while flying (loop)            - only in the air
    takeoff     thump + whoosh as the jumper leaves the table
    landing     soft thud and crunch of the landing
    crash       heavy fall
    cheer       the crowd
    beep        start signal
"""

from __future__ import annotations

import math
import os
import random
import wave
from array import array

try:                                     # needed to hand Windows paths to Panda3D correctly
    from panda3d.core import Filename
except ImportError:                      # plain Python (tests)
    Filename = None

RATE = 22050
SOUND_NAMES = ("ski_slide", "air_rush", "takeoff", "landing", "crash", "cheer", "beep")


# --------------------------------------------------------------------------
#  Synthesis helpers (lists of floats in -1..1)
# --------------------------------------------------------------------------
def _noise(rng: random.Random, n: int) -> list[float]:
    return [rng.uniform(-1.0, 1.0) for _ in range(n)]


def lowpass(x: list[float], fc: float, rate: int = RATE) -> list[float]:
    a = 1.0 - math.exp(-2.0 * math.pi * fc / rate)
    y, s = [], 0.0
    for v in x:
        s += a * (v - s)
        y.append(s)
    return y


def highpass(x: list[float], fc: float, rate: int = RATE) -> list[float]:
    lp = lowpass(x, fc, rate)
    return [v - l for v, l in zip(x, lp)]


def bandpass(x: list[float], lo: float, hi: float, rate: int = RATE) -> list[float]:
    return lowpass(highpass(x, lo, rate), hi, rate)


def normalise(x: list[float], peak: float = 0.8) -> list[float]:
    m = max((abs(v) for v in x), default=0.0) or 1.0
    return [v * peak / m for v in x]


def make_loop(x: list[float], fade: int) -> list[float]:
    """Make x (length L + fade) seamless: the tail is cross-faded into the start."""
    n = len(x) - fade
    y = x[:n]
    for i in range(fade):
        w = i / fade
        y[i] = x[i] * w + x[n + i] * (1.0 - w)
    return y


def _envelope(n: int, attack: float, release: float, rate: int = RATE) -> list[float]:
    a, r = max(1, int(attack * rate)), max(1, int(release * rate))
    env = []
    for i in range(n):
        v = 1.0
        if i < a:
            v = i / a
        if i > n - r:
            v = min(v, (n - i) / r)
        env.append(max(0.0, v))
    return env


def make_slide(seed: int = 1, seconds: float = 2.0) -> list[float]:
    """Skis on snow: band-limited hiss with a slight scratchy modulation and a low rumble."""
    rng = random.Random(seed)
    n = int(seconds * RATE)
    fade = int(0.15 * RATE)
    hiss = bandpass(_noise(rng, n + fade), 900.0, 5200.0)
    rumble = lowpass(_noise(rng, n + fade), 160.0)
    out = []
    for i in range(n + fade):
        t = i / RATE
        mod = 1.0 + 0.22 * math.sin(2 * math.pi * 11.0 * t) + 0.12 * math.sin(2 * math.pi * 3.3 * t + 1.0)
        out.append(hiss[i] * mod + 2.5 * rumble[i])
    return make_loop(normalise(out, 0.75), fade)


def make_air(seed: int = 2, seconds: float = 2.0) -> list[float]:
    """Wind rushing past the helmet: low-passed noise with slow gusts."""
    rng = random.Random(seed)
    n = int(seconds * RATE)
    fade = int(0.2 * RATE)
    body = lowpass(_noise(rng, n + fade), 700.0)
    hiss = bandpass(_noise(rng, n + fade), 1500.0, 3500.0)
    out = []
    for i in range(n + fade):
        t = i / RATE
        gust = 0.75 + 0.25 * math.sin(2 * math.pi * 1.0 * t)      # whole periods in 2 s: loops
        out.append((body[i] * 3.0 + hiss[i] * 0.5) * gust)
    return make_loop(normalise(out, 0.7), fade)


def make_takeoff(seed: int = 3) -> list[float]:
    rng = random.Random(seed)
    n = int(0.5 * RATE)
    whoosh = bandpass(_noise(rng, n), 300.0, 2500.0)
    out, phase = [], 0.0
    for i in range(n):
        t = i / RATE
        f = 45.0 + 100.0 * math.exp(-t * 14.0)
        phase += 2 * math.pi * f / RATE
        thump = math.sin(phase) * math.exp(-t * 11.0)
        out.append(thump * 1.0 + whoosh[i] * 0.9 * (t / 0.5) * math.exp(-t * 3.5))
    return normalise(out, 0.85)


def make_landing(seed: int = 4) -> list[float]:
    rng = random.Random(seed)
    n = int(0.55 * RATE)
    crunch = highpass(_noise(rng, n), 1200.0)
    out, phase = [], 0.0
    for i in range(n):
        t = i / RATE
        f = 40.0 + 60.0 * math.exp(-t * 16.0)
        phase += 2 * math.pi * f / RATE
        out.append(math.sin(phase) * math.exp(-t * 13.0) + crunch[i] * 0.55 * math.exp(-t * 7.0))
    return normalise(out, 0.85)


def make_crash(seed: int = 5) -> list[float]:
    rng = random.Random(seed)
    n = int(1.1 * RATE)
    rough = bandpass(_noise(rng, n), 200.0, 4000.0)
    out, phase = [], 0.0
    for i in range(n):
        t = i / RATE
        f = 35.0 + 70.0 * math.exp(-t * 9.0)
        phase += 2 * math.pi * f / RATE
        bursts = 0.5 + 0.5 * math.sin(2 * math.pi * 9.0 * t)
        out.append(math.sin(phase) * math.exp(-t * 6.0) * 1.2 + rough[i] * bursts * math.exp(-t * 3.2))
    return normalise(out, 0.95)


def make_cheer(seed: int = 6) -> list[float]:
    """A crowd: band-limited noise, swelling and fading, with some clapping bursts."""
    rng = random.Random(seed)
    n = int(2.6 * RATE)
    body = bandpass(_noise(rng, n), 350.0, 2600.0)
    claps = highpass(_noise(rng, n), 2000.0)
    env = _envelope(n, 0.45, 0.9)
    out = []
    for i in range(n):
        t = i / RATE
        swell = 0.55 + 0.45 * math.sin(2 * math.pi * 2.2 * t + 0.6)
        clap = 0.5 * max(0.0, math.sin(2 * math.pi * 7.0 * t)) ** 6
        out.append((body[i] * swell + claps[i] * clap * 0.6) * env[i])
    return normalise(out, 0.8)


def make_beep() -> list[float]:
    out = []
    for freq, length in ((660.0, 0.12), (990.0, 0.20)):
        n = int(length * RATE)
        for i in range(n):
            t = i / RATE
            env = min(1.0, i / 120.0) * min(1.0, (n - i) / 400.0)
            out.append(0.6 * math.sin(2 * math.pi * freq * t) * env)
        out.extend([0.0] * int(0.03 * RATE))
    return out


GENERATORS = {
    "ski_slide": make_slide, "air_rush": make_air, "takeoff": make_takeoff,
    "landing": make_landing, "crash": make_crash, "cheer": make_cheer, "beep": make_beep,
}


def write_wav(path: str, samples: list[float], rate: int = RATE) -> None:
    data = array("h", (int(max(-1.0, min(1.0, v)) * 32767) for v in samples))
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data.tobytes())


def ensure_sounds(folder: str) -> dict[str, str]:
    """Create any missing sound files in `folder`; returns name -> path."""
    os.makedirs(folder, exist_ok=True)
    paths = {}
    for name in SOUND_NAMES:
        path = os.path.join(folder, name + ".wav")
        if not os.path.exists(path):
            write_wav(path, GENERATORS[name]())
        paths[name] = path
    return paths


# --------------------------------------------------------------------------
#  Playback
# --------------------------------------------------------------------------
def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


class SoundManager:
    """Plays the loops and one-shot effects according to what the jumper is doing.

    The ski slide is only audible while the skis are on the ground (inrun and
    outrun) and is silent in the air; the wind rush is only audible in flight.
    `loader` is the Panda3D loader (base.loader); a missing audio device is fine.
    """

    def __init__(self, loader, folder: str, enabled: bool = True) -> None:
        self.enabled = enabled
        self.master = 1.0
        self.sounds: dict[str, object] = {}
        self._playing: dict[str, bool] = {}
        try:
            paths = ensure_sounds(folder)
        except OSError:
            paths = {}
        self.errors: list[str] = []
        for name, path in paths.items():
            try:
                # Panda3D wants its own path syntax (forward slashes, /c/dir on Windows)
                panda_path = Filename.from_os_specific(path).get_fullpath() if Filename else path
                snd = loader.load_sfx(panda_path)
            except Exception as exc:
                snd = None
                self.errors.append(f"{name}: {exc}")
            if snd is not None:
                self.sounds[name] = snd
        for name in ("ski_slide", "air_rush"):
            if name in self.sounds:
                self.sounds[name].set_loop(True)
        # a NullAudioSound (no audio device / failed load) reports a length of 0
        self.working = any(self._is_real(s) for s in self.sounds.values())

    @staticmethod
    def _is_real(snd) -> bool:
        try:
            return snd.length() > 0.0
        except Exception:
            return True

    def describe(self) -> str:
        """One line for the console / help screen."""
        if not self.sounds:
            return "sound: NOT available (" + ("; ".join(self.errors) or "no sound files") + ")"
        if not self.working:
            return "sound: files found but no audio device / loader (silent)"
        return f"sound: {len(self.sounds)} effects loaded" + ("" if self.enabled else " (muted)")

    # ---- helpers ---------------------------------------------------------------
    def _loop(self, name: str, volume: float, rate: float) -> None:
        snd = self.sounds.get(name)
        if snd is None:
            return
        volume = volume * self.master if self.enabled else 0.0
        if volume < 0.02:
            if self._playing.get(name):
                snd.stop()
                self._playing[name] = False
            return
        snd.set_volume(_clamp(volume))
        snd.set_play_rate(_clamp(rate, 0.5, 2.0))
        if not self._playing.get(name):
            snd.play()
            self._playing[name] = True

    def play(self, name: str, volume: float = 1.0) -> None:
        snd = self.sounds.get(name)
        if snd is None or not self.enabled:
            return
        snd.set_volume(_clamp(volume * self.master))
        snd.play()

    # ---- control -----------------------------------------------------------------
    def update(self, phase: str, speed: float) -> None:
        """phase: 'ready', 'inrun', 'flight', 'outrun' or anything else (silence)."""
        on_ground = phase in ("inrun", "outrun") and speed > 0.4
        if on_ground:
            k = _clamp(speed / 28.0)
            base = 0.30 if phase == "inrun" else 0.25
            self._loop("ski_slide", base + 0.65 * k, 0.75 + 0.55 * k)
        else:
            self._loop("ski_slide", 0.0, 1.0)
        if phase == "flight":
            self._loop("air_rush", _clamp((speed - 8.0) / 20.0) * 0.6, 0.8 + 0.5 * _clamp(speed / 32.0))
        else:
            self._loop("air_rush", 0.0, 1.0)

    def stop_all(self) -> None:
        for name in ("ski_slide", "air_rush"):
            self._loop(name, 0.0, 1.0)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        if not enabled:
            self.stop_all()

    def toggle(self) -> bool:
        self.set_enabled(not self.enabled)
        return self.enabled


if __name__ == "__main__":
    print("This file is a library module of the ski jump game.")
    print("Start the game with:   python ski_jump.py")
