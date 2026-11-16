"""Tests for ski_sound (no Panda3D):  python test_sound.py"""
import os, tempfile, unittest, wave, math
import ski_sound as ss


class FakeSound:
    def __init__(self): self.vol=0.0; self.rate=1.0; self.loop=False; self.playing=False; self.plays=0
    def set_loop(self,v): self.loop=v
    def set_volume(self,v): self.vol=v
    def set_play_rate(self,r): self.rate=r
    def play(self): self.playing=True; self.plays+=1
    def stop(self): self.playing=False
class FakeLoader:
    def load_sfx(self,path): assert os.path.exists(path); return FakeSound()


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.paths = ss.ensure_sounds(os.path.join(cls.tmp.name, "sounds"))
    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_all_files_are_valid_wav(self):
        self.assertEqual(set(self.paths), set(ss.SOUND_NAMES))
        for name, path in self.paths.items():
            with wave.open(path) as w:
                self.assertEqual((w.getnchannels(), w.getsampwidth(), w.getframerate()), (1, 2, ss.RATE))
                secs = w.getnframes() / ss.RATE
                self.assertTrue(0.2 < secs < 3.0, (name, secs))

    def test_loops_are_seamless_and_not_silent(self):
        for name in ("ski_slide", "air_rush"):
            x = ss.GENERATORS[name]()
            rms = math.sqrt(sum(v * v for v in x) / len(x))
            self.assertGreater(rms, 0.05, name)
            self.assertLessEqual(max(abs(v) for v in x), 0.8 + 1e-9)
            # the step across the loop point is no bigger than a typical step inside the loop
            steps = sorted(abs(a - b) for a, b in zip(x, x[1:]))
            seam = abs(x[0] - x[-1])
            self.assertLess(seam, steps[int(len(steps) * 0.98)], name)

    def test_slide_is_hissy_and_air_is_low(self):
        def brightness(x):   # share of energy in the sample-to-sample difference (high frequencies)
            d = sum((a - b) ** 2 for a, b in zip(x, x[1:])); e = sum(v * v for v in x) or 1.0
            return d / e
        self.assertGreater(brightness(ss.make_slide()), brightness(ss.make_air()) * 2)

    def test_one_shots_decay(self):
        for name in ("takeoff", "landing", "crash", "cheer"):
            x = ss.GENERATORS[name]()
            tail = max(abs(v) for v in x[-200:]); peak = max(abs(v) for v in x)
            self.assertLess(tail, 0.25 * peak, name)

    def test_files_are_not_regenerated(self):
        path = self.paths["beep"]; t = os.path.getmtime(path)
        ss.ensure_sounds(os.path.dirname(path))
        self.assertEqual(t, os.path.getmtime(path))


class ManagerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.sm = ss.SoundManager(FakeLoader(), os.path.join(self.tmp.name, "s"))
        self.slide = self.sm.sounds["ski_slide"]; self.air = self.sm.sounds["air_rush"]
    def tearDown(self): self.tmp.cleanup()

    def test_slide_only_on_the_ground(self):
        self.assertTrue(self.slide.loop and self.air.loop)
        self.sm.update("ready", 0.0); self.assertFalse(self.slide.playing)
        self.sm.update("inrun", 15.0); self.assertTrue(self.slide.playing); self.assertFalse(self.air.playing)
        v1, r1 = self.slide.vol, self.slide.rate
        self.sm.update("inrun", 25.0)
        self.assertGreater(self.slide.vol, v1); self.assertGreater(self.slide.rate, r1)
        self.sm.update("flight", 27.0)                      # in the air: no sliding sound
        self.assertFalse(self.slide.playing); self.assertTrue(self.air.playing)
        self.sm.update("outrun", 20.0)                      # landed
        self.assertTrue(self.slide.playing); self.assertFalse(self.air.playing)
        self.sm.update("outrun", 0.1)                       # stopped
        self.assertFalse(self.slide.playing)

    def test_mute_and_events(self):
        self.sm.update("inrun", 20.0); self.assertTrue(self.slide.playing)
        self.assertFalse(self.sm.toggle())
        self.assertFalse(self.slide.playing)
        self.sm.update("inrun", 20.0); self.assertFalse(self.slide.playing)
        self.sm.play("landing"); self.assertEqual(self.sm.sounds["landing"].plays, 0)
        self.assertTrue(self.sm.toggle())
        self.sm.play("landing", 0.7); self.assertEqual(self.sm.sounds["landing"].plays, 1)
        self.assertAlmostEqual(self.sm.sounds["landing"].vol, 0.7)

    def test_missing_audio_device_is_ok(self):
        class Broken:
            def load_sfx(self, p): raise RuntimeError("no audio")
        sm = ss.SoundManager(Broken(), os.path.join(self.tmp.name, "b"))
        sm.update("inrun", 20.0); sm.play("beep"); sm.stop_all()


if __name__ == "__main__":
    unittest.main(verbosity=2)
