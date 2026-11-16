"""Tests for ski_physics (no Panda3D needed):  python test_ski.py"""

import math
import unittest

import ski_physics as sp

K100 = sp.HILLS["K100"]
CALM = sp.Weather(0.0, 0.0, 0.0, -4.0, "packed")


def jump(spec=K100, weather=CALM, **kw):
    kw.setdefault("alpha_cmd", 28.0)
    return sp.run_jump(spec, weather, seed=kw.pop("seed", 1), **kw)


class WeatherTests(unittest.TestCase):
    def test_air_density(self):
        self.assertAlmostEqual(sp.air_density(15.0, 0.0), 1.225, places=3)
        # colder air and lower altitude are denser
        self.assertGreater(sp.air_density(-15, 0), sp.air_density(5, 0))
        self.assertGreater(sp.air_density(0, 0), sp.air_density(0, 1500))

    def test_friction_order(self):
        mu = [sp.snow_friction(s, -6.0) for s in sp.SNOW_TYPES]
        self.assertEqual(mu, sorted(mu))
        self.assertTrue(0.012 < mu[0] < 0.02 and 0.04 < mu[-1] < 0.07)
        # best glide near -6 C: colder and warmer are worse
        self.assertGreater(sp.snow_friction("packed", -20), sp.snow_friction("packed", -6))
        self.assertGreater(sp.snow_friction("packed", 2), sp.snow_friction("packed", -6))

    def test_wind_vector(self):
        head = sp.Weather(4.0, 0.0).mean_wind()
        tail = sp.Weather(4.0, 180.0).mean_wind()
        right = sp.Weather(4.0, 90.0).mean_wind()
        self.assertAlmostEqual(head[0], -4.0)             # air moves toward the jumper
        self.assertAlmostEqual(tail[0], 4.0)
        self.assertAlmostEqual(right[1], -4.0)            # from the right = blows to the left
        self.assertGreater(head[2], 0.0)                  # updraft on a head wind
        self.assertEqual(tail[2], 0.0)

    def test_gusts_have_zero_mean_and_scale(self):
        import random
        calm = sp.Wind(sp.Weather(5.0, 0.0, 0.0), random.Random(1))
        calm.step(0.1)
        self.assertAlmostEqual(calm.vector()[0], -5.0)
        gusty = sp.Wind(sp.Weather(5.0, 0.0, 0.4), random.Random(2))
        speeds = []
        for _ in range(4000):
            gusty.step(0.05)
            speeds.append(-gusty.vector()[0])
        mean = sum(speeds) / len(speeds)
        self.assertAlmostEqual(mean, 5.0, delta=0.6)
        self.assertGreater(max(speeds) - min(speeds), 2.0)

    def test_presets_run(self):
        for name, weather in sp.WEATHER_PRESETS.items():
            sim = jump(weather=weather)
            self.assertIsNotNone(sim.landing, name)
            self.assertGreater(sim.landing.distance, 50.0, name)


class HillTests(unittest.TestCase):
    def test_inrun_geometry(self):
        for spec in sp.HILLS.values():
            ip = sp.Inrun(spec, spec.gate_ref)
            d, h = ip.position(ip.length)
            self.assertAlmostEqual(d, 0.0, places=6)
            self.assertAlmostEqual(h, 0.0, places=6)
            self.assertLess(ip.position(0.0)[0], -60.0)
            self.assertGreater(ip.start_height, 30.0)
            self.assertTrue(85.0 < ip.length < 145.0, (spec.key, ip.length))
            # angle goes gamma -> table angle without jumps
            self.assertAlmostEqual(ip.angle(0.0), math.radians(spec.inrun_angle))
            self.assertAlmostEqual(ip.angle(ip.length), math.radians(spec.takeoff_angle))
            prev = ip.angle(0.0)
            for i in range(1, 400):
                a = ip.angle(ip.length * i / 400)
                self.assertLessEqual(a, prev + 1e-12)
                prev = a

    def test_higher_gate_is_longer_and_higher(self):
        low = sp.Inrun(K100, 8)
        high = sp.Inrun(K100, 16)
        self.assertGreater(high.length, low.length)
        self.assertGreater(high.start_height, low.start_height)

    def test_landing_profile(self):
        for spec in sp.HILLS.values():
            pr = sp.LandingProfile(spec)
            self.assertAlmostEqual(pr.ground(0.0), -spec.table_height)
            ys = pr.y
            self.assertTrue(all(b <= a + 1e-9 for a, b in zip(ys, ys[1:])))      # never rises
            self.assertAlmostEqual(math.degrees(pr.slope(pr.x_k)), spec.slope_k, delta=0.1)
            self.assertAlmostEqual(math.degrees(pr.slope(pr.x_end)), 0.0, delta=0.01)  # flat outrun
            # K point is at slope distance K
            self.assertAlmostEqual(pr.surface_distance(pr.x_k), spec.k_point, delta=0.3)
            # vertical drop to K between 0.4 and 0.65 of K
            self.assertTrue(0.40 * spec.k_point < -pr.k_height < 0.65 * spec.k_point, spec.key)

    def test_all_hills_are_defined_sensibly(self):
        for spec in sp.HILLS.values():
            self.assertLess(spec.k_point, spec.hill_size)
            self.assertTrue(30 <= spec.inrun_angle <= 38)
            self.assertTrue(9 <= spec.takeoff_angle <= 12)
        self.assertEqual(sp.DEFAULT_HILL, "K100")
        self.assertEqual(sp.HILLS["K100"].k_point, 100)


class InrunSpeedTests(unittest.TestCase):
    def test_edge_speed_matches_real_hills(self):
        for spec in sp.HILLS.values():
            sim = jump(spec)
            self.assertAlmostEqual(sim.edge_speed, spec.typical_speed, delta=0.35, msg=spec.key)

    def test_energy_is_not_created(self):
        sim = jump()
        drop = sim.inrun.start_height
        self.assertLess(sim.edge_speed, math.sqrt(2 * sp.G * drop))
        self.assertGreater(sim.edge_speed, 0.6 * math.sqrt(2 * sp.G * drop))

    def test_friction_and_gate_and_mass(self):
        ice = jump(weather=sp.Weather(0, 0, 0, -6.0, "ice"))
        slush = jump(weather=sp.Weather(0, 0, 0, 3.0, "slush"))
        self.assertGreater(ice.edge_speed, slush.edge_speed + 1.0)
        self.assertGreater(ice.landing.distance, slush.landing.distance)
        speeds = [jump(gate=g).edge_speed for g in (6, 9, 12, 15, 18)]
        self.assertEqual(speeds, sorted(speeds))
        light, heavy = jump(jumper=sp.Jumper(body_mass=50)), jump(jumper=sp.Jumper(body_mass=90))
        self.assertGreater(heavy.edge_speed, light.edge_speed)      # drag deceleration ~ 1/m
        self.assertGreater(light.landing.distance, heavy.landing.distance)  # but light jumpers glide further

    def test_head_wind_slows_the_inrun(self):
        head = jump(weather=sp.Weather(5.0, 0.0, 0.0))
        tail = jump(weather=sp.Weather(5.0, 180.0, 0.0))
        self.assertLess(head.edge_speed, jump().edge_speed)
        self.assertGreater(tail.edge_speed, jump().edge_speed)


class FlightTests(unittest.TestCase):
    def test_distance_is_realistic(self):
        for key, lo, hi in (("K90", 88, 110), ("K100", 100, 118), ("K120", 118, 140), ("K185", 185, 220)):
            spec = sp.HILLS[key]
            best = max(jump(spec, alpha_cmd=a).landing.distance for a in (24, 28, 32))
            self.assertTrue(lo <= best <= hi, (key, best))
            self.assertLess(best, spec.hill_size + 8)

    def test_default_posture_near_k_point(self):
        sim = sp.run_jump(K100, CALM, seed=1)         # no key presses in flight
        self.assertTrue(95 <= sim.landing.distance <= 118, sim.landing.distance)
        self.assertEqual(sim.landing.telemark, "good")
        self.assertFalse(sim.landing.crashed)

    def test_optimal_take_off_beats_missed_one(self):
        perfect = jump(press_offset=-0.035)
        omitted = jump(press_offset=None)
        self.assertEqual(omitted.timing, None)
        self.assertAlmostEqual(omitted.jump_quality, sp.JumpSimulation.MISS_QUALITY)
        self.assertGreater(perfect.landing.distance - omitted.landing.distance, 15.0)
        self.assertIn("No jump", omitted.takeoff_note)
        self.assertIn("Perfect", perfect.takeoff_note)

    def test_timing_window_is_short(self):
        self.assertAlmostEqual(sp.timing_quality(-0.035), 1.0)
        for e in (-0.3, -0.2, 0.05, 0.1):
            self.assertLess(sp.timing_quality(e), 0.25, e)
        # the good window (>= 0.7) is about 0.1 s wide
        good = [e / 1000 for e in range(-300, 100) if sp.timing_quality(e / 1000) >= 0.7]
        self.assertTrue(0.06 <= max(good) - min(good) <= 0.16, (min(good), max(good)))
        # late is punished harder than early by the same distance from the optimum
        self.assertLess(sp.timing_quality(-0.035 + 0.06), sp.timing_quality(-0.035 - 0.06))

    def test_timing_changes_distance_monotonically_towards_the_optimum(self):
        offsets = (-0.4, -0.2, -0.12, -0.07, -0.035)
        d = [jump(press_offset=o).landing.distance for o in offsets]
        self.assertEqual(d, sorted(d))
        late = jump(press_offset=-0.0001).landing.distance
        self.assertLess(late, d[-1] - 3.0)

    def test_press_after_the_edge_is_too_late(self):
        sim = sp.JumpSimulation(K100, CALM, seed=1)
        sim.start()
        while sim.phase is sp.Phase.INRUN:
            sim.advance(sim.DT)
        sim.press_up()                              # in flight: ignored
        self.assertEqual(sim.up_presses, [])
        self.assertEqual(sim.timing, None)

    def test_head_wind_shortens_and_tail_wind_lengthens(self):
        base = jump().landing.distance
        head = jump(weather=sp.Weather(4.0, 0.0, 0.0)).landing.distance
        tail = jump(weather=sp.Weather(4.0, 180.0, 0.0)).landing.distance
        # a head wind slows the inrun and the flight in these model settings
        self.assertLess(head, base)
        self.assertGreater(tail, base)

    def test_stall_and_wrong_attack_angle(self):
        good = jump(alpha_cmd=28).landing.distance
        flat = jump(alpha_cmd=10).landing.distance
        stalled = jump(alpha_cmd=48).landing.distance
        self.assertGreater(good, flat + 8)
        self.assertGreater(good, stalled + 8)
        cl, cd = sp.aero_coefficients(26.0)
        self.assertTrue(0.4 < cl < 0.7 and 0.25 < cd < 0.5)
        self.assertTrue(1.2 < cl / cd < 1.8)           # typical lift/drag ratio of ski jumpers
        # stall: lift drops and drag rises past the stall angle
        cl_hi, cd_hi = sp.aero_coefficients(46.0)
        self.assertLess(cl_hi, sp.aero_coefficients(34.0)[0])
        self.assertGreater(cd_hi, sp.aero_coefficients(34.0)[1])

    def test_cross_wind_drift_and_lateral_control(self):
        weather = sp.Weather(5.0, 90.0, 0.0)             # from the right
        free = jump(weather=weather, lateral_hold=False)
        held = jump(weather=weather, lateral_hold=True)
        self.assertLess(free.landing.lateral, -1.5)       # pushed to the left
        self.assertLess(abs(held.landing.lateral), 0.8)
        left = jump(weather=sp.Weather(5.0, 270.0, 0.0), lateral_hold=False)
        self.assertGreater(left.landing.lateral, 1.5)

    def test_manual_controls_move_the_jumper(self):
        sim = sp.JumpSimulation(K100, CALM, seed=1)
        sim.start()
        while sim.phase is sp.Phase.INRUN:
            if sim.time_to_edge() < 0.035 and not sim.up_presses:
                sim.press_up()
            sim.advance(sim.DT)
        for _ in range(240):
            sim.advance(sim.DT, sp.Controls(right=True, up=True))
        self.assertGreater(sim.l, 0.2)
        self.assertGreater(sim.alpha_cmd, sp.JumpSimulation.ALPHA_START + 10)
        self.assertLessEqual(sim.alpha_cmd, sp.ALPHA_MAX)

    def test_deterministic_for_a_seed(self):
        a = jump(weather=sp.Weather(4.0, 45.0, 0.4), seed=7)
        b = jump(weather=sp.Weather(4.0, 45.0, 0.4), seed=7)
        self.assertEqual(a.landing.distance, b.landing.distance)
        self.assertEqual(a.score.total, b.score.total)


class LandingAndScoreTests(unittest.TestCase):
    def test_landing_is_on_the_slope(self):
        sim = jump()
        land = sim.landing
        self.assertAlmostEqual(sim.h, sim.profile.ground(sim.d) if sim.phase is sp.Phase.FLIGHT else sim.h)
        self.assertGreater(land.d, 50)
        self.assertLess(land.d, sim.profile.x_l)
        self.assertGreater(land.airtime, 3.0)
        self.assertLess(land.fall_height, 1.5)
        self.assertAlmostEqual(land.fall_height, land.normal_speed ** 2 / (2 * sp.G), places=6)

    def test_bad_attack_angle_gives_a_crash_on_a_big_hill(self):
        sim = jump(sp.HILLS["K120"], alpha_cmd=48)
        self.assertTrue(sim.landing.crashed)
        self.assertLess(sim.score.style_points, 25.0)

    def test_telemark_states(self):
        self.assertEqual(jump(telemark_height=1.5).landing.telemark, "good")
        self.assertEqual(jump(telemark_height=3.0).landing.telemark, "good")
        self.assertEqual(jump(telemark_height=0.4).landing.telemark, "late")
        self.assertEqual(jump(telemark_height=-1).landing.telemark, "none")
        # pressing right after take-off is far too early
        self.assertEqual(jump(telemark_height=50).landing.telemark, "early")

    def test_telemark_is_worth_points_and_early_costs_distance(self):
        good = jump(telemark_height=2.5)
        none = jump(telemark_height=-1)
        self.assertGreater(good.score.style_points, none.score.style_points + 4)
        early = jump(telemark_height=50)
        self.assertLess(early.landing.distance, good.landing.distance - 5)

    def test_scoring_formula(self):
        sim = jump()
        sc = sim.score
        spec = sim.spec
        self.assertAlmostEqual(sc.distance_points, 60 + (sc.distance - spec.k_point) * spec.meter_value)
        self.assertAlmostEqual(sc.gate_points, 0.0)                     # reference gate
        self.assertAlmostEqual(sc.total, sc.distance_points + sc.style_points
                               + sc.wind_points + sc.gate_points)
        self.assertEqual(len(sc.judges), 5)
        self.assertAlmostEqual(sc.style_points, sum(sorted(sc.judges)[1:4]))
        self.assertTrue(40 <= sc.style_points <= 60)
        # distance is measured in half metres
        self.assertEqual(sc.distance * 2, round(sc.distance * 2))

    def test_meter_values_follow_hill_class(self):
        self.assertEqual(sp.HILLS["K90"].meter_value, 2.0)
        self.assertEqual(sp.HILLS["K100"].meter_value, 1.8)
        self.assertEqual(sp.HILLS["K185"].meter_value, 1.2)

    def test_wind_and_gate_compensation_signs(self):
        head = jump(weather=sp.Weather(4.0, 0.0, 0.0)).score
        tail = jump(weather=sp.Weather(4.0, 180.0, 0.0)).score
        self.assertLess(head.wind_points, -10)
        self.assertGreater(tail.wind_points, 10)
        high = jump(gate=16).score
        low = jump(gate=8).score
        self.assertLess(high.gate_points, 0)
        self.assertGreater(low.gate_points, 0)
        # compensation roughly cancels the effect of a lower / higher gate
        base = jump().score.total
        self.assertAlmostEqual(high.total, base, delta=12.0)
        self.assertAlmostEqual(low.total, base, delta=12.0)

    def test_outrun_finishes(self):
        sim = jump()
        self.assertIs(sim.phase, sp.Phase.FINISHED)
        self.assertEqual(sim.v, 0.0)
        self.assertLess(sim.d, sim.profile.x_end)

    def test_crash_stops_quickly(self):
        sim = jump(sp.HILLS["K120"], alpha_cmd=48)
        ok = jump(sp.HILLS["K120"])
        self.assertLess(sim.d - sim.landing.d, ok.d - ok.landing.d)


if __name__ == "__main__":
    unittest.main(verbosity=2)
