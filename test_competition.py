"""Tests for ski_competition (no Panda3D):  python test_competition.py"""
import unittest
import ski_competition as sc
import ski_physics as sp


def rec(total, dist=100.0, crashed=False):
    return sc.JumpRecord(dist, 60.0, total - 60.0, 0.0, 0.0, total, crashed)


class CompetitionTests(unittest.TestCase):
    def test_player_limits_and_names(self):
        with self.assertRaises(ValueError):
            sc.Competition([])
        with self.assertRaises(ValueError):
            sc.Competition(["x"] * 11)
        comp = sc.Competition(["Anna", "  ", ""])
        self.assertEqual([p.name for p in comp.players], ["Anna", "Player 2", "Player 3"])
        self.assertEqual(len(sc.Competition([f"p{i}" for i in range(10)]).players), 10)
        self.assertEqual(len(sc.PLAYER_COLORS), sc.MAX_PLAYERS)

    def test_full_flow_training_and_two_rounds(self):
        comp = sc.Competition(["A", "B", "C"])
        self.assertEqual(comp.round_key, "training")
        self.assertEqual([p.name for p in comp.start_list], ["A", "B", "C"])
        seen = []
        for total in (100, 110, 90):                       # training
            seen.append(comp.current_player.name)
            comp.record(rec(total))
        self.assertEqual(seen, ["A", "B", "C"])
        self.assertTrue(comp.round_complete)
        self.assertIsNone(comp.current_player)
        self.assertFalse(comp.finished)
        self.assertEqual([p.total() for p in comp.players], [0, 0, 0])   # training is not counted
        comp.start_next_round()
        self.assertEqual(comp.round_key, "round1")
        for total in (120.0, 140.0, 130.0):                # round 1: A, B, C in entry order
            comp.record(rec(total))
        self.assertTrue(comp.round_complete)
        comp.start_next_round()
        self.assertEqual(comp.round_key, "round2")
        # reverse standing: lowest first, the leader last
        self.assertEqual([p.name for p in comp.start_list], ["A", "C", "B"])
        self.assertEqual(comp.current_player.name, "A")
        self.assertEqual(comp.next_player.name, "C")
        comp.record(rec(125.0))
        comp.record(rec(150.0))
        self.assertFalse(comp.finished)
        comp.record(rec(120.0))
        self.assertTrue(comp.finished and comp.round_complete)
        with self.assertRaises(RuntimeError):
            comp.start_next_round()
        with self.assertRaises(RuntimeError):
            comp.record(rec(1))
        rows = comp.standings()
        # totals: A 120+125=245, B 140+120=260, C 130+150=280
        self.assertEqual([r.player.name for r in rows], ["C", "B", "A"])
        self.assertEqual([r.rank for r in rows], [1, 2, 3])
        self.assertAlmostEqual(rows[0].total, 280.0)
        self.assertAlmostEqual(rows[0].round1, 130.0)
        self.assertAlmostEqual(rows[0].round2, 150.0)
        self.assertEqual(comp.winner().name, "C")

    def test_cannot_advance_early(self):
        comp = sc.Competition(["A", "B"])
        with self.assertRaises(RuntimeError):
            comp.start_next_round()

    def test_training_table_ranks_by_distance(self):
        comp = sc.Competition(["A", "B", "C"])
        comp.record(rec(100, dist=98.0))
        comp.record(rec(100, dist=104.5))
        table = comp.training_table()
        self.assertEqual([(r, p.name) for r, p, _ in table], [(1, "B"), (2, "A")])
        comp.record(rec(100, dist=98.0))
        table = comp.training_table()
        self.assertEqual([r for r, _, _ in table], [1, 2, 2])         # tie shares the rank

    def test_ties_share_rank_and_next_rank_skips(self):
        comp = sc.Competition(["A", "B", "C", "D"])
        for _ in range(4):
            comp.record(rec(1))                                       # training
        comp.start_next_round()
        for total in (120.0, 120.0, 100.0, 140.0):
            comp.record(rec(total))
        ranks = [(r.rank, r.player.name) for r in comp.standings()]
        self.assertEqual(ranks, [(1, "D"), (2, "A"), (2, "B"), (4, "C")])

    def test_single_player_and_ten_players(self):
        solo = sc.Competition(["Only"])
        for _ in range(3):
            solo.record(rec(100))
            if not solo.finished:
                solo.start_next_round()
        self.assertTrue(solo.finished)
        self.assertEqual(solo.rank_of(solo.players[0]), 1)
        ten = sc.Competition([f"P{i}" for i in range(10)])
        jumps = 0
        while not ten.finished:
            ten.record(rec(100 + ten.current_player.index))
            jumps += 1
            if ten.round_complete and not ten.finished:
                ten.start_next_round()
        self.assertEqual(jumps, 30)

    def test_round2_order_is_reverse_of_round1_standing(self):
        comp = sc.Competition([f"P{i}" for i in range(6)])
        for _ in range(6):
            comp.record(rec(50))
        comp.start_next_round()
        pts = [130, 150, 90, 170, 110, 120]
        for p in pts:
            comp.record(rec(p))
        comp.start_next_round()
        names = [p.name for p in comp.start_list]
        by_points = sorted(range(6), key=lambda i: pts[i])
        self.assertEqual(names, [f"P{i}" for i in by_points])
        self.assertEqual(names[-1], "P3")                              # the leader jumps last

    def test_record_from_simulation_score(self):
        sim = sp.run_jump(sp.HILLS["K100"], sp.Weather(), seed=1, alpha_cmd=28)
        r = sc.JumpRecord.from_score(sim.score, sim.landing)
        self.assertAlmostEqual(r.total, sim.score.total)
        self.assertEqual(r.distance, sim.landing.distance)
        self.assertEqual(len(r.judges), 5)
        comp = sc.Competition(["A"])
        comp.record(r)
        self.assertEqual(comp.players[0].best_distance(), r.distance)


if __name__ == "__main__":
    unittest.main(verbosity=2)
