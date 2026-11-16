"""Hot-seat ski jumping competition (pure Python, no Panda3D).

Up to 10 players jump one after another:
    training jump   - not counted, shows the distances
    1st round       - start order = entry order
    2nd round       - start order = reverse standing after round 1 (the leader jumps last)
A jump is worth distance points + style points + wind and gate compensation, exactly
as computed by ski_physics.JumpSimulation.compute_score().  The winner is the player
with the most points over both rounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MAX_PLAYERS = 10
ROUND_KEYS = ("training", "round1", "round2")
ROUND_TITLES = {"training": "Training jump", "round1": "1st round", "round2": "2nd round (final)"}

# suit colours of the players (r, g, b)
PLAYER_COLORS: tuple[tuple[float, float, float], ...] = (
    (0.90, 0.15, 0.15), (0.15, 0.35, 0.95), (0.10, 0.70, 0.25), (0.98, 0.80, 0.10),
    (0.65, 0.25, 0.85), (0.98, 0.50, 0.10), (0.10, 0.80, 0.85), (0.95, 0.40, 0.70),
    (0.60, 0.60, 0.62), (0.45, 0.28, 0.12),
)


@dataclass(frozen=True)
class JumpRecord:
    distance: float
    distance_points: float
    style: float
    wind: float
    gate: float
    total: float
    crashed: bool = False
    judges: tuple[float, ...] = ()

    @classmethod
    def from_score(cls, score, landing) -> "JumpRecord":
        """Build a record from ski_physics.Score and ski_physics.Landing objects."""
        return cls(score.distance, score.distance_points, score.style_points, score.wind_points,
                   score.gate_points, score.total, bool(landing.crashed), tuple(score.judges))


@dataclass
class Player:
    index: int
    name: str
    jumps: dict[str, JumpRecord] = field(default_factory=dict)

    @property
    def color(self) -> tuple[float, float, float]:
        return PLAYER_COLORS[self.index % len(PLAYER_COLORS)]

    def points(self, round_key: str) -> float | None:
        rec = self.jumps.get(round_key)
        return None if rec is None else rec.total

    def total(self) -> float:
        return sum(r.total for k, r in self.jumps.items() if k != "training")

    def best_distance(self) -> float:
        return max((r.distance for r in self.jumps.values()), default=0.0)


@dataclass(frozen=True)
class StandingRow:
    rank: int
    player: Player
    round1: float | None
    round2: float | None
    total: float
    best_distance: float


class Competition:
    """Keeps the start order, the jump records and the standings."""

    def __init__(self, names: list[str]) -> None:
        if not 1 <= len(names) <= MAX_PLAYERS:
            raise ValueError(f"1 to {MAX_PLAYERS} players are allowed")
        self.players = [Player(i, (n or "").strip() or f"Player {i + 1}") for i, n in enumerate(names)]
        self.round_index = 0
        self.finished = False
        self.round_complete = False
        self._turn = 0
        self._order = self._make_order(0)

    # ---- state ---------------------------------------------------------------
    @property
    def round_key(self) -> str:
        return ROUND_KEYS[self.round_index]

    @property
    def round_title(self) -> str:
        return ROUND_TITLES[self.round_key]

    @property
    def current_player(self) -> Player | None:
        if self.round_complete:
            return None
        return self.players[self._order[self._turn]]

    @property
    def turn_number(self) -> int:
        """1-based number of the jump in this round (the last one once the round is done)."""
        return min(self._turn + 1, len(self._order))

    @property
    def round_size(self) -> int:
        return len(self._order)

    @property
    def start_list(self) -> list[Player]:
        return [self.players[i] for i in self._order]

    @property
    def next_player(self) -> Player | None:
        if self.round_complete or self._turn + 1 >= len(self._order):
            return None
        return self.players[self._order[self._turn + 1]]

    # ---- order ---------------------------------------------------------------------
    def _make_order(self, round_index: int) -> list[int]:
        n = len(self.players)
        if ROUND_KEYS[round_index] != "round2":
            return list(range(n))
        # reverse standing after round 1: lowest points first, the leader last
        return [p.index for p in sorted(self.players, key=lambda p: (p.total(), p.index))]

    # ---- progress ----------------------------------------------------------------------
    def record(self, rec: JumpRecord) -> Player:
        """Store the jump of the current player; returns that player."""
        player = self.current_player
        if player is None:
            raise RuntimeError("the round is complete")
        player.jumps[self.round_key] = rec
        self._turn += 1
        if self._turn >= len(self._order):
            self.round_complete = True
            if self.round_key == ROUND_KEYS[-1]:
                self.finished = True
        return player

    def start_next_round(self) -> None:
        if not self.round_complete:
            raise RuntimeError("the round is not complete")
        if self.finished:
            raise RuntimeError("the competition is finished")
        self.round_index += 1
        self._order = self._make_order(self.round_index)
        self._turn = 0
        self.round_complete = False

    # ---- tables --------------------------------------------------------------------------
    @staticmethod
    def _ranked(items: list[tuple[float, object]]) -> list[tuple[int, object]]:
        """items sorted by value descending -> (rank, item); equal values share the rank."""
        out: list[tuple[int, object]] = []
        last_value = None
        rank = 0
        for pos, (value, item) in enumerate(items, start=1):
            if last_value is None or abs(value - last_value) > 0.05:
                rank = pos
            out.append((rank, item))
            last_value = value
        return out

    def standings(self) -> list[StandingRow]:
        """Standing by points over the rounds jumped so far (training is not counted)."""
        order = sorted(self.players, key=lambda p: (-p.total(), p.index))
        ranked = self._ranked([(p.total(), p) for p in order])
        return [StandingRow(rank, p, p.points("round1"), p.points("round2"), p.total(),
                            max((r.distance for k, r in p.jumps.items() if k != "training"), default=0.0))
                for rank, p in ranked]

    def training_table(self) -> list[tuple[int, Player, float]]:
        """Training jump: ranked by distance (only players who have jumped)."""
        jumped = [p for p in self.players if "training" in p.jumps]
        jumped.sort(key=lambda p: (-p.jumps["training"].distance, p.index))
        ranked = self._ranked([(p.jumps["training"].distance, p) for p in jumped])
        return [(rank, p, p.jumps["training"].distance) for rank, p in ranked]

    def rank_of(self, player: Player) -> int:
        for row in self.standings():
            if row.player is player:
                return row.rank
        raise ValueError("unknown player")

    def winner(self) -> Player:
        return self.standings()[0].player


if __name__ == "__main__":
    print("This file is a library module of the ski jump game.")
    print("Start the game with:   python ski_jump.py")
