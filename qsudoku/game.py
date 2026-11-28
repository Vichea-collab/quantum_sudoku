"""Game state for the Play page: input, notes, hints, mistakes, timer, undo."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .classical import propagate
from .puzzle import CELLS, DIGITS, PEERS, Puzzle, candidates, rc

MAX_MISTAKES = 3


@dataclass
class Move:
    cell: int
    old_value: int
    old_notes: frozenset[int]
    kind: str


@dataclass
class Game:
    puzzle: Puzzle
    max_mistakes: int = MAX_MISTAKES
    values: list[int] = field(default_factory=list)
    notes: list[set[int]] = field(default_factory=list)
    mistakes: int = 0
    hints_used: int = 0
    history: list[Move] = field(default_factory=list)
    wrong_cell: tuple[int, int] | None = None     # last rejected (cell, digit) for UI feedback
    _elapsed: float = 0.0
    _running_since: float | None = None
    _now: callable = time.monotonic

    def __post_init__(self) -> None:
        self.values = list(self.puzzle.grid)
        self.notes = [set() for _ in range(CELLS)]
        self.resume()

    # ----------------------------------------------------------- status ----
    def is_given(self, i: int) -> bool:
        return self.puzzle.grid[i] != 0

    @property
    def solved(self) -> bool:
        return tuple(self.values) == self.puzzle.solution

    @property
    def lost(self) -> bool:
        return self.mistakes >= self.max_mistakes

    @property
    def status(self) -> str:
        return "won" if self.solved else "lost" if self.lost else "playing"

    @property
    def over(self) -> bool:
        return self.status != "playing"

    def remaining(self, d: int) -> int:
        return N_PER_DIGIT - sum(1 for v in self.values if v == d)

    # ------------------------------------------------------------ timer ----
    def resume(self) -> None:
        if self._running_since is None and not self.over:
            self._running_since = self._now()

    def pause(self) -> None:
        if self._running_since is not None:
            self._elapsed += self._now() - self._running_since
            self._running_since = None

    @property
    def elapsed(self) -> float:
        extra = self._now() - self._running_since if self._running_since is not None else 0.0
        return self._elapsed + extra

    def clock(self) -> str:
        s = int(self.elapsed)
        return f"{s // 60:02d}:{s % 60:02d}"

    def _finish_if_over(self) -> None:
        if self.over:
            self.pause()

    # ------------------------------------------------------------ moves ----
    def place(self, i: int, d: int) -> str:
        """Enter digit d in cell i. Returns 'ok' | 'wrong' | 'given' | 'over' | 'same'."""
        if self.over:
            return "over"
        if self.is_given(i):
            return "given"
        if self.values[i] == d:
            return "same"
        if d != self.puzzle.solution[i]:
            self.mistakes += 1
            self.wrong_cell = (i, d)
            self._finish_if_over()
            return "wrong"
        self.wrong_cell = None
        self._commit(i, d, "place")
        return "ok"

    def _commit(self, i: int, d: int, kind: str) -> None:
        self.history.append(Move(i, self.values[i], frozenset(self.notes[i]), kind))
        self.values[i] = d
        self.notes[i].clear()
        for j in PEERS[i]:                    # a placed digit leaves its peers' notes
            self.notes[j].discard(d)
        self._finish_if_over()

    def toggle_note(self, i: int, d: int) -> bool:
        if self.over or self.is_given(i) or self.values[i]:
            return False
        self.history.append(Move(i, self.values[i], frozenset(self.notes[i]), "note"))
        self.notes[i] ^= {d}
        return True

    def erase(self, i: int) -> bool:
        if self.over or self.is_given(i) or not (self.values[i] or self.notes[i]):
            return False
        self.history.append(Move(i, self.values[i], frozenset(self.notes[i]), "erase"))
        self.values[i] = 0
        self.notes[i].clear()
        return True

    def undo(self) -> bool:
        if not self.history or self.over:
            return False
        m = self.history.pop()
        self.values[m.cell] = m.old_value
        self.notes[m.cell] = set(m.old_notes)
        return True

    def auto_notes(self) -> None:
        """Fill every empty cell with its legal candidates."""
        for i in range(CELLS):
            if self.values[i] == 0:
                self.notes[i] = set(candidates(tuple(self.values), i))

    # ------------------------------------------------------------ hints ----
    def hint(self, prefer: int | None = None) -> tuple[int, int, str] | None:
        """Reveal one correct digit with a reason. Prefers the selected cell if it is blank."""
        if self.over:
            return None
        blanks_ = [i for i in range(CELLS) if self.values[i] == 0]
        if not blanks_:
            return None
        grid = tuple(self.values)
        _, moves = propagate(grid)
        pick = None
        if moves:
            pick = next((m for m in moves if m[0] == prefer), moves[0])
        if pick is None:
            cell = prefer if prefer in blanks_ else blanks_[0]
            r, c = rc(cell)
            pick = (cell, self.puzzle.solution[cell],
                    f"Row {r+1}, column {c+1} is {self.puzzle.solution[cell]} (revealed by the solver).")
        cell, d, why = pick
        self.hints_used += 1
        self._commit(cell, d, "hint")
        return cell, d, why

    def reveal(self, cell: int, d: int) -> None:
        """Fill a cell with a known-correct digit (used by the quantum hint); counts as a hint."""
        if self.over or self.is_given(cell) or self.values[cell] or d != self.puzzle.solution[cell]:
            return
        self.hints_used += 1
        self._commit(cell, d, "hint")

    def conflicts_now(self) -> set[int]:
        from .puzzle import conflicts
        return conflicts(tuple(self.values))


N_PER_DIGIT = 4     # each digit appears 4 times on a 4x4 board
