"""Classical baseline: backtracking with optional pruning (constraint propagation)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .puzzle import (CELLS, DIGITS, PEERS, UNITS, Grid, blanks, candidates,
                     is_consistent, rc)


@dataclass
class Step:
    """One event in a solver trace (for the step-by-step animation)."""
    cell: int
    value: int
    kind: str          # "try" | "clash" | "place" | "undo" | "prune"


@dataclass
class ClassicalResult:
    solved: bool
    solution: Grid | None
    tries: int                     # candidate (cell, digit) placements tested
    backtracks: int                # times a placement was undone
    elapsed: float                 # seconds
    pruned_cells: int = 0          # cells filled by propagation before searching
    mode: str = "plain"
    trace: list[Step] = field(default_factory=list)
    success_probability: float = 1.0   # deterministic solver


# ------------------------------------------------------------ propagation ----
def propagate(grid: Grid) -> tuple[Grid | None, list[tuple[int, int, str]]]:
    """Fill every cell the rules can decide (naked + hidden singles).

    Returns (new_grid, moves) where moves = [(cell, digit, reason), ...].
    new_grid is None if a contradiction is found.
    """
    g = list(grid)
    moves: list[tuple[int, int, str]] = []
    if not is_consistent(grid):
        return None, moves
    changed = True
    while changed:
        changed = False
        # naked singles: a cell with exactly one candidate
        for i in range(CELLS):
            if g[i]:
                continue
            c = candidates(tuple(g), i)
            if not c:
                return None, moves
            if len(c) == 1:
                d = next(iter(c))
                used = sorted({g[j] for j in PEERS[i]} - {0})
                r, col = rc(i)
                moves.append((i, d, f"Row {r+1}, column {col+1} can only be {d}: "
                                    f"its row, column and box already contain {', '.join(map(str, used))}."))
                g[i] = d
                changed = True
        # hidden singles: a digit with exactly one possible place in a unit
        for ui, unit in enumerate(UNITS):
            for d in DIGITS:
                if any(g[i] == d for i in unit):
                    continue
                spots = [i for i in unit if g[i] == 0 and d in candidates(tuple(g), i)]
                if not spots:
                    return None, moves
                if len(spots) == 1:
                    i = spots[0]
                    kind = ("row", "column", "box")[ui // 4]
                    r, col = rc(i)
                    moves.append((i, d, f"In this {kind}, {d} has only one place left: "
                                        f"row {r+1}, column {col+1}."))
                    g[i] = d
                    changed = True
    return tuple(g), moves


# --------------------------------------------------------------- backtrack ----
def solve(grid: Grid, pruning: bool = False, record_trace: bool = False) -> ClassicalResult:
    """Backtracking solver for 4x4 Sudoku.

    pruning=False : plain backtracking, blanks filled in reading order.
    pruning=True  : propagate singles first, then pick the most constrained blank.
    `tries` counts every candidate digit tested, including those that clash.
    """
    t0 = time.perf_counter()
    trace: list[Step] = []
    tries = backtracks = 0
    start = grid
    pruned = 0

    if pruning:
        start2, moves = propagate(grid)
        if start2 is None:
            return ClassicalResult(False, None, 0, 0, time.perf_counter() - t0, 0, "pruned")
        pruned = len(moves)
        if record_trace:
            trace += [Step(c, d, "prune") for c, d, _ in moves]
        start = start2

    g = list(start)

    def clash(i: int, d: int) -> bool:
        return any(g[j] == d for j in PEERS[i])

    def pick() -> int:
        free = [i for i in range(CELLS) if g[i] == 0]
        if not free:
            return -1
        if not pruning:
            return free[0]
        return min(free, key=lambda i: len(candidates(tuple(g), i)))

    def rec() -> bool:
        nonlocal tries, backtracks
        i = pick()
        if i == -1:
            return True
        for d in DIGITS:
            tries += 1
            if record_trace:
                trace.append(Step(i, d, "try"))
            if clash(i, d):
                if record_trace:
                    trace.append(Step(i, d, "clash"))
                continue
            g[i] = d
            if record_trace:
                trace.append(Step(i, d, "place"))
            if rec():
                return True
            g[i] = 0
            backtracks += 1
            if record_trace:
                trace.append(Step(i, d, "undo"))
        return False

    ok = is_consistent(start) and rec()
    return ClassicalResult(
        solved=ok, solution=tuple(g) if ok else None, tries=tries, backtracks=backtracks,
        elapsed=time.perf_counter() - t0, pruned_cells=pruned,
        mode="pruned" if pruning else "plain", trace=trace,
    )
