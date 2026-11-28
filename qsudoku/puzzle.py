"""4x4 Sudoku model: grids, units, validity, generation and presets.

A grid is a flat tuple of 16 ints (row-major). 0 means blank, 1..4 are digits.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

N = 4          # digits 1..4, 4 rows, 4 columns
BOX = 2        # 2x2 boxes
CELLS = N * N
DIGITS = tuple(range(1, N + 1))

Grid = tuple[int, ...]

# Level -> number of blanks (from the project proposal: 4 / 6 / 8 blanks)
LEVELS: dict[str, int] = {"Easy": 4, "Medium": 6, "Hard": 8}


def rc(i: int) -> tuple[int, int]:
    return divmod(i, N)


def idx(r: int, c: int) -> int:
    return r * N + c


def box_of(i: int) -> int:
    r, c = rc(i)
    return (r // BOX) * BOX + c // BOX


def _build_units() -> tuple[tuple[int, ...], ...]:
    rows = [tuple(idx(r, c) for c in range(N)) for r in range(N)]
    cols = [tuple(idx(r, c) for r in range(N)) for c in range(N)]
    boxes = [
        tuple(i for i in range(CELLS) if box_of(i) == b) for b in range(N)
    ]
    return tuple(rows + cols + boxes)


UNITS = _build_units()
PEERS: tuple[frozenset[int], ...] = tuple(
    frozenset(j for u in UNITS if i in u for j in u if j != i) for i in range(CELLS)
)


# ---------------------------------------------------------------- parsing ----
def parse(text: str) -> Grid:
    """Parse '1..4 .412 ...' style text. Digits 1-4 are clues; '.', '0', '_' blanks."""
    chars = [ch for ch in text if ch in "01234._"]
    if len(chars) != CELLS:
        raise ValueError(f"need {CELLS} cells, got {len(chars)}")
    return tuple(int(ch) if ch in "1234" else 0 for ch in chars)


def to_string(grid: Grid) -> str:
    rows = ["".join(str(v) if v else "." for v in grid[r * N:(r + 1) * N]) for r in range(N)]
    return "/".join(rows)


# --------------------------------------------------------------- queries ----
def blanks(grid: Grid) -> list[int]:
    return [i for i, v in enumerate(grid) if v == 0]


def candidates(grid: Grid, i: int) -> set[int]:
    """Digits that do not clash with the peers of cell i (cell itself ignored)."""
    used = {grid[j] for j in PEERS[i]}
    return {d for d in DIGITS if d not in used}


def conflicts(grid: Grid) -> set[int]:
    """Cells that take part in a duplicate inside some row, column or box."""
    bad: set[int] = set()
    for unit in UNITS:
        seen: dict[int, int] = {}
        for i in unit:
            v = grid[i]
            if v == 0:
                continue
            if v in seen:
                bad.update((i, seen[v]))
            else:
                seen[v] = i
    return bad


def is_consistent(grid: Grid) -> bool:
    return not conflicts(grid)


def is_solved(grid: Grid) -> bool:
    return all(grid) and is_consistent(grid)


# ------------------------------------------------------ solution counting ----
def count_solutions(grid: Grid, limit: int = 2) -> int:
    """Count completions of `grid`, stopping at `limit` (used for uniqueness)."""
    if not is_consistent(grid):
        return 0
    g = list(grid)

    def rec() -> int:
        best, best_c = -1, None
        for i in range(CELLS):
            if g[i] == 0:
                c = candidates(tuple(g), i)
                if not c:
                    return 0
                if best_c is None or len(c) < len(best_c):
                    best, best_c = i, c
                    if len(c) == 1:
                        break
        if best == -1:
            return 1
        total = 0
        for d in sorted(best_c):
            g[best] = d
            total += rec()
            if total >= limit:
                break
        g[best] = 0
        return total

    return rec()


def all_solutions(grid: Grid, limit: int = 100000) -> list[Grid]:
    """Enumerate all completions of `grid` (4x4 is tiny, so this is cheap)."""
    out: list[Grid] = []
    if not is_consistent(grid):
        return out
    g = list(grid)

    def rec() -> None:
        if len(out) >= limit:
            return
        best, best_c = -1, None
        for i in range(CELLS):
            if g[i] == 0:
                c = candidates(tuple(g), i)
                if not c:
                    return
                if best_c is None or len(c) < len(best_c):
                    best, best_c = i, c
        if best == -1:
            out.append(tuple(g))
            return
        for d in sorted(best_c):
            g[best] = d
            rec()
        g[best] = 0

    rec()
    return out


# ------------------------------------------------------------ generation ----
@dataclass(frozen=True)
class Puzzle:
    grid: Grid
    solution: Grid
    level: str = "Custom"
    name: str = ""

    @property
    def n_blanks(self) -> int:
        return len(blanks(self.grid))

    @property
    def n_clues(self) -> int:
        return CELLS - self.n_blanks


def random_solution(rng: random.Random) -> Grid:
    g = [0] * CELLS

    def rec(pos: int) -> bool:
        if pos == CELLS:
            return True
        opts = sorted(candidates(tuple(g), pos))
        rng.shuffle(opts)
        for d in opts:
            g[pos] = d
            if rec(pos + 1):
                return True
        g[pos] = 0
        return False

    rec(0)
    return tuple(g)


def generate(n_blanks: int, seed: int | None = None, level: str = "Custom") -> Puzzle:
    """Random puzzle with exactly `n_blanks` blanks and a UNIQUE solution."""
    if not 1 <= n_blanks <= 10:
        raise ValueError("n_blanks must be between 1 and 10 for a unique 4x4 puzzle")
    rng = random.Random(seed)
    while True:
        sol = random_solution(rng)
        for _ in range(300):
            holes = set(rng.sample(range(CELLS), n_blanks))
            grid = tuple(0 if i in holes else v for i, v in enumerate(sol))
            if count_solutions(grid, 2) == 1:
                return Puzzle(grid, sol, level)


def new_puzzle(level: str, seed: int | None = None) -> Puzzle:
    return generate(LEVELS[level], seed=seed, level=level)


def _preset(level: str, text: str, name: str) -> Puzzle:
    grid = parse(text)
    sols = all_solutions(grid, 2)
    assert len(sols) == 1, f"preset {name} must have a unique solution"
    return Puzzle(grid, sols[0], level, name)


PRESETS: dict[str, list[Puzzle]] = {}
# Fixed presets (generated once from seeds 11/22/33, each has a unique solution).
_PRESET_DATA = [
    ("Easy", "1..2/2341/..23/3214", "Easy 1"),
    ("Easy", ".132/3.14/..41/1423", "Easy 2"),
    ("Easy", "41.2/.3.4/1243/34.1", "Easy 3"),
    ("Medium", "1..2/2341/..../3214", "Medium 1"),
    ("Medium", ".1.2/3.14/..41/142.", "Medium 2"),
    ("Medium", "41.2/.3../.243/34.1", "Medium 3"),
    ("Hard", ".4.2/2341/4.../.2..", "Hard 1"),
    ("Hard", ".1.2/3..4/..41/14..", "Hard 2"),
    ("Hard", ".1.2/.3../.243/3..1", "Hard 3"),
]
for _lvl, _txt, _name in _PRESET_DATA:
    PRESETS.setdefault(_lvl, []).append(_preset(_lvl, _txt, _name))
