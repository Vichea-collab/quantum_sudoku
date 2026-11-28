"""Play-area controller: turns UI actions into Game moves and Game state into board props.

Kept free of Streamlit so it can be unit-tested. The board component sends actions like
{"type": "digit", "digit": 3, "sel": 5, "nonce": 17}; `apply_action` updates the Game and the
UI state; `board_props` is everything the component needs to draw itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .game import Game
from .puzzle import CELLS, DIGITS, PEERS, rc

GroverFn = Callable[[tuple[int, ...]], object]          # grid -> GroverResult
MeasureFn = Callable[[tuple[int, ...], int], object]    # (grid, rounds) -> GroverResult


@dataclass
class UiState:
    sel: int | None = None
    sel_token: int = 0            # bumped when the server moves the selection (e.g. a hint)
    note_mode: bool = False
    msg_kind: str = "info"
    msg_text: str = "Pick a cell, then a number. You can also use the keyboard."
    last_nonce: int = -1
    game_no: int = 0              # bumped on every new game so the board resets its local state
    measure: dict | None = None   # last "Measure on Qiskit Aer" result for Grover's view

    def say(self, kind: str, text: str) -> None:
        self.msg_kind, self.msg_text = kind, text

    def select(self, cell: int | None) -> None:
        self.sel = cell
        self.sel_token += 1


def new_game_request(ui: UiState, action: dict | None) -> tuple[str, str] | None:
    """If `action` asks for a new game (and isn't a repeat), return (level, source)."""
    from .puzzle import LEVELS
    if not isinstance(action, dict) or action.get("type") != "new":
        return None
    nonce = action.get("nonce")
    if not isinstance(nonce, int) or nonce == ui.last_nonce:
        return None
    level, source = action.get("level"), action.get("source")
    if level not in LEVELS or source not in ("1", "2", "3", "Random"):
        return None
    ui.last_nonce = nonce
    return level, source


def _where(i: int) -> str:
    r, c = rc(i)
    return f"row {r + 1}, column {c + 1}"


def apply_action(game: Game, ui: UiState, action: dict | None, grover: GroverFn | None = None,
                 measure: MeasureFn | None = None) -> bool:
    """Apply one action. Returns False if it was a repeat (same nonce) or malformed."""
    if not isinstance(action, dict):
        return False
    nonce = action.get("nonce")
    if not isinstance(nonce, int) or nonce == ui.last_nonce:
        return False
    ui.last_nonce = nonce

    game.wrong_cell = None        # the red "that was wrong" mark lasts until the next action
    sel = action.get("sel")
    ui.sel = sel if isinstance(sel, int) and 0 <= sel < CELLS else None
    kind = action.get("type")

    if kind == "digit":
        _digit(game, ui, int(action.get("digit", 0)))
    elif kind == "erase":
        if ui.sel is None or not game.erase(ui.sel):
            ui.say("warn", "Select a cell you filled to erase it.")
    elif kind == "undo":
        if not game.undo():
            ui.say("warn", "Nothing to undo yet.")
    elif kind == "notes":
        ui.note_mode = not ui.note_mode
        ui.say("info", "Notes on: numbers become pencil marks." if ui.note_mode
               else "Notes off: numbers fill the cell.")
    elif kind == "autonotes":
        game.auto_notes()
        ui.say("info", "Pencil marks added for every digit each empty cell can still take.")
    elif kind == "hint":
        h = game.hint(prefer=ui.sel)
        if h:
            cell, _, why = h
            ui.select(cell)
            ui.say("info", why)
    elif kind == "qhint":
        _quantum_hint(game, ui, grover)
    elif kind == "measure":
        _measure(game, ui, measure, action.get("k"))
    else:
        return False
    return True


def _digit(game: Game, ui: UiState, d: int) -> None:
    if d not in DIGITS or game.over:
        return
    i = ui.sel
    if i is None:
        ui.say("warn", "Select a cell first, then press a number.")
        return
    if game.is_given(i):
        ui.say("warn", "That cell is part of the puzzle and can't be changed.")
        return
    if ui.note_mode:
        game.toggle_note(i, d)
        return
    res = game.place(i, d)
    if res == "wrong":
        ui.say("error", f"{d} doesn't fit in {_where(i)}. Mistake {game.mistakes} of {game.max_mistakes}.")
    elif res == "ok":
        ui.say("ok", f"{d} is right for {_where(i)}.")


def _quantum_hint(game: Game, ui: UiState, grover: GroverFn | None) -> None:
    grid = tuple(game.values)
    cells = [i for i, v in enumerate(grid) if v == 0]
    if game.over or not cells or grover is None:
        return
    res = grover(grid)
    if not res.solved:
        ui.say("warn", f"Grover measured a filling that breaks a rule "
                       f"({res.exact_probability:.1%} chance of success). Try again.")
        return
    cell = ui.sel if ui.sel in cells else cells[0]
    game.reveal(cell, res.solution[cell])
    ui.select(cell)
    ui.say("quantum", f"Grover searched all {res.problem.space:,} fillings of your "
                      f"{res.problem.n_blanks} blank cells in {res.iterations} rounds and revealed "
                      f"{_where(cell)}.")


def _measure(game: Game, ui: UiState, measure: MeasureFn | None, k) -> None:
    grid = tuple(game.values)
    if measure is None or not isinstance(k, int) or k < 0 or all(grid):
        return
    res = measure(grid, k)
    mask = res.problem.solution_mask()
    top = []
    for x, c in res.top_outcomes[:5]:
        top.append({"digits": "".join(map(str, res.problem.values(x))), "frac": c / res.shots,
                    "ok": bool(mask[x])})
    ui.measure = {"key": "".join(map(str, grid)), "k": k, "shots": res.shots, "top": top,
                  "hit": res.success_probability,
                  "backend": "Qiskit Aer" if res.backend == "aer" else "the NumPy simulator"}


def lens_props(game: Game) -> dict | None:
    """Data for Grover's view of the CURRENT board: the browser turns it into probabilities."""
    from .quantum import AER_QUBIT_LIMIT, SearchProblem, digit_counts, optimal_iterations
    grid = tuple(game.values)
    if all(grid):
        return None
    sp = SearchProblem.from_grid(grid)
    sol, non = digit_counts(sp)
    k_opt = optimal_iterations(sp.space, max(sp.n_solutions, 1))
    return {
        "key": "".join(map(str, grid)),
        "blanks": list(sp.blanks),
        "N": sp.space, "M": sp.n_solutions,
        "sol": sol.tolist(), "non": non.tolist(),
        "kOpt": k_opt, "kMax": max(2 * k_opt + 2, 6),
        "searchQubits": sp.n_search, "helperQubits": sp.n_flags,
        "aer": sp.n_qubits <= AER_QUBIT_LIMIT,
    }


def board_props(game: Game, ui: UiState) -> dict:
    """JSON-safe description of the play area for the board component."""
    status = game.status
    if status == "won":
        msg = ("won", f"Solved in {game.clock()} with {game.mistakes} mistake(s) and "
                      f"{game.hints_used} hint(s). Scroll down to see how Grover finds the same answer.")
    elif status == "lost":
        msg = ("lost", "Three mistakes: this game is over. Press New game to try again, "
                       "or scroll down to see both solvers finish it.")
    else:
        msg = (ui.msg_kind, ui.msg_text)
    wrong = game.wrong_cell[0] if game.wrong_cell and not game.values[game.wrong_cell[0]] else None
    return {
        "values": list(game.values),
        "given": [game.is_given(i) for i in range(CELLS)],
        "notes": [sorted(n) for n in game.notes],
        "conflicts": sorted(game.conflicts_now()),
        "wrong": wrong,
        "peers": [sorted(PEERS[i]) for i in range(CELLS)],
        "sel": ui.sel,
        "selToken": ui.sel_token,
        "noteMode": ui.note_mode,
        "remaining": {str(d): game.remaining(d) for d in DIGITS},
        "mistakes": game.mistakes,
        "maxMistakes": game.max_mistakes,
        "hints": game.hints_used,
        "blanksLeft": sum(1 for v in game.values if not v),
        "totalBlanks": game.puzzle.n_blanks,
        "elapsed": round(game.elapsed, 2),
        "running": status == "playing",
        "status": status,
        "msgKind": msg[0],
        "msgText": msg[1],
        "gameId": ui.game_no,
        "lens": lens_props(game),
        "measure": ui.measure,
    }
