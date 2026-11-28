# Quantum Sudoku Solver

A playable 4×4 Sudoku with two solvers that attack the same puzzle: a **classical backtracking** solver and a **Grover quantum search** built in Qiskit and run on Qiskit Aer. Group 7 · Quantum Computing.

## Set up

On a new machine (Python 3.13):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

In this folder the existing environment lives one level up, so the commands below use `../.venv`.

## Run it

```bash
../.venv/bin/streamlit run app.py        # the app
../.venv/bin/jupyter lab notebooks/      # the notebook (Quantum_Sudoku.ipynb)
../.venv/bin/python -m pytest -q tests   # 53 tests
```

## The app (one page)

Everything follows the puzzle you are playing.

**At the top: one board, two views.**

- **Your board.** Pick a level (Easy 4 / Medium 6 / Hard 8 empty cells) and a puzzle (3 presets or Random). Play with the mouse or the keyboard: arrows move, 1–4 fill, N notes, Backspace erase, Z undo, H hint. Notes show as a 2×2 grid. Hints explain *why*. **Quantum hint** runs Grover on your current board and reveals a cell. Three mistakes allowed; the board lights up when you win.
- **Grover's view** (press G). The same board, drawn as pencil marks whose size follows how likely each digit is after *k* rounds of Grover's search on **your current board**. Drag the rounds slider, click the curve, or press ▶ to watch the right digits grow from 25% to almost 100% (and fade again if you go past the best number of rounds). **Measure** runs the real circuit on Qiskit Aer (Easy) or the NumPy simulator and shows the measured outcomes. Scrubbing is instant because Grover's state only ever holds two amplitudes, so every round has a closed-form answer (tested against the full simulation).

**Below, computed automatically:**

| Section | What it shows |
|---|---|
| **Solve it both ways** | Backtracking and Grover on the same puzzle: both answers, digits tried vs oracle queries vs blind guessing, success probability, and a step-by-step replay of the backtracking. Settings for pruning, shots and simulator (gate-level Aer only offered when the circuit fits, i.e. Easy). |
| **The circuit behind it** | The Qiskit circuit (H, Oracle, Diffuser, measure), a toggle to open the oracle, and what it would cost on real hardware. |
| **Across all three levels** | A batch of random puzzles per level solved both ways. |

Design: IBM Plex Sans (Qiskit's typeface), the colours from the proposal slides (navy, qubit blue, measurement amber, oracle magenta), HTML/SVG charts that match the page, and a layout that works down to phone width.

## How the quantum solver works

- **Encoding.** Each blank holds 1–4 in 2 qubits (`1→00 … 4→11`).
- **Oracle.** One helper qubit per rule — *two blanks that see each other must differ* and *a blank must not equal a clue that sees it*. A flag goes up when its rule is broken; the oracle flips the sign of the state when no flag is up, then undoes the checks.
- **Diffuser.** Inversion about the mean.
- **Rounds.** `round(π/(4θ) − ½)` with `sin θ = √(M/N)`: **12 / 50 / 201** for Easy / Medium / Hard.

## Two backends — be aware of this

| Backend | What it simulates | When |
|---|---|---|
| `aer` | The real gate-level circuit (search + helper qubits) on Qiskit Aer | Automatic up to 18 qubits (Easy, ~1 s) |
| `fast` | Only the search register with NumPy: oracle = sign flip on valid fillings, diffuser = inversion about the mean | Medium and Hard (23–39 qubits) |

Gate-level Aer on Medium took about **45 minutes** on the development machine and Hard does not fit in memory, which is why `fast` exists. The tests check that, on Easy, the circuit and `fast` give identical probabilities (difference below 1e-9) and that the oracle flips exactly the valid fillings. The Circuit Viewer still draws and costs out the real circuit for every level.

## What the numbers say (honestly)

Backtracking solves these 4×4 puzzles in ~10–30 tries because it uses the rules to prune; Grover's oracle is a black box. Grover's advantage is against *unstructured* search: about √N queries instead of N/2 (201 vs ~32,768 on Hard). Both solvers are always correct. The notebook (§12) and the Solve and Compare page show both comparisons side by side, and §13 shows how quickly noise destroys the result on real hardware.

## Layout

```
quantum_sudoku/
  app.py                  Streamlit app (one page)
  components/board/       the top of the page: app bar, board with both views, controls (HTML + JS, no build step)
  qsudoku/
    puzzle.py             grid model, units, generator, presets
    classical.py          backtracking + constraint propagation, step trace
    quantum.py            encoding, oracle, diffuser, Grover, Aer + fast backends
    game.py               game rules (notes, hints, mistakes, undo, timer)
    play.py               turns board actions into moves, and game state into board props
    compare.py            side-by-side comparison and benchmark
    viz.py                circuit drawing + matplotlib charts (used by the notebook)
    render.py             HTML pieces for the sections below the board (panels, bars, tables)
    runtime.py            big-stack thread for Qiskit/Aer inside Streamlit (macOS)
  notebooks/Quantum_Sudoku.ipynb
  tests/                  pytest suite
  .streamlit/config.toml  theme
  requirements.txt        pinned versions
```

## Notes

- Puzzles up to 10 blanks can be searched (4¹⁰ ≈ 1M fillings).
- If a puzzle has several solutions, Grover's round count is tuned for the known number of solutions (in practice that number would come from quantum counting).
- The "Classical pre-pass" option fills every cell the rules can decide before Grover runs. On 4×4 that usually leaves nothing to search, so it is off by default.
