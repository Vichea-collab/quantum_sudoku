"""Quantum Sudoku — one page: play the puzzle, solve it both ways, look inside Grover's search.

Run:  streamlit run app.py
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

from qsudoku import puzzle as pz
from qsudoku import render as R
from qsudoku import viz
from qsudoku.classical import solve as classical_solve
from qsudoku.compare import benchmark
from qsudoku.game import Game
from qsudoku.play import UiState, apply_action, board_props, new_game_request
from qsudoku.puzzle import LEVELS, PRESETS, new_puzzle, rc
from qsudoku.quantum import (AER_QUBIT_LIMIT, SearchProblem, circuit_stats, optimal_iterations,
                             run_grover)
from qsudoku.runtime import run_big_stack

st.set_page_config(page_title="Quantum Sudoku", page_icon="⚛️", layout="wide",
                   initial_sidebar_state="collapsed")

ss = st.session_state
board_component = components.declare_component(
    "qsudoku_board", path=str(Path(__file__).parent / "components" / "board"))
SOURCES = ["1", "2", "3", "Random"]


def heavy(fn, *args, **kwargs):
    """Qiskit-heavy work runs on a big-stack thread (see qsudoku.runtime)."""
    return run_big_stack(fn, *args, **kwargs)


# ================================================================ state ======
def init_state() -> None:
    if "game" in ss:
        return
    ss.game = Game(PRESETS["Easy"][0])
    ss.ui = UiState()
    ss.level = "Easy"
    ss.source = "1"
    ss.random_seed = 1


def puzzle_key(grid: pz.Grid) -> str:
    return pz.to_string(grid)


def start(level: str, source: str) -> None:
    if level not in LEVELS or source not in SOURCES:
        return
    if source == "Random":
        ss.random_seed += 1
        puzzle = new_puzzle(level, seed=ss.random_seed * 7919 + list(LEVELS).index(level))
        name = f"random {level.lower()} puzzle"
    else:
        puzzle = PRESETS[level][int(source) - 1]
        name = f"{level} puzzle {source}"
    ss.game = Game(puzzle)
    ss.ui = UiState(game_no=ss.ui.game_no + 1, last_nonce=ss.ui.last_nonce)
    ss.ui.say("info", f"New {name}: {puzzle.n_blanks} empty cells to fill.")
    ss.level, ss.source = level, source


def on_board() -> None:
    """The board component sent an action (a digit, a new game, a measurement, ...)."""
    action = ss.get("board")
    request = new_game_request(ss.ui, action)
    if request:
        start(*request)
        return
    apply_action(ss.game, ss.ui, action,
                 grover=lambda grid: heavy(run_grover, grid, shots=256),
                 measure=lambda grid, k: heavy(run_grover, grid, iterations=k, shots=1024))


# =============================================================== caching =====
@st.cache_resource(max_entries=32, show_spinner=False)
def solve_both(grid: pz.Grid, cpr: bool, gpr: bool, shots: int, backend: str):
    cl = classical_solve(grid, pruning=cpr, record_trace=True)
    gr = heavy(run_grover, grid, prune=gpr, shots=shots, backend=backend, seed=7)
    return cl, gr


def _circuit_png(grid: pz.Grid, expand: bool) -> bytes:
    fig = viz.plot_circuit(SearchProblem.from_grid(grid), expand_oracle=expand)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


@st.cache_data(max_entries=24, show_spinner=False)
def circuit_png(grid: pz.Grid, expand: bool) -> bytes:
    return heavy(_circuit_png, grid, expand)


@st.cache_data(max_entries=24, show_spinner=False)
def cost_at(grid: pz.Grid, k: int):
    return heavy(circuit_stats, SearchProblem.from_grid(grid), k)


@st.cache_data(show_spinner=False)
def bench_rows(n: int):
    return benchmark(n_puzzles=n, backend="fast", seed=100)


# ================================================================ styling =====
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
:root{
  --bg:#F3F5FA; --surface:#FFFFFF; --ink:#161B3D; --ink2:#4A5275; --line:#DCE1EC;
  --qubit:#3E5BDB; --qubit-soft:#E4EAFE; --amber:#E08A2E; --amber-soft:#FCEBD6;
  --oracle:#9B1B4E; --error:#D93F4C; --error-soft:#FDE6E8; --ok:#2F8F5B;
}
.stApp{background:var(--bg);}
.stApp, .stApp p, .stApp li, .stApp label, .stApp button, .stApp input,
.stApp h1, .stApp h2, .stApp h3, .stApp td, .stApp th, .stMarkdown{
  font-family:'IBM Plex Sans', system-ui, sans-serif; color:var(--ink);}
header[data-testid="stHeader"]{display:none;}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"]{display:none;}
.block-container{max-width:1180px; padding:1.6rem 1.5rem 5rem;}
iframe[title="app.qsudoku_board"]{display:block;}
.stApp .muted{color:var(--ink2);} .stApp .small{font-size:.85rem;}
[data-testid="stHeaderActionElements"]{display:none;}
[data-testid="stPopover"] button, .stButton button{border-radius:12px; border:1px solid var(--line);
  background:var(--surface); color:var(--ink); font-weight:500; min-height:2.75rem; box-shadow:none;}
[data-testid="stPopover"] button:hover, .stButton button:hover{border-color:var(--qubit); color:var(--qubit);}
/* ---------- shared readouts ---------- */
.stats{display:flex; gap:2rem; flex-wrap:wrap; margin:.2rem 0 1rem;}
.stats .k{display:block; font-size:.82rem; color:var(--ink2);}
.stats .v{display:block; font-size:1.55rem; font-weight:600; font-variant-numeric:tabular-nums;}
.panel .stats{flex-direction:column; gap:.85rem; margin:0;} .panel .stats .v{font-size:1.2rem;}
.note{border-left:4px solid var(--qubit); background:var(--surface); padding:.75rem 1rem;
  border-radius:0 12px 12px 0; margin-top:1rem; font-size:.97rem; line-height:1.45;}
.note-ok{border-color:var(--ok);} .note-warn{border-color:var(--amber);}
.note-error{border-color:var(--error); background:var(--error-soft);}
.note-quantum{border-color:var(--oracle);}
.note-won{border-color:var(--ok); font-weight:500;}
.note-lost{border-color:var(--error); background:var(--error-soft); font-weight:500;}

/* ---------- sections ---------- */
.rule{border:0; border-top:1px solid var(--line); margin:4.2rem 0 0;}
.sec{margin:2.2rem 0 1.2rem;}
.sec h2{font-size:1.75rem; font-weight:600; letter-spacing:-.01em; margin:0; padding:0;}
.stApp .sec p{color:var(--ink2); margin:.45rem 0 0; max-width:44rem; line-height:1.5;}
.panel{background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:1.25rem 1.4rem;}
.panel h3{font-size:1.1rem; font-weight:600; margin:0 0 1rem; padding:0;}
.panel .row{display:flex; gap:1.6rem; align-items:flex-start; flex-wrap:nowrap;}
.panel.duo{display:grid; grid-template-columns:1fr 1fr; gap:2.5rem;}
@media (max-width:760px){ .panel.duo{grid-template-columns:1fr; gap:1.5rem;} }
.hb-row .hb-val{min-width:3.6rem;}

/* mini read-only grid */
.mg{display:grid; grid-template-columns:repeat(4,2.4rem); border:2px solid var(--ink); border-radius:8px; overflow:hidden;}
.mg-cell{height:2.4rem; display:flex; align-items:center; justify-content:center; font-weight:600;
  border-right:1px solid var(--line); border-bottom:1px solid var(--line); font-variant-numeric:tabular-nums;}
.mg-given{background:#EEF1F8;}
.mg-new.mg-qubit{color:var(--qubit);} .mg-new.mg-amber{color:var(--amber);}

/* shared box lines for 4x4 html grids */
.bx-r{border-right:2px solid var(--ink) !important;} .bx-b{border-bottom:2px solid var(--ink) !important;}
.last-c{border-right:none !important;} .last-r{border-bottom:none !important;}

/* bars */
.hb-row, .ob-row{display:grid; grid-template-columns:7.6rem 1fr 4.2rem; align-items:center; gap:.8rem; margin:.45rem 0;}
.hb-name{font-weight:500;}
.hb-track{height:.9rem; background:#EDF0F6; border-radius:999px; overflow:hidden;}
.hb-fill{display:block; height:100%; border-radius:999px;}
.hb-val{text-align:right; font-variant-numeric:tabular-nums; color:var(--ink2);}

.circuit{background:#fff; border:1px solid var(--line); border-radius:14px; padding:1.2rem; overflow:auto;}
.circuit img{display:block; margin:0 auto; width:auto; max-width:100%; max-height:520px;}
.circuit.wide img{max-height:none; width:100%; min-width:900px;}
.inline-stats .stats{flex-direction:row; gap:3rem;} .inline-stats .stats .v{font-size:1.45rem;}

table.bench{width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums;}
table.bench th, table.bench td{padding:.6rem .7rem; text-align:left; border-bottom:1px solid var(--line); font-size:.95rem;}
table.bench thead th{font-weight:500; color:var(--ink2); font-size:.85rem;}
.lbw{display:flex; align-items:center; gap:.5rem;} .lbw span:last-child{min-width:3.5rem;}
.lb{display:block; height:.6rem; border-radius:999px; min-width:3px;}

[data-testid="stExpander"] details{border-radius:12px; border-color:var(--line); background:var(--surface);}
@media (max-width: 640px){
  .stats{gap:1.2rem;} .panel .row{flex-wrap:wrap;}
  .hb-row, .ob-row{grid-template-columns:4.8rem 1fr 3.2rem;}
}
</style>
"""


# ============================================================== sections =====
def play_area() -> None:
    g: Game = ss.game
    g.resume()
    props = board_props(g, ss.ui)
    props.update(level=ss.level, source=ss.source)
    board_component(props=props, key="board", on_change=on_board, default=None)


def solve_section() -> None:
    grid = ss.game.puzzle.grid
    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    head, opts = st.columns([5, 1.3], vertical_alignment="bottom")
    head.markdown(R.section_head(
        "Solve it both ways",
        "The same puzzle, the same rules, two very different searches. Backtracking tries digits cell by "
        "cell and steps back at dead ends. Grover's search holds every filling at once and boosts the "
        "ones that obey the rules."), unsafe_allow_html=True)
    with opts.popover("Settings", use_container_width=True):
        cpr = st.toggle("Backtracking: fill rule-decided cells first", value=False)
        gpr = st.toggle("Grover: fill rule-decided cells first", value=False,
                        help="On a 4×4 grid this usually leaves nothing for Grover to search.")
        shots = st.select_slider("Grover shots", [256, 512, 1024, 2048, 4096], value=1024)
        n_q = SearchProblem.from_grid(grid, gpr).n_qubits
        if n_q <= AER_QUBIT_LIMIT:
            backend = st.radio("Grover simulator", ["auto", "aer", "fast"], horizontal=True,
                               help="aer runs the real circuit gate by gate on Qiskit Aer. fast simulates "
                                    "the same maths with NumPy. auto picks aer here.")
        else:
            backend = "fast"
            st.caption(f"This puzzle's circuit needs {n_q} qubits, so Grover runs on the exact NumPy "
                       f"simulator. The gate-level Aer run is offered up to {AER_QUBIT_LIMIT} qubits "
                       "(Easy); Medium took about 45 minutes and Hard does not fit in memory.")

    with st.spinner("Solving…"):
        cl, gr = solve_both(grid, cpr, gpr, shots, backend)
    sp = gr.problem
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown(
            '<div class="panel"><h3>Backtracking</h3><div class="row">'
            f'{R.mini_grid(cl.solution or grid, grid, "qubit")}'
            + R.stats([("Digits tried", f"{cl.tries}"), ("Steps back", f"{cl.backtracks}"),
                       ("Time", f"{cl.elapsed * 1000:.2f} ms")])
            + "</div></div>", unsafe_allow_html=True)
    with right:
        body = (R.mini_grid(gr.solution, grid, "amber") if gr.solution else
                '<p class="muted">No valid filling was measured.</p>')
        st.markdown(
            '<div class="panel"><h3>Grover\'s search</h3><div class="row">' + body
            + R.stats([("Oracle queries", f"{gr.oracle_queries}"),
                       ("Qubits", f"{sp.n_search} + {sp.n_flags} helpers"),
                       ("Simulator", "Qiskit Aer" if gr.backend == "aer" else
                        "NumPy (exact)" if gr.backend == "fast" else "none needed")])
            + "</div></div>", unsafe_allow_html=True)

    blind = (sp.space + 1) / (sp.n_solutions + 1) if sp.n_solutions else sp.space
    effort = ""
    if sp.n_blanks:
        effort = R.effort_bars([("Backtracking", cl.tries, "digits tried", R.QUBIT),
                                ("Grover", gr.oracle_queries, "oracle queries", R.AMBER),
                                ("Blind guessing", blind, "guesses on average", "#AEB5CB")])
    st.markdown(
        '<div class="panel duo" style="margin-top:1rem">'
        '<div><h3>Chance of the right answer</h3>'
        + R.solver_bars([("Backtracking", 1.0 if cl.solved else 0.0, R.QUBIT),
                         ("Grover", gr.success_probability, R.AMBER)])
        + '</div><div><h3>Steps to find it</h3>' + effort + "</div></div>",
        unsafe_allow_html=True)
    if sp.n_blanks:
        st.markdown(R.note("info",
            f"Backtracking wins here because it uses the rules to cut dead branches. Grover's oracle only answers "
            f"“valid or not”, so the fair rival is blind guessing: about <b>{blind:,.0f}</b> guesses against "
            f"Grover's <b>{gr.oracle_queries}</b> queries. That square-root gap is the quantum advantage, and it "
            f"grows with the puzzle."), unsafe_allow_html=True)
    for n in gr.notes:
        st.markdown(R.note("warn", R.esc(n)), unsafe_allow_html=True)
    st.write("")

    if cl.trace:
        with st.expander("Replay backtracking step by step"):
            n = len(cl.trace)
            step = st.slider("Step", 0, n, min(n, 1), key=f"replay_{puzzle_key(grid)}_{cpr}")
            g_now, text = replay(grid, cl.trace, step)
            x, y = st.columns([1, 3], vertical_alignment="center")
            x.markdown(R.mini_grid(g_now, grid), unsafe_allow_html=True)
            y.markdown(f"**Step {step} of {n}.** {text}")


def replay(grid: pz.Grid, trace, upto: int) -> tuple[pz.Grid, str]:
    g = list(grid)
    text = "The starting puzzle."
    for s in trace[:upto]:
        r, c = rc(s.cell)
        where = f"row {r + 1}, column {c + 1}"
        if s.kind == "prune":
            g[s.cell], text = s.value, f"The rules decide {s.value} for {where}."
        elif s.kind == "try":
            text = f"Try {s.value} in {where}."
        elif s.kind == "clash":
            text = f"{s.value} clashes with its row, column or box in {where}."
        elif s.kind == "place":
            g[s.cell], text = s.value, f"Place {s.value} in {where}."
        elif s.kind == "undo":
            g[s.cell], text = 0, f"Dead end: take {s.value} back out of {where}."
    return tuple(g), text


def circuit_section() -> None:
    grid = ss.game.puzzle.grid
    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    sp = SearchProblem.from_grid(grid)
    k = optimal_iterations(sp.space, max(sp.n_solutions, 1))
    st.markdown(R.section_head(
        "The circuit behind it",
        f"This is the Grover circuit for this puzzle, built in Qiskit. Hadamards spread the {sp.n_search} "
        f"search qubits over every filling. The oracle (magenta) marks fillings that obey the rules, and "
        f"the diffuser (navy) boosts them. The pair repeats {k} times, then the search qubits are "
        f"measured (amber)."), unsafe_allow_html=True)
    expand = st.toggle("Show inside the oracle", value=False,
                       help="Each rule gets a helper qubit. CX gates compare two cells, a multi-controlled X "
                            "raises the helper when the rule is broken, a Z flips the sign when no helper is "
                            "raised, and the checks are undone.")
    png = base64.b64encode(circuit_png(grid, expand)).decode()
    st.markdown(f'<div class="circuit{" wide" if expand else ""}"><img alt="Grover circuit diagram" '
                f'src="data:image/png;base64,{png}"></div>',
                unsafe_allow_html=True)
    s = cost_at(grid, max(k, 1))
    st.markdown(
        '<div class="panel inline-stats" style="margin-top:1rem"><h3>What it would cost on real hardware</h3>'
        + R.stats([("Qubits", f"{sp.n_qubits}"),
                   ("Two-qubit gates", f"{s.total_cx:,}"),
                   ("Circuit depth", f"{s.total_depth:,}")])
        + '<p class="muted" style="margin:.8rem 0 0">Qiskit breaks the big multi-controlled gates into H, T '
          'and CX gates. Today\'s hardware makes roughly one error per thousand two-qubit gates, so a circuit '
          'this deep would return noise. That is why this app simulates it.</p></div>',
        unsafe_allow_html=True)


def levels_section() -> None:
    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    head, opts = st.columns([5, 1.3], vertical_alignment="bottom")
    head.markdown(R.section_head(
        "Across all three levels",
        "A batch of random puzzles at each level, solved both ways. Backtracking stays small because it uses "
        "the rules. Grover grows with the square root of the number of fillings, far below blind guessing."),
        unsafe_allow_html=True)
    n = opts.select_slider("Puzzles per level", [3, 5, 10], value=5)
    with st.spinner("Solving puzzles…"):
        st.markdown(R.bench_table(bench_rows(n)), unsafe_allow_html=True)


# ================================================================== main =====
def main() -> None:
    init_state()
    st.markdown(CSS, unsafe_allow_html=True)
    play_area()
    solve_section()
    circuit_section()
    levels_section()


main()
