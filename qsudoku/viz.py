"""Matplotlib figures shared by the notebook and the Streamlit app."""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .classical import ClassicalResult
from .puzzle import N, Grid, rc
from .quantum import (GroverResult, SearchProblem, build_grover_circuit, build_oracle,
                      success_curve, optimal_iterations)

NAVY, BLUE, ORANGE, MAGENTA, RED, GREY = "#1B2142", "#3E5BDB", "#D9822B", "#9B1B4E", "#F0605D", "#C9CDD6"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=NAVY)
    ax.xaxis.label.set_color(NAVY)
    ax.yaxis.label.set_color(NAVY)
    ax.title.set_color(NAVY)


def draw_grid(grid: Grid, given: Grid | None = None, title: str = "", ax=None, figsize=(2.6, 2.6)):
    """Draw a 4x4 board. Cells that differ from `given` are drawn in blue."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    for i, v in enumerate(grid):
        r, c = rc(i)
        if v:
            clue = given is None or given[i] == v
            ax.text(c + .5, N - r - .5, str(v), ha="center", va="center", fontsize=18,
                    fontweight="bold", color=NAVY if clue else BLUE)
    for k in range(N + 1):
        w = 2.5 if k % 2 == 0 else 0.8
        ax.plot([0, N], [k, k], color=NAVY, lw=w)
        ax.plot([k, k], [0, N], color=NAVY, lw=w)
    ax.set_xlim(-.05, N + .05); ax.set_ylim(-.05, N + .05)
    ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title, color=NAVY, fontsize=9)
    return fig


CIRCUIT_STYLE = {
    "backgroundcolor": "#FFFFFF",
    "linecolor": "#9AA3BF",
    "textcolor": "#161B3D",
    "gatetextcolor": "#FFFFFF",
    "subtextcolor": "#FFFFFF",
    "fontsize": 13,
    "displaycolor": {
        "h": ["#3E5BDB", "#FFFFFF"],
        "x": ["#3E5BDB", "#FFFFFF"],
        "z": ["#3E5BDB", "#FFFFFF"],
        "cx": ["#161B3D", "#FFFFFF"],
        "cz": ["#161B3D", "#FFFFFF"],
        "mcx": ["#161B3D", "#FFFFFF"],
        "ccx": ["#161B3D", "#FFFFFF"],
        "measure": ["#E08A2E", "#FFFFFF"],
        "Oracle": ["#9B1B4E", "#FFFFFF"],
        "Diffuser": ["#2C3A8C", "#FFFFFF"],
    },
}


def plot_circuit(problem: SearchProblem, expand_oracle: bool = False):
    """High-level Grover circuit (H | Oracle | Diffuser | measure) or the oracle's insides."""
    if expand_oracle:
        return build_oracle(problem).draw("mpl", fold=40, style=CIRCUIT_STYLE)
    qc = build_grover_circuit(problem, 1, measure=True, boxed=True)
    return qc.draw("mpl", fold=-1, style=CIRCUIT_STYLE)


def outcome_label(problem: SearchProblem, x: int) -> str:
    return "".join(map(str, problem.values(x)))


def plot_histogram(result: GroverResult, top: int = 8, ax=None, log: bool = False):
    """Measurement histogram. Orange = fillings that satisfy every rule."""
    sp = result.problem
    items = result.top_outcomes[:top]
    mask = sp.solution_mask()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3))
    else:
        fig = ax.figure
    labels = [outcome_label(sp, x) for x, _ in items]
    vals = [c / result.shots for _, c in items]
    cols = [ORANGE if mask[x] else BLUE for x, _ in items]
    ax.bar(range(len(items)), vals, color=cols)
    ax.set_xticks(range(len(items)), labels, fontfamily="monospace")
    ax.set_ylabel("Probability"); ax.set_xlabel("Measured filling of the blanks (digits in reading order)")
    ax.set_title(f"Measurement histogram ({result.shots} shots, top {len(items)})")
    if log:
        ax.set_yscale("log"); ax.set_ylim(0.5 / result.shots, 1.5)
    else:
        ax.set_ylim(0, 1.05)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_success_curve(problem: SearchProblem, k_used: int | None = None, ax=None):
    """P(success) after k Grover rounds. Too many rounds rotates past the answer."""
    M = max(problem.n_solutions, 1)
    k_opt = optimal_iterations(problem.space, M)
    k_max = max(2 * k_opt + 2, 6)
    curve = success_curve(problem.space, M, k_max)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3))
    else:
        fig = ax.figure
    ax.plot(range(k_max + 1), curve, color=BLUE, lw=2)
    ax.scatter([k_opt], [curve[k_opt]], color=ORANGE, zorder=3, s=60, label=f"best: k = {k_opt}")
    if k_used is not None and k_used <= k_max and k_used != k_opt:
        ax.scatter([k_used], [curve[k_used]], color=RED, zorder=3, s=60, label=f"used: k = {k_used}")
    ax.set_xlabel("Grover rounds (oracle + diffuser)"); ax.set_ylabel("P(measure a solution)")
    ax.set_title("Success probability vs. number of rounds")
    ax.set_ylim(0, 1.05); ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_solver_bars(classical: ClassicalResult | None, grover: GroverResult | None, ax=None):
    """Success probability of each solver (the 'Solve and Compare' card)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 1.8))
    else:
        fig = ax.figure
    names, vals, cols = [], [], []
    if classical is not None:
        names.append("Classical"); vals.append(1.0 if classical.solved else 0.0); cols.append(BLUE)
    if grover is not None:
        names.append("Grover"); vals.append(grover.success_probability); cols.append(ORANGE)
    ax.barh(names[::-1], vals[::-1], color=cols[::-1])
    for y, v in enumerate(vals[::-1]):
        ax.text(min(v + .02, .9), y, f"{v:.1%}", va="center", color=NAVY, fontsize=10)
    ax.set_xlim(0, 1.05); ax.set_xlabel("Success probability")
    _style(ax)
    fig.tight_layout()
    return fig


def plot_effort(rows: list[dict]):
    """Benchmark: classical tries vs Grover oracle queries vs brute-force average, per level."""
    levels = list(dict.fromkeys(r["level"] for r in rows))
    def mean(key, lv): return float(np.mean([r[key] for r in rows if r["level"] == lv]))
    x = np.arange(len(levels)); w = 0.27
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(x - w, [mean("classical_tries", l) for l in levels], w, color=BLUE, label="Classical backtracking (tries)")
    ax.bar(x, [mean("grover_queries", l) for l in levels], w, color=ORANGE, label="Grover (oracle queries)")
    ax.bar(x + w, [mean("brute_force_avg", l) for l in levels], w, color=GREY, label="Unstructured brute force (avg)")
    ax.set_yscale("log"); ax.set_xticks(x, levels)
    ax.set_ylabel("Steps (log scale)"); ax.set_title("Search effort by level")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_constraint_graph(problem: SearchProblem, ax=None):
    """Blanks as nodes; an edge means 'these two cells must differ' (one oracle flag each)."""
    k = problem.n_blanks
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
    else:
        fig = ax.figure
    ang = np.linspace(0, 2 * np.pi, k, endpoint=False) + np.pi / 2
    xy = np.c_[np.cos(ang), np.sin(ang)]
    for a, b in problem.pairs:
        ax.plot(*zip(xy[a], xy[b]), color=MAGENTA, lw=1.6, alpha=.7, zorder=1)
    for i, (x, y) in enumerate(xy):
        r, c = rc(problem.blanks[i])
        ax.scatter([x], [y], s=900, color="white", edgecolor=NAVY, lw=2, zorder=2)
        ax.text(x, y, f"r{r+1}c{c+1}", ha="center", va="center", fontsize=8, color=NAVY, zorder=3)
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"{len(problem.pairs)} 'must differ' rules between blanks", color=NAVY, fontsize=10)
    return fig
