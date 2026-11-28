"""HTML / SVG fragments for the single-page app. Pure functions that return strings."""
from __future__ import annotations

import html
import math

import numpy as np

from .puzzle import N, Grid, rc

INK, INK2, LINE = "#161B3D", "#4A5275", "#DCE1EC"
QUBIT, QUBIT_SOFT = "#3E5BDB", "#E4EAFE"
AMBER, ORACLE, ERROR = "#E08A2E", "#9B1B4E", "#D93F4C"


def esc(s) -> str:
    return html.escape(str(s))


def _box_classes(i: int) -> list[str]:
    r, c = rc(i)
    out = []
    if c == 1:
        out.append("bx-r")
    if r == 1:
        out.append("bx-b")
    if c == N - 1:
        out.append("last-c")
    if r == N - 1:
        out.append("last-r")
    return out


# ----------------------------------------------------------------- grids ----
def mini_grid(grid: Grid, given: Grid | None = None, accent: str = "qubit") -> str:
    """Small read-only board. Cells that were blank in `given` use the accent colour."""
    cells = []
    for i, v in enumerate(grid):
        cls = ["mg-cell", *_box_classes(i)]
        if v and given is not None and not given[i]:
            cls.append(f"mg-new mg-{accent}")
        elif v:
            cls.append("mg-given")
        cells.append(f'<div class="{" ".join(cls)}">{v or ""}</div>')
    return f'<div class="mg">{"".join(cells)}</div>'


# ----------------------------------------------------------------- charts ----
def solver_bars(rows: list[tuple[str, float, str]]) -> str:
    """[(name, fraction, colour)] -> horizontal bars like the proposal mockup."""
    out = []
    for name, frac, colour in rows:
        out.append(
            f'<div class="hb-row"><span class="hb-name">{esc(name)}</span>'
            f'<span class="hb-track"><span class="hb-fill" style="width:{100 * frac:.1f}%;background:{colour}">'
            f'</span></span><span class="hb-val">{100 * frac:.1f}%</span></div>')
    return f'<div class="hb">{"".join(out)}</div>'


def effort_bars(rows: list[tuple[str, float, str, str]]) -> str:
    """[(name, steps, unit, colour)] on a log scale, so 12 and 32,768 can share one axis."""
    top = math.log10(max(v for _, v, _, _ in rows) + 1)
    out = []
    for name, v, unit, colour in rows:
        w = 100 * math.log10(v + 1) / top if top else 0
        out.append(
            f'<div class="hb-row"><span class="hb-name">{esc(name)}</span>'
            f'<span class="hb-track"><span class="hb-fill" style="width:{max(w, 2):.1f}%;background:{colour}">'
            f'</span></span><span class="hb-val" title="{esc(unit)}">{v:,.0f}</span></div>')
    return f'<div class="hb">{"".join(out)}</div><p class="muted small" style="margin:.5rem 0 0">Log scale.</p>'


# ------------------------------------------------------------------ text ----
def note(kind: str, text: str) -> str:
    return f'<div class="note note-{esc(kind)}" role="status">{text}</div>'


def stats(items: list[tuple[str, str]]) -> str:
    """Compact label/value pairs in a row."""
    cells = "".join(f'<div><span class="k">{esc(k)}</span><span class="v">{esc(v)}</span></div>'
                    for k, v in items)
    return f'<div class="stats">{cells}</div>'


def section_head(title: str, lede: str) -> str:
    return f'<div class="sec"><h2>{esc(title)}</h2><p>{lede}</p></div>'


def bench_table(rows: list[dict]) -> str:
    levels = list(dict.fromkeys(r["level"] for r in rows))
    body = []
    for lv in levels:
        rs = [r for r in rows if r["level"] == lv]
        tries = np.mean([r["classical_tries"] for r in rs])
        q = rs[0]["grover_queries"]
        brute = np.mean([r["brute_force_avg"] for r in rs])
        ok = all(r["grover_correct"] and r["classical_correct"] for r in rs)
        succ = np.mean([r["grover_success"] for r in rs])
        top = math.log10(max(brute, 10))
        bar = lambda v, col: (f'<span class="lb" style="width:{100 * math.log10(max(v, 1)) / top:.0f}%;'
                              f'background:{col}"></span>')
        body.append(
            f'<tr><th>{esc(lv)}</th><td>{rs[0]["fillings"]:,}</td>'
            f'<td><div class="lbw">{bar(tries, QUBIT)}<span>{tries:.1f}</span></div></td>'
            f'<td><div class="lbw">{bar(q, AMBER)}<span>{q}</span></div></td>'
            f'<td><div class="lbw">{bar(brute, "#AEB5CB")}<span>{brute:,.0f}</span></div></td>'
            f'<td>{100 * succ:.1f}%</td><td>{"All correct" if ok else "Mismatch"}</td></tr>')
    return ('<table class="bench"><thead><tr><th>Level</th><th>Fillings</th><th>Backtracking tries</th>'
            '<th>Grover queries</th><th>Blind guessing (avg)</th><th>Grover success</th><th>Answers</th>'
            f'</tr></thead><tbody>{"".join(body)}</tbody></table>'
            '<p class="muted small">Bar lengths use a log scale.</p>')
