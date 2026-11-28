"""Run both solvers on the same puzzle and put the numbers side by side."""
from __future__ import annotations

from dataclasses import dataclass

from . import puzzle as pz
from .classical import ClassicalResult, solve
from .quantum import GroverResult, SearchProblem, run_grover


@dataclass
class Comparison:
    grid: pz.Grid
    classical: ClassicalResult
    grover: GroverResult

    @property
    def brute_force_expected(self) -> float:
        """Average guesses an unstructured classical search would need (N/2 for M=1)."""
        sp = self.grover.problem
        return (sp.space + 1) / (sp.n_solutions + 1) if sp.n_solutions else float(sp.space)

    def rows(self) -> list[dict]:
        c, g = self.classical, self.grover
        return [
            {"metric": "Solved", "Classical": c.solved, "Grover": g.solved},
            {"metric": "Success probability", "Classical": 1.0, "Grover": g.success_probability},
            {"metric": "Search effort", "Classical": f"{c.tries} tries",
             "Grover": f"{g.oracle_queries} oracle queries"},
            {"metric": "Search space", "Classical": "pruned by rules",
             "Grover": f"{g.problem.space:,} fillings"},
            {"metric": "Time (s)", "Classical": round(c.elapsed, 5), "Grover": round(g.elapsed, 3)},
        ]


def compare(grid: pz.Grid, *, classical_pruning: bool = False, grover_prune: bool = False,
            shots: int = 1024, backend: str = "auto", seed: int | None = None) -> Comparison:
    return Comparison(grid,
                      solve(grid, pruning=classical_pruning),
                      run_grover(grid, prune=grover_prune, shots=shots, backend=backend, seed=seed))


def benchmark(levels=("Easy", "Medium", "Hard"), n_puzzles: int = 5, seed: int = 0,
              shots: int = 1024, backend: str = "fast") -> list[dict]:
    """Random puzzles per level -> one row per puzzle."""
    rows = []
    for level in levels:
        for k in range(n_puzzles):
            q = pz.new_puzzle(level, seed=seed + 1000 * list(pz.LEVELS).index(level) + k)
            cmp = compare(q.grid, shots=shots, backend=backend, seed=seed + k)
            sp = cmp.grover.problem
            rows.append({
                "level": level, "blanks": q.n_blanks, "search_qubits": sp.n_search,
                "total_qubits": sp.n_qubits, "fillings": sp.space,
                "classical_tries": cmp.classical.tries,
                "grover_queries": cmp.grover.oracle_queries,
                "brute_force_avg": cmp.brute_force_expected,
                "grover_success": cmp.grover.success_probability,
                "grover_exact": cmp.grover.exact_probability,
                "grover_correct": cmp.grover.solved and cmp.grover.solution == q.solution,
                "classical_correct": cmp.classical.solution == q.solution,
            })
    return rows
