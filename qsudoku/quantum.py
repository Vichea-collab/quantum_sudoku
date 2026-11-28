"""Grover search for 4x4 Sudoku.

Encoding   : every blank cell holds a digit 1..4 in 2 qubits  (1->00, 2->01, 3->10, 4->11).
             Blank i uses qubits 2i (low bit) and 2i+1 (high bit).
Oracle     : one helper ("flag") qubit per rule.
               * pair rule  : two blanks that share a row/column/box must differ
               * clue rule  : a blank must not equal a clue that sees it
             Each flag is raised when its rule is BROKEN. The oracle flips the sign of the
             state when every flag is 0, then uncomputes the flags.
Diffuser   : the standard inversion about the mean.
Backends   : "aer"  - real gate-level circuit on Qiskit Aer (all qubits simulated)
             "fast" - NumPy simulation of the search register only. Same maths
                      (oracle = sign flip, diffuser = inversion about the mean), used when
                      the full circuit is too big for a laptop. Tests check both agree.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile

from .classical import propagate
from .runtime import run_big_stack
from .puzzle import PEERS, Grid, blanks, is_solved

AER_QUBIT_LIMIT = 18       # gate-level Aer: seconds at <=18 qubits (Easy); Medium (23+) took ~45 min
AER_MAX_QUBITS = 30         # 2^30 amplitudes = 16 GB: beyond this Aer cannot hold the state
MAX_BLANKS = 10            # 4^10 = ~1M fillings: the most the NumPy simulator will search


# ------------------------------------------------------------- the problem ----
@dataclass(frozen=True)
class SearchProblem:
    """The part of the puzzle Grover has to search."""
    grid: Grid                                   # clues (+ any cells fixed by pruning)
    blanks: tuple[int, ...]                      # cell index of each search cell
    pairs: tuple[tuple[int, int], ...]           # (a, b): blanks a and b must differ
    forbidden: tuple[frozenset[int], ...]        # digits ruled out per blank by clues
    prefilled: int = 0                           # cells fixed by classical pruning

    @classmethod
    def from_grid(cls, grid: Grid, prune: bool = False) -> "SearchProblem":
        prefilled = 0
        if prune:
            g2, moves = propagate(grid)
            if g2 is not None:
                grid, prefilled = g2, len(moves)
        bl = tuple(blanks(grid))
        if len(bl) > MAX_BLANKS:
            raise ValueError(f"{len(bl)} blanks is too many to simulate (max {MAX_BLANKS}).")
        pos = {cell: k for k, cell in enumerate(bl)}
        pairs = tuple(
            (a, b) for a in range(len(bl)) for b in range(a + 1, len(bl))
            if bl[b] in PEERS[bl[a]]
        )
        forb = tuple(
            frozenset(grid[j] for j in PEERS[cell] if grid[j]) for cell in bl
        )
        return cls(grid, bl, pairs, forb, prefilled)

    # -- sizes
    @property
    def n_blanks(self) -> int:
        return len(self.blanks)

    @property
    def n_search(self) -> int:
        return 2 * self.n_blanks

    @property
    def clue_rules(self) -> tuple[int, ...]:
        """Blank indices that have at least one clue rule (one flag each)."""
        return tuple(i for i, f in enumerate(self.forbidden) if f)

    @property
    def n_flags(self) -> int:
        return len(self.pairs) + len(self.clue_rules)

    @property
    def n_qubits(self) -> int:
        return self.n_search + self.n_flags

    @property
    def space(self) -> int:
        return 4 ** self.n_blanks

    # -- classical view of the same rules (used for the fast backend and checking)
    def values(self, x: int) -> tuple[int, ...]:
        return tuple(((x >> (2 * i)) & 3) + 1 for i in range(self.n_blanks))

    def to_grid(self, x: int) -> Grid:
        g = list(self.grid)
        for cell, v in zip(self.blanks, self.values(x)):
            g[cell] = v
        return tuple(g)

    @lru_cache(maxsize=None)
    def solution_mask(self) -> np.ndarray:
        """Boolean array over all 4^k assignments: True where every rule holds."""
        xs = np.arange(self.space, dtype=np.int64)
        vals = [((xs >> (2 * i)) & 3) + 1 for i in range(self.n_blanks)]
        ok = np.ones(self.space, dtype=bool)
        for a, b in self.pairs:
            ok &= vals[a] != vals[b]
        for i, f in enumerate(self.forbidden):
            for d in f:
                ok &= vals[i] != d
        return ok

    @property
    def n_solutions(self) -> int:
        return int(self.solution_mask().sum())

    def solution_indices(self) -> list[int]:
        return [int(x) for x in np.flatnonzero(self.solution_mask())]

    def __hash__(self) -> int:           # frozen dataclass with ndarray cache -> keep hashable
        return hash((self.grid, self.blanks, self.pairs, self.forbidden))


# ------------------------------------------------------------ Grover maths ----
def optimal_iterations(space: int, n_solutions: int = 1) -> int:
    """Number of Grover rounds k that maximises the success probability."""
    if n_solutions <= 0 or n_solutions >= space:
        return 0
    theta = math.asin(math.sqrt(n_solutions / space))
    return max(0, round(math.pi / (4 * theta) - 0.5))


def theoretical_success(space: int, n_solutions: int, k: int) -> float:
    if n_solutions <= 0:
        return 0.0
    theta = math.asin(math.sqrt(min(1.0, n_solutions / space)))
    return math.sin((2 * k + 1) * theta) ** 2


def success_curve(space: int, n_solutions: int, k_max: int) -> np.ndarray:
    ks = np.arange(k_max + 1)
    theta = math.asin(math.sqrt(n_solutions / space))
    return np.sin((2 * ks + 1) * theta) ** 2


# ---------------------------------------------------------------- circuits ----
def _mcz(qc: QuantumCircuit, qubits: list) -> None:
    """Multi-controlled Z on `qubits` (phase flip when all are 1)."""
    if not qubits:
        return
    if len(qubits) == 1:
        qc.z(qubits[0])
    elif len(qubits) == 2:
        qc.cz(*qubits)
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])


def _registers(problem: SearchProblem):
    s = QuantumRegister(problem.n_search, "s")
    regs = [s]
    f = None
    if problem.n_flags:
        f = QuantumRegister(problem.n_flags, "f")
        regs.append(f)
    return s, f, regs


def build_oracle(problem: SearchProblem) -> QuantumCircuit:
    """Phase oracle: -1 on assignments that satisfy every rule (flags are uncomputed)."""
    s, f, regs = _registers(problem)
    qc = QuantumCircuit(*regs, name="Oracle")
    if f is None:                         # no rules -> everything is a solution
        return qc

    def compute() -> None:
        k = 0
        for a, b in problem.pairs:        # rule: blank a != blank b
            qa0, qa1, qb0, qb1 = s[2 * a], s[2 * a + 1], s[2 * b], s[2 * b + 1]
            qc.cx(qa0, qb0)               # b0 <- a0 xor b0   (0 means equal bit)
            qc.cx(qa1, qb1)
            qc.x([qb0, qb1])
            qc.mcx([qb0, qb1], f[k])      # flag=1  <=>  both bits equal  <=>  rule broken
            qc.x([qb0, qb1])
            qc.cx(qa1, qb1)
            qc.cx(qa0, qb0)
            k += 1
        for i in problem.clue_rules:      # rule: blank i != each forbidden clue digit
            q0, q1 = s[2 * i], s[2 * i + 1]
            for d in sorted(problem.forbidden[i]):
                bits = d - 1
                zeros = [q for q, bit in ((q0, bits & 1), (q1, bits >> 1)) if bit == 0]
                if zeros:
                    qc.x(zeros)
                qc.mcx([q0, q1], f[k])    # digits are mutually exclusive, so XOR == OR
                if zeros:
                    qc.x(zeros)
            k += 1

    compute()
    qc.x(list(f))                         # flag 0 (rule kept)  ->  1
    _mcz(qc, list(f))                     # -1 if every rule is kept
    qc.x(list(f))
    # uncompute: the compute block is self-inverse gate by gate, so reverse it
    undo = QuantumCircuit(*regs)
    undo_start = len(qc.data)
    compute()
    block = qc.data[undo_start:]
    del qc.data[undo_start:]
    for inst in reversed(block):
        qc.append(inst.operation, inst.qubits, inst.clbits)
    return qc


def build_diffuser(n_search: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_search, name="Diffuser")
    qc.h(range(n_search))
    qc.x(range(n_search))
    _mcz(qc, list(range(n_search)))
    qc.x(range(n_search))
    qc.h(range(n_search))
    return qc


def build_grover_circuit(problem: SearchProblem, iterations: int, measure: bool = True,
                         boxed: bool = False) -> QuantumCircuit:
    """Full Grover circuit.  boxed=True hides Oracle/Diffuser inside labelled boxes."""
    s, f, regs = _registers(problem)
    creg = ClassicalRegister(problem.n_search, "c") if measure else None
    qc = QuantumCircuit(*regs, *([creg] if creg else []), name="Grover")
    qc.h(s)
    oracle, diff = build_oracle(problem), build_diffuser(problem.n_search)
    all_q = list(s) + (list(f) if f is not None else [])
    for _ in range(iterations):
        if boxed:
            qc.append(oracle.to_gate(label="Oracle"), all_q)
            qc.append(diff.to_gate(label="Diffuser"), list(s))
        else:
            qc.compose(oracle, all_q, inplace=True)
            qc.compose(diff, list(s), inplace=True)
    if measure:
        qc.measure(s, creg)
    return qc


# ------------------------------------------------------------- statistics ----
@dataclass
class CircuitStats:
    qubits: int
    search_qubits: int
    flag_qubits: int
    iterations: int
    oracle_gates: dict
    depth_per_round: int
    total_depth: int
    total_cx: int


def circuit_stats(problem: SearchProblem, iterations: int) -> CircuitStats:
    """Cost of the circuit after Qiskit breaks big gates into basis gates (h, t, cx, ...)."""
    one = build_grover_circuit(problem, 1, measure=False)
    t = transpile(one, basis_gates=["h", "t", "tdg", "s", "sdg", "x", "z", "cx"],
                  optimization_level=1, seed_transpiler=1)
    ops = dict(t.count_ops())
    d = t.depth()
    return CircuitStats(problem.n_qubits, problem.n_search, problem.n_flags, iterations,
                        ops, d, d * max(iterations, 1), ops.get("cx", 0) * iterations)


# --------------------------------------------------------------- simulators ----
def simulate_fast(problem: SearchProblem, iterations: int) -> np.ndarray:
    """Exact probabilities over the search register (NumPy)."""
    mask = problem.solution_mask()
    amp = np.full(problem.space, 1 / math.sqrt(problem.space))
    for _ in range(iterations):
        amp = np.where(mask, -amp, amp)          # oracle: flip the sign of solutions
        amp = 2 * amp.mean() - amp               # diffuser: inversion about the mean
    return amp ** 2


def digit_marginals(problem: SearchProblem, iterations: int) -> np.ndarray:
    """P(blank i holds digit d) after `iterations` rounds, shape (n_blanks, 4). Exact."""
    probs = simulate_fast(problem, iterations)
    xs = np.arange(problem.space, dtype=np.int64)
    return np.array([np.bincount((xs >> (2 * i)) & 3, weights=probs, minlength=4)
                     for i in range(problem.n_blanks)])


def digit_counts(problem: SearchProblem) -> tuple[np.ndarray, np.ndarray]:
    """For each blank and digit: how many valid fillings / invalid fillings put that digit there.

    Grover's state only ever has two amplitudes (one shared by every valid filling, one by every
    invalid one), so these counts give the per-cell probabilities for ANY number of rounds in
    closed form. The app's board uses this to scrub through rounds instantly.
    """
    xs = np.arange(problem.space, dtype=np.int64)
    mask = problem.solution_mask()
    sol = np.zeros((problem.n_blanks, 4), dtype=np.int64)
    allc = np.zeros((problem.n_blanks, 4), dtype=np.int64)
    for i in range(problem.n_blanks):
        vals = (xs >> (2 * i)) & 3
        sol[i] = np.bincount(vals[mask], minlength=4)
        allc[i] = np.bincount(vals, minlength=4)
    return sol, allc - sol


def closed_form_marginals(problem: SearchProblem, iterations: int) -> np.ndarray:
    """Same result as digit_marginals, from the two-amplitude formula (no simulation)."""
    M, N = problem.n_solutions, problem.space
    sol, non = digit_counts(problem)
    if M == 0:
        return non / max(N, 1)
    p_sol = theoretical_success(N, M, iterations)
    out = p_sol * sol / M
    if N > M:
        out = out + (1 - p_sol) * non / (N - M)
    return out


def statevector_probabilities(problem: SearchProblem, iterations: int) -> np.ndarray:
    """Exact probabilities over the search register from the real circuit (Qiskit)."""
    from qiskit.quantum_info import Statevector
    qc = build_grover_circuit(problem, iterations, measure=False)
    probs = Statevector(qc).probabilities(list(range(problem.n_search)))
    return np.asarray(probs)


def _run_aer_inner(problem: SearchProblem, iterations: int, shots: int, seed: int | None) -> dict[int, int]:
    from qiskit_aer import AerSimulator
    qc = build_grover_circuit(problem, iterations, measure=True)
    sim = AerSimulator(method="statevector", max_parallel_threads=0)
    tqc = transpile(qc, sim, optimization_level=0)
    res = sim.run(tqc, shots=shots, seed_simulator=seed).result()
    return {int(k, 2): v for k, v in res.get_counts().items()}


def _run_aer(problem: SearchProblem, iterations: int, shots: int, seed: int | None) -> dict[int, int]:
    return run_big_stack(_run_aer_inner, problem, iterations, shots, seed)


# ------------------------------------------------------------------ result ----
@dataclass
class GroverResult:
    problem: SearchProblem
    solved: bool
    solution: Grid | None
    answer_index: int | None
    counts: dict[int, int]
    shots: int
    iterations: int
    oracle_queries: int
    success_probability: float          # measured: share of shots that hit a solution
    exact_probability: float            # exact p(success) from the simulated state
    backend: str                        # "aer" | "fast" | "trivial"
    elapsed: float
    n_solutions: int
    notes: list[str] = field(default_factory=list)

    @property
    def top_outcomes(self) -> list[tuple[int, int]]:
        return sorted(self.counts.items(), key=lambda kv: -kv[1])


def run_grover(grid: Grid, *, prune: bool = False, shots: int = 1024, backend: str = "auto",
               seed: int | None = None, iterations: int | None = None,
               aer_limit: int = AER_QUBIT_LIMIT) -> GroverResult:
    """Solve `grid` with Grover search.

    backend: "auto" (Aer if the circuit fits `aer_limit` qubits, else fast), "aer" or "fast".
    iterations: override the optimal number of Grover rounds (for the over-rotation demo).
    """
    t0 = time.perf_counter()
    problem = SearchProblem.from_grid(grid, prune=prune)
    notes: list[str] = []
    M = problem.n_solutions

    if problem.n_blanks == 0:
        ok = is_solved(problem.grid)
        return GroverResult(problem, ok, problem.grid if ok else None, 0, {0: shots}, shots, 0, 0,
                            1.0 if ok else 0.0, 1.0 if ok else 0.0, "trivial",
                            time.perf_counter() - t0, 1 if ok else 0,
                            ["Classical pruning already decided every cell: no quantum search needed."])
    if M == 0:
        notes.append("This puzzle has no solution.")

    k = optimal_iterations(problem.space, M) if iterations is None else iterations
    if M > 1:
        notes.append(f"{M} solutions exist; the number of rounds was tuned for M={M} "
                     "(assumes M is known, e.g. from quantum counting).")

    if backend == "auto":
        backend = "aer" if problem.n_qubits <= aer_limit else "fast"
        if backend == "fast":
            notes.append(f"Circuit needs {problem.n_qubits} qubits (> {aer_limit}): used the fast "
                         "amplitude simulator (same maths as the circuit).")
    if backend == "aer" and problem.n_qubits > AER_MAX_QUBITS:
        raise ValueError(f"The circuit needs {problem.n_qubits} qubits; a laptop state-vector simulator "
                         f"holds about {AER_MAX_QUBITS}. Use backend='fast' instead.")
    if backend == "aer":
        counts = _run_aer(problem, k, shots, seed)
        exact = float(simulate_fast(problem, k)[problem.solution_mask()].sum())
    else:
        probs = simulate_fast(problem, k)
        rng = np.random.default_rng(seed)
        draw = rng.multinomial(shots, probs / probs.sum())
        counts = {int(i): int(c) for i, c in enumerate(draw) if c}
        exact = float(probs[problem.solution_mask()].sum())

    mask = problem.solution_mask()
    hits = sum(c for x, c in counts.items() if mask[x])
    best = max(counts, key=counts.get)
    sol = problem.to_grid(best)
    return GroverResult(problem, bool(mask[best]) and is_solved(sol), sol if mask[best] else None, best,
                        counts, shots, k, k, hits / shots, exact, backend,
                        time.perf_counter() - t0, M, notes)
