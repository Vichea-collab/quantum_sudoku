import numpy as np
import pytest

from qsudoku import puzzle as p
from qsudoku.quantum import (SearchProblem, optimal_iterations, run_grover, simulate_fast,
                             statevector_probabilities, theoretical_success, build_oracle)


def test_iteration_counts_match_proposal_table():
    assert optimal_iterations(4 ** 4) == 12      # Easy   256
    assert optimal_iterations(4 ** 6) == 50      # Medium 4096
    assert optimal_iterations(4 ** 8) == 201     # Hard   65536


def test_problem_sizes_follow_levels():
    for level, n in p.LEVELS.items():
        q = p.PRESETS[level][0]
        sp = SearchProblem.from_grid(q.grid)
        assert sp.n_search == 2 * n and sp.space == 4 ** n
        assert sp.n_solutions == 1
        assert sp.to_grid(sp.solution_indices()[0]) == q.solution


def test_pruning_shrinks_search():
    q = p.PRESETS["Hard"][0]
    assert SearchProblem.from_grid(q.grid, prune=True).n_blanks == 0


@pytest.mark.parametrize("name", ["Easy 1", "Easy 3"])
def test_circuit_matches_fast_simulator_exactly(name):
    q = next(x for x in p.PRESETS["Easy"] if x.name == name)
    sp = SearchProblem.from_grid(q.grid)
    for k in (0, 1, 3, 12):
        assert np.allclose(statevector_probabilities(sp, k), simulate_fast(sp, k), atol=1e-9)


def test_oracle_marks_only_solutions_and_restores_flags():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    q = p.PRESETS["Easy"][1]
    sp = SearchProblem.from_grid(q.grid)
    oracle = build_oracle(sp)
    mask = sp.solution_mask()
    rng = np.random.default_rng(0)
    xs = set(sp.solution_indices()) | set(int(v) for v in rng.integers(0, sp.space, 40))
    for x in xs:                              # basis state |x>|flags=0>
        prep = QuantumCircuit(sp.n_qubits)
        for i in range(sp.n_search):
            if (x >> i) & 1:
                prep.x(i)
        sv0 = Statevector(prep)
        sv1 = sv0.evolve(oracle)
        overlap = np.vdot(sv0.data, sv1.data)  # +1 unchanged, -1 marked, 0 means flags left dirty
        assert np.isclose(overlap, -1.0 if mask[x] else 1.0), (x, overlap)


def test_fast_matches_theory():
    sp = SearchProblem.from_grid(p.PRESETS["Hard"][0].grid)
    for k in (1, 50, 201, 300):
        prob = simulate_fast(sp, k)[sp.solution_mask()].sum()
        assert np.isclose(prob, theoretical_success(sp.space, 1, k))


@pytest.mark.parametrize("level", list(p.LEVELS))
def test_grover_finds_solution(level):
    q = p.PRESETS[level][0]
    r = run_grover(q.grid, shots=1024, seed=7, backend="fast")
    assert r.solved and r.solution == q.solution
    assert r.exact_probability > 0.99


def test_grover_on_aer_easy():
    q = p.PRESETS["Easy"][0]
    r = run_grover(q.grid, shots=512, seed=3, backend="aer")
    assert r.backend == "aer" and r.solved and r.solution == q.solution
    assert r.success_probability > 0.9


def test_over_rotation_hurts():
    q = p.PRESETS["Easy"][0]
    r = run_grover(q.grid, backend="fast", iterations=25, seed=1)
    assert r.exact_probability < 0.5


def test_trivial_when_pruning_solves_everything():
    q = p.PRESETS["Easy"][0]
    r = run_grover(q.grid, prune=True)
    assert r.backend == "trivial" and r.solved and r.solution == q.solution


def test_multiple_solutions_are_counted_and_tuned():
    g = p.parse("1.../..../..../...2")      # 14 blanks -> too big, must be rejected
    with pytest.raises(ValueError):
        SearchProblem.from_grid(g)
    g = p.parse("1234/3412/..../....")      # open puzzle with 4 valid fillings
    sols = p.all_solutions(g)
    r = run_grover(g, backend="fast", seed=2)
    assert r.n_solutions == len(sols) == 4
    assert r.solved and r.solution in sols


def test_unsolvable_puzzle_reports_no_solution():
    g = p.parse("1432/2341/4.1./3214")      # column needs a 1, but the row already has one
    r = run_grover(g, backend="fast", seed=2)
    assert r.n_solutions == 0 and not r.solved


def test_digit_marginals_start_uniform_and_converge():
    from qsudoku.quantum import digit_marginals
    q = p.PRESETS["Easy"][0]
    sp = SearchProblem.from_grid(q.grid)
    m0 = digit_marginals(sp, 0)
    assert m0.shape == (4, 4) and np.allclose(m0, 0.25)
    m = digit_marginals(sp, optimal_iterations(sp.space))
    for k, cell in enumerate(sp.blanks):
        assert m[k, q.solution[cell] - 1] > 0.99
    assert np.allclose(m.sum(axis=1), 1.0)


def test_aer_refuses_circuits_too_big_for_memory():
    q = p.PRESETS["Hard"][0]                     # 39 qubits
    with pytest.raises(ValueError, match="fast"):
        run_grover(q.grid, backend="aer")


@pytest.mark.parametrize("level", list(p.LEVELS))
def test_closed_form_marginals_match_simulation(level):
    from qsudoku.quantum import closed_form_marginals, digit_marginals
    sp = SearchProblem.from_grid(p.PRESETS[level][1].grid)
    for k in (0, 1, 5, optimal_iterations(sp.space), 2 * optimal_iterations(sp.space)):
        assert np.allclose(closed_form_marginals(sp, k), digit_marginals(sp, k), atol=1e-9)


def test_closed_form_handles_several_solutions():
    from qsudoku.quantum import closed_form_marginals, digit_marginals
    sp = SearchProblem.from_grid(p.parse("1234/3412/..../...."))     # 8 blanks, 4 solutions
    assert sp.n_solutions == 4
    for k in (0, 1, 2):
        assert np.allclose(closed_form_marginals(sp, k), digit_marginals(sp, k), atol=1e-9)
