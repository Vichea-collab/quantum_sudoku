from qsudoku import puzzle as p
from qsudoku.classical import propagate, solve


def test_presets_unique_and_blank_counts():
    for level, n in p.LEVELS.items():
        for q in p.PRESETS[level]:
            assert q.n_blanks == n
            assert p.count_solutions(q.grid, 2) == 1
            assert p.is_solved(q.solution)


def test_generator_is_unique_and_seeded():
    a = p.new_puzzle("Hard", seed=5)
    b = p.new_puzzle("Hard", seed=5)
    assert a == b
    assert a.n_blanks == 8 and p.count_solutions(a.grid, 2) == 1


def test_backtracking_matches_known_solution_plain_and_pruned():
    for level in p.LEVELS:
        for q in p.PRESETS[level]:
            for pruning in (False, True):
                r = solve(q.grid, pruning=pruning)
                assert r.solved and r.solution == q.solution
                assert pruning or r.tries >= 1   # pruning may decide everything


def test_pruning_never_costs_more_tries_than_plain_on_presets():
    for level in p.LEVELS:
        for q in p.PRESETS[level]:
            assert solve(q.grid, True).tries <= solve(q.grid, False).tries


def test_unsolvable_puzzle_reports_failure():
    bad = p.parse("11../..../..../....")
    r = solve(bad)
    assert not r.solved and r.solution is None


def test_propagate_explains_moves_and_detects_contradiction():
    q = p.PRESETS["Easy"][0]
    g, moves = propagate(q.grid)
    assert g == q.solution and len(moves) == 4
    assert all(reason for _, _, reason in moves)
    assert propagate(p.parse("11../..../..../...."))[0] is None


def test_trace_recording_counts_tries():
    r = solve(p.PRESETS["Medium"][0].grid, record_trace=True)
    assert sum(1 for s in r.trace if s.kind == "try") == r.tries
