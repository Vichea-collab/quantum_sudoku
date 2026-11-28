from qsudoku import puzzle as p
from qsudoku.game import Game


def make(level="Medium", idx=0, clock=None):
    g = Game(p.PRESETS[level][idx])
    if clock is not None:
        g._now = clock
        g._elapsed, g._running_since = 0.0, clock()
    return g


def first_blank(g):
    return next(i for i, v in enumerate(g.values) if v == 0)


def test_correct_entry_and_win():
    g = make("Easy")
    for i in [i for i, v in enumerate(g.values) if v == 0]:
        assert g.place(i, g.puzzle.solution[i]) == "ok"
    assert g.solved and g.status == "won"


def test_wrong_entry_counts_mistake_and_three_loses():
    g = make()
    i = first_blank(g)
    wrong = next(d for d in (1, 2, 3, 4) if d != g.puzzle.solution[i])
    assert g.place(i, wrong) == "wrong" and g.mistakes == 1 and g.values[i] == 0
    g.place(i, wrong); g.place(i, wrong)
    assert g.lost and g.place(i, g.puzzle.solution[i]) == "over"


def test_givens_are_locked():
    g = make()
    given = next(i for i, v in enumerate(g.values) if v)
    assert g.place(given, 1) == "given" and not g.erase(given) and not g.toggle_note(given, 1)


def test_notes_toggle_and_clear_peers_on_place():
    g = make("Easy")
    i = first_blank(g)
    d = g.puzzle.solution[i]
    peer = next(j for j in p.PEERS[i] if g.values[j] == 0)
    g.toggle_note(peer, d)
    assert d in g.notes[peer]
    g.place(i, d)
    assert d not in g.notes[peer]


def test_undo_restores_value_and_notes():
    g = make()
    i = first_blank(g)
    g.toggle_note(i, 2)
    g.place(i, g.puzzle.solution[i])
    assert g.undo() and g.values[i] == 0 and g.notes[i] == {2}
    assert g.undo() and g.notes[i] == set()
    assert not g.undo()


def test_hint_gives_correct_digit_with_reason_and_counts():
    g = make("Hard")
    cell, d, why = g.hint()
    assert d == g.puzzle.solution[cell] and g.values[cell] == d
    assert why and g.hints_used == 1


def test_hint_prefers_selected_cell():
    g = make("Easy")
    target = [i for i, v in enumerate(g.values) if v == 0][-1]
    cell, _, _ = g.hint(prefer=target)
    assert cell == target


def test_auto_notes_contain_solution_digit():
    g = make("Hard")
    g.auto_notes()
    for i, v in enumerate(g.values):
        if v == 0:
            assert g.puzzle.solution[i] in g.notes[i]


def test_timer_runs_pauses_and_stops_when_over():
    t = [0.0]
    g = make(clock=lambda: t[0])
    t[0] = 65
    assert g.clock() == "01:05"
    g.pause(); t[0] = 500
    assert g.clock() == "01:05"
    g.resume(); t[0] = 510
    assert g.clock() == "01:15"
    i = first_blank(g)
    wrong = next(d for d in (1, 2, 3, 4) if d != g.puzzle.solution[i])
    for _ in range(3):
        g.place(i, wrong)
    t[0] = 9999
    assert g.clock() == "01:15"
