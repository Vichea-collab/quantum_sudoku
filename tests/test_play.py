from types import SimpleNamespace

from qsudoku import puzzle as p
from qsudoku.game import Game
from qsudoku.play import UiState, apply_action, board_props


def setup(level="Easy"):
    return Game(p.PRESETS[level][0]), UiState()


def blank(g):
    return next(i for i, v in enumerate(g.values) if v == 0)


def act(g, ui, n, **kw):
    return apply_action(g, ui, {"nonce": n, **kw})


def test_digit_places_correct_value_and_counts_mistakes():
    g, ui = setup()
    i = blank(g)
    wrong = next(d for d in (1, 2, 3, 4) if d != g.puzzle.solution[i])
    assert act(g, ui, 1, type="digit", digit=wrong, sel=i)
    assert g.mistakes == 1 and ui.msg_kind == "error"
    assert act(g, ui, 2, type="digit", digit=g.puzzle.solution[i], sel=i)
    assert g.values[i] == g.puzzle.solution[i] and ui.msg_kind == "ok"


def test_same_nonce_is_ignored():
    g, ui = setup()
    i = blank(g)
    wrong = next(d for d in (1, 2, 3, 4) if d != g.puzzle.solution[i])
    act(g, ui, 7, type="digit", digit=wrong, sel=i)
    assert not act(g, ui, 7, type="digit", digit=wrong, sel=i)
    assert g.mistakes == 1


def test_digit_without_selection_or_on_given_warns():
    g, ui = setup()
    act(g, ui, 1, type="digit", digit=1, sel=None)
    assert ui.msg_kind == "warn"
    given = next(i for i, v in enumerate(g.values) if v)
    act(g, ui, 2, type="digit", digit=1, sel=given)
    assert ui.msg_kind == "warn" and g.mistakes == 0


def test_notes_mode_toggles_pencil_marks():
    g, ui = setup()
    i = blank(g)
    act(g, ui, 1, type="notes", sel=i)
    assert ui.note_mode
    act(g, ui, 2, type="digit", digit=3, sel=i)
    assert 3 in g.notes[i] and g.values[i] == 0


def test_erase_undo_autonotes_hint():
    g, ui = setup("Hard")
    i = blank(g)
    act(g, ui, 1, type="digit", digit=g.puzzle.solution[i], sel=i)
    act(g, ui, 2, type="erase", sel=i)
    assert g.values[i] == 0
    act(g, ui, 3, type="undo", sel=i)
    assert g.values[i] == g.puzzle.solution[i]
    act(g, ui, 4, type="autonotes", sel=None)
    assert any(g.notes)
    token = ui.sel_token
    act(g, ui, 5, type="hint", sel=None)
    assert g.hints_used == 1 and ui.sel_token == token + 1 and g.values[ui.sel]


def test_quantum_hint_uses_solver_result():
    g, ui = setup()
    sol = g.puzzle.solution
    fake = lambda grid: SimpleNamespace(solved=True, solution=sol, exact_probability=1.0,
                                        iterations=12, problem=SimpleNamespace(space=256, n_blanks=4))
    act(g, ui, 1, type="qhint", sel=None)            # no solver given: nothing happens
    assert g.hints_used == 0
    apply_action(g, ui, {"nonce": 2, "type": "qhint", "sel": None}, grover=fake)
    assert g.hints_used == 1 and ui.msg_kind == "quantum"


def test_unknown_or_malformed_actions_are_rejected():
    g, ui = setup()
    assert not apply_action(g, ui, None)
    assert not apply_action(g, ui, {"type": "digit"})          # no nonce
    assert not act(g, ui, 1, type="explode")


def test_board_props_are_json_safe_and_complete():
    import json
    g, ui = setup()
    props = board_props(g, ui)
    json.dumps(props)
    assert len(props["values"]) == 16 and props["status"] == "playing"
    assert props["remaining"]["2"] == 0          # Easy 1 already has every 2
    i = blank(g)
    for d in (1, 2, 3, 4):
        if d != g.puzzle.solution[i]:
            g.place(i, d)
    props = board_props(g, ui)
    assert props["status"] == "lost" and props["msgKind"] == "lost" and not props["running"]


def test_wrong_mark_clears_on_next_action():
    g, ui = setup()
    i = blank(g)
    wrong = next(d for d in (1, 2, 3, 4) if d != g.puzzle.solution[i])
    act(g, ui, 1, type="digit", digit=wrong, sel=i)
    assert board_props(g, ui)["wrong"] == i
    act(g, ui, 2, type="notes", sel=i)
    assert board_props(g, ui)["wrong"] is None


def test_lens_props_follow_the_current_board():
    from qsudoku.play import lens_props
    g, ui = setup()
    lens = lens_props(g)
    assert lens["N"] == 256 and lens["M"] == 1 and lens["kOpt"] == 12 and len(lens["blanks"]) == 4
    i = blank(g)
    g.place(i, g.puzzle.solution[i])
    lens = lens_props(g)
    assert lens["N"] == 64 and i not in lens["blanks"]
    for c in lens["blanks"]:
        g.place(c, g.puzzle.solution[c])
    assert lens_props(g) is None and board_props(g, ui)["lens"] is None


def test_measure_action_stores_result_for_the_board():
    from qsudoku.quantum import run_grover
    g, ui = setup()
    fn = lambda grid, k: run_grover(grid, iterations=k, backend="fast", seed=1)
    apply_action(g, ui, {"nonce": 1, "type": "measure", "k": 12, "sel": None}, measure=fn)
    m = board_props(g, ui)["measure"]
    assert m["k"] == 12 and m["top"][0]["ok"] and m["hit"] > 0.99
    assert m["key"] == "".join(map(str, g.values))


def test_new_game_request_validates_and_dedupes():
    from qsudoku.play import new_game_request
    ui = UiState()
    assert new_game_request(ui, {"type": "new", "level": "Hard", "source": "2", "nonce": 5}) == ("Hard", "2")
    assert new_game_request(ui, {"type": "new", "level": "Hard", "source": "2", "nonce": 5}) is None
    assert new_game_request(ui, {"type": "new", "level": "Huge", "source": "1", "nonce": 6}) is None
    assert new_game_request(ui, {"type": "new", "level": "Easy", "source": "9", "nonce": 7}) is None
    assert new_game_request(ui, {"type": "digit", "nonce": 8}) is None
