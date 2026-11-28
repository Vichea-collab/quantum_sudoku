from pathlib import Path

from streamlit.testing.v1 import AppTest

from qsudoku.game import Game
from qsudoku.play import apply_action
from qsudoku.puzzle import PRESETS

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def fresh():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def page_text(at):
    return " ".join(m.value for m in at.markdown)


def test_single_page_renders_every_section_without_clicks():
    at = fresh()
    text = page_text(at)
    for heading in ("Solve it both ways", "The circuit behind it", "Across all three levels",
                    "Backtracking", "Grover's search", "What it would cost on real hardware"):
        assert heading in text
    assert 'class="bench"' in text                      # benchmark computed automatically
    assert len(at.sidebar) == 0


def test_board_actions_change_the_game():
    at = fresh()
    g = at.session_state.game
    cell = next(i for i, v in enumerate(g.values) if v == 0)
    apply_action(g, at.session_state.ui, {"type": "digit", "digit": g.puzzle.solution[cell],
                                         "sel": cell, "nonce": 1})
    at.run()
    assert at.session_state.game.values[cell] == g.puzzle.solution[cell] and not at.exception


def test_hard_puzzle_runs_on_fast_backend_and_never_offers_aer():
    at = fresh()
    assert any(r.label == "Grover simulator" for r in at.radio)      # offered on Easy
    at.session_state.game = Game(PRESETS["Hard"][0])
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not any(r.label == "Grover simulator" for r in at.radio)
    assert "NumPy (exact)" in page_text(at)


def test_settings_change_reruns_comparison():
    at = fresh()
    at.toggle[0].set_value(True).run()                   # backtracking with pruning
    assert not at.exception
    assert "Digits tried" in page_text(at)
