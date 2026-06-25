"""Engine math unit tests — values computed by hand to guard the port.
Difficulty scale is -2..2 (matches the authoring template); mult(0)=1.0 == old mult(3)."""
from datetime import timedelta

from app.services import engine
from app.services.engine import Attempt, now_utc


def _att(c, d, ts):
    return Attempt(correct=c, difficulty=d, ts=ts)


def test_difficulty_multiplier():
    assert abs(engine.difficulty_multiplier(0) - 1.0) < 1e-9
    assert abs(engine.difficulty_multiplier(-2) - 0.61920) < 1e-4
    assert abs(engine.difficulty_multiplier(2) - 1.38080) < 1e-4


def test_performance_ewma():
    assert engine.performance([]) == 0.5
    assert abs(engine.performance([0]) - 0.4) < 1e-9
    assert abs(engine.performance([1, 1, 1]) - 0.744) < 1e-9  # 0.5->0.6->0.68->0.744


def test_blended_mastery_single_fresh_correct():
    now = now_utc()
    mastery, p, d, m = engine.blended_mastery([_att(1, 0, now)], now)
    assert abs(p - 0.6) < 1e-9 and abs(d - 1.0) < 1e-9 and abs(m - 1.0) < 1e-9
    assert abs(mastery - 0.84) < 1e-9          # 0.4*0.6 + 0.3*1 + 0.3*1
    assert engine.is_mastered(mastery)


def test_maple_edge_clamps():
    now = now_utc()
    assert abs(engine.maple_edge([_att(1, 0, now)] * 10) - 2.0) < 1e-9    # clamps at +2
    assert abs(engine.maple_edge([_att(0, 0, now)] * 10) - (-2.0)) < 1e-9  # clamps at -2
    # correct then wrong returns to start (0.0)
    assert abs(engine.maple_edge([_att(1, 0, now), _att(0, 0, now)]) - 0.0) < 1e-9


def test_learning_progress():
    now = now_utc()
    assert engine.learning_progress([_att(1, 0, now)]) == 0.2          # <2 attempts
    assert abs(engine.learning_progress([_att(1, 0, now), _att(1, 0, now)]) - 0.75) < 1e-9


def test_memory_decays_over_time():
    now = now_utc()
    fresh = engine.memory_strength([_att(1, 0, now)], now)
    old = engine.memory_strength([_att(1, 0, now - timedelta(days=30))], now)
    assert fresh > old                      # a 30-day-old trace is weaker than a fresh one
    assert abs(fresh - 1.0) < 1e-9


def test_problem_weight_peaks_at_edge():
    assert engine.problem_weight(0, 0.0) == 1.0
    assert engine.problem_weight(-2, 0.0) < engine.problem_weight(-1, 0.0) < 1.0
