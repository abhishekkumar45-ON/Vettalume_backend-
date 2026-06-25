"""Simulation harness (Phase 2) — the calibration release gate.

Manufacture a synthetic population whose true (a, b, c) we KNOW, answer the bank through the 3PL, run
the exact production calibration loop, then measure how well it recovered the truth (correlations +
RMSE). This is the gate: we only trust calibration on real data once it provably recovers parameters
on synthetic data, and it reproduces the reference's honest finding — b is recoverable at ~40
responses/item, a needs far more, c more still.
"""
from __future__ import annotations

import math
import random

from . import calibration, irt


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx < 1e-12 or syy < 1e-12:
        return None  # a fixed (un-fitted) parameter has no variance -> correlation undefined
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def _rmse(xs: list[float], ys: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(xs, ys)) / len(xs))


def make_items(n_items: int, rng: random.Random) -> list[dict]:
    """True item parameters: a in [0.6,2.0], b ~ N(0,1) clipped to +/-2.5, c from a 4/5-option or
    type-in format."""
    items = []
    for _ in range(n_items):
        num_opts = rng.choice([4, 4, 5])
        is_type_in = rng.random() < 0.15
        c = 0.0 if is_type_in else round(1.0 / num_opts, 4)
        a = round(rng.uniform(0.6, 2.0), 4)
        b = round(max(-2.5, min(2.5, rng.gauss(0.0, 1.0))), 4)
        items.append({"a": a, "b": b, "c": c, "num_options": num_opts, "is_type_in": is_type_in})
    return items


def simulate_population(n_students: int, items_true: list[dict], rng: random.Random,
                        responses_per_student: int | None = None):
    """Each student gets a true theta ~ N(0,1) and answers items via the 3PL. Returns the matrix in
    calibrate()'s format plus the ground truth."""
    true_theta = [rng.gauss(0.0, 1.0) for _ in range(n_students)]
    n_items = len(items_true)
    by_learner: dict[int, list[tuple[int, int]]] = {}
    by_item: dict[int, list[tuple[int, int]]] = {}
    fmt: dict[int, tuple[int, bool]] = {ii: (it["num_options"], it["is_type_in"])
                                        for ii, it in enumerate(items_true)}
    for li in range(n_students):
        which = range(n_items) if not responses_per_student else \
            rng.sample(range(n_items), min(responses_per_student, n_items))
        for ii in which:
            it = items_true[ii]
            p = irt.prob_3pl(true_theta[li], it["a"], it["b"], it["c"])
            u = 1 if rng.random() < p else 0
            by_learner.setdefault(li, []).append((ii, u))
            by_item.setdefault(ii, []).append((li, u))
    return by_learner, by_item, fmt, true_theta


def run_recovery(n_students: int = 600, n_items: int = 60, *, responses_per_student: int | None = None,
                 two_pl_at: int = 500, three_pl_at: int = 2000, seed: int = 7) -> dict:
    """Generate -> simulate -> calibrate -> compare. Returns recovery metrics."""
    rng = random.Random(seed)
    items_true = make_items(n_items, rng)
    by_learner, by_item, fmt, _true_theta = simulate_population(
        n_students, items_true, rng, responses_per_student)

    _ab, params, sweeps, converged = calibration.calibrate(
        n_students, by_learner, by_item, fmt, two_pl_at=two_pl_at, three_pl_at=three_pl_at)

    fitted = sorted(params.keys())
    true_b = [items_true[ii]["b"] for ii in fitted]
    est_b = [params[ii]["b"] for ii in fitted]
    true_a = [items_true[ii]["a"] for ii in fitted]
    est_a = [params[ii]["a"] for ii in fitted]
    true_c = [items_true[ii]["c"] for ii in fitted]
    est_c = [params[ii]["c"] for ii in fitted]

    resp_per_item = (sum(len(v) for v in by_item.values()) / max(1, len(by_item)))
    phase = irt.phase_for(int(resp_per_item), two_pl_at=two_pl_at, three_pl_at=three_pl_at)

    return {
        "n_students": n_students, "n_items": n_items,
        "responses": sum(len(v) for v in by_learner.values()),
        "avg_responses_per_item": round(resp_per_item, 1),
        "phase": phase, "sweeps": sweeps, "converged": converged,
        "b_corr": _pearson(true_b, est_b), "b_rmse": round(_rmse(true_b, est_b), 3),
        "a_corr": _pearson(true_a, est_a),
        "c_corr": _pearson(true_c, est_c),
    }


def gate(result: dict, *, b_min: float = 0.7) -> dict:
    """Release gate: calibration passes if it recovered difficulty well enough at this volume."""
    bc = result.get("b_corr")
    passed = bc is not None and bc >= b_min
    return {"passed": passed, "b_corr": bc, "b_min": b_min,
            "verdict": "PASS — difficulty recovered, safe to calibrate real data"
            if passed else "FAIL — difficulty not recovered; do not activate"}
