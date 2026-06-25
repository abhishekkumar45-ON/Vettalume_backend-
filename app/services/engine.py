"""The Learning engine — pure functions, no DB.

Ported verbatim from the dashboard's adaptive-engine reference so behaviour matches the
prototype. Equation numbers refer to the Adaptive Engine doc.

Runs entirely on observed correctness + expert difficulty d in [-2, 2]. No IRT, no calibration.
This is the LEARNING estimator (blended 0-1 mastery), distinct from the mocks' IRT theta.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

# ---- tunable constants (match the dashboard) ----
H = 0.74                       # mastery threshold ("mastered")
EWMA_BETA = 0.8                # eq8a smoothing
EWMA_P0 = 0.5                  # eq8a prior
LP_WINDOW = 4                  # eq1 learning-progress window
MAPLE_START = 0.0              # MAPLE edge starts mid-difficulty (scale is -2..2)
MAPLE_STEP = 0.4               # +/- per correct/incorrect
MAPLE_MIN, MAPLE_MAX = -2.0, 2.0
MEM_TAU_DAYS = 10.0            # eq9/10 memory decay scale (per review slot)
REVIEW_THRESHOLD = 0.60        # eq12 retention below which a mastered concept is due for review
BLEND_P, BLEND_D, BLEND_M = 0.40, 0.30, 0.30   # eq8 weights
PROBLEM_SIGMA = 1.0            # eq4 Gaussian difficulty prior width
MOMENTUM = 1.15               # bandit boost for already-started topics
# Minimum attempts before a concept can be called "mastered", regardless of the blended score.
# 1 = faithful to the prototype (one fresh correct can clear H). Bump to 3 to require more evidence.
MASTERY_MIN_ATTEMPTS = 3       # work up the difficulty ladder, not one lucky correct


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def difficulty_multiplier(d: float) -> float:
    """eq6/8b: harder items count for more. Scale is -2..2 (matches the authored template and the
    IRT b prior). sigmoid(d)+0.5 -> ~0.62 at d=-2, 1.0 at d=0, ~1.38 at d=2."""
    return sigmoid(d) + 0.5


@dataclass
class Attempt:
    correct: int        # 0 or 1
    difficulty: int     # 1..5
    ts: datetime        # when answered (tz-aware UTC)


def performance(corrects: list[int], beta: float = EWMA_BETA, p0: float = EWMA_P0) -> float:
    """eq8a: exponentially-weighted recent accuracy."""
    p = p0
    for c in corrects:
        p = beta * p + (1 - beta) * c
    return p


def difficulty_score(attempts: list[Attempt]) -> float:
    """eq8b: difficulty-weighted accuracy."""
    num = sum(a.correct * difficulty_multiplier(a.difficulty) for a in attempts)
    den = sum(difficulty_multiplier(a.difficulty) for a in attempts)
    return num / den if den else 0.0


def memory_strength(attempts: list[Attempt], now: datetime) -> float:
    """eq9/10: decaying memory traces of correct answers, with later reviews decaying slower
    (spacing effect). Real-time generalisation of the dashboard's session-based recency."""
    corr = [a for a in attempts if a.correct == 1]
    if not corr:
        return 0.0
    s = 0.0
    for i, a in enumerate(corr):
        elapsed_days = max(0.0, (now - a.ts).total_seconds() / 86400.0)
        s += math.exp(-elapsed_days / (MEM_TAU_DAYS * (i + 1)))
    return s / len(corr)


def blended_mastery(attempts: list[Attempt], now: datetime) -> tuple[float, float, float, float]:
    """eq8: returns (mastery, P, D, M)."""
    if not attempts:
        return 0.0, 0.0, 0.0, 0.0
    p = performance([a.correct for a in attempts])
    d = difficulty_score(attempts)
    m = memory_strength(attempts, now)
    return BLEND_P * p + BLEND_D * d + BLEND_M * m, p, d, m


def learning_progress(attempts: list[Attempt]) -> float:
    """eq1->2: mean learning-progress gradient (recent half minus older half of a sliding window).
    The concept bandit's reward signal."""
    if len(attempts) < 2:
        return 0.2
    cs = [a.correct * difficulty_multiplier(a.difficulty) for a in attempts]
    L, rs = LP_WINDOW, []
    h = L // 2
    for k in range(1, len(cs) + 1):
        seq = cs[:k]
        seq = [0.0] * (L - len(seq)) + seq if len(seq) < L else seq[-L:]
        rs.append(sum(seq[h:]) / h - sum(seq[:h]) / h)
    return sum(rs) / len(rs)


def maple_edge(attempts: list[Attempt], start: float = MAPLE_START) -> float:
    """Replay the MAPLE difficulty ladder: +0.4 on correct, -0.4 on wrong, clamped to [1,5]."""
    edge = start
    for a in attempts:
        edge = min(MAPLE_MAX, edge + MAPLE_STEP) if a.correct else max(MAPLE_MIN, edge - MAPLE_STEP)
    return edge


def problem_weight(item_difficulty: float, edge: float, sigma: float = PROBLEM_SIGMA) -> float:
    """eq4: Gaussian difficulty prior centred on the learner's current edge."""
    return math.exp(-((item_difficulty - edge) ** 2) / (2 * sigma * sigma))


def expected_gain(topic_mastery: float, started: bool) -> float:
    """Topic bandit value: room-to-grow toward H, boosted if the topic is already in progress."""
    return (H - topic_mastery) * (MOMENTUM if started else 1.0)


def is_mastered(mastery: float) -> bool:
    return mastery >= H


def concept_mastered(mastery: float, attempts: int, min_attempts: int = MASTERY_MIN_ATTEMPTS) -> bool:
    """Mastery with the minimum-evidence gate applied. ``min_attempts`` may be lowered per concept so
    a concept with fewer items than the global floor is still masterable (see concept_state)."""
    return mastery >= H and attempts >= min_attempts


def is_due_for_review(mastery: float, m: float) -> bool:
    """eq12: a mastered concept whose memory trace has decayed below threshold."""
    return mastery >= H and m < REVIEW_THRESHOLD


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(dt: datetime) -> datetime:
    """Coerce a possibly-naive timestamp (SQLite) to tz-aware UTC."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
