"""Diagnosis chassis (Phase 4) — classify every miss by CAUSE, not aggregate accuracy.

The chassis carries the PRD's six causes (four shared + two exam-native), rolls them into the strategy
decomposition (foundations / execution / selection / vocabulary), attaches a cause mixture to each
skill-graph node, and ranks the learner's leaks so the plan engine can target the biggest first.

Two hard chassis rules from the PRD are enforced here:
  1. Diagnose before prescribing — with no practice data the diagnosis is "insufficient" and the plan
     engine refuses to plan blind.
  2. Gap detection in Part 1, not Part 2 — the primary diagnosis reads the PRACTICE loop only; a mock
     is decomposed by the same taxonomy as a separate rendering (mock_decomposition), never folded
     into the primary diagnosis.
"""
from __future__ import annotations

import enum
import uuid
from collections import defaultdict
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models
from . import ability as ability_svc, engine, knowledge_graph as kg

# chassis rule: the primary diagnosis runs on the practice/review loop ONLY
DIAGNOSIS_CONTEXTS = {models.Context.practice.value}

# heuristic thresholds for the cause layer — explicit and tunable
SLOW_MS = 120_000      # > 2 min on a practice item -> time ran away
FAST_MS = 20_000       # < 20 s -> snap answer
EASY_D = 0             # difficulty at/below this is "easy relative to a capable learner"
RECENCY_DECAY = 0.8    # a miss's weight decays by this per newer attempt on the same concept
LEAK_FLOOR = 0.6       # below this recency-weighted score a node is treated as resolved, not a leak


class Cause(str, enum.Enum):
    concept_gap = "concept_gap"          # foundations: the governing rule/principle is missing
    process_error = "process_error"      # execution: rule held, solving procedure broke
    timing_pressure = "timing_pressure"  # execution: accuracy collapses under pace
    careless_slip = "careless_slip"      # execution: capability present, execution lapse
    vocabulary_gap = "vocabulary_gap"    # GRE-native: a missing word
    selection_error = "selection_error"  # CAT-native: attempted what should have been skipped


# strategy decomposition: which bucket each cause rolls into (PRD Section 04)
STRATEGY_BUCKET = {
    Cause.concept_gap: "foundations",
    Cause.process_error: "execution",
    Cause.timing_pressure: "execution",
    Cause.careless_slip: "execution",
    Cause.vocabulary_gap: "vocabulary",
    Cause.selection_error: "selection",
}

CAUSE_LABEL = {
    Cause.concept_gap: "Concept gap — the governing rule is missing",
    Cause.process_error: "Process error — the rule is held but the procedure broke",
    Cause.timing_pressure: "Timing pressure — accuracy collapses under pace",
    Cause.careless_slip: "Careless slip — capability present, execution lapse",
    Cause.vocabulary_gap: "Vocabulary gap — a missing word",
    Cause.selection_error: "Selection error — attempted a question that should have been skipped",
}


def _is_vocabulary_item(item: models.Item, node: models.KnowledgeNode | None,
                        section_name: str | None) -> bool:
    hay = " ".join(x.lower() for x in (getattr(node, "name", "") or "",
                                       section_name or "", item.format or ""))
    return "vocab" in hay


def classify_miss(*, exam: str, mastery: float, prereq_met: bool, hints_used: int,
                  response_time_ms: int | None, difficulty_d: int,
                  is_vocabulary: bool = False, allow_selection: bool = False) -> Cause:
    """Assign ONE cause to a single wrong answer from the available signals. First match wins."""
    held = mastery >= engine.H
    # exam-native causes first
    if exam.upper() == "GRE" and is_vocabulary:
        return Cause.vocabulary_gap
    if (exam.upper() == "CAT" and allow_selection and difficulty_d >= 1
            and response_time_ms is not None and response_time_ms <= FAST_MS):
        # a snap attempt on a hard item under negative marking — should have been skipped
        return Cause.selection_error
    # foundations: principle missing -> needed a hint and still missed, prereq unmet, or low mastery
    if hints_used > 0 or not prereq_met or not held:
        return Cause.concept_gap
    # concept is held -> the miss is an execution failure; sub-classify
    if response_time_ms is not None and response_time_ms >= SLOW_MS:
        return Cause.timing_pressure
    if response_time_ms is not None and response_time_ms <= FAST_MS and difficulty_d <= EASY_D:
        return Cause.careless_slip
    return Cause.process_error


def _difficulty_weight(d: int) -> float:
    # a miss on a harder item weighs a little more (1.0 at d=-2 up to 2.0 at d=+2)
    return 1.0 + 0.25 * (d + 2)


def _practice_responses(db: Session, learner_id: uuid.UUID, exam: str):
    rows = db.execute(
        select(models.Response, models.Item)
        .join(models.Item, models.Item.item_id == models.Response.item_id)
        .where(models.Response.learner_id == learner_id,
               models.Response.exam_code == exam,
               models.Response.context.in_(list(DIAGNOSIS_CONTEXTS)))
        .order_by(models.Response.created_at.asc())
    ).all()
    return rows


def diagnose(db: Session, learner: models.Account, exam: str, now: datetime | None = None) -> dict:
    """Full diagnosis from the practice loop: per-node cause mixture, leak ranking, strategy
    decomposition, with an honest ability range. Returns status 'insufficient_data' when there is
    nothing to diagnose, so the plan engine will not plan blind."""
    now = now or engine.now_utc()
    rows = _practice_responses(db, learner.id, exam)
    if not rows:
        return {"status": "insufficient_data", "exam": exam,
                "message": "No practice attempts yet — diagnose first, then prescribe."}

    # cache per-node mastery + prereq state and section names
    node_cache: dict[str, dict] = {}
    sec_name: dict[uuid.UUID, str] = {}

    def node_info(node_id: str) -> dict:
        if node_id not in node_cache:
            node = db.get(models.KnowledgeNode, node_id)
            st = kg.concept_state(db, learner.id, node, now) if node else None
            prereqs = kg.node_prereq_detail(db, learner.id, node, now) if node else []
            node_cache[node_id] = {
                "node": node, "name": node.name if node else node_id,
                "mastery": st.mastery if st else 0.0,
                "prereq_met": all(p["met"] for p in prereqs),
                "unmet_prereqs": [p for p in prereqs if not p["met"]],
            }
        return node_cache[node_id]

    cause_totals: dict[str, int] = defaultdict(int)
    total_misses = 0
    total_attempts = 0
    resolved: list[str] = []

    # group by concept so recency (newest attempt = weight 1) can be applied per concept
    by_concept: dict[str, list] = defaultdict(list)
    for resp, item in rows:
        by_concept[item.concept_node_id].append((resp, item))

    nodes_out = {}
    for nid, pairs in by_concept.items():
        info = node_info(nid)
        n_attempts = len(pairs)
        total_attempts += n_attempts
        causes: dict[str, int] = defaultdict(int)
        leak = 0.0
        miss_count = 0
        for idx, (resp, item) in enumerate(pairs):
            if resp.correct:
                continue
            if resp.section_id not in sec_name:
                s = db.get(models.Section, resp.section_id)
                sec_name[resp.section_id] = s.name if s else ""
            cause = classify_miss(
                exam=exam, mastery=info["mastery"], prereq_met=info["prereq_met"],
                hints_used=resp.hints_used, response_time_ms=resp.response_time_ms,
                difficulty_d=resp.difficulty_d,
                is_vocabulary=_is_vocabulary_item(item, info["node"], sec_name[resp.section_id]),
                allow_selection=True)
            causes[cause.value] += 1
            miss_count += 1
            pos_from_newest = n_attempts - 1 - idx
            leak += _difficulty_weight(resp.difficulty_d) * (RECENCY_DECAY ** pos_from_newest)
        if miss_count == 0:
            continue
        if leak < LEAK_FLOOR:
            resolved.append(nid)        # recently fixed — no longer a current leak (don't re-prescribe)
            continue
        mixture = {c: round(v / miss_count, 3) for c, v in causes.items()}
        dominant = max(causes.items(), key=lambda kv: kv[1])[0]
        for c, v in causes.items():
            cause_totals[c] += v
        total_misses += miss_count
        nodes_out[nid] = {
            "node_id": nid, "name": info["name"], "mastery": round(info["mastery"], 3),
            "attempts": n_attempts, "misses": miss_count,
            "miss_rate": round(miss_count / n_attempts, 3) if n_attempts else 0.0,
            "cause_mixture": mixture, "dominant_cause": dominant,
            "strategy_bucket": STRATEGY_BUCKET[Cause(dominant)],
            "leak_score": round(leak, 2),
            "unmet_prereqs": [{"id": p["id"], "name": p["name"], "mastery": p["mastery"]}
                              for p in info["unmet_prereqs"]],
        }

    leaks = sorted(nodes_out.values(),
                   key=lambda x: (x["leak_score"], x["miss_rate"]), reverse=True)

    # strategy decomposition over all misses
    buckets: dict[str, int] = defaultdict(int)
    for c, n in cause_totals.items():
        buckets[STRATEGY_BUCKET[Cause(c)]] += n
    decomposition = {b: round(100 * n / total_misses, 1) for b, n in buckets.items()} \
        if total_misses else {}

    perimeter = _honest_perimeter(db, learner, exam)

    return {
        "status": "ok", "exam": exam,
        "attempts": total_attempts, "misses": total_misses,
        "resolved_nodes": resolved,
        "strategy_decomposition": decomposition,
        "cause_totals": {c: cause_totals[c] for c in sorted(cause_totals)},
        "leaks": leaks,
        "nodes": nodes_out,
        "ability": perimeter,
    }


def _honest_perimeter(db: Session, learner: models.Account, exam: str) -> dict | None:
    """Ability as a RANGE, never a bare point (the platform-wide honesty contract)."""
    latest = ability_svc.latest_ability(db, learner, exam)
    if not latest:
        return None
    theta = latest.get("theta")
    se = latest.get("se")
    band = None
    if theta is not None and se not in (None, float("inf")):
        band = [round(theta - 2 * se, 3), round(theta + 2 * se, 3)]
    return {"theta": theta, "se": se, "band_95": band,
            "note": "estimate from the latest mock, shown as a range to be confirmed"}


def mock_decomposition(db: Session, session: models.MockSession, now: datetime | None = None) -> dict:
    """Decompose a MOCK's misses by the same taxonomy — a rendering only, kept separate from the
    practice-loop diagnosis (chassis rule). Mastery still comes from the practice loop."""
    now = now or engine.now_utc()
    rows = db.execute(
        select(models.Response, models.Item)
        .join(models.Item, models.Item.item_id == models.Response.item_id)
        .where(models.Response.session_id == str(session.id))
    ).all()
    cause_totals: dict[str, int] = defaultdict(int)
    total_misses = 0
    for resp, item in rows:
        if resp.correct:
            continue
        node = db.get(models.KnowledgeNode, item.concept_node_id)
        st = kg.concept_state(db, session.learner_id, node, now) if node else None
        prereqs = kg.node_prereq_detail(db, session.learner_id, node, now) if node else []
        sec = db.get(models.Section, resp.section_id)
        cause = classify_miss(
            exam=session.exam_code, mastery=st.mastery if st else 0.0,
            prereq_met=all(p["met"] for p in prereqs), hints_used=resp.hints_used,
            response_time_ms=resp.response_time_ms, difficulty_d=resp.difficulty_d,
            is_vocabulary=_is_vocabulary_item(item, node, sec.name if sec else ""),
            allow_selection=True)
        cause_totals[cause.value] += 1
        total_misses += 1
    buckets: dict[str, int] = defaultdict(int)
    for c, n in cause_totals.items():
        buckets[STRATEGY_BUCKET[Cause(c)]] += n
    decomposition = {b: round(100 * n / total_misses, 1) for b, n in buckets.items()} \
        if total_misses else {}
    return {"session_id": str(session.id), "exam": session.exam_code, "misses": total_misses,
            "strategy_decomposition": decomposition,
            "cause_totals": {c: cause_totals[c] for c in sorted(cause_totals)},
            "note": "mock decomposition is a rendering; the primary diagnosis runs on practice"}
