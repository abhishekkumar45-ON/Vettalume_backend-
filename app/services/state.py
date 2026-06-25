"""Learner-facing engine seam.

Phase 0 proves the data flow (spine append + exposure + state read). The selection here is a
trivial 'first eligible' pick and mastery is a plain proportion-correct. Phase 1 replaces:
  * eligible_items() ordering          -> the concept + problem MAB
  * _recompute_node_state()            -> blended mastery 0.40*P + 0.30*D + 0.30*M
without changing these function signatures or the schema.
"""
from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models
from ..config import settings
from ..models import MOCK_CONTEXTS


def eligible_items(db: Session, learner_id: uuid.UUID, *, context: str,
                   concept_node_id: str | None = None, exam_code: str | None = None) -> list[models.Item]:
    """Encodes the shared-bank rule:
      * mocks STRICTLY exclude any item the learner has already seen (no contamination);
      * practice allows repeats (the MCM deliberately resurfaces concepts in Phase 1);
      * usage_scope reservation is honoured (mock_only / practice_only).
    """
    seen = set(db.scalars(
        select(models.Exposure.item_id).where(models.Exposure.learner_id == learner_id)
    ).all())

    q = select(models.Item)
    if settings.serve_only_approved:
        q = q.where(models.Item.status == "approved")
    if concept_node_id:
        q = q.where(models.Item.concept_node_id == concept_node_id)
    if exam_code:
        q = q.where(models.Item.exam_code == exam_code)

    out: list[models.Item] = []
    is_mock = context in MOCK_CONTEXTS
    for it in db.scalars(q).all():
        if context == "practice" and it.usage_scope == "mock_only":
            continue
        if is_mock and it.usage_scope == "practice_only":
            continue
        if is_mock and it.item_id in seen:
            continue  # strict exclusion for scored mocks
        out.append(it)
    return out


def record_response(db: Session, learner: models.Account, item: models.Item, *, context: str,
                    answer_given: str | None, correct: bool | None, response_time_ms: int | None,
                    attempt_number: int, hints_used: int, session_id: str | None):
    # Grade server-side when the client did not assert correctness (don't trust the client).
    if correct is None:
        correct = (answer_given is not None
                   and str(answer_given).strip() == str(item.correct_answer).strip())

    resp = models.Response(
        learner_id=learner.id, item_id=item.item_id, item_version=item.version, context=context,
        correct=bool(correct), answer_given=answer_given, response_time_ms=response_time_ms,
        attempt_number=attempt_number, hints_used=hints_used, difficulty_d=item.difficulty_d,
        exam_code=item.exam_code, section_id=item.section_id, session_id=session_id,
    )
    db.add(resp)
    _bump_exposure(db, learner.id, item.item_id, context)
    db.flush()  # session is autoflush=False; make the new response visible to the recompute query
    state = _recompute_node_state(db, learner.id, item.concept_node_id)
    db.commit()
    return resp, bool(correct), state


def _bump_exposure(db: Session, learner_id: uuid.UUID, item_id: str, context: str) -> None:
    exp = db.scalar(select(models.Exposure).where(
        models.Exposure.learner_id == learner_id, models.Exposure.item_id == item_id))
    if exp is None:
        db.add(models.Exposure(learner_id=learner_id, item_id=item_id,
                               last_seen_context=context, times_seen=1))
    else:
        exp.times_seen += 1
        exp.last_seen_context = context


def _recompute_node_state(db: Session, learner_id: uuid.UUID, node_id: str) -> models.LearnerNodeState:
    # Phase 1: real blended mastery = 0.40*P + 0.30*D + 0.30*M, computed from the ordered spine log.
    from . import engine
    from .knowledge_graph import concept_attempts

    now = engine.now_utc()
    attempts = concept_attempts(db, learner_id, node_id)

    st = db.scalar(select(models.LearnerNodeState).where(
        models.LearnerNodeState.learner_id == learner_id,
        models.LearnerNodeState.node_id == node_id))
    if st is None:
        st = models.LearnerNodeState(learner_id=learner_id, node_id=node_id)
        db.add(st)

    mastery, p, d, m = engine.blended_mastery(attempts, now)
    st.performance_p = p
    st.difficulty_score = d
    st.memory_strength = m
    st.mastery = mastery
    st.learned = len(attempts) >= 1
    st.bandit_state = {
        "edge": engine.maple_edge(attempts),
        "learning_progress": engine.learning_progress(attempts),
        "attempts": len(attempts),
        "mastered": engine.concept_mastered(mastery, len(attempts)),
        "due_for_review": engine.is_due_for_review(mastery, m),
    }
    return st


def node_attempt_count(db: Session, learner_id: uuid.UUID, node_id: str) -> int:
    return len(db.scalars(
        select(models.Response.id)
        .join(models.Item, models.Item.item_id == models.Response.item_id)
        .where(models.Response.learner_id == learner_id, models.Item.concept_node_id == node_id)
    ).all())
