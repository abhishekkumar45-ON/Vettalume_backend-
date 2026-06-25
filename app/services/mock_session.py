"""Mock session orchestration (Phase 3) — the per-response checkpoint loop.

start -> (serve_next -> answer)* -> finish.  After EVERY answer the MockSession row is updated with the
running theta/SE/reliability and delivery cursor, so the attempt survives a dropped connection with no
data loss (the PRD's mock-reliability contract). The same step computes marginal reliability and the
SE-stop decision. Mock responses are recorded in a COLD mock context (so they feed IRT/calibration)
but never touch the Learning-side mastery.
"""
from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models
from . import calibration, irt, mock_delivery, mock_scoring


def _mock_context(session: models.MockSession) -> str:
    return "sectional_mock" if session.section_key else "full_mock"


def _learner_mock_seen(db: Session, learner_id, exam: str) -> set[str]:
    rows = db.execute(
        select(models.Response.item_id).where(
            models.Response.learner_id == learner_id, models.Response.exam_code == exam,
            models.Response.context.in_(list(models.MOCK_CONTEXTS)))
    ).all()
    return {iid for (iid,) in rows}


def _recompute_ability(db: Session, session: models.MockSession) -> tuple[float, float]:
    responses = db.scalars(
        select(models.Response).where(models.Response.session_id == str(session.id))
        .order_by(models.Response.created_at.asc())).all()
    triples = []
    for r in responses:
        item = db.get(models.Item, r.item_id)
        if item is None:
            continue
        a, b, c = calibration.active_params(db, item)
        triples.append((a, b, c, 1 if r.correct else 0))
    return irt.eap_ability(triples)


def should_stop(session: models.MockSession, plan_exhausted: bool) -> tuple[bool, str]:
    if plan_exhausted:
        return True, "form_complete"
    if session.n_answered >= session.max_items:
        return True, "max_items_reached"
    if session.mode == "item_adaptive" and session.se <= session.se_target and session.n_answered >= 3:
        return True, "se_target_met"   # precise enough
    return False, ""


# ---------- lifecycle ----------
def start(db: Session, learner: models.Account, exam: str, *, mode: str = "item_adaptive",
          section_key: str | None = None, max_items: int = 25, se_target: float = 0.30,
          routing_size: int = 5, seed: int = 0) -> models.MockSession:
    if mode not in {"item_adaptive", "mst", "fixed_form"}:
        raise ValueError(f"unknown mode '{mode}'")
    seed = seed or uuid.uuid4().int % 1_000_000
    plan = mock_delivery.build_plan(db, exam, mode, section_key, max_items, routing_size, seed)
    session = models.MockSession(
        learner_id=learner.id, exam_code=exam, section_key=section_key, mode=mode,
        status="in_progress", stage=("routing" if mode == "mst" else "main"),
        cursor=0, theta=0.0, se=99.0, reliability=0.0, n_answered=0,
        max_items=max_items, se_target=se_target, plan=plan, seed=seed)
    db.add(session)
    db.flush()
    db.commit()
    return session


def serve_next(db: Session, session: models.MockSession) -> dict:
    """Pick and return the next question (does not record anything). Marks it served and checkpoints."""
    if session.status != "in_progress":
        return {"status": session.status}
    stopped, reason = should_stop(session, plan_exhausted=False)
    if stopped:
        return _finish(db, session, reason)

    item = mock_delivery.next_item(db, session)
    # for item_adaptive, also avoid items this learner saw in earlier mocks
    if item is None or (session.mode == "item_adaptive" and
                        item.item_id in _learner_mock_seen(db, session.learner_id, session.exam_code)):
        if session.mode == "item_adaptive":
            seen = _learner_mock_seen(db, session.learner_id, session.exam_code)
            served = set((session.plan or {}).get("served", []))
            pool = [it for it in mock_delivery.eligible_items(db, session.exam_code, session.section_key)
                    if it.item_id not in served and it.item_id not in seen]
            if not pool:
                return _finish(db, session, "pool_exhausted")
            import random
            rng = random.Random(session.seed + session.n_answered)
            exposure = mock_delivery.exposure_counts(db, session.exam_code)
            item = mock_delivery.select_by_information(db, pool, session.theta, set(), exposure, rng)
        if item is None:
            return _finish(db, session, "form_complete")

    mock_delivery.mark_served(session, item.item_id)
    db.flush()
    db.commit()
    a, b, c = calibration.active_params(db, item)
    return {"status": "serving", "session_id": str(session.id), "n_answered": session.n_answered,
            "stage": session.stage, "panel_taken": session.panel_taken,
            "question": {"item_id": item.item_id, "stem": item.stem, "options": item.options,
                         "format": item.format, "num_options": item.num_options,
                         "difficulty_b": round(b, 2)}}


def answer(db: Session, session: models.MockSession, item: models.Item, *,
           answer_given: str | None, response_time_ms: int | None = None) -> dict:
    """Record the answer in a mock context, recompute theta/SE/reliability, advance, and CHECKPOINT."""
    if session.status != "in_progress":
        return {"status": session.status}

    correct = (answer_given is not None
               and str(answer_given).strip() == str(item.correct_answer).strip())
    db.add(models.Response(
        learner_id=session.learner_id, item_id=item.item_id, item_version=item.version,
        context=_mock_context(session), correct=bool(correct), answer_given=answer_given,
        response_time_ms=response_time_ms, attempt_number=1, hints_used=0,
        difficulty_d=item.difficulty_d, exam_code=session.exam_code, section_id=item.section_id,
        session_id=str(session.id)))
    db.flush()

    # --- the per-response checkpoint: recompute ability + reliability, advance cursor, persist ---
    theta, se = _recompute_ability(db, session)
    session.theta = theta
    session.se = 99.0 if se == float("inf") else se
    session.reliability = irt.marginal_reliability(se)
    session.n_answered += 1
    session.cursor += 1
    db.flush()   # durable checkpoint — resume after a drop reads exactly this

    stopped, reason = should_stop(session, plan_exhausted=mock_delivery.next_item(db, session) is None)
    payload = {"status": "answered", "correct": correct,
               "theta": round(theta, 3), "se": None if se == float("inf") else round(se, 3),
               "reliability": round(session.reliability, 3), "n_answered": session.n_answered,
               "solution": item.solution, "stop": stopped, "stop_reason": reason if stopped else None}
    if stopped:
        payload["final"] = _finish(db, session, reason)
    else:
        db.commit()   # durable checkpoint survives a dropped connection
    return payload


def _finish(db: Session, session: models.MockSession, reason: str) -> dict:
    if session.status == "in_progress":
        from datetime import datetime, timezone
        session.status = "completed"
        session.completed_at = datetime.now(timezone.utc)
        db.flush()
        db.commit()
    return {"status": "completed", "reason": reason, "score": mock_scoring.score_mock(db, session)}


def state(db: Session, session: models.MockSession) -> dict:
    """Resume snapshot — everything needed to continue after a connection loss."""
    plan = session.plan or {}
    return {"session_id": str(session.id), "exam": session.exam_code, "mode": session.mode,
            "section": session.section_key, "status": session.status, "stage": session.stage,
            "panel_taken": session.panel_taken, "n_answered": session.n_answered,
            "served": plan.get("served", []), "theta": round(session.theta, 3),
            "se": None if session.se >= 99 else round(session.se, 3),
            "reliability": round(session.reliability, 3), "max_items": session.max_items,
            "se_target": session.se_target}


def score(db: Session, session: models.MockSession) -> dict:
    return mock_scoring.score_mock(db, session)
