"""Ability scoring (Phase 2) — turn a learner's COLD/mock answers into an IRT theta (-3..+3) + SE.

This is the mock-side estimator, kept strictly separate from the Learning-side 0..1 mastery. It scores
with whatever item parameters are currently active (calibrated rows if present, else the authored
prior), via the same EAP engine the calibration loop uses. Elo is offered as a cheap running score.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models
from ..models import COLD_CONTEXTS
from . import calibration, irt


def _responses_for(db: Session, learner_id, exam: str, *, scope: str,
                   session_id: str | None, contexts) -> list[models.Response]:
    q = (select(models.Response)
         .where(models.Response.learner_id == learner_id, models.Response.exam_code == exam)
         .order_by(models.Response.created_at.asc()))
    if session_id:
        q = q.where(models.Response.session_id == session_id)
    else:
        q = q.where(models.Response.context.in_(list(contexts)))
    return list(db.scalars(q).all())


def score(db: Session, learner: models.Account, exam: str, *, scope: str = "diagnostic",
          session_id: str | None = None, contexts=None, method: str = "eap",
          persist: bool = True) -> dict:
    """EAP (default) or Elo theta + SE over the learner's responses in scope.

    scope: a label for the estimate ("diagnostic", a session_id, "full_mock", ...). When session_id is
    given we score that session; otherwise we score all responses in the given contexts (default: the
    cold contexts). Persists an AbilityEstimate unless persist=False.
    """
    contexts = contexts or COLD_CONTEXTS
    responses = _responses_for(db, learner.id, exam, scope=scope, session_id=session_id, contexts=contexts)
    triples = []
    for r in responses:
        item = db.get(models.Item, r.item_id)
        if item is None:
            continue
        a, b, c = calibration.active_params(db, item)
        triples.append((a, b, c, 1 if r.correct else 0))

    if method == "elo":
        theta = 0.0
        for a, b, c, u in triples:
            theta = irt.elo_update(theta, b, bool(u))
        total_info = sum(irt.information_3pl(theta, a, b, c) for a, b, c, _ in triples)
        se = irt.se_from_info(total_info)
    else:
        theta, se = irt.eap_ability(triples)

    se_out = None if se == float("inf") else round(se, 4)
    result = {"exam": exam, "scope": scope, "method": method, "theta": round(theta, 4),
              "se": se_out, "n_items": len(triples),
              "interval": ([round(theta - 2 * se, 3), round(theta + 2 * se, 3)]
                           if se_out is not None else None)}

    if persist and triples:
        db.add(models.AbilityEstimate(
            learner_id=learner.id, exam_code=exam, scope=scope, theta=theta,
            se=(se if se != float("inf") else 99.0), n_items=len(triples), method=method))
        db.flush()
    return result


def latest_ability(db: Session, learner: models.Account, exam: str) -> dict | None:
    """Most recent stored ability estimate for the learner in an exam."""
    row = db.scalar(select(models.AbilityEstimate)
                    .where(models.AbilityEstimate.learner_id == learner.id,
                           models.AbilityEstimate.exam_code == exam)
                    .order_by(models.AbilityEstimate.created_at.desc()))
    if row is None:
        return None
    return {"exam": exam, "scope": row.scope, "method": row.method, "theta": round(row.theta, 4),
            "se": round(row.se, 4), "n_items": row.n_items, "at": row.created_at.isoformat()}
