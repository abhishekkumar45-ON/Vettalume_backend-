from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models
from ..deps import get_current_learner, get_db
from ..services import mock_delivery, mock_session

router = APIRouter(prefix="/mock", tags=["mocks"])


class AnswerIn(BaseModel):
    item_id: str
    answer_given: Optional[str] = None
    response_time_ms: Optional[int] = None


def _get_session(db: Session, sid: str) -> models.MockSession:
    try:
        key = uuid.UUID(str(sid))
    except ValueError:
        raise HTTPException(404, "mock session not found")
    s = db.get(models.MockSession, key)
    if s is None:
        raise HTTPException(404, "mock session not found")
    return s


@router.post("/start")
def start(exam: str, mode: str = "item_adaptive", section: Optional[str] = None,
          max_items: int = 25, se_target: float = 0.30, routing_size: int = 5, seed: int = 0,
          learner=Depends(get_current_learner), db: Session = Depends(get_db)) -> dict:
    """Begin a mock. mode = item_adaptive (GMAT-style) | mst (GRE-style) | fixed_form (CAT-style).
    Returns the session and the first question."""
    if db.get(models.Exam, exam) is None:
        raise HTTPException(404, f"unknown exam '{exam}'")
    try:
        session = mock_session.start(db, learner, exam, mode=mode, section_key=section,
                                     max_items=max_items, se_target=se_target,
                                     routing_size=routing_size, seed=seed)
    except ValueError as e:
        raise HTTPException(400, str(e))
    first = mock_session.serve_next(db, session)
    return {"session": mock_session.state(db, session), "next": first}


@router.get("/{sid}/next")
def next_question(sid: str, db: Session = Depends(get_db)) -> dict:
    return mock_session.serve_next(db, _get_session(db, sid))


@router.post("/{sid}/answer")
def submit_answer(sid: str, body: AnswerIn, db: Session = Depends(get_db)) -> dict:
    session = _get_session(db, sid)
    item = db.get(models.Item, body.item_id)
    if item is None:
        raise HTTPException(404, "item not found")
    return mock_session.answer(db, session, item, answer_given=body.answer_given,
                               response_time_ms=body.response_time_ms)


@router.get("/{sid}")
def resume(sid: str, db: Session = Depends(get_db)) -> dict:
    """The checkpointed state — call this to resume a mock after a dropped connection."""
    return mock_session.state(db, _get_session(db, sid))


@router.get("/{sid}/score")
def score(sid: str, db: Session = Depends(get_db)) -> dict:
    return mock_session.score(db, _get_session(db, sid))


@router.get("/exposure/report")
def exposure(exam: str, top: int = 15, db: Session = Depends(get_db)) -> dict:
    """Population exposure — how often each item has been served across mocks (exposure control input)."""
    counts = mock_delivery.exposure_counts(db, exam)
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top]
    total = sum(counts.values())
    return {"exam": exam, "items_served": len(counts), "total_administrations": total,
            "most_exposed": [{"item_id": i, "times_served": n} for i, n in ranked]}
