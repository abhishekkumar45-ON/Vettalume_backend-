from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models
from ..deps import get_current_learner, get_db
from ..services import billing, debrief, honesty

router = APIRouter(prefix="/review", tags=["review"])
honesty_router = APIRouter(prefix="/honesty", tags=["honesty"])


def _require_exam(db: Session, exam: str) -> None:
    if db.get(models.Exam, exam) is None:
        raise HTTPException(404, f"unknown exam '{exam}'")


@router.get("/mock/{sid}")
def mock_debrief(sid: str, learner=Depends(get_current_learner),
                 db: Session = Depends(get_db)) -> dict:
    """Full mock debrief. The item-by-item review with solutions is a paid surface; the free tier
    gets the honest score, cause decomposition, timing, and counts. Gating only bites when
    enforce_entitlements is on."""
    try:
        key = uuid.UUID(str(sid))
    except ValueError:
        raise HTTPException(404, "mock session not found")
    session = db.get(models.MockSession, key)
    if session is None:
        raise HTTPException(404, "mock session not found")
    state = billing.enforce(db, learner, session.exam_code, need="any")
    return debrief.debrief_mock(db, session, full=state["paid"])


@router.get("/queue")
def review_queue(exam: str, learner=Depends(get_current_learner),
                 db: Session = Depends(get_db)) -> dict:
    _require_exam(db, exam)
    return debrief.review_queue(db, learner, exam)


@router.get("/progress")
def progress(exam: str, learner=Depends(get_current_learner),
             db: Session = Depends(get_db)) -> dict:
    _require_exam(db, exam)
    return debrief.progress(db, learner, exam)


@honesty_router.get("/accuracy")
def accuracy(exam: str | None = None, kind: str | None = None,
             db: Session = Depends(get_db)) -> dict:
    """The platform's published accuracy record: band coverage and mean error over verified
    outcomes. Provisional until enough outcomes accumulate."""
    return honesty.accuracy_record(db, account=None, exam=exam, kind=kind)
