from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models
from ..deps import get_current_learner, get_db
from ..services import billing, warmstart

router = APIRouter(prefix="/account", tags=["account"])


def _require_exam(db: Session, exam: str) -> None:
    if db.get(models.Exam, exam) is None:
        raise HTTPException(404, f"unknown exam '{exam}'")


@router.get("/profile")
def profile(learner=Depends(get_current_learner), db: Session = Depends(get_db)) -> dict:
    """One identity, the learner's course entitlements (AC-01)."""
    return {"id": str(learner.id), "email": learner.email, "display_name": learner.display_name,
            "entitlements": billing.all_entitlements(db, learner)}


@router.get("/warm-start")
def warm_start_preview(exam: str, learner=Depends(get_current_learner),
                       db: Session = Depends(get_db)) -> dict:
    """Preview the priors that would transfer into this course from the learner's other courses,
    without persisting (AC-02). Transferred priors are always provisional — estimates to confirm."""
    _require_exam(db, exam)
    return warmstart.warm_start(db, learner, exam, persist=False)


@router.post("/warm-start")
def warm_start_apply(exam: str, learner=Depends(get_current_learner),
                     db: Session = Depends(get_db)) -> dict:
    """Compute and persist transferred priors for this course."""
    _require_exam(db, exam)
    return warmstart.warm_start(db, learner, exam, persist=True)
