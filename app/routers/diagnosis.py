from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models
from ..deps import get_current_learner, get_db
from ..services import diagnosis as dx
from ..services import plan as plan_svc

router = APIRouter(prefix="/diagnosis", tags=["diagnosis"])
plan_router = APIRouter(prefix="/plan", tags=["plan"])


def _require_exam(db: Session, exam: str) -> None:
    if db.get(models.Exam, exam) is None:
        raise HTTPException(404, f"unknown exam '{exam}'")


@router.get("")
def get_diagnosis(exam: str, learner=Depends(get_current_learner),
                  db: Session = Depends(get_db)) -> dict:
    """Classify the learner's practice misses by cause: per-node cause mixture, leak ranking, the
    foundations/execution/selection strategy decomposition, and an honest ability range."""
    _require_exam(db, exam)
    return dx.diagnose(db, learner, exam)


@router.get("/leaks")
def get_leaks(exam: str, top: int = 10, learner=Depends(get_current_learner),
              db: Session = Depends(get_db)) -> dict:
    _require_exam(db, exam)
    diag = dx.diagnose(db, learner, exam)
    if diag.get("status") != "ok":
        return diag
    return {"exam": exam, "strategy_decomposition": diag["strategy_decomposition"],
            "leaks": diag["leaks"][:top]}


@router.get("/mock/{sid}")
def mock_decomposition(sid: str, db: Session = Depends(get_db)) -> dict:
    """Decompose one mock's misses by the same taxonomy (a rendering, separate from the practice
    diagnosis)."""
    try:
        key = uuid.UUID(str(sid))
    except ValueError:
        raise HTTPException(404, "mock session not found")
    session = db.get(models.MockSession, key)
    if session is None:
        raise HTTPException(404, "mock session not found")
    return dx.mock_decomposition(db, session)


@plan_router.post("/generate")
def generate(exam: str, learner=Depends(get_current_learner),
             db: Session = Depends(get_db)) -> dict:
    """Generate or re-generate the study plan from the current diagnosis. Refuses to plan when there
    is not enough practice signal. On a re-plan, returns the diff and a plain-language explanation."""
    _require_exam(db, exam)
    return plan_svc.generate_plan(db, learner, exam)


@plan_router.get("")
def get_plan(exam: str, learner=Depends(get_current_learner),
             db: Session = Depends(get_db)) -> dict:
    _require_exam(db, exam)
    return plan_svc.current_plan(db, learner, exam)


@plan_router.get("/history")
def history(exam: str, learner=Depends(get_current_learner),
            db: Session = Depends(get_db)) -> dict:
    _require_exam(db, exam)
    return {"exam": exam, "versions": plan_svc.plan_history(db, learner, exam)}
