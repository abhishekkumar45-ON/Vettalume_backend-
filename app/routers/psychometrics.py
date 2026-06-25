from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models
from ..config import settings
from ..deps import get_current_learner, get_db
from ..services import ability as ability_svc
from ..services import calibration, sim_harness

router = APIRouter(prefix="/irt", tags=["psychometrics"])


def _run_dict(run: models.CalibrationRun) -> dict:
    return {"run_id": str(run.id), "exam": run.exam_code, "status": run.status,
            "n_items": run.n_items, "n_responses": run.n_responses, "n_learners": run.n_learners,
            "iterations": run.iterations, "converged": run.converged, "activated": run.activated,
            "summary": run.summary, "created_at": run.created_at.isoformat() if run.created_at else None}


@router.post("/calibrate")
def calibrate(exam: str, two_pl_at: int = 500, three_pl_at: int = 2000, activate: bool = True,
              db: Session = Depends(get_db)) -> dict:
    """DEV — run the calibration worker over this exam's COLD responses and write a versioned
    parameter set. `activate` makes it the live params (mirrored onto Item.irt_*)."""
    if not settings.dev_mode:
        raise HTTPException(403, "calibration endpoint is dev-only")
    if db.get(models.Exam, exam) is None:
        raise HTTPException(404, f"unknown exam '{exam}'")
    run = calibration.run_calibration(db, exam, two_pl_at=two_pl_at, three_pl_at=three_pl_at,
                                      activate=activate)
    return _run_dict(run)


@router.get("/runs")
def runs(exam: Optional[str] = None, db: Session = Depends(get_db)) -> dict:
    q = select(models.CalibrationRun).order_by(models.CalibrationRun.created_at.desc())
    if exam:
        q = q.where(models.CalibrationRun.exam_code == exam)
    return {"runs": [_run_dict(r) for r in db.scalars(q).all()]}


@router.get("/runs/{run_id}")
def run_detail(run_id: str, db: Session = Depends(get_db)) -> dict:
    try:
        key = uuid.UUID(str(run_id))
    except ValueError:
        raise HTTPException(404, "run not found")
    run = db.get(models.CalibrationRun, key)
    if run is None:
        raise HTTPException(404, "run not found")
    params = db.scalars(select(models.IrtParameter)
                        .where(models.IrtParameter.run_id == run.id)).all()
    return {**_run_dict(run),
            "parameters": [{"item_id": p.item_id, "a": round(p.a, 4), "b": round(p.b, 4),
                            "c": round(p.c, 4), "phase": p.phase, "n_responses": p.n_responses,
                            "active": p.active} for p in params]}


@router.get("/item/{item_id}")
def item_params(item_id: str, db: Session = Depends(get_db)) -> dict:
    item = db.get(models.Item, item_id)
    if item is None:
        raise HTTPException(404, "item not found")
    a, b, c = calibration.active_params(db, item)
    history = db.scalars(select(models.IrtParameter)
                         .where(models.IrtParameter.item_id == item_id)
                         .order_by(models.IrtParameter.created_at.desc())).all()
    active_row = next((h for h in history if h.active), None)
    return {
        "item_id": item_id, "authored_difficulty_d": item.difficulty_d, "format": item.format,
        "num_options": item.num_options,
        "active_params": {"a": round(a, 4), "b": round(b, 4), "c": round(c, 4),
                          "source": "calibrated" if active_row else "authored_prior",
                          "phase": active_row.phase if active_row else None},
        "history": [{"run_id": str(h.run_id), "a": round(h.a, 4), "b": round(h.b, 4),
                     "c": round(h.c, 4), "phase": h.phase, "n_responses": h.n_responses,
                     "active": h.active, "at": h.created_at.isoformat() if h.created_at else None}
                    for h in history],
    }


@router.post("/score")
def score(exam: str, scope: str = "diagnostic", session_id: Optional[str] = None,
          method: str = "eap", learner=Depends(get_current_learner),
          db: Session = Depends(get_db)) -> dict:
    """Score the current learner's COLD/mock responses into an IRT theta (-3..+3) + SE via EAP
    (or method=elo). Persists the estimate."""
    if db.get(models.Exam, exam) is None:
        raise HTTPException(404, f"unknown exam '{exam}'")
    return ability_svc.score(db, learner, exam, scope=scope, session_id=session_id, method=method)


@router.get("/ability")
def ability(exam: str, learner=Depends(get_current_learner), db: Session = Depends(get_db)) -> dict:
    latest = ability_svc.latest_ability(db, learner, exam)
    if latest is None:
        return {"exam": exam, "theta": None, "note": "no ability scored yet — POST /irt/score after a diagnostic or mock"}
    return latest


@router.post("/simulate")
def simulate(students: int = 600, items: int = 60, seed: int = 7,
             two_pl_at: int = 500, three_pl_at: int = 2000, b_min: float = 0.7) -> dict:
    """DEV — the release gate. Generate a synthetic population with known a,b,c, run the exact
    calibration loop, and report parameter recovery. Calibration is only safe to trust if it
    recovers difficulty here."""
    if not settings.dev_mode:
        raise HTTPException(403, "simulate endpoint is dev-only")
    result = sim_harness.run_recovery(n_students=students, n_items=items, seed=seed,
                                      two_pl_at=two_pl_at, three_pl_at=three_pl_at)
    return {**result, "gate": sim_harness.gate(result, b_min=b_min)}
