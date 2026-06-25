"""The Honest Perimeter (Phase 5) — the platform-wide prediction-honesty discipline.

Three rules, enforced here so no surface can quietly break them:
  1. Ranges, not bare points. Every prediction is wrapped in a band.
  2. No inflation. A prediction the platform cannot defend with verified evidence is marked
     `provisional` and `is_claim = False` — shown as an estimate to be confirmed, never as a claim.
  3. A published accuracy record. Every emitted prediction is logged; once a real outcome is known
     the band's coverage and the point error are computed. The aggregate over those rows is the
     accuracy the platform publishes about itself.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models

Z95 = 1.959963985            # 95% normal quantile
MIN_BASIS_N = 5              # below this many supporting observations a prediction is provisional
PUBLISH_MIN_N = 20          # below this many resolved outcomes the accuracy record is not yet published


def _envelope(point, band, se, basis_n: int, kind: str) -> dict:
    finite_band = band is not None and band[0] is not None and band[1] is not None
    calibrated = finite_band and basis_n >= MIN_BASIS_N
    return {
        "kind": kind,
        "point": point,
        "band_95": [round(band[0], 3), round(band[1], 3)] if finite_band else None,
        "confidence": 0.95 if finite_band else None,
        "se": round(se, 4) if se not in (None, float("inf")) else None,
        "basis_n": basis_n,
        "basis": "calibrated" if calibrated else "provisional",
        "is_claim": bool(calibrated),
        "note": ("a measured estimate with a verified band"
                 if calibrated else
                 "provisional — shown as an estimate to be confirmed, not a published claim"),
    }


def perimeter(point: float | None, se: float | None, *, kind: str = "ability",
              basis_n: int = 0, z: float = Z95) -> dict:
    """Symmetric range from a point and its standard error (point and se in the same units)."""
    if point is None or se in (None, float("inf")):
        return _envelope(point, None, se, basis_n, kind)
    return _envelope(point, [point - z * se, point + z * se], se, basis_n, kind)


def perimeter_band(point: float | None, low: float | None, high: float | None, *,
                   kind: str = "score", se: float | None = None, basis_n: int = 0) -> dict:
    """Range from an explicit band (used for scores, where the band comes from mapping the ability
    band through a scoring adapter and so is not symmetric in score units)."""
    band = None if low is None or high is None else [min(low, high), max(low, high)]
    return _envelope(point, band, se, basis_n, kind)


def record_prediction(db: Session, account: models.Account | None, exam: str, env: dict) -> str:
    """Log an emitted prediction so its accuracy can be measured once an outcome is known."""
    band = env.get("band_95") or [None, None]
    rec = models.PredictionRecord(
        account_id=(account.id if account else None), exam_code=exam, kind=env.get("kind", "score"),
        point=float(env["point"]), band_low=band[0], band_high=band[1],
        se=env.get("se"), basis=env.get("basis", "provisional"))
    db.add(rec)
    db.flush()
    db.commit()
    return str(rec.id)


def record_outcome(db: Session, prediction_id: str, actual: float) -> dict:
    rec = db.get(models.PredictionRecord, uuid.UUID(str(prediction_id)))
    if rec is None:
        return {"status": "not_found"}
    rec.outcome = float(actual)
    rec.within_band = (rec.band_low is not None and rec.band_high is not None
                       and rec.band_low <= actual <= rec.band_high)
    rec.resolved_at = datetime.now(timezone.utc)
    db.commit()
    return {"status": "ok", "within_band": rec.within_band,
            "error": round(abs(rec.point - actual), 3)}


def accuracy_record(db: Session, account: models.Account | None = None, exam: str | None = None,
                    kind: str | None = None) -> dict:
    """The published accuracy record: of resolved predictions, what fraction of actual outcomes fell
    inside the 95% band (coverage), and the mean absolute point error. Coverage near 0.95 means the
    bands are honest. Not published until enough outcomes exist (PUBLISH_MIN_N)."""
    q = select(models.PredictionRecord).where(models.PredictionRecord.outcome.is_not(None))
    if account is not None:
        q = q.where(models.PredictionRecord.account_id == account.id)
    if exam:
        q = q.where(models.PredictionRecord.exam_code == exam)
    if kind:
        q = q.where(models.PredictionRecord.kind == kind)
    rows = list(db.scalars(q).all())
    n = len(rows)
    if n == 0:
        return {"resolved": 0, "published": False,
                "note": "no verified outcomes yet — nothing to publish"}
    with_band = [r for r in rows if r.band_low is not None]
    coverage = (sum(1 for r in with_band if r.within_band) / len(with_band)) if with_band else None
    mae = sum(abs(r.point - r.outcome) for r in rows) / n
    return {
        "exam": exam or "all", "kind": kind or "all", "resolved": n,
        "band_coverage_95": round(coverage, 3) if coverage is not None else None,
        "target_coverage": 0.95,
        "mean_abs_error": round(mae, 3),
        "published": n >= PUBLISH_MIN_N,
        "note": ("published accuracy record" if n >= PUBLISH_MIN_N
                 else f"provisional — {n}/{PUBLISH_MIN_N} outcomes needed before publishing"),
    }
