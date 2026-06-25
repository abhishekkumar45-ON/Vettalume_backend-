"""Cross-exam warm-start (Phase 6, AC-02).

The platform's structural advantage: a learner's second course starts warmer than cold. Three shared
constructs transfer across courses (PRD Section 06): quant core, Reading Comprehension, and the
data-logic engine (GMAT Data Insights <-> CAT DILR). When a course is newly entitled, ability measured
on those constructs in the learner's OTHER courses seeds a prior for the new one.

The honesty rule (PRD: "cross-course signal transfer is overclaimed"): a transferred prior is ALWAYS
shown as an estimate to be confirmed, never as established ability. So the transfer inflates the SE
(cross-exam uncertainty) and the Honest Perimeter marks it provisional (is_claim = False) regardless
of how much source data backs it.
"""
from __future__ import annotations

import math
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models
from . import honesty
from . import mock_scoring

CONSTRUCTS = ("quant", "rc", "data_logic")
TRANSFER_VAR = 0.25   # added variance for crossing exams — widens the band, keeps the prior honest

_CONSTRUCT_TOKENS = {
    "quant": ("quant", "qa", "number", "algebra", "arith"),
    "rc": ("verbal", "varc", "rc", "reading", "va"),
    "data_logic": ("data", "dilr", "insight", "logic", "di"),
}


def construct_of(section: models.Section) -> str | None:
    """Map an exam-scoped section to a shared construct by its key/name (quant | rc | data_logic)."""
    hay = f"{section.key} {section.name}".lower()
    for construct in CONSTRUCTS:                      # quant, then rc, then data_logic (priority order)
        if any(tok in hay for tok in _CONSTRUCT_TOKENS[construct]):
            return construct
    return None


def _section_constructs(db: Session, exam: str) -> dict[str, list[str]]:
    """construct -> [section keys] for one exam."""
    out: dict[str, list[str]] = {}
    for sec in db.scalars(select(models.Section).where(models.Section.exam_code == exam)).all():
        c = construct_of(sec)
        if c:
            out.setdefault(c, []).append(sec.key)
    return out


def _construct_ability(db: Session, learner: models.Account, construct: str,
                       exclude_exam: str) -> tuple[float, float, int] | None:
    """EAP ability over the learner's COLD responses (mocks/diagnostic) in the construct's sections,
    pooled across every course except the target."""
    rows = db.scalars(select(models.Response).where(
        models.Response.learner_id == learner.id,
        models.Response.exam_code != exclude_exam,
        models.Response.context.in_(list(models.COLD_CONTEXTS)))).all()
    if not rows:
        return None
    sec_construct: dict = {}
    matched = []
    for r in rows:
        if r.section_id not in sec_construct:
            sec = db.get(models.Section, r.section_id)
            sec_construct[r.section_id] = construct_of(sec) if sec else None
        if sec_construct[r.section_id] == construct:
            matched.append(r)
    if not matched:
        return None
    theta, se, n = mock_scoring._eap_for(db, matched)
    return theta, se, n


def warm_start(db: Session, learner: models.Account, target_exam: str, *,
               persist: bool = True) -> dict:
    """Compute (and optionally persist) transferred priors for a newly entitled course."""
    constructs = _section_constructs(db, target_exam)
    transferred = []
    for construct, section_keys in constructs.items():
        src = _construct_ability(db, learner, construct, exclude_exam=target_exam)
        if src is None:
            continue
        theta, se, n = src
        se_in = math.sqrt((se if se != float("inf") else 1.0) ** 2 + TRANSFER_VAR)
        # basis_n forced to 0 -> the Honest Perimeter marks every transfer provisional (estimate to confirm)
        env = honesty.perimeter(round(theta, 3), round(se_in, 3), kind="ability", basis_n=0)
        transferred.append({
            "construct": construct, "target_sections": sorted(section_keys),
            "source_responses": n, "source_se": round(se, 3) if se != float("inf") else None,
            "prior": env,
        })
        if persist:
            scope = f"warm_start:{construct}"
            existing = db.scalar(select(models.AbilityEstimate).where(
                models.AbilityEstimate.learner_id == learner.id,
                models.AbilityEstimate.exam_code == target_exam,
                models.AbilityEstimate.scope == scope))
            if existing is None:
                db.add(models.AbilityEstimate(learner_id=learner.id, exam_code=target_exam,
                                              scope=scope, theta=round(theta, 3),
                                              se=round(se_in, 3), n_items=0, method="transfer"))
            else:
                existing.theta, existing.se = round(theta, 3), round(se_in, 3)
    if persist and transferred:
        db.commit()
    return {
        "exam": target_exam,
        "warm_started": bool(transferred),
        "transferred": transferred,
        "note": ("transferred priors are estimates to be confirmed on this exam, not established ability"
                 if transferred else
                 "cold start — no shared-construct signals from other courses yet"),
    }
