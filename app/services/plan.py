"""Plan engine (Phase 4) — turn a diagnosis into a prioritized, prerequisite-ordered study plan, and
re-plan continuously while explaining the change in plain language.

The two PRD principles this enforces: "diagnose before prescribing" (it refuses to plan when the
diagnosis is insufficient) and "plans adapt; candidates do not repeat what is broken" (each re-plan
diffs against the prior version — closed leaks drop out, new leaks enter — and the change is explained
in words, not just a new list).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models
from . import diagnosis as dx
from . import knowledge_graph as kg

MAX_PLAN_ITEMS = 8

# cause -> remediation: knowing vs doing get different work (PRD "separate knowing from doing")
REMEDIATION = {
    dx.Cause.concept_gap: ("learn_concept", "Learn the governing concept, then re-practise"),
    dx.Cause.process_error: ("drill_method", "Drill the solving procedure until it is automatic"),
    dx.Cause.timing_pressure: ("timed_drill", "Timed sets to hold accuracy under pace"),
    dx.Cause.careless_slip: ("accuracy_reps", "Accuracy-discipline reps and a checking routine"),
    dx.Cause.vocabulary_gap: ("vocabulary_engine", "Build the missing words in the vocabulary engine"),
    dx.Cause.selection_error: ("selection_trainer", "Selection trainer: when to attempt vs skip"),
}


def _remediation(cause_value: str) -> tuple[str, str]:
    return REMEDIATION[dx.Cause(cause_value)]


def _build_items(db: Session, learner: models.Account, diag: dict, now: datetime) -> list[dict]:
    """From the leak ranking, emit prerequisite-ordered plan items. An unmet prerequisite of a leak
    is scheduled BEFORE the leak itself (you cannot fix mixtures while ratios is broken)."""
    leaks = diag["leaks"][:MAX_PLAN_ITEMS]
    scheduled: dict[str, dict] = {}
    order: list[str] = []

    def add(node_id: str, name: str, cause: str, leak_score: float, mastery: float,
            reason: str, prereq_for: str | None = None):
        if node_id in scheduled:
            return
        action, label = _remediation(cause)
        scheduled[node_id] = {
            "node_id": node_id, "name": name, "cause": cause,
            "strategy_bucket": dx.STRATEGY_BUCKET[dx.Cause(cause)],
            "action": action, "action_label": label,
            "leak_score": leak_score, "mastery": mastery, "why": reason,
            "prerequisite_for": prereq_for,
        }
        order.append(node_id)

    for leak in leaks:
        # schedule any unmet prerequisites first, as foundations (concept gap)
        for p in leak.get("unmet_prereqs", []):
            add(p["id"], p["name"], dx.Cause.concept_gap.value, leak["leak_score"], p["mastery"],
                reason=f"Prerequisite for {leak['name']}; mastery {p['mastery']} is below threshold.",
                prereq_for=leak["name"])
        add(leak["node_id"], leak["name"], leak["dominant_cause"], leak["leak_score"],
            leak["mastery"],
            reason=(f"{leak['misses']} miss(es), mostly "
                    f"{dx.CAUSE_LABEL[dx.Cause(leak['dominant_cause'])].split(' — ')[0].lower()}; "
                    f"mastery {leak['mastery']}."))

    # priority = schedule order (prereqs already precede their dependents)
    for i, nid in enumerate(order, start=1):
        scheduled[nid]["priority"] = i
    return [scheduled[nid] for nid in order]


def _diff(old_items: list[dict] | None, new_items: list[dict]) -> dict:
    old_by = {it["node_id"]: it for it in (old_items or [])}
    new_by = {it["node_id"]: it for it in new_items}
    added = [it for it in new_items if it["node_id"] not in old_by]
    removed = [it for it in (old_items or []) if it["node_id"] not in new_by]
    kept = [nid for nid in new_by if nid in old_by]
    reprioritized = [
        {"node_id": nid, "from": old_by[nid].get("priority"), "to": new_by[nid].get("priority")}
        for nid in kept if old_by[nid].get("priority") != new_by[nid].get("priority")]
    return {"added": added, "removed": removed, "reprioritized": reprioritized}


def _explain(diff: dict, prior_diag: dict | None, diag: dict, first: bool) -> str:
    if first:
        return ("Initial plan from your diagnostic: targeting the biggest leaks, foundations first.")
    parts: list[str] = []
    for it in diff["removed"]:
        parts.append(f"Closed {it['name']} — dropping it from the plan.")
    for it in diff["added"]:
        cause = dx.CAUSE_LABEL[dx.Cause(it['cause'])].split(' — ')[0].lower()
        parts.append(f"New {cause} leak in {it['name']} — added.")
    if diff["reprioritized"]:
        parts.append(f"Reordered {len(diff['reprioritized'])} item(s) as leaks shifted.")
    # strategy-mix movement
    if prior_diag and prior_diag.get("strategy_decomposition") and diag.get("strategy_decomposition"):
        of = prior_diag["strategy_decomposition"].get("foundations", 0)
        nf = diag["strategy_decomposition"].get("foundations", 0)
        if abs(nf - of) >= 5:
            verb = "rose" if nf > of else "fell"
            parts.append(f"Foundations share of misses {verb} {of}% -> {nf}%.")
    return " ".join(parts) if parts else "No material change since the last plan."


def generate_plan(db: Session, learner: models.Account, exam: str,
                  now: datetime | None = None) -> dict:
    """Diagnose, then (only if there is something to act on) build and persist a new plan version,
    diffed against the prior one with a plain-language explanation."""
    now = now or kg.engine.now_utc()
    diag = dx.diagnose(db, learner, exam, now)
    prior = _current(db, learner, exam)

    if diag.get("status") != "ok" or not diag.get("leaks"):
        return {"status": "refused",
                "reason": "Diagnose first — not enough practice signal to build a plan.",
                "diagnosis": diag}

    items = _build_items(db, learner, diag, now)
    diff = _diff(prior.items if prior else None, items)
    explanation = _explain(diff, prior.diagnosis if prior else None, diag, first=prior is None)

    diag_snapshot = {"strategy_decomposition": diag["strategy_decomposition"],
                     "cause_totals": diag["cause_totals"], "misses": diag["misses"],
                     "ability": diag.get("ability")}
    if prior is not None:
        prior.status = "superseded"
    version = (prior.version + 1) if prior else 1
    plan = models.StudyPlan(learner_id=learner.id, exam_code=exam, version=version, status="active",
                            items=items, diagnosis=diag_snapshot,
                            rationale={"diff": {k: [i.get("node_id") for i in v] if isinstance(v, list)
                                                else v for k, v in diff.items()},
                                       "explanation": explanation})
    db.add(plan)
    db.flush()
    db.commit()
    return {"status": "ok", "version": version, "exam": exam, "items": items,
            "diagnosis_summary": diag_snapshot,
            "change": {"explanation": explanation,
                       "added": [i["node_id"] for i in diff["added"]],
                       "removed": [i["node_id"] for i in diff["removed"]],
                       "reprioritized": diff["reprioritized"]}}


def _current(db: Session, learner: models.Account, exam: str) -> models.StudyPlan | None:
    return db.scalar(select(models.StudyPlan)
                     .where(models.StudyPlan.learner_id == learner.id,
                            models.StudyPlan.exam_code == exam,
                            models.StudyPlan.status == "active")
                     .order_by(models.StudyPlan.version.desc()))


def current_plan(db: Session, learner: models.Account, exam: str) -> dict:
    plan = _current(db, learner, exam)
    if plan is None:
        return {"status": "none", "exam": exam,
                "message": "No plan yet — POST /plan/generate after some practice."}
    return {"status": "ok", "version": plan.version, "exam": exam, "items": plan.items,
            "diagnosis_summary": plan.diagnosis, "rationale": plan.rationale,
            "created_at": plan.created_at.isoformat() if plan.created_at else None}


def plan_history(db: Session, learner: models.Account, exam: str) -> list[dict]:
    rows = db.scalars(select(models.StudyPlan)
                      .where(models.StudyPlan.learner_id == learner.id,
                             models.StudyPlan.exam_code == exam)
                      .order_by(models.StudyPlan.version.asc())).all()
    return [{"version": p.version, "status": p.status, "n_items": len(p.items or []),
             "explanation": (p.rationale or {}).get("explanation"),
             "created_at": p.created_at.isoformat() if p.created_at else None} for p in rows]
