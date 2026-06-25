"""Analysis / debrief + shared review surfaces (Phase 5).

This is the shared mock-debrief contract: a completed mock becomes an honest score (a range, logged to
the accuracy record), a cause decomposition (reusing the Phase-4 taxonomy), a timing read, and a
section breakdown. The shared review surfaces expose the learner's missed items (with solutions) and
spaced reviews, plus a progress view (ability trend, mastery, open leaks, published accuracy).
"""
from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models
from . import ability as ability_svc
from . import diagnosis as dx
from . import honesty
from . import knowledge_graph as kg
from . import learning
from . import mock_scoring

Z = honesty.Z95


def _mock_responses(db: Session, session: models.MockSession):
    return list(db.scalars(select(models.Response).where(
        models.Response.session_id == str(session.id))).all())


def _honest_headline(db: Session, session: models.MockSession, report: dict) -> dict:
    """Wrap the headline number in a 95% band, mapping the ability band through the adapter."""
    exam = session.exam_code.upper()
    theta, se = session.theta, session.se
    se_eff = None if (se is None or se >= 99) else se
    n = session.n_answered
    if exam == "GMAT" and se_eff is not None:
        sections = mock_scoring.section_breakdown(db, session)
        lo = {k: {**v, "theta": v["theta"] - Z * se_eff} for k, v in sections.items()}
        hi = {k: {**v, "theta": v["theta"] + Z * se_eff} for k, v in sections.items()}
        total_lo = mock_scoring.composite_score(theta - Z * se_eff, lo)["total"]
        total_hi = mock_scoring.composite_score(theta + Z * se_eff, hi)["total"]
        return honesty.perimeter_band(report["total"], total_lo, total_hi,
                                      kind="score", se=se_eff, basis_n=n)
    if exam == "CAT":
        band = report.get("percentile_band_95") or [None, None]
        return honesty.perimeter_band(report.get("percentile"), band[0], band[1],
                                      kind="percentile", se=se_eff, basis_n=n)
    # GRE / generic: publish the native ability range honestly; score-space band pending panel mapping
    env = honesty.perimeter(theta, se_eff, kind="ability", basis_n=n)
    env["note"] += " (score-space band pending panel-aware mapping)"
    return env


def _timing(responses) -> dict:
    timed = [r.response_time_ms for r in responses if r.response_time_ms is not None]
    if not timed:
        return {"n_timed": 0}
    total = sum(timed)
    return {"n_timed": len(timed), "total_ms": total, "avg_ms": round(total / len(timed)),
            "slow_items": sum(1 for t in timed if t >= dx.SLOW_MS),
            "fast_items": sum(1 for t in timed if t <= dx.FAST_MS)}


def _item_review(db: Session, responses, *, only_misses: bool = True) -> list[dict]:
    out = []
    for r in responses:
        if only_misses and r.correct:
            continue
        item = db.get(models.Item, r.item_id)
        if item is None:
            continue
        node = db.get(models.KnowledgeNode, item.concept_node_id)
        out.append({"item_id": item.item_id, "concept": node.name if node else item.concept_node_id,
                    "difficulty_d": item.difficulty_d, "your_answer": r.answer_given,
                    "correct_answer": item.correct_answer, "correct": r.correct,
                    "solution": item.solution})
    return out


def debrief_mock(db: Session, session: models.MockSession, *, full: bool,
                 record: bool = True) -> dict:
    """Full mock debrief. `full` (a paid surface) includes the item-by-item review with solutions;
    the free tier gets the honest score, decomposition, timing, and counts only."""
    score = mock_scoring.score_mock(db, session)
    headline = _honest_headline(db, session, score["report"])
    if record:
        honesty.record_prediction(db, db.get(models.Account, session.learner_id),
                                  session.exam_code, headline)
    responses = _mock_responses(db, session)
    decomposition = dx.mock_decomposition(db, session)
    out = {
        "session_id": str(session.id), "exam": session.exam_code, "status": session.status,
        "honest_score": headline, "score_detail": score["report"],
        "ability": {"theta": score["theta"], "se": score["se"],
                    "reliability": score["reliability"]},
        "sections": mock_scoring.section_breakdown(db, session),
        "decomposition": decomposition,
        "timing": _timing(responses),
        "n_answered": session.n_answered, "misses": sum(1 for r in responses if not r.correct),
        "full": full,
    }
    if full:
        out["review_items"] = _item_review(db, responses, only_misses=True)
    else:
        out["review_items_available"] = sum(1 for r in responses if not r.correct)
        out["upgrade_note"] = "Full item-by-item review with solutions is a paid feature."
    return out


def review_queue(db: Session, learner: models.Account, exam: str) -> dict:
    """Shared review surface: items the learner's most recent attempt got wrong (across practice and
    mocks), grouped by concept and carrying solutions, plus due spaced reviews."""
    rows = db.scalars(select(models.Response).where(
        models.Response.learner_id == learner.id, models.Response.exam_code == exam)
        .order_by(models.Response.created_at.asc())).all()
    latest: dict[str, models.Response] = {}
    for r in rows:
        latest[r.item_id] = r            # last write wins -> most recent attempt per item
    by_concept: dict[str, list] = {}
    for item_id, r in latest.items():
        if r.correct:
            continue
        item = db.get(models.Item, item_id)
        if item is None:
            continue
        node = db.get(models.KnowledgeNode, item.concept_node_id)
        cname = node.name if node else item.concept_node_id
        by_concept.setdefault(cname, []).append(
            {"item_id": item.item_id, "difficulty_d": item.difficulty_d,
             "your_answer": r.answer_given, "correct_answer": item.correct_answer,
             "solution": item.solution})
    due = learning.due_reviews(db, learner, exam)
    total = sum(len(v) for v in by_concept.values())
    return {"exam": exam, "to_review": total, "by_concept": by_concept, "due_reviews": due}


def progress(db: Session, learner: models.Account, exam: str) -> dict:
    """Progress analytics: honest ability trend, topic mastery, open leaks, published accuracy."""
    hist = db.scalars(select(models.AbilityEstimate).where(
        models.AbilityEstimate.learner_id == learner.id,
        models.AbilityEstimate.exam_code == exam)
        .order_by(models.AbilityEstimate.created_at.asc())).all()
    trend = [{"at": a.created_at.isoformat() if a.created_at else None,
              **honesty.perimeter(a.theta, None if a.se in (None,) or (a.se or 0) >= 99 else a.se,
                                  kind="ability", basis_n=a.n_items)} for a in hist]
    topics = db.scalars(select(models.KnowledgeNode).where(
        models.KnowledgeNode.exam_code == exam,
        models.KnowledgeNode.kind == models.NodeKind.topic.value)).all()
    mastery = [{"topic": t.name, "mastery": round(kg.topic_mastery(db, learner.id, t), 3)}
               for t in topics]
    diag = dx.diagnose(db, learner, exam)
    leaks = {"status": diag["status"],
             "open_leaks": len(diag.get("leaks", [])) if diag["status"] == "ok" else 0,
             "strategy_decomposition": diag.get("strategy_decomposition", {})}
    return {"exam": exam, "ability_trend": trend, "mastery": mastery, "leaks": leaks,
            "accuracy_record": honesty.accuracy_record(db, learner, exam)}
