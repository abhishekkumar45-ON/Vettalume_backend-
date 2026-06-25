from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models
from ..deps import get_db

router = APIRouter(prefix="/catalog", tags=["catalog"])


@router.get("/tree")
def get_tree(exam: str, db: Session = Depends(get_db)) -> dict:
    """Section -> topic -> concept hierarchy for an exam (the 'tree' / filing taxonomy)."""
    if db.get(models.Exam, exam) is None:
        raise HTTPException(404, f"unknown exam '{exam}'")

    sections = db.scalars(select(models.Section).where(models.Section.exam_code == exam)).all()
    nodes = db.scalars(select(models.KnowledgeNode).where(models.KnowledgeNode.exam_code == exam)).all()

    by_section: dict = {s.id: {"section": s.key, "name": s.name, "topics": []} for s in sections}
    children: dict[str, list] = {}
    for n in nodes:
        if n.parent_id:
            children.setdefault(n.parent_id, []).append(n)

    topics = []
    for n in nodes:
        if n.kind == "topic":
            topic = {
                "id": n.id, "name": n.name, "section": next(
                    (s.key for s in sections if s.id == n.section_id), None),
                "concepts": [{"id": c.id, "name": c.name} for c in children.get(n.id, [])],
            }
            topics.append(topic)

    # concepts with no topic parent (e.g. mounted GMAT/GRE) -> a synthetic per-section topic so they surface
    topic_ids = {n.id for n in nodes if n.kind == "topic"}
    sec_key = {s.id: s.key for s in sections}
    sec_name = {s.id: s.name for s in sections}
    orphans: dict = {}
    for n in nodes:
        if n.kind == "concept" and (n.parent_id is None or n.parent_id not in topic_ids):
            orphans.setdefault(n.section_id, []).append(n)
    for sec_id, concepts in orphans.items():
        topics.append({
            "id": "_general_" + sec_key.get(sec_id, str(sec_id)),
            "name": sec_name.get(sec_id, "General"),
            "section": sec_key.get(sec_id),
            "concepts": [{"id": c.id, "name": c.name} for c in concepts],
        })

    return {"exam": exam, "sections": [{"key": s.key, "name": s.name} for s in sections], "topics": topics}
