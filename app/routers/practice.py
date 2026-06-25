from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models
from ..deps import get_current_learner, get_db
from ..schemas import AnswerIn, AnswerOut, ItemPublic, NodeStateOut, StateOut
from ..services.state import eligible_items, node_attempt_count, record_response

router = APIRouter(prefix="/practice", tags=["practice"])


@router.get("/next", response_model=ItemPublic)
def next_item(node_id: str, learner=Depends(get_current_learner), db: Session = Depends(get_db)) -> ItemPublic:
    """Return the next question for a concept. Phase 0 picks the first eligible item; Phase 1's
    problem bandit replaces the selection. The learner never sees difficulty, the answer, or the
    solution here."""
    if db.get(models.KnowledgeNode, node_id) is None:
        raise HTTPException(404, f"unknown node '{node_id}'")
    candidates = eligible_items(db, learner.id, context="practice", concept_node_id=node_id)
    if not candidates:
        raise HTTPException(404, "no eligible items for this concept")
    it = candidates[0]
    return ItemPublic(item_id=it.item_id, stem=it.stem, options=it.options,
                      format=it.format, num_options=it.num_options)


@router.post("/answer", response_model=AnswerOut)
def submit_answer(body: AnswerIn, learner=Depends(get_current_learner), db: Session = Depends(get_db)) -> AnswerOut:
    item = db.get(models.Item, body.item_id)
    if item is None:
        raise HTTPException(404, f"unknown item '{body.item_id}'")

    _resp, correct, state = record_response(
        db, learner, item, context=body.context.value, answer_given=body.answer_given,
        correct=body.correct, response_time_ms=body.response_time_ms,
        attempt_number=body.attempt_number, hints_used=body.hints_used, session_id=body.session_id,
    )
    attempts = node_attempt_count(db, learner.id, item.concept_node_id)
    return AnswerOut(correct=correct, solution=item.solution, node_id=item.concept_node_id,
                     mastery=round(state.mastery, 4), attempts=attempts)


@router.get("/state", response_model=StateOut)
def get_state(exam: str, learner=Depends(get_current_learner), db: Session = Depends(get_db)) -> StateOut:
    nodes = db.scalars(select(models.KnowledgeNode).where(models.KnowledgeNode.exam_code == exam)).all()
    states = {s.node_id: s for s in db.scalars(
        select(models.LearnerNodeState).where(models.LearnerNodeState.learner_id == learner.id)).all()}

    out: list[NodeStateOut] = []
    for n in nodes:
        st = states.get(n.id)
        attempts = node_attempt_count(db, learner.id, n.id) if st else 0
        out.append(NodeStateOut(
            node_id=n.id, name=n.name,
            learned=bool(st.learned) if st else False,
            mastery=round(st.mastery, 4) if st else 0.0,
            attempts=attempts,
        ))
    return StateOut(exam=exam, nodes=out)
