"""Phase 5 — analysis/debrief + shared review surfaces."""
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app import models
from app.services import debrief, honesty, state


def _session():
    eng = create_engine("sqlite+pysqlite:///:memory:",
                        connect_args={"check_same_thread": False}, poolclass=StaticPool)
    models.Base.metadata.create_all(eng)
    return Session(eng, autoflush=False)


def _seed(db, n_items=6):
    db.add(models.Exam(code="CAT", name="CAT")); db.flush()
    sec = models.Section(exam_code="CAT", key="QA", name="Quant"); db.add(sec); db.flush()
    db.add(models.KnowledgeNode(id="ratio", exam_code="CAT", section_id=sec.id,
                                kind="concept", name="Ratios")); db.flush()
    for i in range(n_items):
        db.add(models.Item(item_id=f"r{i}", version=1, content_hash=f"h{i}", exam_code="CAT",
                           section_id=sec.id, concept_node_id="ratio", difficulty_d=0, format="mcq",
                           num_options=4, stem=f"q{i}", options=["A", "B", "C", "D"],
                           correct_answer="A", solution=f"A is correct ({i}).", status="approved"))
    db.flush()
    a = models.Account(email="r@t.com", display_name="r"); db.add(a); db.flush(); db.commit()
    return a, sec


def _completed_mock(db, a, sec, n_items, n_answered, *, theta=0.5, se=0.3):
    ms = models.MockSession(learner_id=a.id, exam_code="CAT", section_key=None, mode="fixed_form",
                            status="completed", stage="done", cursor=n_answered, theta=theta, se=se,
                            reliability=0.9, n_answered=n_answered, max_items=n_items, se_target=0.3,
                            plan={}, seed=1)
    db.add(ms); db.flush()
    its = db.scalars(select(models.Item)).all()[:n_answered]
    for i, it in enumerate(its):
        correct = i % 3 != 0
        db.add(models.Response(learner_id=a.id, item_id=it.item_id, item_version=1,
                               context="full_mock", correct=correct,
                               answer_given=("A" if correct else "B"),
                               response_time_ms=(150000 if i == 0 else 45000), attempt_number=1,
                               hints_used=0, difficulty_d=0, exam_code="CAT", section_id=sec.id,
                               session_id=str(ms.id)))
    db.commit()
    return ms


def test_debrief_free_has_honest_score_decomposition_timing_no_items():
    db = _session(); a, sec = _seed(db)
    ms = _completed_mock(db, a, sec, 6, 6)
    out = debrief.debrief_mock(db, ms, full=False)
    hs = out["honest_score"]
    assert hs["kind"] == "percentile" and hs["band_95"] is not None and hs["is_claim"] is True
    assert "foundations" in out["decomposition"]["strategy_decomposition"]
    assert out["timing"]["slow_items"] == 1
    assert "review_items" not in out and out["review_items_available"] == 2


def test_debrief_full_includes_item_review_with_solutions():
    db = _session(); a, sec = _seed(db)
    ms = _completed_mock(db, a, sec, 6, 6)
    out = debrief.debrief_mock(db, ms, full=True)
    assert len(out["review_items"]) == 2
    assert all(r["solution"] for r in out["review_items"])
    assert all(r["correct"] is False for r in out["review_items"])


def test_debrief_score_is_provisional_when_too_few_items():
    db = _session(); a, sec = _seed(db)
    ms = _completed_mock(db, a, sec, 6, 3)   # 3 answered -> below MIN_BASIS_N
    out = debrief.debrief_mock(db, ms, full=False)
    assert out["honest_score"]["is_claim"] is False
    assert out["honest_score"]["basis"] == "provisional"


def test_debrief_records_a_prediction():
    db = _session(); a, sec = _seed(db)
    ms = _completed_mock(db, a, sec, 6, 6)
    debrief.debrief_mock(db, ms, full=False, record=True)
    recs = db.scalars(select(models.PredictionRecord).where(
        models.PredictionRecord.account_id == a.id)).all()
    assert len(recs) == 1 and recs[0].exam_code == "CAT"


def test_review_queue_groups_misses_by_concept_with_solutions():
    db = _session(); a, sec = _seed(db)
    its = db.scalars(select(models.Item)).all()
    state.record_response(db, a, its[0], context="practice", answer_given="B", correct=False,
                          response_time_ms=40000, attempt_number=1, hints_used=0, session_id=None)
    state.record_response(db, a, its[1], context="practice", answer_given="B", correct=False,
                          response_time_ms=40000, attempt_number=1, hints_used=0, session_id=None)
    q = debrief.review_queue(db, a, "CAT")
    assert q["to_review"] == 2 and "Ratios" in q["by_concept"]
    assert all(it["solution"] for it in q["by_concept"]["Ratios"])


def test_review_queue_drops_items_once_corrected():
    db = _session(); a, sec = _seed(db)
    it = db.scalars(select(models.Item)).first()
    state.record_response(db, a, it, context="practice", answer_given="B", correct=False,
                          response_time_ms=40000, attempt_number=1, hints_used=0, session_id=None)
    assert debrief.review_queue(db, a, "CAT")["to_review"] == 1
    state.record_response(db, a, it, context="practice", answer_given="A", correct=True,
                          response_time_ms=40000, attempt_number=1, hints_used=0, session_id=None)
    # most recent attempt is correct -> no longer in the queue
    assert debrief.review_queue(db, a, "CAT")["to_review"] == 0


def test_progress_returns_trend_leaks_and_accuracy():
    db = _session(); a, sec = _seed(db)
    db.add(models.AbilityEstimate(learner_id=a.id, exam_code="CAT", scope="exam",
                                  theta=0.4, se=0.3, n_items=10, method="eap")); db.commit()
    its = db.scalars(select(models.Item)).all()
    state.record_response(db, a, its[0], context="practice", answer_given="B", correct=False,
                          response_time_ms=40000, attempt_number=1, hints_used=0, session_id=None)
    pr = debrief.progress(db, a, "CAT")
    assert len(pr["ability_trend"]) == 1 and pr["ability_trend"][0]["band_95"] is not None
    assert pr["leaks"]["status"] == "ok"
    assert "accuracy_record" in pr
