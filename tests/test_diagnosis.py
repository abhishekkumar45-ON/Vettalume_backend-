"""Phase 4 tests — diagnosis chassis (cause taxonomy, cause mixture, leak ranking, strategy
decomposition) and the plan engine (prereq-ordered, diagnose-before-prescribing, re-plan + explain)."""
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app import models
from app.services import diagnosis as dx
from app.services import plan as plan_svc
from app.services import state


def _session():
    eng = create_engine("sqlite+pysqlite:///:memory:",
                        connect_args={"check_same_thread": False}, poolclass=StaticPool)
    models.Base.metadata.create_all(eng)
    return Session(eng, autoflush=False)


def _seed(db, exam="CAT"):
    """Topic Quant with three concepts; mix-basic requires ratio-basic (a real prereq edge)."""
    db.add(models.Exam(code=exam, name=exam)); db.flush()
    sec = models.Section(exam_code=exam, key="QA", name="Quant"); db.add(sec); db.flush()
    db.add(models.KnowledgeNode(id="quant", exam_code=exam, section_id=sec.id, kind="topic", name="Quant"))
    for cid, nm in [("ratio-basic", "Ratios"), ("avg-simple", "Averages"), ("mix-basic", "Mixtures")]:
        db.add(models.KnowledgeNode(id=cid, exam_code=exam, section_id=sec.id, kind="concept",
                                    name=nm, parent_id="quant"))
    db.flush()
    db.add(models.PrereqEdge(node_id="mix-basic", prereq_node_id="ratio-basic"))
    n = 0
    for cid in ["ratio-basic", "avg-simple", "mix-basic"]:
        for _ in range(8):
            db.add(models.Item(item_id=f"{cid}-{n}", version=1, content_hash=f"h{n}", exam_code=exam,
                               section_id=sec.id, concept_node_id=cid, difficulty_d=0, format="mcq",
                               num_options=4, stem=f"q{n}", options=["A", "B", "C", "D"],
                               correct_answer="A", status="approved")); n += 1
    db.flush(); db.commit()
    learner = models.Account(email="dx@t.com", display_name="dx"); db.add(learner); db.flush(); db.commit()
    return learner


def _practice(db, learner, concept, correct, *, rt, hints=0, n=1):
    its = db.scalars(select(models.Item).where(models.Item.concept_node_id == concept)).all()
    for k in range(n):
        it = its[k % len(its)]
        state.record_response(db, learner, it, context="practice",
                              answer_given=("A" if correct else "Z"), correct=correct,
                              response_time_ms=rt, attempt_number=1, hints_used=hints, session_id=None)


# ---------------- cause inference (pure) ----------------
def test_classify_concept_gap_from_low_mastery_hints_or_prereq():
    assert dx.classify_miss(exam="CAT", mastery=0.2, prereq_met=True, hints_used=0,
                            response_time_ms=60000, difficulty_d=0) == dx.Cause.concept_gap
    assert dx.classify_miss(exam="CAT", mastery=0.9, prereq_met=True, hints_used=2,
                            response_time_ms=60000, difficulty_d=0) == dx.Cause.concept_gap
    assert dx.classify_miss(exam="CAT", mastery=0.9, prereq_met=False, hints_used=0,
                            response_time_ms=60000, difficulty_d=0) == dx.Cause.concept_gap


def test_classify_execution_causes_when_concept_held():
    held = dict(exam="CAT", mastery=0.9, prereq_met=True, hints_used=0)
    assert dx.classify_miss(**held, response_time_ms=200000, difficulty_d=0) == dx.Cause.timing_pressure
    assert dx.classify_miss(**held, response_time_ms=5000, difficulty_d=0) == dx.Cause.careless_slip
    assert dx.classify_miss(**held, response_time_ms=60000, difficulty_d=1) == dx.Cause.process_error


def test_classify_exam_native_causes():
    assert dx.classify_miss(exam="GRE", mastery=0.2, prereq_met=True, hints_used=0,
                            response_time_ms=30000, difficulty_d=0,
                            is_vocabulary=True) == dx.Cause.vocabulary_gap
    assert dx.classify_miss(exam="CAT", mastery=0.2, prereq_met=True, hints_used=0,
                            response_time_ms=5000, difficulty_d=2,
                            allow_selection=True) == dx.Cause.selection_error


# ---------------- diagnosis ----------------
def test_diagnose_builds_mixture_leaks_and_decomposition():
    db = _session(); learner = _seed(db)
    _practice(db, learner, "ratio-basic", True, rt=40000, n=6)   # held
    _practice(db, learner, "ratio-basic", False, rt=200000)      # timing
    _practice(db, learner, "ratio-basic", False, rt=8000)        # careless
    _practice(db, learner, "avg-simple", False, rt=60000, hints=2, n=3)  # concept gap
    diag = dx.diagnose(db, learner, "CAT")
    assert diag["status"] == "ok"
    names = {L["name"] for L in diag["leaks"]}
    assert {"Ratios", "Averages"} <= names
    ratios = next(L for L in diag["leaks"] if L["name"] == "Ratios")
    assert set(ratios["cause_mixture"]) == {"timing_pressure", "careless_slip"}
    assert ratios["strategy_bucket"] == "execution"
    assert "foundations" in diag["strategy_decomposition"]
    assert abs(sum(diag["strategy_decomposition"].values()) - 100.0) < 0.5


def test_resolved_concept_drops_from_leaks_after_recent_correct_work():
    db = _session(); learner = _seed(db)
    _practice(db, learner, "avg-simple", False, rt=60000, hints=2, n=3)
    assert any(L["name"] == "Averages" for L in dx.diagnose(db, learner, "CAT")["leaks"])
    _practice(db, learner, "avg-simple", True, rt=40000, n=12)    # fix it
    diag = dx.diagnose(db, learner, "CAT")
    assert not any(L["name"] == "Averages" for L in diag["leaks"])
    assert "avg-simple" in diag["resolved_nodes"]


def test_diagnose_ignores_mock_responses():
    db = _session(); learner = _seed(db)
    it = db.scalars(select(models.Item).where(models.Item.concept_node_id == "ratio-basic")).first()
    # a mock-context miss must NOT create a practice-loop diagnosis (gap detection in Part 1, not Part 2)
    state.record_response(db, learner, it, context="full_mock", answer_given="Z", correct=False,
                          response_time_ms=60000, attempt_number=1, hints_used=0, session_id="s1")
    assert dx.diagnose(db, learner, "CAT")["status"] == "insufficient_data"


# ---------------- plan engine ----------------
def test_plan_refuses_without_practice_signal():
    db = _session(); learner = _seed(db)
    out = plan_svc.generate_plan(db, learner, "CAT")
    assert out["status"] == "refused"


def test_plan_schedules_unmet_prerequisite_before_its_dependent():
    db = _session(); learner = _seed(db)
    # Ratios left unpractised -> mastery below threshold -> a genuine unmet prerequisite of Mixtures
    _practice(db, learner, "mix-basic", False, rt=50000, n=3)     # mixtures is a leak
    out = plan_svc.generate_plan(db, learner, "CAT")
    assert out["status"] == "ok"
    order = [it["node_id"] for it in out["items"]]
    assert "ratio-basic" in order and "mix-basic" in order
    assert order.index("ratio-basic") < order.index("mix-basic")
    ratio_item = next(it for it in out["items"] if it["node_id"] == "ratio-basic")
    assert ratio_item["prerequisite_for"] == "Mixtures"


def test_replan_diffs_and_explains_a_closed_leak():
    db = _session(); learner = _seed(db)
    _practice(db, learner, "avg-simple", False, rt=60000, hints=2, n=3)
    _practice(db, learner, "ratio-basic", False, rt=60000, n=3)   # a second leak that persists
    v1 = plan_svc.generate_plan(db, learner, "CAT")
    assert v1["version"] == 1 and "avg-simple" in [i["node_id"] for i in v1["items"]]
    _practice(db, learner, "avg-simple", True, rt=40000, n=12)    # close avg-simple only
    v2 = plan_svc.generate_plan(db, learner, "CAT")
    assert v2["version"] == 2
    assert "avg-simple" in v2["change"]["removed"]
    assert "ratio-basic" in [i["node_id"] for i in v2["items"]]   # the other leak remains
    assert "Averages" in v2["change"]["explanation"]


def test_plan_versions_supersede_and_history_tracks():
    db = _session(); learner = _seed(db)
    _practice(db, learner, "ratio-basic", False, rt=60000, n=3)
    plan_svc.generate_plan(db, learner, "CAT")
    plan_svc.generate_plan(db, learner, "CAT")
    actives = db.scalars(select(models.StudyPlan).where(
        models.StudyPlan.status == "active")).all()
    assert len(actives) == 1 and actives[0].version == 2
    assert [h["version"] for h in plan_svc.plan_history(db, learner, "CAT")] == [1, 2]


def test_honest_perimeter_present_when_ability_exists():
    db = _session(); learner = _seed(db)
    db.add(models.AbilityEstimate(learner_id=learner.id, exam_code="CAT", scope="exam",
                                  theta=0.5, se=0.3, n_items=10, method="eap")); db.commit()
    _practice(db, learner, "ratio-basic", False, rt=60000, n=2)
    diag = dx.diagnose(db, learner, "CAT")
    assert diag["ability"] is not None
    assert diag["ability"]["band_95"] == [-0.1, 1.1]
