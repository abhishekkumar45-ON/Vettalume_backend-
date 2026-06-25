"""Phase 3 mock tests — delivery engines, scoring adapters, exposure control, the per-response
checkpoint, reliability, the SE-stop, and the learning/mock separation invariant."""
import random

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app import models
from app.services import irt, knowledge_graph as kg
from app.services import mock_delivery, mock_scoring, mock_session


def _session():
    eng = create_engine("sqlite+pysqlite:///:memory:",
                        connect_args={"check_same_thread": False}, poolclass=StaticPool)
    models.Base.metadata.create_all(eng)
    return Session(eng, autoflush=False)


def _seed_exam(db, exam="GMAT", per_band=8, sections=("Quant", "Verbal"), seed=1):
    """An exam with two sections and items spread across all five difficulty bands (-2..2),
    approved, 5-option MCQ with correct answer 'A'."""
    rng = random.Random(seed)
    db.add(models.Exam(code=exam, name=exam)); db.flush()
    sec_objs = []
    for k in sections:
        s = models.Section(exam_code=exam, key=k, name=k); db.add(s); db.flush(); sec_objs.append(s)
    topic = models.KnowledgeNode(id=f"{exam}-t", exam_code=exam, section_id=sec_objs[0].id,
                                 kind="topic", name="T")
    concept = models.KnowledgeNode(id=f"{exam}-c", exam_code=exam, section_id=sec_objs[0].id,
                                   kind="concept", name="C", parent_id=f"{exam}-t")
    db.add_all([topic, concept]); db.flush()
    n = 0
    for d in (-2, -1, 0, 1, 2):
        for _ in range(per_band):
            sec = sec_objs[n % len(sec_objs)]
            db.add(models.Item(item_id=f"{exam}-{n}", version=1, content_hash=f"{exam}h{n}",
                               exam_code=exam, section_id=sec.id, concept_node_id=f"{exam}-c",
                               difficulty_d=d, format="mcq", num_options=5,
                               stem=f"q{n}", options=["A", "B", "C", "D", "E"],
                               correct_answer="A", status="approved"))
            n += 1
    db.flush()
    return n


def _drive(db, learner, exam, mode, *, max_items=12, acc=1.0, seed=1, section=None):
    rng = random.Random(seed)
    s = mock_session.start(db, learner, exam, mode=mode, section_key=section,
                           max_items=max_items, seed=seed)
    out = mock_session.serve_next(db, s)
    final = None
    while out.get("status") == "serving":
        it = db.get(models.Item, out["question"]["item_id"])
        give = "A" if rng.random() < acc else "Z"
        res = mock_session.answer(db, s, it, answer_given=give)
        if res.get("stop"):
            final = res["final"]; break
        out = mock_session.serve_next(db, s)
        if out.get("status") == "completed":
            final = out; break
    if final is None:
        final = mock_session.score(db, s)
    db.refresh(s)
    return s, final


# ---------------- delivery engines ----------------
def test_item_adaptive_runs_and_scores():
    db = _session(); _seed_exam(db, "GMAT")
    learner = models.Account(email="a@t.com", display_name="a"); db.add(learner); db.flush()
    s, final = _drive(db, learner, "GMAT", "item_adaptive", max_items=10, acc=1.0)
    assert s.n_answered == 10
    assert final["score"]["report"]["scale"] == "GMAT 205-805"


def test_mst_routes_strong_learner_to_hard_panel():
    db = _session(); _seed_exam(db, "GMAT")
    learner = models.Account(email="b@t.com", display_name="b"); db.add(learner); db.flush()
    s, final = _drive(db, learner, "GMAT", "mst", max_items=12, acc=1.0, seed=2)
    assert s.panel_taken == "hard"        # all-correct routing -> hard panel
    assert s.stage == "panel"


def test_fixed_form_serves_planned_sequence_in_order():
    db = _session(); _seed_exam(db, "CAT")
    learner = models.Account(email="c@t.com", display_name="c"); db.add(learner); db.flush()
    s = mock_session.start(db, learner, "CAT", mode="fixed_form", max_items=8, seed=3)
    planned = list((s.plan or {}).get("form", []))
    served_order = []
    out = mock_session.serve_next(db, s)
    while out.get("status") == "serving":
        iid = out["question"]["item_id"]; served_order.append(iid)
        it = db.get(models.Item, iid)
        res = mock_session.answer(db, s, it, answer_given="A")
        if res.get("stop"):
            break
        out = mock_session.serve_next(db, s)
    assert served_order == planned[:len(served_order)]   # exact planned order, no adaptivity


# ---------------- scoring adapters ----------------
def test_composite_scale_bounds():
    # single section present: Verbal/DI fall back to overall ability, so all three measures = 75
    r = mock_scoring.composite_score(0.0, {"Quant": {"theta": 0.0}})
    assert r["section_scores"]["Quant"] == 75
    assert r["total"] == 505                       # (75+75+75-180)*20/3 + 205
    assert mock_scoring.composite_score(5.0, {})["total"] == 805    # all 90 -> ceiling
    assert mock_scoring.composite_score(-5.0, {})["total"] == 205   # all 60 -> floor


def test_composite_total_is_derived_from_section_scores():
    full = mock_scoring.composite_score(1.0, {"Quant": {"theta": 2.0},
                                              "Verbal": {"theta": 1.0},
                                              "Data Insights": {"theta": 0.0}})
    assert full["section_scores"] == {"Quant": 85, "Verbal": 80, "Data Insights": 75}
    # (85 + 80 + 75 - 180) * 20/3 + 205 = 605
    assert full["total"] == 605


def test_sectional_scale_bounds():
    r = mock_scoring.sectional_score({"Verbal": {"theta": 0.0}, "Quant": {"theta": 3.0}}, "hard")
    assert r["section_scores"]["Verbal"] == 150
    assert 130 <= r["section_scores"]["Quant"] <= 170 and r["section_scores"]["Quant"] == 170
    assert r["panel_taken"] == "hard"


def test_percentile_and_call_probability():
    # very strong + precise -> high percentile, confident yes on a moderate cutoff
    strong = mock_scoring.percentile_call_score(2.5, 0.2, {})
    assert strong["percentile"] > 99
    assert any(c["verdict"] == "confident yes" for c in strong["calls"])
    # average + uncertain -> calls are low / not confident
    weak = mock_scoring.percentile_call_score(0.0, 0.9, {})
    assert weak["percentile"] == 50.0
    assert all(c["verdict"] != "confident yes" for c in weak["calls"])


# ---------------- exposure control ----------------
def test_exposure_counts_grow_with_administrations():
    db = _session(); _seed_exam(db, "GMAT")
    learner = models.Account(email="e@t.com", display_name="e"); db.add(learner); db.flush()
    _drive(db, learner, "GMAT", "item_adaptive", max_items=8, acc=1.0)
    exposure = mock_delivery.exposure_counts(db, "GMAT")
    assert sum(exposure.values()) == 8 and all(v >= 1 for v in exposure.values())


def test_exposure_cap_excludes_overexposed_when_alternatives_exist():
    db = _session(); _seed_exam(db, "GMAT", per_band=4)
    items = mock_delivery.eligible_items(db, "GMAT")
    hot = items[0].item_id
    exposure = {hot: 99}
    rng = random.Random(0)
    picks = {mock_delivery.select_by_information(db, items, 0.0, set(), exposure, rng,
                                                 exposure_cap=10).item_id for _ in range(20)}
    assert hot not in picks      # capped out while alternatives remain


# ---------------- per-response checkpoint / resume ----------------
def test_checkpoint_persists_state_for_resume():
    db = _session(); _seed_exam(db, "GMAT")
    learner = models.Account(email="r@t.com", display_name="r"); db.add(learner); db.flush()
    s = mock_session.start(db, learner, "GMAT", mode="item_adaptive", max_items=20, seed=7)
    for _ in range(4):
        out = mock_session.serve_next(db, s)
        it = db.get(models.Item, out["question"]["item_id"])
        mock_session.answer(db, s, it, answer_given="A")
    sid = str(s.id)
    # simulate a fresh process: re-fetch the row by id (as the router does) and read its state
    import uuid as _uuid
    s2 = db.get(models.MockSession, _uuid.UUID(sid))
    snap = mock_session.state(db, s2)
    assert snap["n_answered"] == 4
    assert len(snap["served"]) == 4
    assert snap["status"] == "in_progress"
    # responses are on the durable spine, tied to the session
    n_resp = len(db.scalars(select(models.Response)
                            .where(models.Response.session_id == sid)).all())
    assert n_resp == 4


# ---------------- reliability + SE-stop ----------------
def test_reliability_increases_as_se_drops():
    assert irt.marginal_reliability(float("inf")) == 0.0
    assert irt.marginal_reliability(0.5) > 0.0
    assert irt.marginal_reliability(0.2) > irt.marginal_reliability(0.5)
    assert abs(irt.marginal_reliability(0.5) - 0.75) < 1e-9   # 1 - 0.25


def test_se_stop_ends_adaptive_before_max():
    db = _session(); _seed_exam(db, "GMAT", per_band=20)
    learner = models.Account(email="s@t.com", display_name="s"); db.add(learner); db.flush()
    # loose SE target so a consistent taker triggers the stop before max_items
    s = mock_session.start(db, learner, "GMAT", mode="item_adaptive",
                           max_items=80, se_target=0.7, seed=11)
    rng = random.Random(1)
    out = mock_session.serve_next(db, s); stopped_reason = None
    while out.get("status") == "serving":
        it = db.get(models.Item, out["question"]["item_id"])
        res = mock_session.answer(db, s, it, answer_given="A" if rng.random() < 0.85 else "Z")
        if res.get("stop"):
            stopped_reason = res["stop_reason"]; break
        out = mock_session.serve_next(db, s)
    db.refresh(s)
    assert stopped_reason == "se_target_met"
    assert s.n_answered < 80


# ---------------- learning / mock separation ----------------
def test_mock_responses_do_not_touch_learning_mastery():
    db = _session(); _seed_exam(db, "GMAT")
    learner = models.Account(email="sep@t.com", display_name="sep"); db.add(learner); db.flush()
    _drive(db, learner, "GMAT", "item_adaptive", max_items=10, acc=1.0)
    concept = db.get(models.KnowledgeNode, "GMAT-c")
    st = kg.concept_state(db, learner.id, concept)
    assert st.attempts == 0 and st.mastery == 0.0   # mock answers never feed the 0..1 mastery
