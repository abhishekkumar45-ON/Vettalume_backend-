"""Phase 2 psychometrics tests.

The reference's worked numbers are the source of truth for the math; the calibration/store/ability
paths are checked end-to-end on a small synthetic cold-response set.
"""
import random

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app import models
from app.services import ability as ability_svc
from app.services import calibration, irt, sim_harness


# ---------------- pure math: golden vectors from the reference ----------------
def test_eap_ability_matches_reference_ladder():
    seq = [(1, 0.0, 0.2, 1), (1, 0.5, 0.2, 1), (1, 1.0, 0.2, 1), (1, 1.5, 0.2, 0)]
    t1, se1 = irt.eap_ability(seq[:1])
    t2, _ = irt.eap_ability(seq[:2])
    t3, _ = irt.eap_ability(seq[:3])
    t4, _ = irt.eap_ability(seq[:4])
    assert abs(t1 - 0.275) < 0.01      # +0.28 in the doc
    assert abs(t2 - 0.547) < 0.01      # +0.55
    assert abs(t3 - 0.812) < 0.01      # +0.81
    assert abs(t4 - 0.568) < 0.01      # +0.57
    assert abs(se1 - 2.42) < 0.05      # SE after Q1


def test_eap_single_wrong_is_negative_and_larger():
    t_right, _ = irt.eap_ability([(1, 0.0, 0.2, 1)])
    t_wrong, _ = irt.eap_ability([(1, 0.0, 0.2, 0)])
    assert abs(t_wrong - (-0.41)) < 0.01
    assert abs(t_wrong) > abs(t_right)     # the guessing floor makes a miss carry more weight


def test_information_and_se_vectors():
    assert abs(irt.information_3pl(0.55, 1, 0.55, 0.2) - 0.167) < 0.005
    assert abs(irt.information_3pl(0.55, 2, 0.55, 0.2) - 0.667) < 0.01
    assert abs(irt.information_3pl(0.55, 1, 0.55, 0.0) - 0.250) < 0.005
    assert abs(irt.se_from_info(0.170) - 2.42) < 0.05


def test_elo_step_update():
    assert abs(irt.elo_update(0.0, 0.0, True) - 0.25) < 1e-9
    assert abs(irt.sigmoid(0.25 - 0.5) - 0.438) < 0.001
    # wrong moves down
    assert irt.elo_update(0.0, 0.0, False) < 0.0


def test_prob_3pl_worked_example():
    # theta=0.8, b=1, a=1.5, c=0.2 -> ~0.54 in the doc
    assert abs(irt.prob_3pl(0.8, 1.5, 1.0, 0.2) - 0.54) < 0.01


def test_phase_thresholds():
    assert irt.phase_for(40) == "b"
    assert irt.phase_for(600) == "2pl"
    assert irt.phase_for(2500) == "3pl"
    assert irt.default_c(4) == 0.25 and irt.default_c(5) == 0.2 and irt.default_c(None, True) == 0.0


# ---------------- MLE recovers a known item curve ----------------
def test_fit_item_recovers_known_b():
    rng = random.Random(11)
    true_a, true_b, true_c = 1.5, 1.0, 0.2
    thetas = [rng.gauss(0, 1) for _ in range(500)]
    us = [1 if rng.random() < irt.prob_3pl(t, true_a, true_b, true_c) else 0 for t in thetas]
    fit = irt.fit_item(thetas, us, phase="2pl", b0=0.0, a0=1.0, c0=0.2)
    assert abs(fit["b"] - true_b) < 0.35
    assert abs(fit["a"] - true_a) < 0.6


# ---------------- simulation harness: the release gate ----------------
def test_sim_harness_recovers_difficulty_gate_passes():
    r = sim_harness.run_recovery(n_students=200, n_items=30, seed=3)
    assert r["b_corr"] is not None and r["b_corr"] >= 0.7
    g = sim_harness.gate(r, b_min=0.7)
    assert g["passed"] is True


# ---------------- DB calibration + versioned store ----------------
def _session():
    eng = create_engine("sqlite+pysqlite:///:memory:",
                        connect_args={"check_same_thread": False}, poolclass=StaticPool)
    models.Base.metadata.create_all(eng)
    return Session(eng, autoflush=False)


def _seed_cold_exam(db, n_learners=120, n_items=20, seed=5):
    """Build an exam with items and a synthetic COLD (full_mock) response set from known abilities."""
    rng = random.Random(seed)
    exam = models.Exam(code="CAT", name="CAT")
    sec = models.Section(exam_code="CAT", key="QA", name="QA")
    db.add_all([exam, sec]); db.flush()
    topic = models.KnowledgeNode(id="t1", exam_code="CAT", section_id=sec.id, kind="topic", name="T")
    db.add(topic)
    concept = models.KnowledgeNode(id="c1", exam_code="CAT", section_id=sec.id, kind="concept",
                                   name="C", parent_id="t1")
    db.add(concept); db.flush()

    items_true = []
    for i in range(n_items):
        b = round(max(-2.5, min(2.5, rng.gauss(0, 1))), 3)
        items_true.append((f"IT-{i}", b))
        db.add(models.Item(item_id=f"IT-{i}", version=1, content_hash=f"h{i}", exam_code="CAT",
                           section_id=sec.id, concept_node_id="c1", difficulty_d=0, format="mcq",
                           num_options=4, stem=f"q{i}", options=["1", "2", "3", "4"],
                           correct_answer="1", status="approved"))
    db.flush()

    abilities = [rng.gauss(0, 1) for _ in range(n_learners)]
    for li in range(n_learners):
        learner = models.Account(email=f"cal{li}@t.com", display_name=f"l{li}")
        db.add(learner); db.flush()
        for iid, b in items_true:
            p = irt.prob_3pl(abilities[li], 1.0, b, 0.25)
            db.add(models.Response(learner_id=learner.id, item_id=iid, item_version=1,
                                   context="full_mock", correct=(rng.random() < p),
                                   difficulty_d=0, exam_code="CAT", section_id=sec.id))
    db.flush()
    return dict(items_true=items_true)


def test_calibration_writes_versioned_params_and_recovers_order():
    db = _session()
    info = _seed_cold_exam(db, n_learners=150, n_items=20, seed=5)
    run = calibration.run_calibration(db, "CAT")

    assert run.status == "complete"
    assert run.n_items == 20 and run.n_learners == 150
    rows = db.scalars(select(models.IrtParameter).where(models.IrtParameter.run_id == run.id)).all()
    assert len(rows) == 20
    assert all(r.active for r in rows)                 # this run is live
    assert all(r.phase == "b" for r in rows)           # 150 resp/item -> b-only phase

    # active params mirrored onto the item
    item0 = db.get(models.Item, "IT-0")
    assert item0.irt_b is not None and item0.version == 2

    # recovered b correlates with true b
    est = {r.item_id: r.b for r in rows}
    true = dict(info["items_true"])
    xs = [true[i] for i in est]; ys = [est[i] for i in est]
    corr = sim_harness._pearson(xs, ys)
    assert corr is not None and corr >= 0.7


def test_recalibration_supersedes_previous_active():
    db = _session()
    _seed_cold_exam(db, n_learners=120, n_items=10, seed=9)
    run1 = calibration.run_calibration(db, "CAT")
    run2 = calibration.run_calibration(db, "CAT")

    runs = db.scalars(select(models.CalibrationRun)).all()
    assert len(runs) == 2 and run1.id != run2.id
    # exactly one active row per item, and it belongs to run2
    for iid in [f"IT-{i}" for i in range(10)]:
        active = db.scalars(select(models.IrtParameter)
                            .where(models.IrtParameter.item_id == iid,
                                   models.IrtParameter.active.is_(True))).all()
        assert len(active) == 1 and active[0].run_id == run2.id


def test_calibration_no_cold_responses_fails_gracefully():
    db = _session()
    db.add(models.Exam(code="CAT", name="CAT")); db.flush()
    run = calibration.run_calibration(db, "CAT")
    assert run.status == "failed" and run.activated is False


# ---------------- ability scoring ----------------
def test_ability_score_eap_on_mock_session():
    db = _session()
    _seed_cold_exam(db, n_learners=10, n_items=20, seed=2)
    calibration.run_calibration(db, "CAT")
    learner = db.scalar(select(models.Account))
    out = ability_svc.score(db, learner, "CAT", scope="full_mock", method="eap")
    assert -3.0 <= out["theta"] <= 3.0
    assert out["n_items"] > 0
    assert out["se"] is None or out["se"] > 0
    # persisted
    assert ability_svc.latest_ability(db, learner, "CAT") is not None
