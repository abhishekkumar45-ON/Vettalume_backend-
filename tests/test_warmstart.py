"""Phase 6 — cross-exam warm-start (AC-02) and mounting the GMAT/GRE lines."""
import itertools

from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app import models
from app.services import billing, mount, warmstart


def _session():
    eng = create_engine("sqlite+pysqlite:///:memory:",
                        connect_args={"check_same_thread": False}, poolclass=StaticPool)
    models.Base.metadata.create_all(eng)
    db = Session(eng, autoflush=False)
    billing.ensure_catalog(db)
    mount.mount_gmat_gre_if_empty(db)
    return db


def _cat_with_cold(db, *, quant=None, rc=None):
    """Seed CAT with QA(quant)/VARC(rc) sections and optional cold (mock) responses for one learner."""
    db.add(models.Exam(code="CAT", name="CAT")); db.flush()
    secs = {}
    for k, n in [("QA", "Quant"), ("VARC", "Verbal Ability & RC")]:
        s = models.Section(exam_code="CAT", key=k, name=n); db.add(s); db.flush(); secs[k] = s
        db.add(models.KnowledgeNode(id=f"cat-{k.lower()}", exam_code="CAT", section_id=s.id,
                                    kind="concept", name=n)); db.flush()
        for i in range(8):
            db.add(models.Item(item_id=f"cat-{k}-{i}", version=1, content_hash=f"c{k}{i}",
                               exam_code="CAT", section_id=s.id, concept_node_id=f"cat-{k.lower()}",
                               difficulty_d=0, format="mcq", num_options=4, stem="q",
                               options=["A", "B", "C", "D"], correct_answer="A", status="approved"))
    db.flush()
    a = models.Account(email="poly@x.com", display_name="poly"); db.add(a); db.flush(); db.commit()

    def cold(k, pattern):
        its = db.scalars(select(models.Item).where(models.Item.section_id == secs[k].id)).all()
        for it, ok in zip(its, itertools.cycle(pattern)):
            db.add(models.Response(learner_id=a.id, item_id=it.item_id, item_version=1,
                                   context="full_mock", correct=ok, answer_given=("A" if ok else "B"),
                                   response_time_ms=45000, attempt_number=1, hints_used=0,
                                   difficulty_d=0, exam_code="CAT", section_id=secs[k].id,
                                   session_id="cat-mock"))
    if quant is not None:
        cold("QA", quant)
    if rc is not None:
        cold("VARC", rc)
    db.commit()
    return a


# ---------------- construct inference ----------------
def test_construct_of_maps_sections():
    db = _session()
    by_key = {s.key: warmstart.construct_of(s)
              for s in db.scalars(select(models.Section).where(
                  models.Section.exam_code == "GMAT")).all()}
    assert by_key == {"QA": "quant", "VA": "rc", "DI": "data_logic"}


# ---------------- transfer ----------------
def test_warm_start_transfers_provisional_priors_with_inflated_se():
    db = _session()
    a = _cat_with_cold(db, quant=[True, True, True, True], rc=[True, False, True, False])
    out = warmstart.warm_start(db, a, "GMAT", persist=True)
    assert out["warm_started"] is True
    constructs = {t["construct"] for t in out["transferred"]}
    assert {"quant", "rc"} <= constructs
    for t in out["transferred"]:
        assert t["prior"]["is_claim"] is False           # a transfer is never a claim
        assert t["prior"]["basis"] == "provisional"
        if t["source_se"] is not None:
            assert t["prior"]["se"] > t["source_se"]      # SE inflated for crossing exams


def test_warm_start_persists_transfer_estimates():
    db = _session()
    a = _cat_with_cold(db, quant=[True, True, True, False])
    warmstart.warm_start(db, a, "GMAT", persist=True)
    rows = db.scalars(select(models.AbilityEstimate).where(
        models.AbilityEstimate.learner_id == a.id,
        models.AbilityEstimate.exam_code == "GMAT")).all()
    assert any(r.scope == "warm_start:quant" and r.method == "transfer" for r in rows)


def test_cold_start_when_no_shared_signals():
    db = _session()
    a = models.Account(email="cold@x.com", display_name="cold"); db.add(a); db.flush(); db.commit()
    out = warmstart.warm_start(db, a, "GMAT", persist=True)
    assert out["warm_started"] is False and out["transferred"] == []


def test_grant_free_auto_warm_starts_the_course():
    db = _session()
    a = _cat_with_cold(db, quant=[True, True, True, True])
    billing.grant_free_tier(db, a, "GMAT")               # should auto-warm from CAT quant
    rows = db.scalars(select(models.AbilityEstimate).where(
        models.AbilityEstimate.learner_id == a.id,
        models.AbilityEstimate.exam_code == "GMAT")).all()
    assert any(r.scope == "warm_start:quant" for r in rows)


# ---------------- mounting GMAT/GRE ----------------
def test_mount_creates_gmat_gre_and_is_idempotent():
    db = _session()                                       # already mounts once
    assert db.get(models.Exam, "GMAT") is not None and db.get(models.Exam, "GRE") is not None
    again = mount.mount_gmat_gre_if_empty(db)
    assert again["mounted"] == []                         # nothing to mount the second time
    gmat_items = db.scalar(select(func.count(models.Item.item_id)).where(
        models.Item.exam_code == "GMAT"))
    assert gmat_items > 0
    # all mounted GMAT items are approved so mocks can serve them
    approved = db.scalar(select(func.count(models.Item.item_id)).where(
        models.Item.exam_code == "GMAT", models.Item.status == "approved"))
    assert approved == gmat_items
