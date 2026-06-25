"""Phase 5 — Honest Perimeter: ranges, provisional/claim gating, and the published accuracy record."""
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app import models
from app.services import honesty


def _session():
    eng = create_engine("sqlite+pysqlite:///:memory:",
                        connect_args={"check_same_thread": False}, poolclass=StaticPool)
    models.Base.metadata.create_all(eng)
    return Session(eng, autoflush=False)


def _acct(db):
    a = models.Account(email="h@t.com", display_name="h"); db.add(a); db.flush(); db.commit()
    return a


def test_perimeter_band_is_point_plus_minus_z_se():
    env = honesty.perimeter(0.5, 0.3, kind="ability", basis_n=10)
    assert env["band_95"] == [round(0.5 - honesty.Z95 * 0.3, 3), round(0.5 + honesty.Z95 * 0.3, 3)]
    assert env["confidence"] == 0.95


def test_low_basis_is_provisional_not_a_claim():
    prov = honesty.perimeter(0.5, 0.3, basis_n=honesty.MIN_BASIS_N - 1)
    assert prov["is_claim"] is False and prov["basis"] == "provisional"
    claim = honesty.perimeter(0.5, 0.3, basis_n=honesty.MIN_BASIS_N)
    assert claim["is_claim"] is True and claim["basis"] == "calibrated"


def test_missing_se_yields_no_band_and_provisional():
    env = honesty.perimeter(0.5, None, basis_n=100)
    assert env["band_95"] is None and env["is_claim"] is False


def test_perimeter_band_sorts_endpoints():
    env = honesty.perimeter_band(70.0, 86.0, 46.0, kind="percentile", basis_n=10)
    assert env["band_95"] == [46.0, 86.0]


def test_record_prediction_then_outcome_sets_within_band():
    db = _session(); a = _acct(db)
    pid = honesty.record_prediction(db, a, "CAT",
                                    honesty.perimeter(0.0, 0.3, kind="percentile", basis_n=10))
    inside = honesty.record_outcome(db, pid, 0.2)
    assert inside["within_band"] is True
    pid2 = honesty.record_prediction(db, a, "CAT",
                                     honesty.perimeter(0.0, 0.3, kind="percentile", basis_n=10))
    outside = honesty.record_outcome(db, pid2, 2.0)
    assert outside["within_band"] is False


def test_accuracy_record_coverage_and_publish_gate():
    db = _session(); a = _acct(db)
    # 3 inside, 1 outside -> coverage 0.75; below PUBLISH_MIN_N so not yet published
    for actual, _inside in [(0.1, True), (-0.1, True), (0.3, True), (5.0, False)]:
        pid = honesty.record_prediction(db, a, "CAT",
                                        honesty.perimeter(0.0, 0.3, kind="ability", basis_n=10))
        honesty.record_outcome(db, pid, actual)
    rec = honesty.accuracy_record(db, a, "CAT")
    assert rec["resolved"] == 4
    assert rec["band_coverage_95"] == 0.75
    assert rec["published"] is False


def test_accuracy_record_empty_when_no_outcomes():
    db = _session(); a = _acct(db)
    honesty.record_prediction(db, a, "CAT", honesty.perimeter(0.0, 0.3, basis_n=10))
    rec = honesty.accuracy_record(db, a, "CAT")
    assert rec["resolved"] == 0 and rec["published"] is False
