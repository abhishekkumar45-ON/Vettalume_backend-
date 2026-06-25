"""Chapter-analytics tests — drive a tiny approved bank and assert the analysis shape + invariants."""
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app import models
from app.services import analytics, learning
from app.services.question_bank import import_question_bank


def _session():
    eng = create_engine("sqlite+pysqlite:///:memory:",
                        connect_args={"check_same_thread": False}, poolclass=StaticPool)
    models.Base.metadata.create_all(eng)
    return Session(eng, autoflush=False)


def _rows():
    base = {"Exam": "X", "Section": "S", "Topic": "Chapter One", "Source": "t", "Status": "approved"}
    mk = lambda i, sub, d, correct: {
        **base, "Question ID": f"X-{i}", "Subtopic": sub, "Prerequisites": "-",
        "Difficulty (-2 to 2)": d, "Question type": "MCQ", "Question text": f"q{i}?",
        "Option A": "1", "Option B": "2", "Option C": "3", "Option D": "4", "Correct answer": correct,
    }
    return [
        mk(1, "Alpha", 0, "A"), mk(2, "Alpha", 1, "B"),
        mk(3, "Beta", -1, "C"), mk(4, "Beta", 2, "D"),
    ]


def test_chapter_analysis_shape_and_kpis():
    db = _session()
    import_question_bank(db, _rows())
    learner = models.Account(email="ana@t.com", display_name="ana")
    db.add(learner); db.flush()

    # drive a few steps, answering correctly
    for _ in range(4):
        nxt = learning.next_step(db, learner, "X")
        if nxt.get("status") != "ok":
            break
        item = db.get(models.Item, nxt["question"]["item_id"])
        learning.answer(db, learner, item, answer_given=item.correct_answer,
                        response_time_ms=30_000, session_id=None)

    node = analytics.resolve_chapter(db, "X", "Chapter One", None)
    assert node is not None
    a = analytics.chapter_analysis(db, learner, node)

    assert a["chapter"]["name"] == "Chapter One" and a["chapter"]["section"] == "S"
    assert a["kpis"]["concepts_total"] == 2
    assert a["kpis"]["questions_answered"] >= 1
    assert 0.0 <= a["kpis"]["overall_accuracy"] <= 1.0
    assert len(a["difficulty_spread"]) == 5            # always D1..D5
    assert len(a["subtopics"]) == 2
    # strongest is sorted high->low, weakest low->high
    s = [x["mastery"] for x in a["strongest"]]
    w = [x["mastery"] for x in a["weakest"]]
    assert s == sorted(s, reverse=True)
    assert w == sorted(w)
    # answered count across bands matches total questions answered
    assert sum(b["answered"] for b in a["difficulty_spread"]) == a["kpis"]["questions_answered"]


def test_resolve_chapter_unknown_returns_none():
    db = _session()
    import_question_bank(db, _rows())
    assert analytics.resolve_chapter(db, "X", "Nope", None) is None
