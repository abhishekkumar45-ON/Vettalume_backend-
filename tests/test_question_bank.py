"""Question-bank importer tests — uses an isolated in-memory DB per test (no seed)."""
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app import models
from app.services import knowledge_graph as kg
from app.services.question_bank import import_question_bank
from app.services.state import record_response


def _rows():
    base = {"Exam": "CAT", "Section": "QA", "Topic": "Averages", "Question format": "MCQ",
            "Passage / Set ID": "-", "Source": "test", "Status": "approved"}
    return [
        {**base, "Question ID": "AVQ-001", "Subtopic": "Simple Averages", "Prerequisites": "-",
         "Difficulty (-2 to 2)": -1, "Question text": "Average of 7, 3, 5, 9?",
         "Option A": "5", "Option B": "6", "Option C": "7", "Option D": "8",
         "Correct answer": "B", "Solution / explanation": "24/4 = 6",
         "Expected time (sec)": 55, "IRT b": -1},
        {**base, "Question ID": "AVQ-002", "Subtopic": "Weighted Averages",
         "Prerequisites": "Simple Averages", "Difficulty (-2 to 2)": 0,
         "Question text": "3 students @80 and 2 @90, overall?",
         "Option A": "84", "Option B": "85", "Option C": "86", "Option D": "88",
         "Correct answer": "A", "Solution / explanation": "420/5 = 84",
         "Expected time (sec)": 80, "IRT b": 0},
        {**base, "Question ID": "AVQ-003", "Subtopic": "Average Speed",
         "Prerequisites": "Simple Averages", "Difficulty (-2 to 2)": 1,
         "Question text": "60 there, 40 back, average speed?",
         "Option A": "48", "Option B": "50", "Option C": "52", "Option D": "45",
         "Correct answer": "A", "Solution / explanation": "4800/100 = 48",
         "Expected time (sec)": 110, "IRT b": 1},
    ]


def _fresh_session():
    eng = create_engine("sqlite+pysqlite:///:memory:",
                        connect_args={"check_same_thread": False}, poolclass=StaticPool)
    models.Base.metadata.create_all(eng)
    return Session(eng)


def test_importer_builds_kg_and_questions():
    db = _fresh_session()
    rep = import_question_bank(db, _rows())
    assert rep["status"] == "committed"
    assert rep["knowledge_graph"]["topics_created"] == 1        # Averages
    assert rep["knowledge_graph"]["concepts_created"] == 3      # Simple, Weighted, Average Speed
    assert rep["knowledge_graph"]["prereq_edges_created"] == 2  # both require Simple Averages
    assert rep["questions"]["inserted"] == 3

    item = db.get(models.Item, "AVQ-001")
    assert item.correct_answer == "6"          # resolved from letter "B"
    assert item.difficulty_d == -1 and item.status == "approved"
    assert item.irt_b is None                  # derived field NOT authored on import


def test_importer_concept_locking_releases_on_mastery():
    db = _fresh_session()
    import_question_bank(db, _rows())
    learner = models.Account(email="x@test.com", display_name="x")
    db.add(learner)
    db.flush()

    weighted = db.scalar(select(models.KnowledgeNode)
                         .where(models.KnowledgeNode.name == "Weighted Averages"))
    # locked while its prerequisite (Simple Averages) is unmastered
    assert kg.is_concept_locked(db, learner.id, weighted) is True

    # one correct answer on a Simple Averages item -> mastery crosses H
    simple_item = db.get(models.Item, "AVQ-001")
    record_response(db, learner, simple_item, context="practice", answer_given="6",
                    correct=None, response_time_ms=1000, attempt_number=1,
                    hints_used=0, session_id=None)
    db.flush()
    assert kg.is_concept_locked(db, learner.id, weighted) is False


def _fresh_session_noautoflush():
    """Mirror the app's SessionLocal (autoflush=False) — needed to catch the in-batch
    duplicate-edge bug that autoflush would otherwise paper over."""
    eng = create_engine("sqlite+pysqlite:///:memory:",
                        connect_args={"check_same_thread": False}, poolclass=StaticPool)
    models.Base.metadata.create_all(eng)
    return Session(eng, autoflush=False)


def _rich_rows():
    """Covers the by-exam workbook's tricky shapes: 5-option MCQ, multi-answer, a sequence
    answer, the rich 'Question type' archetype, and a duplicate prerequisite edge."""
    g = {"Exam": "GMAT", "Section": "Quant", "Topic": "Arithmetic", "Source": "t", "Status": "approved"}
    v = {"Exam": "GRE", "Section": "Verbal", "Topic": "Vocab", "Source": "t", "Status": "approved"}
    c = {"Exam": "CAT", "Section": "VARC", "Topic": "Para Jumble", "Source": "t", "Status": "approved"}
    return [
        # 5-option MCQ, correct is the 5th letter
        {**g, "Question ID": "G-1", "Subtopic": "Percentages", "Prerequisites": "-",
         "Difficulty (-2 to 2)": 0, "Question type": "Problem Solving", "Question text": "5-option?",
         "Option A": "1", "Option B": "2", "Option C": "3", "Option D": "4", "Option E": "5",
         "Correct answer": "E"},
        # two questions on the SAME subtopic, both citing the SAME prereq -> one edge only
        {**g, "Question ID": "G-2", "Subtopic": "Ratios", "Prerequisites": "Percentages",
         "Difficulty (-2 to 2)": 1, "Question type": "Problem Solving", "Question text": "ratio q1?",
         "Option A": "1", "Option B": "2", "Option C": "3", "Option D": "4", "Correct answer": "A"},
        {**g, "Question ID": "G-3", "Subtopic": "Ratios", "Prerequisites": "Percentages",
         "Difficulty (-2 to 2)": 1, "Question type": "Problem Solving", "Question text": "ratio q2?",
         "Option A": "1", "Option B": "2", "Option C": "3", "Option D": "4", "Correct answer": "B"},
        # multi-answer -> stored as tita with resolved option VALUES
        {**v, "Question ID": "V-1", "Subtopic": "Synonyms", "Prerequisites": "-",
         "Difficulty (-2 to 2)": 1, "Question type": "Sentence Equivalence", "Question text": "two words?",
         "Option A": "scathing", "Option B": "kind", "Option C": "caustic", "Option D": "warm",
         "Correct answer": "A, C"},
        # sequence answer, no options -> tita with the literal
        {**c, "Question ID": "C-1", "Subtopic": "Ordering", "Prerequisites": "-",
         "Difficulty (-2 to 2)": 1, "Question type": "Parajumble",
         "Question text": "order S,P,R,Q", "Correct answer": "SPRQ"},
    ]


def test_importer_handles_rich_bank_and_dedupes_edges():
    db = _fresh_session_noautoflush()
    rep = import_question_bank(db, _rich_rows())
    assert rep["status"] == "committed"
    assert rep["questions"]["inserted"] == 5
    # the duplicate prereq edge (Ratios -> Percentages, cited twice) collapses to ONE
    assert rep["knowledge_graph"]["prereq_edges_created"] == 1

    g1 = db.get(models.Item, "G-1")
    assert g1.format == "mcq" and len(g1.options) == 5 and g1.correct_answer == "5"
    assert g1.archetype_id == "Problem Solving"

    v1 = db.get(models.Item, "V-1")           # multi-answer -> tita, resolved to values
    assert v1.format == "tita" and v1.correct_answer == "scathing, caustic"

    c1 = db.get(models.Item, "C-1")           # sequence -> tita, literal kept
    assert c1.format == "tita" and c1.correct_answer == "SPRQ"


def test_draft_items_are_not_served_by_default():
    from app.services.state import eligible_items
    rows = _rich_rows()
    rows[0]["Status"] = "draft"               # G-1 becomes a draft
    db = _fresh_session_noautoflush()
    import_question_bank(db, rows)
    learner = models.Account(email="d@d.com", display_name="d"); db.add(learner); db.flush()
    g1 = db.get(models.Item, "G-1")
    elig = eligible_items(db, learner.id, context="practice", concept_node_id=g1.concept_node_id)
    assert g1 not in elig                      # approved-only gate hides the draft
