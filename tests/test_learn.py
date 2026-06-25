import os
import uuid

os.environ["DATABASE_URL"] = "sqlite+pysqlite:///:memory:"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


def _login(c):
    # Unique learner per test: the in-memory DB is shared across tests in one process, so
    # scoping by a fresh learner keeps each test's responses isolated.
    email = f"{uuid.uuid4().hex[:10]}@test.com"
    lid = c.post("/auth/dev-login", json={"email": email}).json()["learner_id"]
    return {"X-Learner-Id": lid}


def test_next_returns_learn_with_theory():
    with TestClient(app) as c:
        H = _login(c)
        nxt = c.get("/learn/next", params={"exam": "CAT"}, headers=H).json()
        assert nxt["status"] == "ok"
        assert nxt["mode"] == "learn"
        assert "question" in nxt and "correct_answer" not in nxt["question"]
        # first time on a concept -> teaching content is included
        assert "theory" in nxt


def test_answer_updates_mastery_and_breakdown():
    with TestClient(app) as c:
        H = _login(c)
        nxt = c.get("/learn/next", params={"exam": "CAT"}, headers=H).json()
        item_id = nxt["question"]["item_id"]
        # answer correctly using the seeded correct answer for avg items
        # avg-simple seed items: seed-avg-1 -> "6", seed-avg-2 -> "25"
        correct = {"seed-avg-1": "6", "seed-avg-2": "25"}[item_id]
        out = c.post("/learn/answer", json={"item_id": item_id, "answer_given": correct}, headers=H).json()
        assert out["correct"] is True
        assert set(out["breakdown"].keys()) == {"P", "D", "M"}
        assert out["mastery"] > 0
        assert isinstance(out["mastered"], bool)   # mastery now needs evidence across the difficulty ladder, not one correct


def test_zpd_unlock_mixtures_after_mastering_ratio():
    with TestClient(app) as c:
        H = _login(c)

        # Mixtures starts LOCKED (prereq: Ratio mastery >= 0.74)
        m0 = c.get("/learn/map", params={"exam": "CAT"}, headers=H).json()
        mix0 = next(t for t in m0["topics"] if t["id"] == "mixtures")
        assert mix0["locked"] is True

        # Master ratio-basic by answering a ratio item correctly
        ratio_answers = {
            "seed-ratio-1": "2:3", "seed-ratio-2": "6",
            "seed-ratio-3": "20 and 30", "seed-ratio-4": "3:10",
        }
        # submit one correct ratio answer (one fresh correct masters the single-concept Ratio topic)
        c.post("/learn/answer", json={"item_id": "seed-ratio-1", "answer_given": "2:3"}, headers=H).json()

        # Now Mixtures should be UNLOCKED
        m1 = c.get("/learn/map", params={"exam": "CAT"}, headers=H).json()
        mix1 = next(t for t in m1["topics"] if t["id"] == "mixtures")
        ratio1 = next(t for t in m1["topics"] if t["id"] == "ratio")
        assert ratio1["mastery"] >= 0.74
        assert mix1["locked"] is False


def test_concept_detail_endpoint():
    with TestClient(app) as c:
        H = _login(c)
        c.post("/learn/answer", json={"item_id": "seed-avg-1", "answer_given": "6"}, headers=H)
        d = c.get("/learn/concept/avg-simple", headers=H).json()
        assert d["concept_id"] == "avg-simple"
        assert d["attempts"] == 1 and d["learned"] is True
        assert "P" in d["breakdown"]
