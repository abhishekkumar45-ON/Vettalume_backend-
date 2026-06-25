import os

# Point the app at in-memory SQLite BEFORE importing it (config reads env at import time).
os.environ["DATABASE_URL"] = "sqlite+pysqlite:///:memory:"

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402


def test_walking_skeleton_end_to_end():
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"

        # dev login -> learner_id is the bearer token
        login = c.post("/auth/dev-login", json={"email": "a@b.com"}).json()
        lid = login["learner_id"]
        H = {"X-Learner-Id": lid}

        # an admin is required for all content upload (no outsiders). Register one and use its JWT.
        settings.admin_emails = "smoke-admin@vettalume.test"
        areg = c.post("/auth/register", json={
            "email": "smoke-admin@vettalume.test", "password": "adminpass123"}).json()
        AH = {"Authorization": "Bearer " + areg["access_token"]}

        # unauthenticated content upload is rejected (the perimeter)
        assert c.post("/ingest/items", json=[]).status_code == 401

        # seeded tree is present
        tree = c.get("/catalog/tree", params={"exam": "CAT"}).json()
        assert tree["exam"] == "CAT"
        assert any(t["id"] == "averages" for t in tree["topics"])

        # ingest a new item
        item = {
            "item_id": "t-avg-1", "exam_code": "CAT", "section_key": "QA",
            "concept_node_id": "avg-simple", "difficulty_d": 0, "format": "mcq",
            "options": ["6", "7", "8", "9"], "correct_answer": "6",
            "stem": "Mean of 4, 8, 6?", "solution": "(4+8+6)/3 = 6",
        }
        rep = c.post("/ingest/items", json=[item], headers=AH).json()
        assert rep["status"] == "committed" and rep["inserted"] == 1

        # idempotent re-ingest is a no-op
        rep2 = c.post("/ingest/items", json=[item], headers=AH).json()
        assert rep2["unchanged"] == 1 and rep2["inserted"] == 0

        # next item never leaks the answer/solution/difficulty
        nxt = c.get("/practice/next", params={"node_id": "avg-simple"}, headers=H).json()
        assert "correct_answer" not in nxt and "solution" not in nxt and "difficulty_d" not in nxt

        # answer correctly -> graded server-side, mastery updates to the blended value
        ans = c.post("/practice/answer",
                     json={"item_id": "t-avg-1", "context": "practice", "answer_given": "6"},
                     headers=H).json()
        assert ans["correct"] is True
        assert ans["attempts"] >= 1
        assert ans["mastery"] == 0.84   # 0.40*0.6 + 0.30*1 + 0.30*1 (one fresh correct at d=3)

        # state reflects the attempt
        st = c.get("/practice/state", params={"exam": "CAT"}, headers=H).json()
        avg = next(n for n in st["nodes"] if n["node_id"] == "avg-simple")
        assert avg["attempts"] >= 1 and avg["learned"] is True and avg["mastery"] == 0.84

        # rejection path: an unknown concept tag rejects the whole batch
        bad = {**item, "item_id": "t-bad", "concept_node_id": "does-not-exist"}
        rej = c.post("/ingest/items", json=[bad], headers=AH).json()
        assert rej["status"] == "rejected" and rej["errors"]

        # authored-vs-derived boundary: trying to author irt_b is a 422 (extra='forbid')
        sneaky = {**item, "item_id": "t-sneaky", "irt_b": -0.5}
        r = c.post("/ingest/items", json=[sneaky], headers=AH)
        assert r.status_code == 422
