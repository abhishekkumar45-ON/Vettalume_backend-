"""End-to-end demo against a running server.

    python scripts/demo_flow.py            # uses http://localhost:8000
    BASE_URL=http://host:8000 python scripts/demo_flow.py
"""
import os

import httpx

BASE = os.environ.get("BASE_URL", "http://localhost:8000")


def main() -> None:
    with httpx.Client(base_url=BASE, timeout=10) as c:
        print("health:", c.get("/health").json())

        login = c.post("/auth/dev-login", json={"email": "aarav@vettalume.test"}).json()
        lid = login["learner_id"]
        H = {"X-Learner-Id": lid}
        print("learner:", lid)

        item = {
            "item_id": "demo-avg-1", "exam_code": "CAT", "section_key": "QA",
            "concept_node_id": "avg-simple", "difficulty_d": 2, "format": "mcq",
            "options": ["6", "7", "5", "8"], "correct_answer": "6",
            "stem": "Mean of 7, 3, 5, 9?", "solution": "(7+3+5+9)/4 = 6",
            "provenance": {"source": "demo_flow"},
        }
        print("ingest:", c.post("/ingest/items", json=[item]).json())

        nxt = c.get("/practice/next", params={"node_id": "avg-simple"}, headers=H).json()
        print("next  :", nxt)

        ans = c.post("/practice/answer",
                     json={"item_id": nxt["item_id"], "context": "practice",
                           "answer_given": "6", "response_time_ms": 12000},
                     headers=H).json()
        print("answer:", ans)

        st = c.get("/practice/state", params={"exam": "CAT"}, headers=H).json()
        print("state :")
        for n in st["nodes"]:
            print(f"   {n['node_id']:<14} mastery={n['mastery']:<6} attempts={n['attempts']} learned={n['learned']}")


if __name__ == "__main__":
    main()
