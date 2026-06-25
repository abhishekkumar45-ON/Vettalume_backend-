"""Watch the Learning engine drive one learner through CAT Quant.

    python scripts/learning_sim.py            # against http://localhost:8000
    BASE_URL=http://host:8000 python scripts/learning_sim.py

Answers the seeded questions correctly so you can see mastery climb, the MAPLE edge adapt,
topics get mastered, and a locked topic (Mixtures) open once its prerequisite (Ratio) is mastered.
"""
import os
import uuid

import httpx

BASE = os.environ.get("BASE_URL", "http://localhost:8000")

# Correct answers for the seeded items (the API never exposes these; the sim knows the seed).
SEED_ANSWERS = {
    "seed-avg-1": "6", "seed-avg-2": "25", "seed-avgw-1": "66",
    "seed-ratio-1": "2:3", "seed-ratio-2": "6", "seed-ratio-3": "20 and 30", "seed-ratio-4": "3:10",
    "seed-mix-1": "30%", "seed-mix-2": "1:3",
}


def locked_set(client, headers):
    m = client.get("/learn/map", params={"exam": "CAT", "section": "QA"}, headers=headers).json()
    return {t["id"] for t in m["topics"] if t["locked"]}, m


def main():
    with httpx.Client(base_url=BASE, timeout=10) as c:
        lid = c.post("/auth/dev-login",
                     json={"email": f"sim-{uuid.uuid4().hex[:8]}@vettalume.test"}).json()["learner_id"]
        H = {"X-Learner-Id": lid}

        print(f"learner {lid}\n")
        locked, _ = locked_set(c, H)
        print(f"initially locked topics: {sorted(locked) or '(none)'}\n")

        for step in range(1, 31):
            nxt = c.get("/learn/next", params={"exam": "CAT", "section": "QA"}, headers=H).json()
            if nxt.get("status") != "ok":
                print(f"\n[{step}] engine says: {nxt.get('message', nxt.get('status'))}")
                break

            item_id = nxt["question"]["item_id"]
            ans = SEED_ANSWERS.get(item_id, "")
            tag = "LEARN " if nxt["mode"] == "learn" else "revise"
            print(f"[{step:>2}] {tag} {nxt['topic']['name']:<22} :: {nxt['concept']['name']:<16} "
                  f"q={item_id:<13} ", end="")

            out = c.post("/learn/answer",
                         json={"item_id": item_id, "answer_given": ans, "response_time_ms": 9000},
                         headers=H).json()
            flag = "OK " if out["correct"] else "x  "
            b = out["breakdown"]
            print(f"{flag} mastery={out['mastery']:.2f} "
                  f"(P={b['P']:.2f} D={b['D']:.2f} M={b['M']:.2f}) edge={out['edge']:.1f}"
                  f"{'  <-- MASTERED' if out['mastered'] else ''}")

            # did anything just unlock?
            now_locked, _ = locked_set(c, H)
            opened = locked - now_locked
            if opened:
                print(f"      >>> UNLOCKED: {sorted(opened)}  (prerequisite mastered)")
            locked = now_locked

        print("\nfinal map:")
        _, m = locked_set(c, H)
        for t in m["topics"]:
            state = "LOCKED" if t["locked"] else ("MASTERED" if t["mastery"] >= 0.74 else "in progress")
            star = " *recommended" if t["recommended"] else ""
            print(f"  {t['name']:<24} mastery={t['mastery']:.2f}  [{state}]{star}")


if __name__ == "__main__":
    main()
