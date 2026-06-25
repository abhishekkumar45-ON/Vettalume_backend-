#!/usr/bin/env python3
"""Interactive MAB session — solve the questions the engine picks for you, and watch it react.

This is the Learning-loop counterpart to scripts/irt_mock.py. The multi-armed bandit chooses the
topic, the concept, and the problem difficulty (via the MAPLE edge). You answer; it tells you whether
you were right, shows your mastery on that concept and the current MAPLE difficulty edge, and
announces the moments that matter: when you MASTER a concept, and when you UNLOCK a new one because
its prerequisite just cleared (ZPD gating). Each run is a fresh learner on a throwaway in-memory DB.

    python scripts/mab_play.py [PATH_TO_BANK.xlsx] [--exam CAT] [--max 40]
    python scripts/mab_play.py ~/Downloads/Vettalume_Question_Bank_by_CAT_GMAT.xlsx --exam GMAT

Answer with the option letter (A/B/C/...).  's' = skip · 'm' = show the map · 'q' = quit.
Run from the repo root with the venv active. Uses the real engine code — no mocks.
"""
import argparse
import os
import sys

os.environ.setdefault("SERVE_ONLY_APPROVED", "false")          # so mostly-draft banks still serve
os.environ["DATABASE_URL"] = "sqlite+pysqlite:///:memory:"      # always a throwaway DB
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openpyxl  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app import models  # noqa: E402
from app.services import learning  # noqa: E402
from app.services.question_bank import import_question_bank  # noqa: E402

LETTERS = "ABCDE"


def load_rows(path: str) -> list[dict]:
    wb = openpyxl.load_workbook(path, data_only=True)
    rows: list[dict] = []
    for ws in wb.worksheets:
        it = ws.iter_rows(values_only=True)
        try:
            header = [str(h).strip() if h is not None else "" for h in next(it)]
        except StopIteration:
            continue
        for r in it:
            if r is None or all(c is None for c in r):
                continue
            row = dict(zip(header, r))
            if not row.get("Exam"):
                row["Exam"] = ws.title
            rows.append(row)
    return rows


def unlocked_concepts(db, learner, exam):
    """Returns ({concept_id: (name, mastery, mastered)} for UNLOCKED concepts, full_map)."""
    m = learning.learning_map(db, learner, exam)
    out = {}
    for t in m["topics"]:
        for c in t.get("concepts", []):
            if not c.get("locked", False):
                out[c["id"]] = (c["name"], c.get("mastery", 0.0), c.get("mastered", False))
    return out, m


def print_map(m):
    print("\n  MASTERY MAP:")
    for t in m["topics"]:
        lk = "x" if t["locked"] else " "
        bar = "#" * round(t["mastery"] * 20)
        flag = "LOCKED" if t["locked"] else ("<- recommended next" if t.get("recommended") else "")
        print(f"   [{lk}] {t['name'][:26]:26} {t['mastery'] * 100:>4.0f}% |{bar:<20}| {flag}")
    print()


def main():
    ap = argparse.ArgumentParser(description="Interactive MAB Learning session over a question bank.")
    ap.add_argument("bank", nargs="?",
                    default=os.path.expanduser("~/Downloads/Vettalume_Question_Bank_by_CAT_GMAT.xlsx"))
    ap.add_argument("--exam", default="CAT")
    ap.add_argument("--max", type=int, default=40)
    args = ap.parse_args()

    if not os.path.exists(args.bank):
        sys.exit(f"bank file not found:\n  {args.bank}\n"
                 f"pass the path explicitly:  python scripts/mab_play.py /path/to/bank.xlsx --exam {args.exam}")

    eng = create_engine("sqlite+pysqlite:///:memory:",
                        connect_args={"check_same_thread": False}, poolclass=StaticPool)
    models.Base.metadata.create_all(eng)
    db = Session(eng, autoflush=False)

    rep = import_question_bank(db, load_rows(args.bank))
    learner = models.Account(email="play@local", display_name="play")
    db.add(learner)
    db.flush()

    print(f"\n=== MAB session — {args.exam} ===")
    print(f"imported {rep['questions']['inserted']} questions | knowledge graph {rep['knowledge_graph']}")
    print("The engine picks each question for you. Answer with the option letter.")
    print("  's' = skip · 'm' = show mastery map · 'q' = quit\n")

    prev_unlocked, _ = unlocked_concepts(db, learner, args.exam)
    n = 0
    while n < args.max:
        nxt = learning.next_step(db, learner, args.exam)
        if nxt.get("status") != "ok":
            print(f"\n-- {nxt.get('status')}: {nxt.get('message', '')}")
            break
        item = db.get(models.Item, nxt["question"]["item_id"])
        opts = item.options or []
        print(f"--- Q{n + 1}   topic: {nxt['topic']['name']}   concept: {nxt['concept']['name']}"
              f"   mode: {nxt['mode']}   difficulty {item.difficulty_d:+d}")
        print(f"  {item.stem}")
        for i, o in enumerate(opts):
            print(f"    {LETTERS[i]}. {o}")
        try:
            ans = input("  Your answer: ").strip().upper()
        except EOFError:
            break

        if ans == "Q":
            break
        if ans == "M":
            _, m = unlocked_concepts(db, learner, args.exam)
            print_map(m)
            continue
        if ans in ("S", ""):
            print("  (skipped)\n")
            continue

        chosen = opts[LETTERS.index(ans)] if (len(ans) == 1 and ans in LETTERS[:len(opts)]) else ans
        out = learning.answer(db, learner, item, answer_given=chosen,
                              response_time_ms=1500, session_id=None)

        mark = "\u2713 correct" if out["correct"] else "\u2717 wrong"
        print(f"  {mark}   (correct answer: {item.correct_answer})")
        sol = (item.solution or "").strip()
        if sol:
            print(f"  solution: {sol}")
        print(f"  -> mastery {out['mastery'] * 100:>4.0f}%   |   MAPLE edge {out['edge']:+.1f}")

        now_unlocked, _ = unlocked_concepts(db, learner, args.exam)
        for cid, (name, mast, mastered) in now_unlocked.items():
            if cid not in prev_unlocked:
                print(f"  \U0001F513 UNLOCKED a new concept: {name}  (its prerequisite just cleared)")
            elif mastered and not prev_unlocked[cid][2]:
                print(f"  \u2B50 MASTERED concept: {name}")
        prev_unlocked = now_unlocked
        print()
        n += 1

    _, m = unlocked_concepts(db, learner, args.exam)
    print_map(m)
    print(f"answered {n} question(s). fresh session each run.")


if __name__ == "__main__":
    main()
