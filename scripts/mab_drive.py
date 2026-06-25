"""Deterministic MAB drive — watch the engine without hand-typing answers.

Imports a question-bank workbook into a throwaway in-memory DB and walks a single learner through
the Learning loop. Because it reads each served item's correct answer straight from the DB, it can
answer with a *controllable* accuracy, so you can drive the bandits, mastery, MAPLE edge, and ZPD
unlocking deterministically.

    python scripts/mab_drive.py /path/to/Vettalume_Question_Bank_by_Exam.xlsx
    python scripts/mab_drive.py bank.xlsx --exam GMAT --steps 40 --accuracy 1.0
    python scripts/mab_drive.py bank.xlsx --exam CAT  --steps 60 --accuracy 0.7 --seed 3

--accuracy is the probability the simulated learner answers correctly (1.0 = always right).
Run from the repo root with the venv active. Uses the real engine code — no mocks.
"""
from __future__ import annotations

import argparse
import os
import random
import sys

# make `app` importable when run as `python scripts/mab_drive.py` from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openpyxl  # noqa: E402


def load_rows(path: str) -> list[dict]:
    """Same sheet-walking the upload endpoint does: tab name = exam when there's no Exam column."""
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Drive the Vettalume MAB over a question bank.")
    ap.add_argument("xlsx", help="path to the question-bank .xlsx")
    ap.add_argument("--exam", default="CAT")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--accuracy", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--include-drafts", action="store_true",
                    help="serve draft items too (otherwise only 'approved' are used)")
    args = ap.parse_args()
    random.seed(args.seed)

    # must be set BEFORE importing app.config (engine + settings are built at import time).
    # this script always runs on its own throwaway in-memory DB, so force SQLite regardless of
    # whatever DATABASE_URL the shell has — that's why it never needs psycopg2/Postgres.
    os.environ["DATABASE_URL"] = "sqlite+pysqlite:///:memory:"
    if args.include_drafts:
        os.environ["SERVE_ONLY_APPROVED"] = "false"

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from sqlalchemy.pool import StaticPool

    from app import models
    from app.services import learning
    from app.services.question_bank import import_question_bank

    eng = create_engine("sqlite+pysqlite:///:memory:",
                        connect_args={"check_same_thread": False}, poolclass=StaticPool)
    models.Base.metadata.create_all(eng)
    db = Session(eng)

    rep = import_question_bank(db, load_rows(args.xlsx))
    print(f"imported {rep['questions']['inserted']} questions | KG {rep['knowledge_graph']} "
          f"| {len(rep['warnings'])} prereq warnings"
          + ("  | serving DRAFTS too" if args.include_drafts else "  | approved-only"))
    print(f"driving '{args.exam}' for {args.steps} steps at accuracy={args.accuracy} (seed {args.seed})\n")

    learner = models.Account(email="mab@drive.local", display_name="mab")
    db.add(learner)
    db.flush()

    hdr = f"{'#':>3}  {'topic':22} {'concept':24} {'mode':6} {'d':>2} {'ok':>3}  {'mast':>5} {'edge':>5}"
    print(hdr)
    print("-" * len(hdr))
    for step in range(1, args.steps + 1):
        nxt = learning.next_step(db, learner, args.exam)
        if nxt.get("status") != "ok":
            print(f"{step:>3}  -- {nxt.get('status')}: {nxt.get('message', '')}")
            break
        item = db.get(models.Item, nxt["question"]["item_id"])
        correct_choice = random.random() < args.accuracy
        given = item.correct_answer if correct_choice else "__deliberately_wrong__"
        out = learning.answer(db, learner, item, answer_given=given,
                              response_time_ms=1500, session_id=None)
        mark = "✓" if out["correct"] else "✗"
        print(f"{step:>3}  {nxt['topic']['name'][:22]:22} {nxt['concept']['name'][:24]:24} "
              f"{nxt['mode']:6} {item.difficulty_d:>+2} {mark:>3}  "
              f"{out['mastery'] * 100:>4.0f}% {out['edge']:>+5.1f}")

    print("\nFINAL MAP:")
    m = learning.learning_map(db, learner, args.exam)
    for t in m["topics"]:
        flag = "LOCKED" if t["locked"] else ("<- recommended next" if t["recommended"] else "")
        bar = "#" * round(t["mastery"] * 20)
        print(f"  [{'x' if t['locked'] else ' '}] {t['name'][:24]:24} {t['mastery'] * 100:>4.0f}% "
              f"|{bar:<20}| {flag}")


if __name__ == "__main__":
    main()
