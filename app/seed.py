"""Idempotent seed: a slice of CAT/QA mirroring the dashboard, so the app demos end-to-end."""
from __future__ import annotations

from sqlalchemy import select

from . import models
from .db import SessionLocal

AVG_SIMPLE_THEORY = {
    "lines": [
        "The average (arithmetic mean) is the total divided by how many values there are.",
        "average = (sum of all values) / (number of values)",
        "For 4, 8, 6 -> (4 + 8 + 6) / 3 = 6.",
    ]
}


def seed_if_empty() -> None:
    db = SessionLocal()
    try:
        if db.scalar(select(models.Exam).limit(1)) is not None:
            return  # already seeded

        cat = models.Exam(code="CAT", name="Common Admission Test")
        db.add(cat)

        qa = models.Section(exam_code="CAT", key="QA", name="Quantitative Ability")
        varc = models.Section(exam_code="CAT", key="VARC", name="Verbal Ability & Reading Comprehension")
        dilr = models.Section(exam_code="CAT", key="DILR", name="Data Interpretation & Logical Reasoning")
        db.add_all([qa, varc, dilr])
        db.flush()

        # topics + concepts (the tree)
        averages = models.KnowledgeNode(id="averages", exam_code="CAT", section_id=qa.id,
                                        kind="topic", name="Averages")
        ratio = models.KnowledgeNode(id="ratio", exam_code="CAT", section_id=qa.id,
                                     kind="topic", name="Ratio & Proportion")
        mixtures = models.KnowledgeNode(id="mixtures", exam_code="CAT", section_id=qa.id,
                                        kind="topic", name="Mixtures & Alligations")
        db.add_all([averages, ratio, mixtures])
        db.flush()

        concepts = [
            models.KnowledgeNode(id="avg-simple", exam_code="CAT", section_id=qa.id, kind="concept",
                                 name="Simple Averages", parent_id="averages", theory=AVG_SIMPLE_THEORY),
            models.KnowledgeNode(id="avg-weighted", exam_code="CAT", section_id=qa.id, kind="concept",
                                 name="Weighted Averages", parent_id="averages"),
            models.KnowledgeNode(id="ratio-basic", exam_code="CAT", section_id=qa.id, kind="concept",
                                 name="Basic Ratios", parent_id="ratio"),
            models.KnowledgeNode(id="mix-basic", exam_code="CAT", section_id=qa.id, kind="concept",
                                 name="Basic Mixtures", parent_id="mixtures"),
        ]
        db.add_all(concepts)
        db.flush()  # parents before children: the items below FK-reference these concepts (Postgres enforces it)

        # a prerequisite edge (the DAG): Mixtures requires Ratio
        db.add(models.PrereqEdge(node_id="mixtures", prereq_node_id="ratio"))

        # a few seed items in the shared bank
        items = [
            models.Item(item_id="seed-avg-1", version=1, content_hash="seed1", exam_code="CAT",
                        section_id=qa.id, concept_node_id="avg-simple", difficulty_d=-1, format="mcq",
                        num_options=4, stem="Find the mean of 7, 3, 5, 9.",
                        options=["6", "7", "5", "8"], correct_answer="6",
                        solution="(7+3+5+9)/4 = 24/4 = 6.", time_benchmark_s=55,
                        provenance={"source": "seed"}),
            models.Item(item_id="seed-avg-2", version=1, content_hash="seed2", exam_code="CAT",
                        section_id=qa.id, concept_node_id="avg-simple", difficulty_d=0, format="mcq",
                        num_options=4, stem="The mean of 10, 20, 30, 40 is?",
                        options=["25", "20", "30", "27.5"], correct_answer="25",
                        solution="(10+20+30+40)/4 = 100/4 = 25.", time_benchmark_s=80,
                        provenance={"source": "seed"}),
            models.Item(item_id="seed-avgw-1", version=1, content_hash="seed3", exam_code="CAT",
                        section_id=qa.id, concept_node_id="avg-weighted", difficulty_d=0, format="tita",
                        num_options=0, stem="20 boys average 60, 30 girls average 70. Overall average?",
                        correct_answer="66", solution="(20*60+30*70)/50 = 3300/50 = 66.",
                        time_benchmark_s=80, provenance={"source": "seed"}),
        ]
        # ratio-basic items (mastering these unlocks the Mixtures topic via the prereq edge)
        for i, (stem, opts, ans, sol, d) in enumerate([
            ("Simplify the ratio 12:18.", ["2:3", "3:2", "4:6", "6:9"], "2:3", "Divide both by 6 -> 2:3.", -1),
            ("If a:b = 2:3 and b = 9, find a.", ["6", "4", "9", "12"], "6", "a = (2/3)*9 = 6.", -1),
            ("Divide 50 in the ratio 2:3.", ["20 and 30", "25 and 25", "10 and 40", "15 and 35"],
             "20 and 30", "Parts = 5; 50/5 = 10; 2*10 and 3*10 -> 20 and 30.", 0),
            ("If x:y = 3:4 and y:z = 2:5, find x:z.", ["3:10", "6:20", "3:5", "12:20"], "3:10",
             "x:y:z = 3:4 and 2:5 -> scale y: 6:8 and 8:20 -> 6:8:20 -> x:z = 6:20 = 3:10.", 1),
        ]):
            items.append(models.Item(item_id=f"seed-ratio-{i+1}", version=1, content_hash=f"seedr{i}",
                                     exam_code="CAT", section_id=qa.id, concept_node_id="ratio-basic",
                                     difficulty_d=d, format="mcq", num_options=len(opts), stem=stem,
                                     options=opts, correct_answer=ans, solution=sol,
                                     time_benchmark_s=None, provenance={"source": "seed"}))
        # mix-basic items (only reachable after Mixtures unlocks)
        for i, (stem, opts, ans, sol, d) in enumerate([
            ("5 L of 20% acid is mixed with 5 L of 40% acid. Final concentration?",
             ["30%", "25%", "35%", "60%"], "30%", "(0.2*5 + 0.4*5)/10 = 3/10 = 30%.", 0),
            ("In what ratio mix 10% and 30% solutions to get 25%?",
             ["1:3", "3:1", "1:1", "2:3"], "1:3", "Alligation: (30-25):(25-10) = 5:15 = 1:3.", 1),
        ]):
            items.append(models.Item(item_id=f"seed-mix-{i+1}", version=1, content_hash=f"seedm{i}",
                                     exam_code="CAT", section_id=qa.id, concept_node_id="mix-basic",
                                     difficulty_d=d, format="mcq", num_options=len(opts), stem=stem,
                                     options=opts, correct_answer=ans, solution=sol,
                                     time_benchmark_s=None, provenance={"source": "seed"}))
        db.add_all(items)

        # a demo learner with a CAT entitlement
        demo = models.Account(email="demo@vettalume.test", display_name="Demo")
        db.add(demo)
        db.flush()
        db.add(models.Entitlement(account_id=demo.id, exam_code="CAT"))

        db.commit()
    finally:
        db.close()
