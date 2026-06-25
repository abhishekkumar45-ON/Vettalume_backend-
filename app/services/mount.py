"""Mount the GMAT and GRE course lines (Phase 6).

The platform was validated on GMAT and GRE, which already assume the shared chassis. This seeds their
exam structures — sections with construct-mappable keys (QA -> quant, VA -> rc, DI -> data_logic),
a few concepts each, and a small set of approved items — so mocks, diagnosis, billing, and cross-exam
warm-start work across all three courses, not just the CAT demo. Idempotent.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models

# exam -> [(section_key, section_name, [(concept_id, concept_name)])]
_BLUEPRINT = {
    "GMAT": [
        ("QA", "Quantitative", [("gmat-arith", "Arithmetic"), ("gmat-algebra", "Algebra")]),
        ("VA", "Verbal", [("gmat-cr", "Critical Reasoning"), ("gmat-rc", "Reading Comprehension")]),
        ("DI", "Data Insights", [("gmat-ds", "Data Sufficiency"), ("gmat-msr", "Multi-Source Reasoning")]),
    ],
    "GRE": [
        ("QA", "Quantitative Reasoning", [("gre-arith", "Arithmetic"), ("gre-algebra", "Algebra")]),
        ("VA", "Verbal Reasoning", [("gre-rc", "Reading Comprehension"), ("gre-vocab", "Vocabulary")]),
    ],
}

_DIFFICULTIES = [-2, -1, 0, 1, 2, 0]   # six items per concept, spread across the band


def _mount_exam(db: Session, code: str, name: str, blueprint: list) -> None:
    if db.get(models.Exam, code) is not None:
        return
    db.add(models.Exam(code=code, name=name))
    db.flush()
    n = 0
    for sec_key, sec_name, concepts in blueprint:
        sec = models.Section(exam_code=code, key=sec_key, name=sec_name)
        db.add(sec)
        db.flush()
        for cid, cname in concepts:
            db.add(models.KnowledgeNode(id=cid, exam_code=code, section_id=sec.id,
                                        kind=models.NodeKind.concept.value, name=cname))
            db.flush()
            for d in _DIFFICULTIES:
                db.add(models.Item(
                    item_id=f"{cid}-{n}", version=1, content_hash=f"{code.lower()}-{n}",
                    exam_code=code, section_id=sec.id, concept_node_id=cid, difficulty_d=d,
                    format="mcq", num_options=4, stem=f"{cname} sample item {n}",
                    options=["A", "B", "C", "D"], correct_answer="A",
                    solution=f"Worked solution for {cname} item {n}.", status="approved"))
                n += 1
    db.commit()


def mount_gmat_gre_if_empty(db: Session) -> dict:
    """Seed GMAT and GRE structures if their Exam rows are absent. Returns what was mounted."""
    mounted = []
    for code, name in (("GMAT", "GMAT"), ("GRE", "GRE")):
        if db.get(models.Exam, code) is None:
            _mount_exam(db, code, name, _BLUEPRINT[code])
            mounted.append(code)
    return {"mounted": mounted}
