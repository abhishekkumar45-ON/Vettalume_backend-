"""Mock scoring adapters (Phase 3) — turn the shared ability estimate into each exam's reported score.

The psychometric core emits one thing: a theta with SE (overall and per-section). Each adapter maps
that into its exam's scale, exactly the split in the PRD:
  * composite        (GMAT): theta -> 205..805 total + 60..90 section scores
  * sectional        (GRE):  per-section theta -> 130..170 + the panel taken; essay scored separately
  * percentile_call  (CAT):  theta -> percentile (normal CDF) -> per-institute call probability vs a
                             cutoff, with a confident yes / no / too-close read from the SE band
"""
from __future__ import annotations

from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models
from . import calibration, irt

# illustrative CAT targets (percentile cutoffs) — the real per-institute table lives in the CAT adapter
CAT_INSTITUTES = [
    {"name": "IIM Ahmedabad", "overall_pct": 99.0},
    {"name": "IIM Bangalore", "overall_pct": 98.5},
    {"name": "IIM Calcutta", "overall_pct": 98.0},
    {"name": "IIM Lucknow", "overall_pct": 96.0},
    {"name": "FMS Delhi", "overall_pct": 98.5},
    {"name": "MDI Gurgaon", "overall_pct": 94.0},
]


def _eap_for(db: Session, responses: list[models.Response]) -> tuple[float, float, int]:
    triples = []
    for r in responses:
        item = db.get(models.Item, r.item_id)
        if item is None:
            continue
        a, b, c = calibration.active_params(db, item)
        triples.append((a, b, c, 1 if r.correct else 0))
    theta, se = irt.eap_ability(triples)
    return theta, se, len(triples)


def section_breakdown(db: Session, session: models.MockSession) -> dict:
    """Per-section theta/SE for the mock session's responses."""
    responses = list(db.scalars(
        select(models.Response).where(models.Response.session_id == str(session.id))).all())
    by_sec: dict = defaultdict(list)
    for r in responses:
        by_sec[r.section_id].append(r)
    out = {}
    for sec_id, rs in by_sec.items():
        sec = db.get(models.Section, sec_id)
        theta, se, n = _eap_for(db, rs)
        key = sec.key if sec else str(sec_id)
        out[key] = {"name": sec.name if sec else key, "theta": round(theta, 3),
                    "se": None if se == float("inf") else round(se, 3), "n_items": n}
    return out


# ---------- scale maps ----------
def _round10(x: float) -> int:
    return int(round(x / 10.0) * 10)


def _section_scaled(theta: float) -> int:
    """GMAT section scaled score: 5*theta + 75, clamped to the 60-90 section scale."""
    return int(max(60, min(90, round(5 * theta + 75))))


def _gmat_total_from_sections(q: int, v: int, di: int) -> int:
    """GMAT Focus total derived from the three section scaled scores:
        total = (Q + V + DI - 180) * 20/3 + 205
    which maps the section-score sum [180, 270] onto the [205, 805] total scale. Rounded to the
    nearest reported GMAT score (10-point steps ending in 5)."""
    raw = (q + v + di - 180) * 20.0 / 3.0 + 205.0
    return int(max(205, min(805, 205 + 10 * round((raw - 205) / 10.0))))


def composite_score(theta: float, sections: dict) -> dict:
    """GMAT — 60..90 section scaled scores (5*theta + 75) and a 205..805 total DERIVED from them
    via (Q + V + DI - 180) * 20/3 + 205. A measure with no responses (e.g. a single-section mock)
    falls back to overall ability so the total stays well-defined."""
    section_scores = {k: _section_scaled(v["theta"]) for k, v in sections.items()}
    fallback = _section_scaled(theta)

    def measure(*tokens: str) -> int:
        for k, v in sections.items():
            if any(tok in k.lower() for tok in tokens):
                return _section_scaled(v["theta"])
        return fallback

    q, v, di = measure("quant"), measure("verbal"), measure("data", "insight")
    total = _gmat_total_from_sections(q, v, di)
    return {"scale": "GMAT 205-805", "total": total, "section_scores": section_scores,
            "measures": {"Quant": q, "Verbal": v, "Data Insights": di}}


def sectional_score(sections: dict, panel_taken: str | None) -> dict:
    """GRE — 130..170 per measure plus the routed panel; the essay (0-6) is scored outside IRT."""
    secs = {k: max(130, min(170, round(150 + 6.667 * v["theta"]))) for k, v in sections.items()}
    return {"scale": "GRE 130-170", "section_scores": secs, "panel_taken": panel_taken,
            "analytical_writing": {"scale": "0-6", "score": None, "note": "essay scored separately, not by IRT"}}


def percentile_call_score(theta: float, se: float, sections: dict) -> dict:
    """CAT — percentile from the ability (normal-CDF norm) and a per-institute call probability.

    call probability = P(true percentile >= cutoff) given theta +/- SE
                     = 1 - Phi((theta_cutoff - theta) / SE),  theta_cutoff = Phi^-1(cutoff/100)
    The verdict reads the SE band: clearly above, clearly below, or straddling the line.
    """
    percentile = round(irt.normal_cdf(theta) * 100, 1)
    se_eff = 0.3 if (se == float("inf") or se <= 0) else se
    lo_pct = round(irt.normal_cdf(theta - 2 * se_eff) * 100, 1)
    hi_pct = round(irt.normal_cdf(theta + 2 * se_eff) * 100, 1)

    calls = []
    for inst in CAT_INSTITUTES:
        cut = inst["overall_pct"]
        theta_cut = irt.normal_ppf(cut / 100.0)
        call_p = 1.0 - irt.normal_cdf((theta_cut - theta) / se_eff)
        if lo_pct >= cut:
            verdict = "confident yes"
        elif hi_pct < cut:
            verdict = "confident no"
        else:
            verdict = "too close to call"
        calls.append({"institute": inst["name"], "cutoff_percentile": cut,
                      "call_probability": round(call_p, 3), "verdict": verdict})
    return {"scale": "CAT percentile + call", "percentile": percentile,
            "percentile_band_95": [lo_pct, hi_pct], "section_thetas": sections, "calls": calls}


# ---------- dispatch ----------
def score_mock(db: Session, session: models.MockSession) -> dict:
    sections = section_breakdown(db, session)
    base = {"exam": session.exam_code, "mode": session.mode, "theta": round(session.theta, 3),
            "se": None if session.se >= 99 else round(session.se, 3),
            "reliability": round(session.reliability, 3), "n_answered": session.n_answered}
    exam = session.exam_code.upper()
    if exam == "GMAT":
        report = composite_score(session.theta, sections)
    elif exam == "GRE":
        report = sectional_score(sections, session.panel_taken)
    elif exam == "CAT":
        report = percentile_call_score(session.theta, session.se if session.se < 99 else float("inf"), sections)
    else:
        report = composite_score(session.theta, sections)   # generic fallback
    return {**base, "report": report}
