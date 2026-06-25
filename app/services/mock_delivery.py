"""Mock delivery engines (Phase 3) — three ways to administer a mock, one psychometric core.

  * item_adaptive (GMAT): continuous next-item selection by maximum information at the current theta,
                          with exposure control applied first (the IRT doc's rule order).
  * mst          (GRE):   a fixed routing module, then route by interim theta into an easy / medium /
                          hard panel assembled by difficulty band.
  * fixed_form   (CAT):   a pre-assembled linear form, served in order, no adaptivity.

All three read item parameters from the active calibrated set (calibration.active_params), falling back
to the authored prior, and exclude items already served in the session.
"""
from __future__ import annotations

import random

from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models
from ..config import settings
from . import calibration, irt

# difficulty bands on the -2..+2 authored scale
EASY_D = {-2, -1}
MED_D = {-1, 0, 1}
HARD_D = {1, 2}
ROUTING_PANEL_THRESHOLDS = (-0.5, 0.5)   # interim theta -> easy / medium / hard
TOP_K = 6                                # randomesque window for exposure control


# ---------- pools & exposure ----------
def eligible_items(db: Session, exam: str, section_key: str | None = None) -> list[models.Item]:
    q = select(models.Item).where(models.Item.exam_code == exam)
    if settings.serve_only_approved:
        q = q.where(models.Item.status == "approved")
    items = list(db.scalars(q).all())
    if section_key:
        sec_ids = [s.id for s in db.scalars(
            select(models.Section).where(models.Section.exam_code == exam)).all()
            if section_key.lower() in (s.key.lower(), s.name.lower())]
        items = [it for it in items if it.section_id in sec_ids]
    return items


def exposure_counts(db: Session, exam: str) -> dict[str, int]:
    """How many times each item has been served in any mock (across the whole population)."""
    rows = db.execute(
        select(models.Response.item_id)
        .where(models.Response.exam_code == exam,
               models.Response.context.in_(list(models.MOCK_CONTEXTS)))
    ).all()
    out: dict[str, int] = {}
    for (iid,) in rows:
        out[iid] = out.get(iid, 0) + 1
    return out


def _info(db: Session, item: models.Item, theta: float) -> float:
    a, b, c = calibration.active_params(db, item)
    return irt.information_3pl(theta, a, b, c)


def select_by_information(db: Session, items: list[models.Item], theta: float, served: set[str],
                          exposure: dict[str, int], rng: random.Random,
                          exposure_cap: int | None = None) -> models.Item | None:
    """Exposure-aware maximum-information pick. Exposure limits are applied FIRST: drop over-exposed
    items when alternatives exist; then take the top-K most informative and randomesque-pick weighted
    toward the less-exposed, so the bank doesn't over-serve the same sharp items."""
    pool = [it for it in items if it.item_id not in served]
    if not pool:
        return None
    if exposure_cap is not None:
        under = [it for it in pool if exposure.get(it.item_id, 0) < exposure_cap]
        if under:
            pool = under
    pool.sort(key=lambda it: _info(db, it, theta), reverse=True)
    top = pool[:TOP_K]
    weights = [1.0 / (1.0 + exposure.get(it.item_id, 0)) for it in top]
    return rng.choices(top, weights=weights, k=1)[0]


# ---------- form / panel assembly ----------
def _band(items: list[models.Item], band: set[int]) -> list[str]:
    return [it.item_id for it in items if it.difficulty_d in band]


def build_plan(db: Session, exam: str, mode: str, section_key: str | None, max_items: int,
               routing_size: int, seed: int) -> dict:
    """Build the delivery plan. item_adaptive needs none; fixed_form pre-orders a balanced form; mst
    lays out a routing module plus easy/medium/hard panels."""
    items = eligible_items(db, exam, section_key)
    rng = random.Random(seed)
    if mode == "item_adaptive":
        return {"served": []}

    if mode == "fixed_form":
        # balanced linear form: cycle difficulty bands so it isn't sorted by difficulty
        buckets = {d: [it.item_id for it in items if it.difficulty_d == d]
                   for d in (-2, -1, 0, 1, 2)}
        for d in buckets:
            rng.shuffle(buckets[d])
        order = []
        wheel = [0, 1, -1, 2, -2, 0, 1, -1]   # medium-weighted interleave
        while len(order) < min(max_items, len(items)):
            progressed = False
            for d in wheel:
                if buckets.get(d):
                    order.append(buckets[d].pop())
                    progressed = True
                    if len(order) >= min(max_items, len(items)):
                        break
            if not progressed:
                break
        return {"form": order, "served": []}

    if mode == "mst":
        med = _band(items, MED_D)
        rng.shuffle(med)
        routing = med[:routing_size]
        used = set(routing)
        panels = {
            "easy": [i for i in _band(items, EASY_D) if i not in used],
            "medium": [i for i in _band(items, MED_D) if i not in used],
            "hard": [i for i in _band(items, HARD_D) if i not in used],
        }
        for k in panels:
            rng.shuffle(panels[k])
        return {"routing": routing, "panels": panels, "routing_size": len(routing), "served": []}

    raise ValueError(f"unknown mock mode '{mode}'")


def route_panel(theta: float) -> str:
    lo, hi = ROUTING_PANEL_THRESHOLDS
    return "easy" if theta < lo else "hard" if theta > hi else "medium"


# ---------- the next-item decision ----------
def next_item(db: Session, session: models.MockSession) -> models.Item | None:
    """Return the next item for this session, or None when the form/panel is exhausted."""
    plan = session.plan or {}
    served = set(plan.get("served", []))

    if session.mode == "item_adaptive":
        items = eligible_items(db, session.exam_code, session.section_key)
        rng = random.Random(session.seed + session.n_answered)
        exposure = exposure_counts(db, session.exam_code)
        return select_by_information(db, items, session.theta, served, exposure, rng)

    if session.mode == "fixed_form":
        form = plan.get("form", [])
        if session.cursor >= len(form):
            return None
        return db.get(models.Item, form[session.cursor])

    if session.mode == "mst":
        routing = plan.get("routing", [])
        rsize = plan.get("routing_size", len(routing))
        if session.stage == "routing" and session.cursor < rsize:
            return db.get(models.Item, routing[session.cursor])
        # routing finished -> lock the panel by interim theta (once), then serve from it
        if session.panel_taken is None:
            session.panel_taken = route_panel(session.theta)
            session.stage = "panel"
        panel = plan.get("panels", {}).get(session.panel_taken, [])
        idx = session.cursor - rsize
        if idx >= len(panel):
            return None
        return db.get(models.Item, panel[idx])

    return None


def mark_served(session: models.MockSession, item_id: str) -> None:
    plan = dict(session.plan or {})
    served = list(plan.get("served", []))
    if item_id not in served:
        served.append(item_id)
    plan["served"] = served
    session.plan = plan
