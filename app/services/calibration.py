"""Calibration worker (Phase 2) — discover item (a, b, c) from the cold response matrix.

Algorithm (straight from the reference):
  0. bootstrap   : ability ~ logit(score),  b ~ -logit(item p-correct),  a=1, c=format floor
  A. ability step: re-estimate every learner's theta by EAP, holding item params fixed
     (then recenter abilities to mean 0 / sd 1 to anchor the scale)
  B. item step   : re-fit every item's curve by Newton MLE, holding abilities fixed —
                   PHASED so we only unlock what the data supports:
                     < ~500 responses  -> b only        (a=1, c=floor)
                     < ~2000 responses -> b, a (2PL)     (c=floor)
                     >= ~2000          -> b, a, c (3PL)
  loop A/B until parameters stop moving.

Only COLD responses feed calibration (diagnostic + mock contexts) — practice/learning responses are
contaminated by teaching and are excluded, exactly as the shared-bank rule requires.

Results are written to the versioned store (CalibrationRun + IrtParameter) and, on activation, mirrored
onto Item.irt_a/b/c with a version bump.
"""
from __future__ import annotations

import statistics
import uuid

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from .. import models
from ..models import COLD_CONTEXTS
from . import irt


# ---------- live parameter lookup ----------
def active_params(db: Session, item: models.Item) -> tuple[float, float, float]:
    """The (a, b, c) the mock scorer/selector should use for an item: the active calibrated row if one
    exists, otherwise a sensible prior — a=1, b from the authored difficulty, c from the format."""
    row = db.scalar(
        select(models.IrtParameter)
        .where(models.IrtParameter.item_id == item.item_id, models.IrtParameter.active.is_(True))
    )
    if row is not None:
        return row.a, row.b, row.c
    c = irt.default_c(item.num_options, is_type_in=(item.format == "tita"))
    return 1.0, float(item.difficulty_d), c


# ---------- cold response matrix ----------
def cold_response_matrix(db: Session, exam: str):
    """Return (learner_ids, item_ids, by_learner, by_item, fmt) from COLD responses for the exam.

    by_learner[li] = list of (ii, u);  by_item[ii] = list of (li, u);  fmt[ii] = (num_options, is_type_in)
    """
    rows = db.execute(
        select(models.Response.learner_id, models.Response.item_id, models.Response.correct,
               models.Item.num_options, models.Item.format)
        .join(models.Item, models.Item.item_id == models.Response.item_id)
        .where(models.Response.exam_code == exam,
               models.Response.context.in_(list(COLD_CONTEXTS)))
    ).all()

    learner_ids: list[uuid.UUID] = []
    item_ids: list[str] = []
    l_index: dict[uuid.UUID, int] = {}
    i_index: dict[str, int] = {}
    fmt: dict[int, tuple[int, bool]] = {}
    by_learner: dict[int, list[tuple[int, int]]] = {}
    by_item: dict[int, list[tuple[int, int]]] = {}

    for lid, iid, correct, num_opts, f in rows:
        if lid not in l_index:
            l_index[lid] = len(learner_ids); learner_ids.append(lid)
        if iid not in i_index:
            i_index[iid] = len(item_ids); item_ids.append(iid)
            fmt[i_index[iid]] = (num_opts, f == "tita")
        li, ii, u = l_index[lid], i_index[iid], 1 if correct else 0
        by_learner.setdefault(li, []).append((ii, u))
        by_item.setdefault(ii, []).append((li, u))
    return learner_ids, item_ids, by_learner, by_item, fmt


def _recenter(thetas: list[float]) -> list[float]:
    """Anchor the ability scale to mean 0 / sd 1 each sweep (identifiability)."""
    if len(thetas) < 2:
        return thetas
    mu = statistics.fmean(thetas)
    sd = statistics.pstdev(thetas)
    if sd < 1e-6:
        return [0.0 for _ in thetas]
    return [(t - mu) / sd for t in thetas]


def calibrate(n_learners: int, by_learner: dict, by_item: dict, fmt: dict, *,
              two_pl_at: int = 500, three_pl_at: int = 2000, max_sweeps: int = 25,
              tol: float = 1e-3, min_responses_per_item: int = 1):
    """Pure calibration loop over an in-memory matrix — shared by the DB worker and the simulation
    harness so the release gate exercises the exact production math.

    Returns (abilities[list], params[dict ii->{a,b,c,phase}], sweeps, converged).
    """
    # Step 0: bootstrap
    abilities = [0.0] * n_learners
    for li, resp in by_learner.items():
        score = sum(u for _, u in resp) / len(resp)
        abilities[li] = irt.logit(score)
    abilities = _recenter(abilities)

    params: dict[int, dict] = {}
    for ii in by_item:
        resp = by_item.get(ii, [])
        p_correct = (sum(u for _, u in resp) / len(resp)) if resp else 0.5
        num_opts, is_type_in = fmt[ii]
        params[ii] = {"a": 1.0, "b": -irt.logit(p_correct),
                      "c": irt.default_c(num_opts, is_type_in), "phase": "b"}

    sweeps = 0
    converged = False
    for sweeps in range(1, max_sweeps + 1):
        new_ab = abilities[:]
        for li, resp in by_learner.items():
            items = [(params[ii]["a"], params[ii]["b"], params[ii]["c"], u) for ii, u in resp]
            new_ab[li], _ = irt.eap_ability(items)
        new_ab = _recenter(new_ab)

        max_move = 0.0
        for ii, resp in by_item.items():
            if len(resp) < min_responses_per_item:
                continue
            thetas = [new_ab[li] for li, _ in resp]
            us = [u for _, u in resp]
            phase = irt.phase_for(len(resp), two_pl_at=two_pl_at, three_pl_at=three_pl_at)
            fit = irt.fit_item(thetas, us, phase=phase,
                               a0=params[ii]["a"], b0=params[ii]["b"], c0=params[ii]["c"])
            max_move = max(max_move, abs(fit["b"] - params[ii]["b"]),
                           abs(fit["a"] - params[ii]["a"]), abs(fit["c"] - params[ii]["c"]))
            params[ii] = {"a": fit["a"], "b": fit["b"], "c": fit["c"], "phase": fit["phase"]}

        abilities = new_ab
        if max_move < tol:
            converged = True
            break
    return abilities, params, sweeps, converged


# ---------- the worker ----------
def run_calibration(db: Session, exam: str, *, two_pl_at: int = 500, three_pl_at: int = 2000,
                    max_sweeps: int = 25, tol: float = 1e-3, min_responses_per_item: int = 1,
                    activate: bool = True) -> models.CalibrationRun:
    learner_ids, item_ids, by_learner, by_item, fmt = cold_response_matrix(db, exam)
    n_resp = sum(len(v) for v in by_learner.values())

    if not item_ids or not learner_ids:
        run = models.CalibrationRun(exam_code=exam, status="failed", n_items=len(item_ids),
                                    n_responses=n_resp, n_learners=len(learner_ids), iterations=0,
                                    converged=False, activated=False,
                                    summary={"reason": "no cold responses to calibrate on"})
        db.add(run); db.flush()
        return run

    _abilities, params, sweeps, converged = calibrate(
        len(learner_ids), by_learner, by_item, fmt, two_pl_at=two_pl_at, three_pl_at=three_pl_at,
        max_sweeps=max_sweeps, tol=tol, min_responses_per_item=min_responses_per_item)

    # ---- persist (versioned) ----
    phase_counts: dict[str, int] = {}
    for ii in params:
        ph = params[ii].get("phase", "b")
        phase_counts[ph] = phase_counts.get(ph, 0) + 1

    run = models.CalibrationRun(
        exam_code=exam, status="complete", n_items=len(item_ids), n_responses=n_resp,
        n_learners=len(learner_ids), iterations=sweeps, converged=converged, activated=activate,
        summary={"phase_counts": phase_counts, "two_pl_at": two_pl_at, "three_pl_at": three_pl_at},
    )
    db.add(run); db.flush()

    for ii, iid in enumerate(item_ids):
        p = params[ii]
        n_i = len(by_item.get(ii, []))
        if activate:
            db.execute(update(models.IrtParameter)
                       .where(models.IrtParameter.item_id == iid, models.IrtParameter.active.is_(True))
                       .values(active=False))
        db.add(models.IrtParameter(run_id=run.id, item_id=iid, a=p["a"], b=p["b"], c=p["c"],
                                   phase=p.get("phase", "b"), n_responses=n_i, active=activate))
        if activate:
            item = db.get(models.Item, iid)
            if item is not None:
                item.irt_a, item.irt_b, item.irt_c = p["a"], p["b"], p["c"]
                item.version = (item.version or 1) + 1
    db.flush()
    return run
