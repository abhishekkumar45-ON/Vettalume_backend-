"""Billing and entitlements (Phase 5).

One account, one wallet, independently-purchasable per-course entitlements, multi-exam bundles with a
cross-course discount, and per-course free tiers (BL-01..05, AC-01). This is the records-and-logic
layer: purchase() records an Order and grants Entitlements but does NOT move money — a real payment
provider (Stripe for USD, Razorpay for INR) slots in at the marked point.

Entitlement tier is encoded in Entitlement.status: free | active (paid) | expired. Enforcement is
gated behind settings.enforce_entitlements so the open demo keeps working until it is switched on.
"""
from __future__ import annotations

import uuid

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models
from ..config import settings

# One coherent catalog. Prices are illustrative; the point is multi-currency + tiers + a bundle.
_CATALOG = [
    dict(code="gmat_free", kind="free", exam_code="GMAT", name="GMAT Free",
         currency="USD", amount_cents=0, limits={"full_mocks": 1, "debrief": "summary"}),
    dict(code="gmat_summit", kind="paid", exam_code="GMAT", name="GMAT Summit",
         currency="USD", amount_cents=19900, limits=None),
    dict(code="gre_free", kind="free", exam_code="GRE", name="GRE Free",
         currency="USD", amount_cents=0, limits={"full_mocks": 2, "debrief": "summary"}),
    dict(code="gre_core", kind="paid", exam_code="GRE", name="GRE Core",
         currency="USD", amount_cents=14900, limits=None),
    dict(code="cat_free", kind="free", exam_code="CAT", name="CAT Free",
         currency="INR", amount_cents=0, limits={"full_mocks": 1, "debrief": "summary"}),
    dict(code="cat_pro", kind="paid", exam_code="CAT", name="CAT Pro",
         currency="INR", amount_cents=1499900, limits=None),
    dict(code="bundle_gmat_gre", kind="bundle", exam_code=None, name="GMAT + GRE Bundle",
         currency="USD", amount_cents=29900, bundle_exams=["GMAT", "GRE"], limits=None),
]


def ensure_catalog(db: Session) -> None:
    """Idempotently load the SKU catalog (safe to call on every boot)."""
    changed = False
    for spec in _CATALOG:
        if db.get(models.PricePlan, spec["code"]) is None:
            db.add(models.PricePlan(active=True, period="one_time", **spec))
            changed = True
    if changed:
        db.commit()


def catalog(db: Session) -> list[dict]:
    rows = db.scalars(select(models.PricePlan).where(models.PricePlan.active.is_(True))).all()
    return [{"code": p.code, "kind": p.kind, "exam": p.exam_code, "name": p.name,
             "currency": p.currency, "amount": p.amount_cents / 100, "amount_cents": p.amount_cents,
             "period": p.period, "bundle_exams": p.bundle_exams, "limits": p.limits} for p in rows]


def _entitlement(db: Session, account: models.Account, exam: str) -> models.Entitlement | None:
    return db.scalar(select(models.Entitlement).where(
        models.Entitlement.account_id == account.id, models.Entitlement.exam_code == exam))


def entitlement_state(db: Session, account: models.Account, exam: str) -> dict:
    ent = _entitlement(db, account, exam)
    status = ent.status if ent else None
    tier = "paid" if status == "active" else ("free" if status == "free" else None)
    return {"exam": exam, "status": status, "tier": tier,
            "entitled": status in ("free", "active"), "paid": status == "active"}


def all_entitlements(db: Session, account: models.Account) -> list[dict]:
    rows = db.scalars(select(models.Entitlement).where(
        models.Entitlement.account_id == account.id)).all()
    return [{"exam": e.exam_code, "status": e.status,
             "tier": "paid" if e.status == "active" else e.status} for e in rows]


def _upsert_entitlement(db: Session, account: models.Account, exam: str, status: str) -> None:
    ent = _entitlement(db, account, exam)
    if ent is None:
        db.add(models.Entitlement(account_id=account.id, exam_code=exam, status=status))
    else:
        ent.status = status


def grant_free_tier(db: Session, account: models.Account, exam: str) -> dict:
    """Grant the per-course free tier if the learner has no entitlement yet (idempotent), then
    warm-start the course from any shared signals on the learner's other courses (AC-02)."""
    if _entitlement(db, account, exam) is None:
        _upsert_entitlement(db, account, exam, "free")
        db.commit()
        from . import warmstart
        warmstart.warm_start(db, account, exam, persist=True)
    return entitlement_state(db, account, exam)


def _component_total_cents(db: Session, exams: list[str]) -> int:
    total = 0
    for ex in exams:
        paid = db.scalar(select(models.PricePlan).where(
            models.PricePlan.exam_code == ex, models.PricePlan.kind == "paid",
            models.PricePlan.active.is_(True)))
        if paid:
            total += paid.amount_cents
    return total


def purchase(db: Session, account: models.Account, plan_code: str) -> dict:
    """Record a purchase and grant entitlements. Does NOT charge — integrate a PSP here."""
    plan = db.get(models.PricePlan, plan_code)
    if plan is None or not plan.active:
        raise HTTPException(404, f"unknown plan '{plan_code}'")
    if plan.kind == "free":
        raise HTTPException(400, "free tiers are granted, not purchased — use /billing/grant-free")

    exams = plan.bundle_exams if plan.kind == "bundle" else [plan.exam_code]
    for ex in exams:
        _upsert_entitlement(db, account, ex, "active")

    savings_cents = 0
    if plan.kind == "bundle":
        savings_cents = max(0, _component_total_cents(db, exams) - plan.amount_cents)

    order = models.Order(account_id=account.id, plan_code=plan.code, currency=plan.currency,
                         amount_cents=plan.amount_cents, status="paid",
                         claim_state={"refundable": True, "guarantee": "per_course",
                                      "claimed": False})
    db.add(order)
    db.flush()
    db.commit()
    # warm-start every newly entitled course from the learner's other courses (AC-02)
    from . import warmstart
    for ex in exams:
        warmstart.warm_start(db, account, ex, persist=True)
    return {
        "status": "ok", "order_id": str(order.id), "plan": plan.code, "currency": plan.currency,
        "amount": plan.amount_cents / 100, "granted_exams": exams,
        "bundle_savings": round(savings_cents / 100, 2) if savings_cents else None,
        "note": "order recorded and entitlement(s) granted — no real charge "
                "(integrate Stripe/Razorpay at billing.purchase())",
    }


def free_tier_usage(db: Session, account: models.Account, exam: str,
                    resource: str = "full_mocks") -> dict:
    """How much of a metered free-tier resource the learner has consumed."""
    ent = _entitlement(db, account, exam)
    if ent and ent.status == "active":
        return {"resource": resource, "metered": False, "tier": "paid"}
    plan = db.scalar(select(models.PricePlan).where(
        models.PricePlan.exam_code == exam, models.PricePlan.kind == "free"))
    limit = (plan.limits or {}).get(resource) if plan else None
    used = 0
    if resource == "full_mocks":
        used = db.scalar(select(models.func.count(models.MockSession.id)).where(
            models.MockSession.learner_id == account.id,
            models.MockSession.exam_code == exam,
            models.MockSession.section_key.is_(None))) or 0
    remaining = None if limit is None else max(0, limit - used)
    return {"resource": resource, "metered": True, "tier": "free", "used": used,
            "limit": limit, "remaining": remaining,
            "exhausted": (limit is not None and used >= limit)}


def enforce(db: Session, account: models.Account, exam: str, *, need: str = "any",
            resource: str | None = None) -> dict:
    """Access guard for course-scoped surfaces. No-op unless settings.enforce_entitlements is on.
    need='paid' requires an active paid entitlement; need='any' allows the free tier but blocks once a
    metered free-tier resource is exhausted. Auto-grants the free tier so the free experience works."""
    state = entitlement_state(db, account, exam)
    if not settings.enforce_entitlements:
        return state
    if need == "paid" and not state["paid"]:
        raise HTTPException(402, {"error": "paid_entitlement_required", "exam": exam,
                                  "see": "/billing/catalog"})
    if not state["entitled"]:
        state = grant_free_tier(db, account, exam)   # baseline free access
    if resource is not None and state["tier"] == "free":
        usage = free_tier_usage(db, account, exam, resource)
        if usage.get("exhausted"):
            raise HTTPException(402, {"error": "free_tier_limit_reached", "resource": resource,
                                      "exam": exam, "see": "/billing/catalog"})
    return state
