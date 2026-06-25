"""The locked Vettalume data model (Phase 0).

Enum-like fields are stored as plain strings for portability; the Python enums below are the
source of truth and are validated at the API boundary (schemas.py). They can be hardened into
native Postgres enums later without changing application code.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base
from .types import JSONType


# ---- controlled vocabularies (validated in schemas.py) ----
class Context(str, enum.Enum):
    diagnostic = "diagnostic"
    practice = "practice"
    sectional_mock = "sectional_mock"
    full_mock = "full_mock"


# "Cold" = exam conditions. ONLY these responses are admissible for IRT calibration.
# Practice is taught-first/untimed, so it never calibrates item difficulty (it drives the MAB).
COLD_CONTEXTS = {Context.diagnostic.value, Context.sectional_mock.value, Context.full_mock.value}
MOCK_CONTEXTS = {Context.sectional_mock.value, Context.full_mock.value}


class ItemFormat(str, enum.Enum):
    mcq = "mcq"
    tita = "tita"  # type-in-the-answer (no guessing floor, no negative marking)


class UsageScope(str, enum.Enum):
    both = "both"
    mock_only = "mock_only"        # reserved holdout: never served in practice
    practice_only = "practice_only"


class NodeKind(str, enum.Enum):
    topic = "topic"
    concept = "concept"  # subtopic / leaf concept


# ---- account & entitlements ----
class Account(Base):
    __tablename__ = "accounts"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True)
    display_name: Mapped[str] = mapped_column(String(120), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Entitlement(Base):
    __tablename__ = "entitlements"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    account_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("accounts.id", ondelete="CASCADE"))
    exam_code: Mapped[str] = mapped_column(ForeignKey("exams.code"))
    status: Mapped[str] = mapped_column(String(20), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    __table_args__ = (UniqueConstraint("account_id", "exam_code", name="uq_entitlement"),)


# ---- exam catalog ----
class Exam(Base):
    __tablename__ = "exams"
    code: Mapped[str] = mapped_column(String(16), primary_key=True)  # CAT / GMAT / GRE
    name: Mapped[str] = mapped_column(String(120))


class Section(Base):
    __tablename__ = "sections"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    exam_code: Mapped[str] = mapped_column(ForeignKey("exams.code"))
    key: Mapped[str] = mapped_column(String(32))  # QA / VARC / DILR ...
    name: Mapped[str] = mapped_column(String(120))
    __table_args__ = (UniqueConstraint("exam_code", "key", name="uq_section"),)


# ---- knowledge graph: tree (parent_id) + DAG (PrereqEdge) ----
class KnowledgeNode(Base):
    __tablename__ = "knowledge_nodes"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # authored, e.g. 'avg-simple'
    exam_code: Mapped[str] = mapped_column(ForeignKey("exams.code"))
    section_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("sections.id"))
    kind: Mapped[str] = mapped_column(String(16))  # NodeKind
    name: Mapped[str] = mapped_column(String(160))
    parent_id: Mapped[Optional[str]] = mapped_column(ForeignKey("knowledge_nodes.id"), nullable=True)
    theory: Mapped[Optional[dict]] = mapped_column(JSONType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class PrereqEdge(Base):
    __tablename__ = "prereq_edges"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    node_id: Mapped[str] = mapped_column(ForeignKey("knowledge_nodes.id"))          # this node ...
    prereq_node_id: Mapped[str] = mapped_column(ForeignKey("knowledge_nodes.id"))   # ... needs this
    __table_args__ = (UniqueConstraint("node_id", "prereq_node_id", name="uq_prereq"),)


# ---- the shared item bank ----
class Item(Base):
    """Phase-0 simplification: item_id is the PK (one row per item) and `version` bumps on a
    content change (which also nulls the calibration fields, since changed content must be
    re-calibrated). The production-correct design is IMMUTABLE (item_id, version) rows that
    responses bind to; that machinery is added in Phase 2, when calibration makes version
    binding load-bearing. The response already snapshots item_version, so the seam exists."""
    __tablename__ = "items"
    item_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    content_hash: Mapped[str] = mapped_column(String(64))

    # routing tags (controlled vocabulary -> FKs)
    exam_code: Mapped[str] = mapped_column(ForeignKey("exams.code"))
    section_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("sections.id"))
    concept_node_id: Mapped[str] = mapped_column(ForeignKey("knowledge_nodes.id"))
    archetype_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)  # TODO Phase-1: FK
    grid_cell: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)     # e.g. 'CALC-L2'

    # authored psychometrics & format
    difficulty_d: Mapped[int] = mapped_column(Integer)  # 1..5, expert-set, never shown to learner
    format: Mapped[str] = mapped_column(String(8))      # ItemFormat
    num_options: Mapped[int] = mapped_column(Integer, default=4)
    negative_marking: Mapped[bool] = mapped_column(Boolean, default=False)

    # payload
    stem: Mapped[str] = mapped_column(Text)
    options: Mapped[Optional[list]] = mapped_column(JSONType, nullable=True)
    correct_answer: Mapped[str] = mapped_column(String(255))
    distractor_map: Mapped[Optional[dict]] = mapped_column(JSONType, nullable=True)
    solution: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    time_benchmark_s: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # provenance & scope
    provenance: Mapped[Optional[dict]] = mapped_column(JSONType, nullable=True)
    usage_scope: Mapped[str] = mapped_column(String(16), default=UsageScope.both.value)
    passage_set_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(20), default="approved", index=True)

    # DERIVED — written by calibration only, NEVER authored via the Excel
    irt_a: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    irt_b: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    irt_c: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    empirical: Mapped[Optional[dict]] = mapped_column(JSONType, nullable=True)
    calibration_run_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    calibrated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# ---- the event spine (append-only) ----
class Response(Base):
    __tablename__ = "responses"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    learner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("accounts.id"), index=True)
    item_id: Mapped[str] = mapped_column(ForeignKey("items.item_id"), index=True)
    item_version: Mapped[int] = mapped_column(Integer)
    context: Mapped[str] = mapped_column(String(16), index=True)  # THE discriminator (Context)
    correct: Mapped[bool] = mapped_column(Boolean)
    answer_given: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    response_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    attempt_number: Mapped[int] = mapped_column(Integer, default=1)
    hints_used: Mapped[int] = mapped_column(Integer, default=0)
    difficulty_d: Mapped[int] = mapped_column(Integer)  # snapshot
    exam_code: Mapped[str] = mapped_column(String(16), index=True)   # denormalized for calib scans
    section_id: Mapped[uuid.UUID] = mapped_column(Uuid, index=True)
    session_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    @property
    def admissible_for_calibration(self) -> bool:
        return self.context in COLD_CONTEXTS


# ---- exposure ledger (drives the shared-bank eligibility rule) ----
class Exposure(Base):
    __tablename__ = "exposure"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    learner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("accounts.id"), index=True)
    item_id: Mapped[str] = mapped_column(ForeignKey("items.item_id"), index=True)
    last_seen_context: Mapped[str] = mapped_column(String(16))
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    times_seen: Mapped[int] = mapped_column(Integer, default=0)
    __table_args__ = (UniqueConstraint("learner_id", "item_id", name="uq_exposure"),)


# ---- per-learner per-node state (mastery store) ----
class LearnerNodeState(Base):
    __tablename__ = "learner_node_state"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    learner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("accounts.id"), index=True)
    node_id: Mapped[str] = mapped_column(ForeignKey("knowledge_nodes.id"), index=True)
    learned: Mapped[bool] = mapped_column(Boolean, default=False)
    performance_p: Mapped[float] = mapped_column(Float, default=0.0)
    difficulty_score: Mapped[float] = mapped_column(Float, default=0.0)
    memory_strength: Mapped[float] = mapped_column(Float, default=0.0)
    mastery: Mapped[float] = mapped_column(Float, default=0.0)
    # Phase 1 stores bandit weights / reward history / MCM traces here as JSON.
    bandit_state: Mapped[Optional[dict]] = mapped_column(JSONType, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    __table_args__ = (UniqueConstraint("learner_id", "node_id", name="uq_learner_node"),)


# ============================================================================
# Phase 2 — psychometric IRT: versioned parameter store + ability estimates
# ============================================================================
class CalibrationRun(Base):
    """One execution of the calibration worker. Every IrtParameter row points back to the run that
    produced it, so the parameter store is fully versioned and auditable."""
    __tablename__ = "calibration_runs"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    exam_code: Mapped[str] = mapped_column(String(16), index=True)
    status: Mapped[str] = mapped_column(String(16), default="complete")  # complete | failed | gated
    n_items: Mapped[int] = mapped_column(Integer, default=0)
    n_responses: Mapped[int] = mapped_column(Integer, default=0)
    n_learners: Mapped[int] = mapped_column(Integer, default=0)
    iterations: Mapped[int] = mapped_column(Integer, default=0)
    converged: Mapped[bool] = mapped_column(Boolean, default=False)
    activated: Mapped[bool] = mapped_column(Boolean, default=False)  # did it become the live params?
    summary: Mapped[Optional[dict]] = mapped_column(JSONType, nullable=True)  # phase counts, gate, notes
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True)


class IrtParameter(Base):
    """A calibrated (a, b, c) for one item from one run. The row with active=True is the live
    parameter set the mock scorer/selector uses; older rows are retained for rollback and drift."""
    __tablename__ = "irt_parameters"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("calibration_runs.id"), index=True)
    item_id: Mapped[str] = mapped_column(ForeignKey("items.item_id"), index=True)
    a: Mapped[float] = mapped_column(Float)
    b: Mapped[float] = mapped_column(Float)
    c: Mapped[float] = mapped_column(Float)
    phase: Mapped[str] = mapped_column(String(8))          # "b" | "2pl" | "3pl"
    n_responses: Mapped[int] = mapped_column(Integer, default=0)
    se_b: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now())
    __table_args__ = (UniqueConstraint("run_id", "item_id", name="uq_run_item"),)


class AbilityEstimate(Base):
    """A scored ability (theta) on the -3..+3 scale for a learner over a cold/mock scope, with its SE.
    Append-only: a learner accrues estimates over diagnostics and mocks."""
    __tablename__ = "ability_estimates"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    learner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("accounts.id"), index=True)
    exam_code: Mapped[str] = mapped_column(String(16), index=True)
    scope: Mapped[str] = mapped_column(String(64))         # "diagnostic" | session_id | "full_mock" ...
    theta: Mapped[float] = mapped_column(Float)
    se: Mapped[float] = mapped_column(Float)
    n_items: Mapped[int] = mapped_column(Integer, default=0)
    method: Mapped[str] = mapped_column(String(8), default="eap")   # "eap" | "elo"
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True)


# ============================================================================
# Phase 3 — mocks: per-response checkpointed session (delivery + scoring state)
# ============================================================================
class MockSession(Base):
    """A mock attempt. This row IS the reliability checkpoint: it is upserted after every single
    response, so a dropped connection loses nothing — resume reads the latest state. It holds the
    live ability (theta/se), the delivery plan (fixed form sequence / MST panels), and the cursor."""
    __tablename__ = "mock_sessions"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    learner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("accounts.id"), index=True)
    exam_code: Mapped[str] = mapped_column(String(16), index=True)
    section_key: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    mode: Mapped[str] = mapped_column(String(16))         # item_adaptive | mst | fixed_form
    status: Mapped[str] = mapped_column(String(16), default="in_progress")  # in_progress|completed|abandoned
    stage: Mapped[str] = mapped_column(String(16), default="main")          # routing|panel|main (MST)
    panel_taken: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)  # easy|medium|hard
    cursor: Mapped[int] = mapped_column(Integer, default=0)                 # fixed/MST position
    theta: Mapped[float] = mapped_column(Float, default=0.0)
    se: Mapped[float] = mapped_column(Float, default=99.0)
    reliability: Mapped[float] = mapped_column(Float, default=0.0)
    n_answered: Mapped[int] = mapped_column(Integer, default=0)
    max_items: Mapped[int] = mapped_column(Integer, default=25)
    se_target: Mapped[float] = mapped_column(Float, default=0.30)
    plan: Mapped[Optional[dict]] = mapped_column(JSONType, nullable=True)   # form/panels + served list
    seed: Mapped[int] = mapped_column(Integer, default=0)                   # exposure RNG seed
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


# ============================================================================
# Phase 4 — plan engine: versioned study plans (each re-plan is a new version, diffed vs the prior)
# ============================================================================
class StudyPlan(Base):
    """A study plan generated from a diagnosis. Versions accumulate per (learner, exam) so the plan
    engine can diff a re-plan against the prior version and explain the change in plain language."""
    __tablename__ = "study_plans"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    learner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("accounts.id"), index=True)
    exam_code: Mapped[str] = mapped_column(String(16), index=True)
    version: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default="active")  # active | superseded
    items: Mapped[Optional[list]] = mapped_column(JSONType, nullable=True)      # ordered plan items
    diagnosis: Mapped[Optional[dict]] = mapped_column(JSONType, nullable=True)  # snapshot it was built from
    rationale: Mapped[Optional[dict]] = mapped_column(JSONType, nullable=True)  # diff + plain-language change
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now())


# ============================================================================
# Phase 5 — billing/entitlements + Honest-Perimeter accuracy record (new tables only;
# the existing Entitlement table is reused, with status encoding the tier: free|active|expired)
# ============================================================================
class PricePlan(Base):
    """SKU catalog. A plan is either a per-course tier (free|paid) or a multi-exam bundle.
    Multi-currency (USD for GMAT/GRE, INR for CAT) lives here without forking the platform (BL-01)."""
    __tablename__ = "price_plans"
    code: Mapped[str] = mapped_column(String(48), primary_key=True)   # e.g. 'gmat_summit'
    kind: Mapped[str] = mapped_column(String(16))                     # free | paid | bundle
    exam_code: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)  # None for bundles
    name: Mapped[str] = mapped_column(String(120))
    currency: Mapped[str] = mapped_column(String(8))                  # USD | INR
    amount_cents: Mapped[int] = mapped_column(Integer, default=0)     # minor units (cents/paise)
    period: Mapped[str] = mapped_column(String(16), default="one_time")
    bundle_exams: Mapped[Optional[list]] = mapped_column(JSONType, nullable=True)  # for bundles
    limits: Mapped[Optional[dict]] = mapped_column(JSONType, nullable=True)        # free-tier caps
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class Order(Base):
    """A recorded purchase. This is the billing-records layer (eligibility + claim state, BL-05);
    it does NOT move money — a real PSP (Stripe/Razorpay) integrates at billing.purchase()."""
    __tablename__ = "orders"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    account_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("accounts.id"), index=True)
    plan_code: Mapped[str] = mapped_column(String(48))
    currency: Mapped[str] = mapped_column(String(8))
    amount_cents: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default="paid")   # paid | refunded
    claim_state: Mapped[Optional[dict]] = mapped_column(JSONType, nullable=True)  # guarantee/refund
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now())


class PredictionRecord(Base):
    """One emitted prediction and (once known) its verified outcome. The aggregate over these rows
    IS the Honest Perimeter's published accuracy record: coverage of the bands and mean error."""
    __tablename__ = "prediction_records"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    account_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("accounts.id"), nullable=True, index=True)
    exam_code: Mapped[str] = mapped_column(String(16), index=True)
    kind: Mapped[str] = mapped_column(String(24))                     # score | percentile | ability
    point: Mapped[float] = mapped_column(Float)
    band_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    band_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    se: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    basis: Mapped[str] = mapped_column(String(16), default="provisional")  # calibrated | provisional
    outcome: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    within_band: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now())
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


# ============================================================================
# Phase 6 — real auth: password credentials, one-to-one with Account (new table; the accounts
# table is unchanged so existing databases need no migration)
# ============================================================================
class Credential(Base):
    """A learner's password credential (PBKDF2-SHA256 hash). Separate from Account so the existing
    accounts table is untouched and dev-login / passwordless accounts remain valid."""
    __tablename__ = "credentials"
    account_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("accounts.id", ondelete="CASCADE"), primary_key=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class AdminUser(Base):
    """Content-admin role grant (additive table — no column added to `accounts`, so it works on
    existing DBs via create_all). An account is an admin iff it has a row here OR its email is in
    settings.admin_emails. Admin accounts gate ALL content-authoring endpoints; "no outsiders"."""

    __tablename__ = "admin_users"

    account_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("accounts.id"), primary_key=True)
    role: Mapped[str] = mapped_column(String(32), default="admin")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
