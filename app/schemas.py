from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Benchmark seconds by difficulty (-2..2 scale); used to auto-fill timing.
BENCH = {-2: 40, -1: 55, 0: 80, 1: 110, 2: 150}


class ItemFormatIn(str, Enum):
    mcq = "mcq"
    tita = "tita"


class UsageScopeIn(str, Enum):
    both = "both"
    mock_only = "mock_only"
    practice_only = "practice_only"


class ContextIn(str, Enum):
    diagnostic = "diagnostic"
    practice = "practice"
    sectional_mock = "sectional_mock"
    full_mock = "full_mock"


class ItemIn(BaseModel):
    # extra='forbid' is the enforcement of the authored-vs-derived boundary: any attempt to
    # author irt_a/irt_b/irt_c (or any unknown field) via the upload is rejected outright.
    model_config = ConfigDict(extra="forbid")

    item_id: str
    exam_code: str
    section_key: str
    concept_node_id: str
    archetype_id: Optional[str] = None
    grid_cell: Optional[str] = None
    difficulty_d: int = Field(ge=-2, le=2)   # authored expert difficulty, -2..2 (prior for IRT b)
    format: ItemFormatIn = ItemFormatIn.mcq
    num_options: int = Field(default=4, ge=0)
    negative_marking: bool = False
    stem: str
    options: Optional[list[str]] = None
    correct_answer: str
    distractor_map: Optional[dict[str, Any]] = None
    solution: Optional[str] = None
    time_benchmark_s: Optional[int] = None
    passage_set_id: Optional[str] = None     # groups questions that share a passage / data set
    status: str = "approved"                 # only 'approved' items are served to learners
    provenance: Optional[dict[str, Any]] = None
    usage_scope: UsageScopeIn = UsageScopeIn.both


class IngestError(BaseModel):
    index: int
    item_id: Optional[str] = None
    error: str


class IngestReport(BaseModel):
    status: str  # "committed" | "rejected"
    received: int
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    errors: list[IngestError] = []


class AnswerIn(BaseModel):
    item_id: str
    context: ContextIn = ContextIn.practice
    answer_given: Optional[str] = None
    correct: Optional[bool] = None  # if omitted, the server grades from the item's correct_answer
    response_time_ms: Optional[int] = None
    attempt_number: int = 1
    hints_used: int = 0
    session_id: Optional[str] = None


class AnswerOut(BaseModel):
    correct: bool
    solution: Optional[str] = None
    node_id: str
    mastery: float
    attempts: int


class ItemPublic(BaseModel):
    """What a learner is allowed to see for the next question. Note: NO correct_answer, NO
    solution, and NO difficulty (difficulty is never shown to the learner, per the engine)."""
    item_id: str
    stem: str
    options: Optional[list[str]] = None
    format: str
    num_options: int


class NodeStateOut(BaseModel):
    node_id: str
    name: str
    learned: bool
    mastery: float
    attempts: int


class StateOut(BaseModel):
    exam: str
    nodes: list[NodeStateOut]


class LearnAnswerIn(BaseModel):
    item_id: str
    answer_given: Optional[str] = None
    response_time_ms: Optional[int] = None
    session_id: Optional[str] = None


class DevLoginIn(BaseModel):
    email: str

    @field_validator("email")
    @classmethod
    def _normalize_email(cls, v: str) -> str:
        return v.strip().lower()
    display_name: Optional[str] = None
    exam_code: str = "CAT"  # auto-grants an entitlement so the demo can run end-to-end


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    learner_id: str


class TreeNode(BaseModel):
    id: str
    name: str
    concepts: list[dict] = []


class TreeOut(BaseModel):
    exam: str
    topics: list[dict]
