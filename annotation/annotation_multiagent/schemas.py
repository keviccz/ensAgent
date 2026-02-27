from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, TypedDict


AllowedLabel = Literal[
    "Layer 1",
    "Layer 2",
    "Layer 3",
    "Layer 4",
    "Layer 5",
    "Layer 6",
    "White Matter",
    "Mixed L6/White Matter",
    "Mixed L1/L2",
    "Mixed L2/L3",
    "Mixed L3/L4",
    "Mixed L4/L5",
    "Mixed L5/L6",
]


class EvidenceItem(TypedDict, total=False):
    id: str
    type: str
    items: List[str]
    direction: str
    strength: float
    note: str
    source_agent: str


class ExpertOutput(TypedDict, total=False):
    agent: Literal["Marker", "Pathway", "Spatial", "VLM"]
    agent_score: float
    label_support: Dict[str, float]  # label -> support
    celltype_support: Dict[str, float]  # cell_type -> support
    reasoning: str
    evidence: List[EvidenceItem]
    quality_flags: List[str]


class CriticIssue(TypedDict, total=False):
    type: str
    severity: float
    blocker: bool
    rerun_agent: Optional[str]
    fix_hint: str
    linked_evidence_ids: List[str]


class RerunRequest(TypedDict, total=False):
    rerun: bool
    agents: List[Literal["Marker", "Pathway", "Spatial", "VLM"]]
    reason: str
    needed_inputs: List[str]


class CriticOutput(TypedDict, total=False):
    critic_score: float
    reasoning: str
    feedback_by_agent: Dict[str, List[str]]
    issues: List[CriticIssue]
    rerun_request: RerunRequest


class AnnotationDraft(TypedDict, total=False):
    domain_id: int
    biological_identity: str
    biological_identity_conf: float
    alternatives: List[Dict[str, Any]]
    margin: float
    primary_cell_types: List[str]
    key_evidence: List[str]
    reasoning: str
    # Short free-text summary of the biological function of this domain.
    function: str


class GateResult(TypedDict, total=False):
    peer_score_weighted: float
    peer_score_unweighted_avg: float
    final_score: float
    passed: bool
    fail_reasons: List[str]


class RoundLog(TypedDict, total=False):
    run_info: Dict[str, Any]
    experts: Dict[str, ExpertOutput]
    proposer: Dict[str, Any]
    critic: CriticOutput
    gate: GateResult


@dataclass
class AnnotationSettings:
    label_space: List[str]
    weights: Dict[str, float]
    standard_score: float
    max_rounds: int
    language: str
    log_dir: str
    vlm_required: bool
    vlm_min_score: float
    vlm_image_path: str


