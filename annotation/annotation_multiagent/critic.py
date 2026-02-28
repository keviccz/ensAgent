from __future__ import annotations

import os
from typing import Any, Dict, List

from .llm_clients import chat_json
from .schemas import AnnotationDraft, CriticOutput, ExpertOutput


def _load_kb_text(kb_path: str) -> str:
    try:
        with open(kb_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def critic_agent(
    client: Any,
    model: str,
    *,
    annotation: AnnotationDraft,
    experts: Dict[str, ExpertOutput],
    kb_path: str,
    standard_score: float,
    language: str = "en",
) -> CriticOutput:
    kb = _load_kb_text(kb_path)

    # Compact expert summary for the critic
    expert_brief: Dict[str, Any] = {}
    for k, v in experts.items():
        expert_brief[k] = {
            "agent_score": v.get("agent_score", 0.0),
            "label_support": v.get("label_support", {}),
            "quality_flags": v.get("quality_flags", []),
            "evidence": v.get("evidence", [])[:3],
            "reasoning": (v.get("reasoning") or "")[:400],
        }

    prompt = f"""
You are the Critic Agent for a multi-agent DLPFC annotation system.
You MUST validate the Annotator's annotation against:
1) The provided expert evidence (Marker/Pathway/Spatiality/Visual)
2) The provided knowledge base rules (KB)

Output ENGLISH only. Return JSON only. No markdown.

Standard score threshold: {standard_score}

KB:
{kb}

Annotator annotation (JSON):
{annotation}

Expert brief (JSON):
{expert_brief}

Return a JSON object:
{{
  "critic_score": <0-1>,
  "reasoning": "<3-8 sentences explaining the score and the main problems>",
  "feedback_by_agent": {{
     "Marker": ["...","..."],
     "Pathway": ["..."],
     "Spatiality": ["..."],
     "Visual": ["..."]
  }},
  "issues": [
    {{
      "type": "<enum>",
      "severity": <0-1>,
      "blocker": <true/false>,
      "rerun_agent": "<Marker|Pathway|Spatiality|Visual|None>",
      "fix_hint": "<short actionable hint>",
      "linked_evidence_ids": ["E1","E2"]
    }}
  ],
  "rerun_request": {{
    "rerun": <true/false>,
    "agents": ["Visual"],
    "reason": "<short>",
    "needed_inputs": ["..."]
  }}
}}

Rules:
- If Visual evidence is missing/failed (quality_flags contains vlm_required_failed, vlm_unavailable, vlm_parse_failed, vlm_low_quality), you MUST add a blocker issue type "vlm_required_failed" and request rerun of Visual.
- If you find a White Matter trap violation, mark it as blocker with high severity.
- If the main problem can be fixed WITHOUT rerunning experts (e.g., choose an appropriate Mixed label), set rerun_request.rerun=false and make fix_hint explicitly name the target label(s).
- If the main problem needs MORE evidence, you MUST set rerun_request.rerun=true and select the MINIMUM set of agents needed (e.g., Spatiality conflict -> Spatiality; Visual ambiguity -> Visual; marker ambiguity -> Marker).
- IMPORTANT: Pathway evidence is functional/cell-program oriented and may not directly determine cortical layer labels.
"""
    return chat_json(
        client,
        model,
        [{"role": "system", "content": "Output JSON only."}, {"role": "user", "content": prompt}],
        max_tokens=1600,
        temperature=0.2,
    )  # type: ignore[return-value]

