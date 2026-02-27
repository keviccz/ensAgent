from __future__ import annotations

import base64
import os
from typing import Any, Dict, List, Optional

from openai import AzureOpenAI

from .llm_clients import chat_json
from .schemas import ExpertOutput


def _encode_image_to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    # png is fine; Azure examples often use jpeg but data URL mime can be png.
    return f"data:image/png;base64,{b64}"


def marker_celltype_agent(
    client: AzureOpenAI,
    model: str,
    *,
    domain_id: int,
    top_degs: List[Dict[str, Any]],
    rerun_guidance: Optional[str] = None,
    language: str = "en",
) -> ExpertOutput:
    # Include more context so layer-specific markers (often not top-5 by logFC) are visible.
    deg_text = ", ".join([f"{d['gene']}(logFC={d['logfc']})" for d in top_degs[:30]])
    guidance_block = ""
    if rerun_guidance:
        guidance_block = f"""
Rerun context (IMPORTANT):
{rerun_guidance}

When rerun context mentions disagreement:
- Re-check whether the provided DEGs better support neuronal layers vs glial/WM.
- You MUST NOT mention any gene not in the provided list.
"""
    prompt = f"""
You are the Marker/CellType Expert for human DLPFC spatial transcriptomics.
You will be given top marker genes (DEGs with logFC) for one domain.
Your job:
1) Infer major cell types supported by these DEGs.
2) Provide a distribution over the allowed cortical identity labels.
3) Provide a domain-level expert score (0-1) reflecting evidence strength and specificity.

Important:
- Output ENGLISH only.
- You MUST NOT mention any gene that is not listed in the input DEGs.
- Do NOT claim "downregulated/absent" genes unless they explicitly appear in the provided DEGs with negative logFC (rare). If all provided logFC are positive, do not use downregulation language.
- Return JSON only, no markdown.

Layer-marker rules (use as grounded priors when these genes appear in the input):
- Layer 1: RELN, CXCL14, CPLX3
- Layer 2/3 (IT): CUX1, CUX2, CALB1, SATB2
- Layer 4 (IT): RORB (strong), PCP4 (supportive)
- Layer 5: BCL11B (CTIP2), FEZF2, TSHZ2
- Layer 6: TLE4, FOXP2, CRYM
- White Matter / oligodendrocytes: MBP, PLP1, MOG, MAG, MOBP, CNP, CLDN11

Non-specific neuronal genes (do NOT use alone to decide layer): NEFL, NEFM, MAP1B, TUBA1B and other generic axonal/cytoskeletal genes.

Allowed labels:
["Layer 1","Layer 2","Layer 3","Layer 4","Layer 5","Layer 6","White Matter","Mixed L6/White Matter","Mixed L1/L2","Mixed L2/L3","Mixed L3/L4","Mixed L4/L5","Mixed L5/L6","Mixed","Unknown"]

Input DEGs (top): {deg_text}
{guidance_block}

Return JSON with fields:
{{
  "agent": "Marker",
  "agent_score": <0-1>,
  "label_support": {{"Layer 1":0.xx, ... (top 3-5 labels only)}},
  "celltype_support": {{"Excitatory neurons":0.xx, ... (top 5-8 types)}},
  "reasoning": "<2-5 sentences>",
  "evidence": [{{"id":"M1","type":"deg_marker","items":["GENE1","GENE2"],"direction":"up","strength":0.xx,"note":"..."}}],
  "quality_flags": []
}}
"""
    return chat_json(
        client,
        model,
        [{"role": "system", "content": "Output JSON only."}, {"role": "user", "content": prompt}],
        max_tokens=1200,
        temperature=0.2,
    )  # type: ignore[return-value]


def pathway_agent(
    client: AzureOpenAI,
    model: str,
    *,
    domain_id: int,
    top_pathways: List[Dict[str, Any]],
    prior_celltype_support: Optional[Dict[str, float]] = None,
    rerun_guidance: Optional[str] = None,
    language: str = "en",
) -> ExpertOutput:
    # Build a more informative and less noisy pathway summary
    pw_text = "; ".join(
        [
            f"{p['term']}({p.get('direction','')}, NES={p['nes']}, p={p.get('pval','')})"
            for p in top_pathways[:10]
        ]
    )
    pri_ct = prior_celltype_support or {}
    guidance_block = ""
    if rerun_guidance:
        guidance_block = f"""
Rerun context (IMPORTANT):
{rerun_guidance}

When rerun context mentions disagreement:
- Focus on functional programs/cell states and be explicit about uncertainty.
- Do NOT output cortical layer labels.
"""
    prompt = f"""
You are the Pathway Expert for human DLPFC spatial transcriptomics.
You will be given the top enriched pathways for one domain.
Your job:
1) Infer biological processes and likely cell-type / cell-program context.
2) Provide cell type / state support (NOT cortical layer labels).
3) Provide a domain-level expert score (0-1) reflecting pathway coherence and relevance.

Important:
- Output ENGLISH only.
- Do not discuss color quality or image aesthetics.
- Do NOT claim pathways not present in the input list.
- Return JSON only.

Input pathways: {pw_text}

Pathway-to-celltype prior (computed upstream; treat as a constraint, not a suggestion):
- prior_celltype_support: {pri_ct}
{guidance_block}

Return JSON with fields:
{{
  "agent": "Pathway",
  "agent_score": <0-1>,
  "celltype_support": <MUST match prior_celltype_support keys and approximate values>,
  "reasoning": "<2-5 sentences>",
  "evidence": [{{"id":"P1","type":"pathway","items":["TERM"],"direction":"enriched","strength":0.xx,"note":"..."}}],
  "quality_flags": []
}}
"""
    out = chat_json(
        client,
        model,
        [{"role": "system", "content": "Output JSON only."}, {"role": "user", "content": prompt}],
        max_tokens=1200,
        temperature=0.2,
    )  # type: ignore[assignment]

    # Enforce priors to stabilize celltype prediction; LLM is used mainly for reasoning/evidence wording.
    out = dict(out)
    out["agent"] = "Pathway"
    # Pathway is not used for layer prediction; do not output label_support.
    out.pop("label_support", None)
    if prior_celltype_support is not None:
        out["celltype_support"] = prior_celltype_support
    return out  # type: ignore[return-value]


def spatial_anatomy_agent(
    client: AzureOpenAI,
    model: str,
    *,
    domain_id: int,
    spatial_summary: Dict[str, Any],
    rerun_guidance: Optional[str] = None,
    language: str = "en",
) -> ExpertOutput:
    guidance_block = ""
    if rerun_guidance:
        guidance_block = f"""
Rerun context (IMPORTANT):
{rerun_guidance}

When rerun context mentions disagreement with other experts, you MUST:
- Re-check `normalized_depth` (and `normalized_depth_raw` if present) and ensure your label is consistent with depth thresholds.
- Prefer Mixed labels near boundaries rather than extreme superficial/deep labels unless depth is clearly in that regime.
- Explain why your updated spatial label resolves the disagreement, using ONLY fields from Spatial summary.
"""
    prompt = f"""
You are the Spatiality Expert for human DLPFC cortical layers.
Use ONLY spatial priors (center/edge, density/compactness descriptors) and do NOT use gene/pathway evidence.

Important:
- Output ENGLISH only.
- Return JSON only.

Allowed labels:
["Layer 1","Layer 2","Layer 3","Layer 4","Layer 5","Layer 6","White Matter","Mixed L6/White Matter","Mixed L1/L2","Mixed L2/L3","Mixed L3/L4","Mixed L4/L5","Mixed L5/L6","Mixed","Unknown"]

Spatial summary (JSON): {spatial_summary}
{guidance_block}

Return JSON with fields:
{{
  "agent": "Spatial",
  "agent_score": <0-1>,
  "label_support": {{"Layer 1":0.xx, ... (top 3-5 labels only)}},
  "celltype_support": {{}},
  "reasoning": "<2-5 sentences>",
  "evidence": [{{"id":"S1","type":"spatial","items":["edge/center","density"],"direction":"prior","strength":0.xx,"note":"..."}}],
  "quality_flags": []
}}

Guidance:
- Use `normalized_depth` if present (0=superficial, 1=deepest). Prefer mixed labels near boundaries:
  * depth <0.15 -> Mixed L1/L2 or Layer 1/2
  * 0.15-0.35 -> Mixed L2/L3 or Layer 2/3
  * 0.35-0.50 -> Mixed L3/L4 or Layer 3/4
  * 0.50-0.70 -> Mixed L4/L5 or Layer 4/5 (use Mixed L4/L5 near boundary)
  * 0.70-0.85 -> Mixed L5/L6 or Layer 5/6
  * >0.85 -> Mixed L6/White Matter, Layer 6, or White Matter depending on depth and density
"""
    return chat_json(
        client,
        model,
        [{"role": "system", "content": "Output JSON only."}, {"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.2,
    )  # type: ignore[return-value]


def vlm_agent(
    client: AzureOpenAI,
    model: str,
    *,
    domain_id: int,
    image_path: str,
    rerun_guidance: Optional[str] = None,
    language: str = "en",
) -> ExpertOutput:
    if not os.path.exists(image_path):
        return {
            "agent": "VLM",
            "agent_score": 0.0,
            "label_support": {"Unknown": 1.0},
            "celltype_support": {},
            "reasoning": f"VLM image not found: {image_path}",
            "evidence": [{"id": "V0", "type": "vlm_required_failed", "items": [image_path], "strength": 1.0}],
            "quality_flags": ["vlm_unavailable", "vlm_required_failed"],
        }

    data_url = _encode_image_to_data_url(image_path)
    guidance_block = ""
    if rerun_guidance:
        guidance_block = f"""
Rerun context (IMPORTANT):
{rerun_guidance}

When rerun context mentions disagreement:
- Double-check you are focusing on the correct domain ID using the legend.
- Re-evaluate band-like morphology and relative position (superficial vs deep) conservatively.
"""
    prompt = f"""
You are the Visual Expert for a DLPFC spatial clustering visualization.
You will be shown ONE figure titled "Spatial Clustering - DLPFC 151507" with a legend mapping Domain IDs 1..7 to colors.

Task:
- Focus ONLY on **Domain {domain_id}** (use the legend for identification).
- Analyze domain-specific spatial structure: contiguity, fragmentation, band-like morphology, boundary smoothness, over-smoothing/blockiness, and anatomical plausibility as a cortical layer band.
- DO NOT comment on color quality (contrast, saturation) or generic image aesthetics.

Output ENGLISH only. Return JSON only (no markdown).

Allowed labels:
["Layer 1","Layer 2","Layer 3","Layer 4","Layer 5","Layer 6","White Matter","Mixed L6/White Matter","Mixed L1/L2","Mixed L2/L3","Mixed L3/L4","Mixed L4/L5","Mixed L5/L6","Mixed","Unknown"]
{guidance_block}

Return JSON with fields:
{{
  "agent": "VLM",
  "agent_score": <0-1>,
  "label_support": {{"Layer 1":0.xx, ... (top 3-5 labels only)}},
  "celltype_support": {{}},
  "reasoning": "<2-5 sentences>",
  "evidence": [{{"id":"V1","type":"vlm_shape","items":["contiguous band","fragmentation"],"direction":"visual","strength":0.xx,"note":"..."}}],
  "quality_flags": []
}}
"""
    try:
        out = chat_json(
            client,
            model,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            max_tokens=1400,
            temperature=0.0,
        )  # type: ignore[assignment]

        # Soft validation: if label_support is missing or empty, mark uncertain rather than hallucinate.
        if not isinstance(out, dict):
            raise ValueError("VLM output is not a JSON object")
        if not out.get("label_support"):
            out = dict(out)
            out["agent"] = "VLM"
            out["agent_score"] = float(out.get("agent_score", 0.0)) * 0.5
            out["label_support"] = {"Unknown": 1.0}
            flags = list(out.get("quality_flags", []) or [])
            flags.append("vlm_uncertain")
            out["quality_flags"] = flags
        return out  # type: ignore[return-value]
    except Exception as e:
        # If the deployment does not support vision, treat as required failure.
        return {
            "agent": "VLM",
            "agent_score": 0.0,
            "label_support": {"Unknown": 1.0},
            "celltype_support": {},
            "reasoning": f"VLM call failed (deployment may not support vision): {type(e).__name__}: {e}",
            "evidence": [{"id": "V_ERR", "type": "vlm_required_failed", "items": [type(e).__name__], "strength": 1.0}],
            "quality_flags": ["vlm_parse_failed", "vlm_required_failed"],
        }


