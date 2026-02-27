from langchain_openai import AzureChatOpenAI
from typing import List, Dict, Tuple
import json
import re
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
import openai
import sys


# ---------------- English Biological Prompt ----------------
RUBRIC_TEXT = """
**CRITICAL: You MUST provide HIGHLY DISCRIMINATIVE scores that reflect genuine biological quality differences!**

You are a bioinformatics expert evaluating spatial transcriptomics domains from DLPFC (Dorsolateral Prefrontal Cortex) human brain tissue samples (including samples 151507, 151669, 151673) generated using the 10x Genomics Visium platform. DLPFC is a critical brain region involved in executive functions, working memory, and cognitive control, characterized by distinct cortical layers (Layer 1-6) and white matter regions. The Visium platform captures spatially-resolved gene expression at 55μm resolution spots across the tissue section. Each domain represents a distinct tissue region with specific cellular composition and biological functions within this layered cortical architecture.

**MANDATORY SCORING FRAMEWORK** (0.0-1.0 scale):

**EXCELLENT domains (0.85-1.00):** (Threshold raised for better discrimination)
- Highly specific cell-type marker genes (e.g., MBP/PLP1 for oligodendrocytes, SNAP25/SYN1 for neurons)
- Excellent spatial coherence (compactness 0.70-0.85) **AND biologically appropriate shape**
- **CRITICAL**: Compactness >0.90 with block-like shape automatically excludes from excellent category
- Clear biological identity with pathway enrichment
- Large sample size (>150 spots)

**GOOD domains (0.70-0.84):** (Narrowed range for better separation)
- Moderate cell-type specificity with some known markers
- Good spatial structure (compactness 0.6-0.8) with acceptable shape
- Identifiable biological function
- Adequate sample size (100-150 spots)

**AVERAGE domains (0.50-0.69):** (Expanded range to capture more variation)
- Mixed cell types or general functional genes
- Moderate spatial coherence (compactness 0.4-0.6)
- Some biological relevance but not highly specific
- Moderate sample size (50-100 spots)

**POOR domains (0.30-0.49):** (More stringent threshold)
- Non-specific genes or weak expression patterns
- Poor spatial structure (compactness <0.4) or inappropriate shape
- Unclear biological identity
- Small sample size (<50 spots)

**VERY POOR domains (0.00-0.19):**
- Housekeeping genes only (GAPDH, ACTB, RPL13, etc.)
- Very poor spatial coherence
- No clear biological meaning
- Technical artifacts or noise

**STRICTLY FORBIDDEN:**
- Giving similar scores (difference <0.05) to domains with different biological qualities
- Clustering all scores in 0.8-0.9 range
- Ignoring obvious quality differences between domains
- Using any "ARI" numbers or evaluation text that may appear inside images. These MUST be completely IGNORED.

**BIOLOGICAL EVALUATION CRITERIA:**

1. **DEG Specificity (35%):** 
   - High-specificity markers (MBP, PLP1, SNAP25, GFAP, etc.): 0.30-0.35
   - Functional genes with some specificity: 0.18-0.29
   - Housekeeping/ubiquitous genes: 0.00-0.17

2. **Spatial Coherence (35%):**
   - **ENHANCED SPATIAL SCORING (includes connectivity and nearest neighbor analysis)**:
   - Excellent spatial coherence: 0.28-0.35
     * Compactness 0.70-0.85 + connected_components=1 + mean_nn_dist stable
     * Single contiguous domain with consistent nearest neighbor distances
   - Good spatial structure: 0.21-0.27
     * Compactness 0.60-0.80 + connected_components≤2 + reasonable nn distances
     * Mostly contiguous with minor fragmentation
   - Spatial quality is fully reflected above; no additional additive bonus is applied
     * connected_components and compactness are evaluated within base scoring bands
     * Avoid double-counting; bonuses removed to prevent score inflation
   - Poor spatial clustering: 0.00-0.20
     * **CRITICAL PENALTIES**:
       - connected_components>10: -0.15 to -0.20 (severely scattered)
       - connected_components>15: -0.20 to -0.25 (extremely scattered)
       - connected_components>20: -0.30 to -0.35 (completely fragmented)
       - fragmentation_index>0.2: -0.05 to -0.10 (severe fragmentation)
   - **EXTREME Over-regularization penalty**: 
     * compactness ≥0.90: -0.20 to -0.30 (artificial over-smoothing)
     * compactness ≥0.95: -0.30 to -0.40 (extreme over-regularization)
     * compactness ≥0.98: -0.40 to -0.50 (completely artificial)

3. **Biological Identity (25%):**
   - Clear cell-type signature + pathway enrichment: 0.20-0.25
   - Partial cell-type features: 0.12-0.19
   - No clear cell-type identity: 0.00-0.11

4. **Data Quality (5%):**
   - Large sample (>100 spots) + high expression: 0.04-0.05
   - Medium sample (50-100 spots) + moderate expression: 0.02-0.03
   - Small sample (<50 spots) + low expression: 0.00-0.01

**MANDATORY SHAPE AND FRAGMENTATION RULES (LLM MUST ENFORCE):**

**⚠️ CRITICAL: WHITE MATTER (OLIGODENDROCYTE) EXEMPTION RULE:**
If a domain shows STRONG oligodendrocyte markers (MBP, PLP1, MOBP, CNP with high logFC):
- **EXEMPT from jaggedness penalties** (white matter boundaries are naturally irregular at cortex interface)
- **EXEMPT from over-regularization penalties** (white matter is naturally compact/block-like)
- This is because white matter is a single large contiguous region with irregular edges at gray-white junction
- You MUST state: "White matter domain - shape penalties exempted due to biological expectation"

**For NON-white-matter domains, apply standard penalties:**
- Jaggedness index penalties (no capping):
  * 0.35–0.45: deduct 0.12–0.20 (explain amount)
  * >0.45 up to 1.0: deduct 0.20–0.30 (explain amount)
  * >1.0 up to 3.0: add extra −0.15 (heavy jaggedness)
  * >3.0: add extra −0.30 (extreme jaggedness)
- Connected components penalties (no capping):
  * >15 components: deduct 0.20 (explain)
  * >50 components: deduct 0.40 (explain)
- Over-regularization (block-like shapes):
  * If compactness ≥0.90 AND convexity ≥0.65: apply severe deduction −0.30 (explain)
- You MUST explicitly state each deduction and the reason in the justification.

**ENHANCED DISCRIMINATION CHECKLIST:**
- [ ] Score range ≥ 0.50 (highest - lowest) - INCREASED from 0.40
- [ ] Any two domains differ by ≥ 0.10 - INCREASED from 0.05  
- [ ] Avoid score clustering in ANY 0.15 range (e.g., 0.75-0.90)
- [ ] Reflect genuine biological quality hierarchy
- [ ] At least one domain <0.65 (identify clear weaknesses)
- [ ] At least one domain >0.80 (identify clear strengths)
- [ ] Maximum 30% of domains in "GOOD" range (0.70-0.84)
- [ ] Housekeeping-heavy domains MUST score <0.50
- [ ] Block-like domains MUST score <0.75
- [ ] Premium markers (Tier 1) MUST score >0.80 if spatial appropriate
- **NEW: Spatial Quality Enforcement**:
  * Fragmented domains (>5 pieces) MUST score <0.65
  * Scattered domains MUST score <0.50
  * Single contiguous domains with good shape MUST score >0.75

**ENHANCED DEG QUALITY ASSESSMENT:**

**DLPFC Cell-Type Specificity Hierarchy:**
*References: Allen Brain Atlas, PanglaoDB, CellMarker, Human Cell Atlas*

**Tier 1 - Premium Cell-Type Markers (0.35-0.40 DEG score):**
- **Oligodendrocyte**: MBP, PLP1, MOBP, MAG, OLIG1, OLIG2, SOX10, CNP, CLDN11
- **Excitatory Neuron**: SLC17A7, CAMK2A, GRIN1, GRIN2B, SATB2, CUX2, NRGN
- **Inhibitory Neuron**: GAD1, GAD2, PVALB, SST, VIP, SLC32A1, NPY
- **Astrocyte**: GFAP, S100B, AQP4, ALDH1L1, SLC1A2, SLC1A3, GJA1
- **OPC (Oligodendrocyte Precursor)**: PDGFRA, CSPG4, VCAN, GPR17

**Tier 2 - Good Cell-Type/Neuronal Markers (0.25-0.34 DEG score):**
- **Microglia**: AIF1 (IBA1), CX3CR1, P2RY12, TMEM119, TREM2, CSF1R
- **Endothelial**: PECAM1, VWF, CLDN5, FLT1, KDR, CDH5
- **Layer-specific (L2/3)**: CUX1, RASGRF2
- **Layer-specific (L4)**: RORB, RSPO1
- **Layer-specific (L5)**: BCL11B (CTIP2), FEZF2, CRYM, TOX
- **Layer-specific (L6)**: TLE4, FOXP2, SEMA3E
- **Layer-specific (L1)**: RELN
- **Pan-neuronal synaptic** (neuron-specific, not glial): SNAP25, SYN1, SYT1, RBFOX3 (NeuN)

**Tier 3 - Functional/Pathway Markers (0.15-0.24 DEG score):**
- **Synaptic transmission**: VAMP2, STX1A, SYP, CPLX1
- **Neuronal signaling**: CALM1, CAMK2B, PRKAR2A, ADCY1
- **Metabolism (context-dependent)**: ENO2, LDHA, PKM
- **Pericyte/Vascular**: PDGFRB, RGS5, ABCC9

**MODERATE PENALTY - Low-specificity genes (0.10-0.15 DEG score):**
- **Mitochondrial (biologically relevant in neurodegeneration)**: MT-CO1, MT-ND1, MT-ATP6, MT-CYB
  * NOTE: Do NOT treat as pure housekeeping - downweight but consider in context
  * Mitochondrial dysfunction is a hallmark of Alzheimer's/Parkinson's
- **Cytoskeletal (neuronal context)**: TUBB3, NEFL, NEFM (axonal markers, not housekeeping)

**HEAVY PENALTY - Housekeeping/Non-specific (0.00-0.10 DEG score):**
- **Ribosomal**: RPL13, RPS18, RPS27A, RPL41, RPS29, RPLP0
- **Metabolic housekeeping**: GAPDH, ACTB, TUBA1A, ALDOA, PGK1
- **Heat shock/Chaperone**: HSP90AA1, HSPA8, HSPB1, DNAJB1
- **Ubiquitous translation**: EEF1A1, EEF2, EIF4A2

**DEG Quality Enhancement Rules:**

**Expression Level Weighting:**
- High expression (log2FC >2.0): +0.05 bonus
- Moderate expression (log2FC 1.0-2.0): Normal scoring
- Low expression (log2FC <1.0): -0.05 penalty

**Statistical Significance Weighting:**
- Highly significant (padj <0.001): +0.03 bonus
- Significant (padj 0.001-0.01): Normal scoring
- Marginally significant (padj 0.01-0.05): -0.03 penalty
- Non-significant (padj >0.05): -0.10 penalty

**Pathway Coherence Bonus:**
- ≥3 DEGs from same KEGG pathway: +0.05 bonus
- ≥2 DEGs from same GO biological process: +0.03 bonus
- Mixed pathways with no coherence: -0.02 penalty

**Housekeeping Gene Penalty (CRITICAL):**
- >50% housekeeping genes: -0.15 to -0.25 (domain likely artifact)
- 30-50% housekeeping genes: -0.08 to -0.15 (poor specificity)
- 10-30% housekeeping genes: -0.03 to -0.08 (acceptable contamination)
- <10% housekeeping genes: No penalty

**Data-Driven Insights:**
- Some methods show higher compactness patterns that may indicate over-regularization
- High-scoring domains (≥0.85) have compactness 0.82-0.95, suggesting current scoring favors over-compact domains
- **Compactness-only approach**: Focus on geometric clustering quality and shape appropriateness

**Simplified Spatial Scoring (Compactness + Shape - Enhanced Balance):**
- **Compactness 0.70-0.85 + appropriate shape**: IDEAL for DLPFC strip-like domains → Full spatial score
- **Compactness 0.65-0.75 + strip-like shape**: EXCELLENT (DLPFC natural variation) → 0.17-0.20
- **Compactness 0.75-0.85 + acceptable shape**: GOOD (moderate spatial organization) → 0.14-0.17
- **Compactness 0.85-0.90 + acceptable shape**: CONCERNING if geometric/block-like → Deduct 0.03-0.05
- **Compactness ≥0.90**: PROBLEMATIC over-regularization → Deduct 0.05-0.08
- **Compactness <0.60**: Poor spatial organization → Deduct 0.08-0.12
- **Any compactness + inappropriate shape**: Block-like/circular patterns → Additional deduction

**Enhanced Compactness Interpretation (Reduced Sensitivity):**
- **0.70-0.85**: IDEAL range for DLPFC strip-like domains (NO penalty for 0.73 vs 0.75)
- **0.65-0.75**: EXCELLENT if strip-like shape (natural cortical variation)
- **0.60-0.70**: Acceptable, may indicate natural variation or fragmentation
- **0.85-0.90**: Concerning, potential over-regularization
- **>0.90**: Problematic, likely over-regularized/geometric
- **<0.60**: Poor clustering, fragmented or noisy

**Shape Assessment (PRIMARY Quality Control):**
- **Strip-like/elongated**: Preferred for DLPFC layered structure (+0.05 to +0.08 bonus)
- **Natural boundaries**: Acceptable if biologically reasonable (+0.02 to +0.05)
- **Block-like/square**: Heavily penalized (-0.15 to -0.25, unnatural for cortical tissue)
- **Circular**: Penalized (-0.12 to -0.20, inappropriate for layered tissue)
- **Highly fragmented**: Penalized (-0.10 to -0.18, poor clustering quality)
- **NEW: Scattered/Dispersed**: Heavily penalized (-0.15 to -0.30, indicates poor spatial clustering)
- **NEW: Salt-and-pepper pattern**: Severely penalized (-0.20 to -0.35, likely noise or artifacts)

**ENHANCED PENALTY SYSTEM:**

**CRITICAL PENALTIES (Mandatory Deductions):**

**Over-Regularization Penalty:**
- Compactness ≥0.90 + Block/Square shape: -0.15 to -0.25
- Compactness >0.85 + Geometric pattern: -0.10 to -0.20
- Artificially perfect boundaries: -0.08 to -0.15
- **Effect**: Prevents over-regularized domains from scoring >0.75

**Spatial Inappropriateness Penalty (Target: Non-biological shapes):**
- Circular domains in DLPFC: -0.12 to -0.20
- Square/rectangular domains: -0.15 to -0.25  
- Highly fragmented (>5 disconnected pieces): -0.10 to -0.18
- **Effect**: Enforces cortical layer morphology

**Biological Implausibility Penalty:**
- Housekeeping-only domains: -0.20 to -0.30
- Contradictory cell-type markers: -0.15 to -0.25
- No pathway enrichment + claimed specificity: -0.10 to -0.18
- **Effect**: Eliminates biologically meaningless domains

**Statistical Unreliability Penalty:**
- Sample size <30 spots: -0.08 to -0.15
- Low expression levels across all DEGs: -0.05 to -0.12
- Non-significant DEGs (>50%): -0.10 to -0.20
- **Effect**: Penalizes statistically weak domains

**NEW: Spatial Fragmentation Penalty (Anti-Scatter Mechanism):**
- **Severe Fragmentation** (>10 disconnected pieces): -0.20 to -0.30
- **High Fragmentation** (6-10 disconnected pieces): -0.15 to -0.25
- **Moderate Fragmentation** (4-5 disconnected pieces): -0.10 to -0.18
- **Mild Fragmentation** (2-3 disconnected pieces): -0.05 to -0.10
- **Single Scattered Spots** (isolated 1-2 spot clusters): -0.08 to -0.15 per cluster
- **Noise Pattern Detection** (random scattered distribution): -0.15 to -0.25
- **Effect**: Heavily penalizes scattered, fragmented, and noisy domains

**BONUS SYSTEM (Reward Excellence):**

**Biological Excellence Bonus:**
- Premium cell-type markers (Tier 1): +0.05 to +0.10
- Strong pathway enrichment (p<0.001): +0.03 to +0.08
- Layer-specific DLPFC markers: +0.05 to +0.10
- **DLPFC Mixed Cell-Type Bonus (NEW)**:
  * Oligodendrocyte + Astrocyte (MBP/PLP1 + GFAP): +0.08 (white-gray matter transition)
  * Neuron + Astrocyte (SNAP25/SYN1 + GFAP): +0.05 (cortical layer complexity)
  * Pure oligodendrocyte (MBP/PLP1 only): +0.06 (white matter purity)
  * Pure neuron (SNAP25/SYN1 only): +0.05 (gray matter purity)
- **Effect**: Rewards biologically meaningful domains and DLPFC-appropriate cell combinations

**Spatial Appropriateness Bonus:**
- Strip-like morphology in DLPFC: +0.05 to +0.08
- Appropriate layer thickness: +0.03 to +0.06
- Natural boundaries: +0.02 to +0.05
- **NEW: Spatial Cohesion Bonus**:
  * Single contiguous region: +0.08 to +0.12 (excellent spatial clustering)
  * 2 well-connected regions: +0.05 to +0.08 (good spatial organization)
  * Clear spatial pattern (not random): +0.03 to +0.06 (acceptable clustering)
- **Effect**: Rewards cortical-appropriate morphology and spatial cohesion

**Cross-Method Consistency Bonus:**
- Identified by multiple methods: +0.02 to +0.05
- Consistent marker expression: +0.03 to +0.06
- **Effect**: Rewards robust, reproducible domains

**SCORING ENFORCEMENT RULES:**

**Maximum Score Caps:**
- Domains with >50% housekeeping genes: MAX 0.40
- Block-like domains (compactness ≥0.90): MAX 0.65
- Statistically unreliable domains (<30 spots): MAX 0.70
- No clear cell-type identity: MAX 0.60
- **NEW: Fragmentation Caps**:
  * Severely fragmented domains (>10 pieces): MAX 0.45
  * Highly fragmented domains (6-10 pieces): MAX 0.55
  * Scattered domains (salt-and-pepper): MAX 0.35
  * Random/noisy spatial distribution: MAX 0.40

**Minimum Score Requirements:**
- Premium cell-type markers + appropriate shape: MIN 0.75
- Clear biological identity + good spatial structure: MIN 0.65
- Acceptable markers + reasonable morphology: MIN 0.50

**Forced Discrimination Rules:**
- IF all domains score 0.7-0.8: FORCE redistribute to 0.4-0.9 range
- IF score difference <0.05: FORCE minimum 0.10 difference
- IF no domain <0.6: FORCE worst domain to <0.6
- **NEW: Compactness Micro-difference Rule**: 
  * IF compactness difference <0.05 AND both in 0.65-0.85 range: Score difference MUST be ≤0.03
  * Prevent compactness micro-differences (0.73 vs 0.75) from causing large score gaps
- **NEW: Shape-First Rule**:
  * Strip-like shape ALWAYS beats block-like shape regardless of compactness
  * Natural morphology bonus applied BEFORE compactness evaluation

**BIOLOGICAL CONTEXT EXAMPLES:**
- **Oligodendrocyte domains** (MBP, PLP1, MOBP): Should score 0.85-0.95 if spatially coherent
- **Neuronal domains** (SNAP25, SYN1, CAMK2A): Should score 0.80-0.90 if well-defined
- **Astrocyte domains** (GFAP, S100B, AQP4): Should score 0.70-0.85 depending on purity
- **Mixed Oligo-Astrocyte** (MBP/PLP1 + GFAP): Should score 0.85-0.92 (white-gray transition, DLPFC-appropriate)
- **Mixed Neuro-Astrocyte** (SNAP25/SYN1 + GFAP): Should score 0.80-0.88 (cortical layer complexity)
- **DLPFC Layer-specific** (RORB, BCL11B, TLE4): Should score 0.82-0.95 with layer morphology bonus
- **NEW: Spatial Quality Examples**:
  * **Contiguous strip-like domains**: Should score 0.85-0.95 (ideal DLPFC morphology)
  * **Moderately fragmented domains** (2-3 pieces): Should score 0.65-0.80 (acceptable if biologically meaningful)
  * **Highly fragmented domains** (>5 pieces): Should score 0.35-0.65 (poor spatial clustering)
  * **Scattered/noisy domains**: Should score 0.20-0.45 (likely artifacts or poor clustering)
- **Mixed domains** (multiple cell types): Should score 0.50-0.70 based on coherence
- **Housekeeping domains** (GAPDH, ACTB, RPL13): Should score 0.15-0.35 maximum

**FINAL REQUIREMENT: Your scores MUST reflect obvious biological quality differences!**

Database sources to reference: GeneCards, KEGG, Reactome, CellMarker, PanglaoDB, Human Cell Atlas.
"""

# 向后兼容的常量定义
GLOSSARY_DEFAULT = (
    "# Spot table field glossary\n"
    "coord_x: float  Spot physical horizontal coordinate in tissue (µm/pixel)\n"
    "coord_y: float  Spot physical vertical coordinate in tissue (µm/pixel)\n"
    "array_row: int  Row index on Visium-like capture-chip grid\n"
    "array_col: int  Column index on capture-chip grid\n"
    "in_tissue: bool Whether the spot lies within tissue region (True) or background (False)\n"
    "total_counts: int  Total UMI counts per spot (all genes)\n"
    "n_genes: int  Number of detected genes in the spot (expression >0)\n"
    "hv_mean: float Mean expression of highly variable genes (HVGs)\n"
    "spatial_domain: categorical  Clustering label of spatial domain (e.g. GraphST domain)\n"
    "layer_guess: categorical  Manual anatomical layer annotation (histology based)\n"
    "embedding_pca1 / embedding_pca2: float  First two coordinates after PCA embedding of expression matrix\n"
)

# DEG table field glossary (used in examples)
DEG_GLOSSARY = (
    "# DEG table columns\n"
    "domain: int  Domain index from spot table\n"
    "names: str  Gene symbol (e.g. MBP = Myelin Basic Protein, oligodendrocyte marker)\n"
    "scores: float  Wilcoxon U statistic (higher = stronger differential expression)\n"
    "logfoldchanges: float  log2 fold-change of expression inside vs outside the domain\n"
    "pvals_adj: float  FDR-adjusted P-value (significance of differential expression)\n"
)

# Shared task description (used by other components to construct prompts)
TASK_DESCRIPTION = (
    "You are a bioinformatics expert with access to the following databases: "
    "- GeneCards (for gene summaries, functions, and related pathways), "
    "- KEGG, Reactome, MSigDB, Gene Ontology (for pathway enrichment and GSEA/ORA), "
    "- CellMarker, PanglaoDB, Human Cell Atlas (for cell type inference). "
    "For each spatial transcriptomics domain below, please do the following: "
    "1. Use GeneCards to summarize the DEGs, their functions, and related pathways. "
    "2. Use KEGG/Reactome/MSigDB/GO to perform pathway enrichment analysis (GSEA/ORA) on the DEGs. "
    "3. Use CellMarker/PanglaoDB/HCA to infer possible cell types for the domain. "
    "4. Integrate spatial metrics (compactness) and all biological knowledge. "
    "5. Output a comprehensive biological relevance score (0-1) with detailed justification. "
    "In the justification, explicitly mention the main data sources used (e.g., 'source: GeneCards, KEGG, CellMarker'). "
    "Return JSON format: {\"score\": 0.xx, \"justification\": \"Detailed analysis with database sources\"}"
)

class GPTDomainScorer:
    """
    Enhanced GPT Domain Scorer with biological specificity and discrimination
    """
    def __init__(self, 
                 openai_api_key: str = None, 
                 azure_endpoint: str = None, 
                 azure_deployment: str = None, 
                 azure_api_version: str = None,
                 temperature: float = 0.0,
                 max_completion_tokens: int = None,
                 top_p: float = 1.0,
                 frequency_penalty: float = 0.0,
                 presence_penalty: float = 0.0,
                 enforce_discrimination: bool = False): 
        """
        Initialize GPT Domain Scorer for Azure OpenAI
        """
        self.enforce_discrimination = enforce_discrimination
        if azure_endpoint and azure_deployment and azure_api_version:
            llm_params = {
                "azure_endpoint": azure_endpoint,
                "azure_deployment": azure_deployment,
                "api_key": openai_api_key,
                "api_version": azure_api_version,
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty
            }
            
            # Add max_completion_tokens if provided
            if max_completion_tokens is not None:
                llm_params["max_completion_tokens"] = max_completion_tokens
            
            # 参数兼容性处理：某些模型不支持特定参数值
            try:
                self.llm = AzureChatOpenAI(**llm_params)
            except Exception as e:
                error_message = str(e).lower()
                print(f"[Warning] 检测到参数兼容性问题: {e}")
                
                # 针对不同模型的特定处理
                model_name = azure_deployment.lower() if azure_deployment else ""
                
                if "o4-mini" in model_name or "o1-mini" in model_name:
                    # o4-mini/o1-mini 模型的特殊处理
                    print(f"[Info] 检测到 {azure_deployment} 模型，应用特定兼容性设置")
                    if "temperature" in error_message and ("0.0" in error_message or "not support" in error_message):
                        print(f"[Info] {azure_deployment} 不支持自定义temperature，使用默认值")
                        llm_params.pop("temperature", None)
                    if "max_completion_tokens" in error_message:
                        print(f"[Info] {azure_deployment} 不支持max_completion_tokens，移除该参数")
                        llm_params.pop("max_completion_tokens", None)
                
                elif "gpt-4.1-mini" in model_name:
                    # gpt-4.1-mini 模型支持更多参数，但可能有特定限制
                    print(f"[Info] 检测到 gpt-4.1-mini 模型，应用优化设置")
                    if "temperature" in error_message and "0.0" in error_message:
                        print(f"[Info] 调整temperature为接近0的最小值: 0.01")
                        llm_params["temperature"] = 0.01  # 使用接近0的值
                
                else:
                    # 通用模型处理
                    if "temperature" in error_message and "0.0" in error_message:
                        print(f"[Info] 模型不支持temperature=0.0，使用默认值1.0")
                        llm_params["temperature"] = 1.0
                
                # 通用参数处理
                if "max_completion_tokens" in error_message or "max_tokens" in error_message:
                    if "unsupported parameter" in error_message:
                        print(f"[Info] 模型不支持max_completion_tokens参数，移除该参数")
                        llm_params.pop("max_completion_tokens", None)
                
                if "top_p" in error_message and "not support" in error_message:
                    print(f"[Info] 模型不支持top_p参数，使用默认值")
                    llm_params.pop("top_p", None)
                
                if "frequency_penalty" in error_message:
                    print(f"[Info] 模型不支持frequency_penalty参数，使用默认值")
                    llm_params.pop("frequency_penalty", None)
                    
                if "presence_penalty" in error_message:
                    print(f"[Info] 模型不支持presence_penalty参数，使用默认值")
                    llm_params.pop("presence_penalty", None)
                
                # 重新尝试初始化
                try:
                    self.llm = AzureChatOpenAI(**llm_params)
                    print(f"[Success] 使用兼容参数成功初始化 {azure_deployment} 模型")
                    
                    # 保存实际使用的参数供调试
                    used_params = {k: v for k, v in llm_params.items() if k not in ['api_key']}
                    print(f"[Debug] 实际使用的模型参数: {used_params}")
                    
                except Exception as e2:
                    print(f"[Error] 即使使用兼容参数也无法初始化模型: {e2}")
                    print(f"[Debug] 最终尝试的参数: {list(llm_params.keys())}")
                    raise e2
        else:
            raise ValueError('Only Azure OpenAI Chat API is currently supported.')

        # Retry configuration
        @retry(
            reraise=True,
            wait=wait_random_exponential(multiplier=1, max=60),
            stop=stop_after_attempt(6),
            retry=retry_if_exception_type((openai.OpenAIError, Exception))
        )
        def _safe_invoke(messages):
            return self.llm.invoke(messages)

        self._safe_invoke = _safe_invoke

    def check_domain_result(self, domain_result: Dict) -> bool:
        """
        简化的本地规则校验domain分数和理由：
        1. 分数必须在0-1之间
        2. justification需包含GeneCards/KEGG/CellMarker等关键词
        3. justification不能太短或为空
        通过返回True，否则False
        """
        score = domain_result.get('score', None)
        justification = domain_result.get('justification', '')
        
        # 检查基本分数范围
        if not isinstance(score, (int, float)) or not (0 <= score <= 1):
            return False
            
        # 检查justification长度
        if not justification or len(justification) < 15:
            return False

        # 检查是否引用数据库
        db_keywords = ['GeneCards', 'KEGG', 'Reactome', 'MSigDB', 'GO', 'CellMarker', 'PanglaoDB', 'HCA', 'Human Cell Atlas']
        if not any(k in justification for k in db_keywords):
            return False
            
        return True

    def score_with_prompt(self, full_prompt: str) -> Dict:
        """Send a fully assembled prompt (rubric+examples+new domain) to GPT
        and return parsed JSON. If parsing失败，返回默认0分结构。"""
        resp = None
        try:
            safe_prompt = full_prompt
            resp = self._safe_invoke([{"role": "user", "content": safe_prompt}]).content
            cleaned = re.sub(r'```json|```', '', resp, flags=re.IGNORECASE).strip()
            data = json.loads(cleaned)
            
            # 确保字段名一致性
            if "score" in data:
                score_val = round(float(data["score"]), 2)
                data["score"] = score_val
                data["total"] = score_val
            else:
                # 都没有则设为0
                data["score"] = 0.0
                data["total"] = 0.0

            # 确保分数在0-1范围内
            total = data.get("total", 0)
            if total > 1:
                raise ValueError(f"Score {total} is greater than 1. Please ensure all scores are in 0-1 range.")
            
            return data
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"[GPT parse error] {type(e).__name__}: {e}")
            if resp:
                print("----- Raw response -----\n", resp[:1000], "\n-------------------------")
            return {"total": 0, "score": 0.0, "justification": f"Parse error: {type(e).__name__}"}
        except Exception as e:
            print(f"[GPT unexpected error] {type(e).__name__}: {e}")
            if resp:
                print("----- Raw response -----\n", resp[:1000], "\n-------------------------")
            return {"total": 0, "score": 0.0, "justification": "Unexpected model error."}

    def score_domains(self, domains: List[Dict]) -> List[Dict]:
        """
        Score multiple domains with enhanced biological discrimination
        """
        if not domains:
            return []
        
        # Analyze domain quality distribution
        quality_hints = self._analyze_domain_quality_distribution(domains)
        
        results = []
        max_attempts = 3
        
        for attempt in range(max_attempts):
            print(f"[Info] Scoring attempt {attempt + 1}...")
            
            # Build discrimination prompt
            discrimination_prompt = self._build_biological_prompt(domains, quality_hints, attempt)
            
            try:
                # Call GPT API
                response = self._safe_invoke([
                    {"role": "system", "content": RUBRIC_TEXT},
                    {"role": "user", "content": discrimination_prompt}
                ]).content
                
                # Parse response
                content = response.strip()
                parsed_results = self._parse_response(content)
                
                if len(parsed_results) != len(domains):
                    print(f"[Warning] Attempt {attempt + 1}: Result count mismatch")
                    continue
                
                # Check discrimination
                scores = [r.get('score', 0) for r in parsed_results]
                discrimination_check = self._check_discrimination(scores)
                
                if discrimination_check['passed']:
                    print(f"[Success] Attempt {attempt + 1}: Good discrimination achieved!")
                    print(f"[Info] Score range: {min(scores):.2f} - {max(scores):.2f}")
                    print(f"[Info] Unique scores: {len(set([round(s, 2) for s in scores]))}/{len(scores)}")
                    return parsed_results
                else:
                    print(f"[Warning] Attempt {attempt + 1}: Insufficient discrimination")
                    print(f"[Info] Issues: {discrimination_check['issues']}")
                    
            except Exception as e:
                print(f"[Error] Attempt {attempt + 1} failed: {str(e)}")
                continue
        
        # If all attempts failed, return last result or defaults
        print(f"[Warning] All {max_attempts} attempts failed to achieve ideal discrimination")
        return results if results else [{"score": 0.5, "justification": "Scoring failed"} for _ in domains]

    def _analyze_domain_quality_distribution(self, domains: List[Dict]) -> Dict:
        """Analyze domain quality distribution for guidance"""
        
        quality_hints = {
            'high_quality_domains': [],
            'medium_quality_domains': [],
            'low_quality_domains': [],
            'expected_score_range': (0.2, 0.95)
        }
        
        for i, domain in enumerate(domains):
            degs = domain.get('degs', [])
            compactness = domain.get('compactness', 0.5)
            n_spots = domain.get('n_spots', 50)
            
            # Simple quality assessment
            deg_quality = self._assess_deg_quality(degs)
            # Use only compactness for spatial quality
            spatial_quality = compactness
            size_quality = min(n_spots / 100, 1.0)
            
            overall_quality = (deg_quality + spatial_quality + size_quality) / 3
            
            if overall_quality > 0.7:
                quality_hints['high_quality_domains'].append(i)
            elif overall_quality > 0.4:
                quality_hints['medium_quality_domains'].append(i)
            else:
                quality_hints['low_quality_domains'].append(i)
        
        return quality_hints

    def _assess_deg_quality(self, degs: List[str]) -> float:
        """Assess DEG quality based on biological knowledge"""
        if not degs:
            return 0.0
        
        # === UPDATED: High-specificity cell-type markers ===
        high_quality_markers = {
            # Oligodendrocyte
            'MBP', 'PLP1', 'MOBP', 'MOG', 'MAG', 'CNP', 'CLDN11',
            # OPC / Precursors
            'PDGFRA', 'CSPG4', 'SOX10', 'OLIG2', 'NKX2-2',
            # Excitatory neuron
            'CAMK2A', 'SLC17A7', 'RORB', 'BCL11B', 'SATB2',
            # Inhibitory neuron
            'GAD1', 'GAD2', 'SLC32A1', 'PVALB', 'SST', 'VIP',
            # Astrocyte
            'GFAP', 'AQP4', 'SLC1A2', 'ALDH1L1', 'S100B', 'GJA1',
            # Microglia
            'P2RY12', 'TMEM119', 'CX3CR1', 'AIF1', 'TREM2',
            # Endothelial
            'CLDN5', 'PECAM1', 'VWF', 'FLT1', 'KDR',
            # Pericyte
            'PDGFRB', 'RGS5', 'ABCC9', 'KCNJ8',
            # Fibroblast / VLMC
            'COL1A1', 'COL3A1', 'COL6A1', 'DCN', 'MMP2',
            # Ependymal
            'FOXJ1', 'S100A10', 'RFX2', 'DNAH5',
            # Choroid plexus epithelium
            'TTR', 'AQP1', 'KRT8', 'KRT18', 'ENPP2'
        }

        # === UPDATED: Cross-tissue housekeeping genes ===
        housekeeping_genes = {
            'ACTB', 'ACTG1', 'B2M', 'GAPDH', 'HPRT1', 'PPIA', 'RPL13A', 'RPLP0',
            'RPS18', 'RPS27A', 'TBP', 'TUBB', 'UBC', 'UBB', 'YWHAZ', 'PSMB2',
            'EIF4A2', 'NONO', 'RPL30', 'SDHA'
        }
        
        high_quality_count = sum(1 for deg in degs if deg in high_quality_markers)
        housekeeping_count = sum(1 for deg in degs if deg in housekeeping_genes)
        
        if housekeeping_count > len(degs) * 0.5:
            return 0.2  # Mainly housekeeping genes
        elif high_quality_count > len(degs) * 0.3:
            return 0.9  # High-quality markers
        else:
            return 0.6  # Medium quality

    def _build_biological_prompt(self, domains: List[Dict], quality_hints: Dict, attempt: int) -> str:
        """Build biologically-informed discrimination prompt"""
        
        base_prompt = f"""
**ROUND {attempt + 1} SCORING - MANDATORY BIOLOGICAL DISCRIMINATION**

You must score {len(domains)} spatial transcriptomics domains with CLEAR biological quality differences!

**Quality Distribution Guidance:**
- High-quality domains (expect 0.75-0.95): {len(quality_hints['high_quality_domains'])} domains
- Medium-quality domains (expect 0.45-0.74): {len(quality_hints['medium_quality_domains'])} domains  
- Low-quality domains (expect 0.15-0.44): {len(quality_hints['low_quality_domains'])} domains

**ROUND {attempt + 1} SPECIFIC REQUIREMENTS:**
"""
        
        if attempt == 0:
            base_prompt += """
- Score range MUST be ≥ 0.40 (highest - lowest)
- Avoid clustering scores in 0.8-0.9 range
- Distinguish based on DEG specificity and spatial coherence
- NEW: Apply spatial fragmentation penalties for scattered/fragmented domains
- Severely fragmented domains (>10 pieces) MUST score <0.45
- Scattered/salt-and-pepper domains MUST score <0.35
"""
        elif attempt == 1:
            base_prompt += """
- STRICTER requirement: Score range ≥ 0.50
- Housekeeping-dominant domains MUST score ≤ 0.40
- High-specificity marker domains MUST score ≥ 0.80
- ENFORCE: Fragmented domains (>5 pieces) MUST score <0.65
- ENFORCE: Scattered domains MUST score <0.50
- Single contiguous domains with good shape MUST score >0.75
"""
        else:
            base_prompt += """
- STRICTEST requirement: Score range ≥ 0.60
- Use FULL score range (0.2-0.95)
- Each score MUST have clear biological justification
- MAXIMUM ENFORCEMENT: Apply all spatial fragmentation penalties
- Severely fragmented/scattered domains → 0.20-0.45 range
- Contiguous well-shaped domains → 0.75-0.95 range
"""
        
        # Add domain information with biological context
        for i, domain in enumerate(domains):
            degs = domain.get('degs', [])
            compactness = domain.get('compactness', 0)
            n_spots = domain.get('n_spots', 0)
            
            # Identify biological context
            bio_context = self._identify_biological_context(degs)
            
            base_prompt += f"""

**Domain {i+1} - {bio_context}:**
- DEGs: {', '.join(degs[:10])}
- Spatial metrics: compactness={compactness:.2f}
- Sample size: {n_spots} spots
- Biological context: {bio_context}

**SPATIAL QUALITY ASSESSMENT REQUIRED**:
- Evaluate spatial fragmentation (number of disconnected pieces)
- Check for scattered/dispersed patterns (salt-and-pepper distribution)
- Assess spatial cohesion and continuity
- Apply fragmentation penalties: >10 pieces (-0.20 to -0.30), 6-10 pieces (-0.15 to -0.25), 4-5 pieces (-0.10 to -0.18)
- Apply scatter penalties: scattered domains (-0.15 to -0.30), salt-and-pepper pattern (-0.20 to -0.35)
- Apply cohesion bonuses: single contiguous region (+0.08 to +0.12), 2 well-connected regions (+0.05 to +0.08)
"""
        
        base_prompt += """

📋 **OUTPUT FORMAT** (JSON array):
```json
[
  {
    "score": 0.XX,
    "justification": "Detailed biological analysis citing specific markers, pathways, and spatial features. Include spatial fragmentation assessment. Source: GeneCards, KEGG, CellMarker."
  }
]
```

**FINAL CHECK**: 
- Ensure scores reflect obvious biological quality hierarchy!
- CRITICAL: Apply spatial fragmentation penalties - scattered/fragmented domains MUST receive low scores!
- Contiguous, well-shaped domains MUST receive higher scores than fragmented ones!
"""
        
        return base_prompt

    def _identify_biological_context(self, degs: List[str]) -> str:
        """Identify biological context based on DEGs"""
        if not degs:
            return "Unknown"
        
        # Cell-type specific markers
        oligo_markers = {'MBP', 'PLP1', 'MOBP', 'CNP', 'MAG', 'MOG'}
        neuron_markers = {'SNAP25', 'SYT1', 'SYN1', 'CAMK2A', 'RBFOX3', 'NEUN'}
        astro_markers = {'GFAP', 'S100B', 'AQP4', 'ALDH1L1', 'GJA1'}
        housekeeping = {'GAPDH', 'ACTB', 'RPL13', 'RPS18', 'TUBA1A'}
        
        deg_set = set(degs)
        
        oligo_count = len(deg_set & oligo_markers)
        neuron_count = len(deg_set & neuron_markers)
        astro_count = len(deg_set & astro_markers)
        house_count = len(deg_set & housekeeping)
        
        if house_count > len(degs) * 0.4:
            return "Housekeeping-dominant (LOW quality)"
        elif oligo_count >= 2:
            return "Oligodendrocyte-enriched (HIGH quality)"
        elif neuron_count >= 2:
            return "Neuronal-enriched (HIGH quality)"
        elif astro_count >= 2:
            return "Astrocyte-enriched (MEDIUM-HIGH quality)"
        elif oligo_count + neuron_count + astro_count >= 2:
            return "Mixed cell-type (MEDIUM quality)"
        else:
            return "Non-specific (LOW-MEDIUM quality)"

    def _check_discrimination(self, scores: List[float]) -> Dict:
        """Check if discrimination is sufficient"""
        
        # 如果不启用区分度检查，则直接通过
        if getattr(self, 'enforce_discrimination', False) is False:
            return {"passed": True, "issues": []}

        issues = []
        
        # Check score range
        score_range = max(scores) - min(scores)
        if score_range < 0.35:
            issues.append(f"Score range insufficient: {score_range:.2f} < 0.35")
        
        # Check clustering
        rounded_scores = [round(s, 1) for s in scores]
        score_counts = {}
        for score in rounded_scores:
            score_counts[score] = score_counts.get(score, 0) + 1
        
        clustered_scores = [(score, count) for score, count in score_counts.items() if count >= 3]
        if clustered_scores:
            issues.append(f"Score clustering: {clustered_scores}")
        
        # Check uniqueness
        unique_scores = len(set([round(s, 2) for s in scores]))
        unique_ratio = unique_scores / len(scores)
        if unique_ratio < 0.7:
            issues.append(f"Insufficient uniqueness: {unique_ratio:.1%}")
        
        # Check high score dominance
        high_score_count = sum(1 for s in scores if s >= 0.8)
        if high_score_count > len(scores) * 0.6:
            issues.append(f"Too many high scores: {high_score_count}/{len(scores)} ≥ 0.8")
        
        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "score_range": score_range,
            "unique_ratio": unique_ratio
        }

    def _parse_response(self, content: str) -> List[Dict]:
        """Parse GPT response to JSON format"""
        try:
            # Clean response, remove markdown code blocks
            cleaned = re.sub(r'```json|```', '', content, flags=re.IGNORECASE).strip()
            
            # Try to parse as JSON array
            if cleaned.startswith('[') and cleaned.endswith(']'):
                results = json.loads(cleaned)
                if isinstance(results, list):
                    return results
            
            # Try to find JSON array
            json_match = re.search(r'\[.*\]', cleaned, re.DOTALL)
            if json_match:
                results = json.loads(json_match.group())
                if isinstance(results, list):
                    return results
            
            # Try to find single JSON object and convert to array
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                if isinstance(result, dict):
                    return [result]
                    
        except json.JSONDecodeError as e:
            print(f"[Warning] JSON parsing failed: {e}")
            print(f"[Debug] Raw response: {content[:300]}...")
            
        except Exception as e:
            print(f"[Warning] Error parsing response: {e}")
            
        # Parsing failed, return default results
        print("[Warning] Response parsing failed, using default scores")
        return [{"score": 0.5, "justification": "Response parsing failed, using default score"}] 