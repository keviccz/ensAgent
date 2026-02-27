#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
import sys
from datetime import datetime
import time
from typing import List, Dict, Any
from azure_client import AzureOpenAIClient
from image_manager import ImageManager

class AutoClusteringAnalyzer:
    """自动聚类分析器，对domain1-7进行批量分析"""
    
    def __init__(self):
        """初始化自动聚类分析器"""
        self.azure_client = AzureOpenAIClient()
        self.image_manager = ImageManager()
        self.output_dir = "output"
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 存储所有domain的评分结果
        self.all_scores = {}
        # 尝试加载历史评分，支持增量补充
        self._load_existing_scores()
        
        # 为每个domain定制的分析提示
        self.domain_prompts = {
            1: self._get_domain_prompt(1),
            2: self._get_domain_prompt(2),
            3: self._get_domain_prompt(3),
            4: self._get_domain_prompt(4),
            5: self._get_domain_prompt(5),
            6: self._get_domain_prompt(6),
            7: self._get_domain_prompt(7)
        }

    def _load_existing_scores(self) -> None:
        """从已有 JSON 中加载历史评分，方便单 domain 增量运行"""
        try:
            json_filename = "all_domains_scores.json"
            json_path = os.path.join(self.output_dir, json_filename)
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.all_scores = data
        except Exception:
            # 静默失败，避免影响主流程
            pass
    
    def _get_domain_prompt(self, domain_num: int) -> str:
        """Get analysis prompt for specified domain"""
        base_prompt = f"""Please act as a professional bioinformatics expert and perform a comparative analysis of these DLPFC (Dorsolateral Prefrontal Cortex) slice clustering images, focusing on domain{domain_num} region.

**TISSUE CONTEXT**: You are analyzing human DLPFC brain tissue samples (151507) generated using the 10x Genomics Visium spatial transcriptomics platform. DLPFC is a critical cortical region involved in executive functions and working memory, characterized by distinct cortical layers (Layer 1-6) and underlying white matter. The Visium platform captures spatially-resolved gene expression at 55μm resolution spots across the tissue section. This layered architecture contains different cell types including neurons, oligodendrocytes, astrocytes, and microglia arranged in functionally distinct layers.

## Analysis Requirements:

### 1. Cross-Method Clustering Image Comparison (Focus on domain{domain_num})
Based on clustering images provided by all methods, perform cross-method comparison. Main comparison aspects include:

**Continuity Analysis:**
- Assess whether each method's clustering in domain{domain_num} is continuous
- Identify obvious breaks or discontinuous regions
- Evaluate clustering boundary clarity and completeness

**Directional Similarity:**
- Compare clustering directions of different methods in domain{domain_num}
- Assess whether methods' clustering shows similar directionality or trends
- Analyze spatial distribution patterns of clustering

**Effect Consistency:**
- Check whether different methods show consistent clustering effects in domain{domain_num}
- Identify differences in clustering morphology, density, boundaries, etc.
- Evaluate clustering result stability and reproducibility
- **Specificity Detection**: Identify abnormal methods that significantly deviate from mainstream morphology and assess their reasonableness

### 2. Analysis Combined with DLPFC Slice Knowledge
Please combine biological and anatomical knowledge of DLPFC slices:

**Biological Background:**
- DLPFC is an important brain cortical region with layered structure
- Different layers should exhibit different cell types and gene expression patterns
- Spatially adjacent cells typically have similar functional characteristics

**Anatomical Features and Expected Clustering Patterns:**
- Ideal clustering should present clear layered structure with distinct boundaries between layers
- Clustering regions should exhibit strip-like or stripe-like continuous distribution patterns
- Different clustering layers should be arranged roughly parallel, reflecting cortical hierarchical organization
- Each clustering layer should maintain relatively uniform density and continuity internally
- Clustering boundaries should be clear and smooth, avoiding over-segmentation
- **Boundary Smoothness Requirements**: Domain edges should be relatively smooth and natural
  * Jagged, serrated, or heavily irregular boundaries indicate technical quality issues
  * Natural biological boundaries should show smooth transitions rather than artificial roughness
  * Excessive edge irregularity suggests clustering algorithm limitations or parameter issues
- Spatially adjacent regions should have biological relevance
- Clustering results should avoid isolated small blocks or noise points
- Overall clustering patterns should reflect orderly spatial organizational structure
- Excellent clustering should maintain clear spatial boundaries and natural organization
- Subtle boundary changes and natural transitions should be considered as embodiments of biological reasonableness

## CRITICAL: Shape Detection Enhancement

### Mandatory Shape Verification Protocol:
1. **Individual Method Shape Assessment**:
   - Be cautious of block-like/square clusters produced by any method
   - Do NOT be misled by color clarity or boundary sharpness
   
2. **Common Shape Issues**:
   - Block-like clustering: Nearly square/rectangular domains
   - Sharp geometric boundaries: Unnatural, non-biological edges  
   - Over-regularized shapes: Too geometric, not organic layered structure
   
3. **Mandatory Cross-Method Comparison**:
   - Count shape types: Strip-like vs Block-like vs Circular
   - If a method is ONLY block-like while others are strip-like → HEAVY deduction -0.25 to -0.35
   - If a method shape differs significantly from majority → Moderate deduction -0.15 to -0.25
   
4. **Scoring Enforcement Rules**:
   - Unique shape anomaly → Mandatory deduction 0.25-0.35
   - Geometric over-regularization → Deduction 0.15-0.25
   
5. **Verification Checklist**:
   - [ ] Described GraphST specific shape features?
   - [ ] Compared GraphST shape with other methods?
   - [ ] Applied appropriate deductions for shape anomalies?
   - [ ] GraphST final score reflects shape problems?

**WARNING**: Do not be deceived by any method's color contrast or boundary clarity. Focus on SHAPE biological reasonableness!

**Strict Shape Assessment Standards:**
- **Ideal Morphology**: Strip-like, arc-shaped, parallel layered distribution, presenting continuous band-like or stripe-like structures
- **Acceptable Morphology**: Slightly curved strips, irregular but continuous layered structures
- **Problematic Morphology**: Block-like, square, circular, isolated clusters, irregular fragmented distribution
- **Serious Problems**: Complete block-like clustering, multiple isolated circular or square regions

**Specific Scoring Rules:**
- **Strip-like continuous distribution**: Normal scoring
- **Slightly irregular but maintaining strip-like shape**: Normal scoring
- **Obviously block-like/cluster-like distribution**: Deduct 0.15-0.20 points
- **Isolated circular or square regions**: Deduct 0.20-0.25 points

**Common Problem Identification (Special Attention):**
- If any domain shows "jagged boundaries" or "irregular clusters", deduct 0.15 points
- Prioritize "spatial continuity and strip-like shape" over "color vividness"
- Block-like clustering violates DLPFC cortical layered anatomical principles and must be strictly penalized

**Biological Shape Reasonableness Check:**
- DLPFC cortex should present parallel layered structure; any shape violating this principle should be penalized
- Morphology should be evaluated for biological plausibility without fixed priority ordering.
- Clustering shapes should reflect real neuroanatomical structures, not artificial geometric figures

**Cross-Method Shape Feature Assessment (Core Evaluation):**

**Shape Feature Cross-Method Comparison Process:**
1. **Overall Morphology Statistics**: Count main shape features of all 8 methods in domain{domain_num}
2. **Mainstream Morphology Identification**: Determine morphological patterns of majority (≥6 methods) as mainstream standard
3. **Abnormal Morphology Detection**: Identify abnormal methods that significantly deviate from mainstream morphology
4. **Difference Degree Quantification**: Determine deduction magnitude based on shape difference degree

**Specific Shape Comparison Rules:**

**A. When mainstream morphology is strip-like/arc-shaped (6+ methods exhibit):**
- This method is also strip-like/arc-shaped → Normal scoring, no deduction
- This method is slightly irregular strip-like → Minor deduction -0.05
- This method is obviously block-like/square → Heavy deduction -0.20 to -0.30
- This method is circular/cluster-like → Severe deduction -0.25 to -0.35

**B. When mainstream morphology is irregular but continuous (6+ methods exhibit):**
- This method is also irregular continuous → Normal scoring
- This method is regular strip-like → Minor bonus +0.05
- This method is obviously block-like → Moderate deduction -0.15 to -0.25
- This method is fragmented → Heavy deduction -0.20 to -0.30

**C. When mainstream morphology is mixed (no clear dominance):**
- This method conforms to any reasonable morphology → Normal scoring
- This method is obviously unreasonable morphology (block-like/circular) → Moderate deduction -0.10 to -0.20

**Shape Feature Identification Guide:**
- **Strip-like**: Long strip, arc-shaped, layered distribution, length-width ratio >2:1
- **Block-like**: Approximately square, rectangular, length-width ratio close to 1:1, sharp boundaries
- **Circular**: Approximately circular, elliptical clusters, tightly aggregated
- **Fragmented**: Multiple isolated small blocks, lacking continuity
- **Irregular continuous**: Irregular shape but maintaining spatial continuity

**Cross-Method Deduction Execution:**
- **Must Count First**: Clearly state "X methods exhibit strip-like, Y methods exhibit block-like"
- **Must Compare**: Clearly point out "specific differences between this method and mainstream morphology"
- **Must Deduct**: Strictly execute corresponding deductions based on difference degree
- **Must Record**: Clearly mark cross-method comparison deduction values in scoring table

**Specific Issues to Identify and Penalize:**
- **CRITICAL: Serrated/Zigzag Lines and Boundaries**: Focus on linear structure quality
  * **Serrated Lines**: Lines resembling saw teeth or zigzag patterns - deduct 0.15-0.25 points (specify: "serrated line structure -0.2 points")
  * **Jagged Linear Domains**: Domain boundaries with sharp, irregular projections - deduct 0.10-0.20 points (specify: "jagged linear boundaries -0.15 points")
  * **Line Discontinuities**: Connected lines that show breaks or interruptions - deduct 0.12-0.22 points (specify: "line discontinuity interrupts structure -0.18 points")
- **CRITICAL: Domain Adjacency Pattern Assessment**: Analyze spatial neighboring relationships
  * **Adjacency Analysis Protocol**:
    1. Identify which domains are adjacent to this domain{domain_num} (above, below, left, right sides)
    2. Count adjacency patterns across all 8 methods: "X methods show domain{domain_num} adjacent to domainY above, Z methods show domainA below"
    3. Determine mainstream adjacency pattern (≥5 methods showing same pattern)
    4. Apply deductions for methods deviating from mainstream adjacency
  * **Adjacency Deviation Penalties**:
    - **Major Adjacency Deviation**: Missing mainstream adjacent domain - deduct 0.15-0.25 points (specify: "domain{domain_num} lacks adjacent domain3 above while 6+ methods have it -0.2 points")
    - **Moderate Adjacency Deviation**: Different adjacent domain than mainstream - deduct 0.10-0.18 points (specify: "domain{domain_num} connects to domain5 above while mainstream connects to domain3 -0.15 points")
    - **Minor Adjacency Inconsistency**: Partial adjacency mismatch - deduct 0.08-0.12 points (specify: "partial adjacency deviation from mainstream pattern -0.1 points")
  * **Adjacency Consistency Requirement**: Must explicitly state "Domain{domain_num} adjacency: Above=domainX, Below=domainY, consistent/inconsistent with mainstream pattern"
- **Cross-Method Domain Splitting Assessment**: Compare domain splitting patterns across all methods
  * If this method shows unique domain splitting that other methods don't have: deduct 0.15-0.25 points
  * If majority of methods (6+ methods) show similar splitting: no deduction for splitting
  * Must specify: "unique domain splitting compared to other methods -0.2 points" or "splitting consistent with other methods, no deduction"

**Special Attention:**
- If a method is the only block-like/circular while other 7 are strip-like → Must heavily deduct
- If a method's boundary direction is completely opposite to other methods → Must moderately deduct
- If a method's continuity is obviously inferior to other methods → Must lightly deduct
- Cross-method comparison deduction is mandatory and cannot be exempted due to other advantages

### 3. Scoring and Ranking
Based on cross-method comparison and DLPFC slice knowledge:

**CRITICAL: WEIGHTED SCORING SYSTEM (0-1 points):**

**Each method's TOTAL SCORE must be calculated as:**
**TOTAL = (Continuity × 0.3) + (Biological_Reasonableness × 0.3) + (Technical_Quality × 0.4)**

**Scoring Components:**
- **Continuity (0.3 weight)**: Clustering continuity and completeness, allowing reasonable biological variation
  * **CRITICAL LINE CONTINUITY ASSESSMENT**: Evaluate linear structure integrity
    - **Line Breaks/Discontinuities**: If connected lines show breaks or gaps - deduct 0.10-0.20 points (specify: "line discontinuity in central region -0.15 points")
    - **Fragmented Linear Structures**: If continuous lines become fragmented - deduct 0.15-0.25 points (specify: "fragmented line structure -0.2 points")
    - **Complete Line Interruption**: If entire linear domains are broken - deduct 0.20-0.30 points (specify: "complete line interruption -0.25 points")
  * **MANDATORY**: Must specify exact deduction reasons (e.g., "central domain discontinuity -0.15 points", "line breaks in strip structure -0.18 points")
  * Rate this component on 0-1 scale FIRST
  * This score will be MULTIPLIED by 0.3 in final calculation
  
- **Biological Reasonableness (0.3 weight)**: Degree of conformity with DLPFC anatomical structure, **strictly assess shape reasonableness and cross-method consistency**
  * **MANDATORY**: Must specify exact deduction reasons for shape issues:
    - Strip-like/band-like morphology: Normal scoring (specify why: "strip-like matches layer structure")
    - Block-like/cluster-like morphology: Deduct 0.15-0.25 (specify: "block-like morphology violates DLPFC layer anatomy -0.2 points")
    - Isolated circular/square regions: Deduct 0.20-0.25 (specify: "isolated circular regions violate biology -0.25 points")
  * **Cross-method comparison shape deduction (mandatory execution with reasons)**:
    - Slightly inconsistent: Deduct 0.05-0.10 (specify: "slight deviation from mainstream morphology -0.05 points")
    - Obviously inconsistent: Deduct 0.15-0.25 (specify: "only block-like method while others strip-like -0.2 points")
    - Severely inconsistent: Deduct 0.25-0.35 (specify exact morphology difference)
  * **MANDATORY**: Each deduction must cite specific morphology observations
  * Rate this component on 0-1 scale FIRST
  * This score will be MULTIPLIED by 0.3 in final calculation
  
- **Technical Quality (0.4 weight)**: Clustering boundary clarity, spatial organization quality, overall clustering precision
  * **MANDATORY**: Must specify exact deduction reasons for technical issues:
    - Noise/precision problems (specify: "excessive noise points in upper domain -0.05 points", "clustering imprecision -0.1 points")
    - **CRITICAL: Linear Edge Quality Assessment**: Evaluate line smoothness and regularity
      * **Serrated/Zigzag Lines**: Lines with tooth-like, saw-toothed patterns - deduct 0.15-0.25 points (specify: "serrated line patterns -0.2 points")
      * **Jagged Line Segments**: Linear structures with sharp, irregular protrusions - deduct 0.10-0.20 points (specify: "jagged line segments -0.15 points")
      * **Rough Linear Boundaries**: Unsmooth, bumpy line edges - deduct 0.08-0.15 points (specify: "rough linear boundaries -0.12 points")
      * **Fragmented Line Patterns**: Broken, discontinuous linear edges - deduct 0.15-0.25 points (specify: "fragmented line patterns -0.2 points")
      * Smooth, natural linear boundaries: Normal scoring
    - Spatial organization issues (specify: "disorganized spatial structure -0.1 points", "insufficient clustering precision -0.1 points")
    - Domain adjacency problems (specify: "domain{domain_num} adjacency deviates from mainstream pattern -0.15 points", "missing expected adjacent domains -0.2 points")
    - Noise/precision problems (specify exact locations and severity)
  * Focus on boundary definition, spatial coherence, and clustering precision without considering color factors
  * **MANDATORY**: Each deduction must cite specific technical observations including edge quality
  * Rate this component on 0-1 scale FIRST  
  * This score will be MULTIPLIED by 0.4 in final calculation

**MANDATORY CALCULATION EXAMPLE:**
Method X: Continuity=0.8, Biological=0.6, Technical=0.9
TOTAL = (0.8 × 0.3) + (0.6 × 0.3) + (0.9 × 0.4) = 0.24 + 0.18 + 0.36 = 0.78

**Ranking Requirements:**
- Rank all methods based on total score (high to low)
- Provide specific scores and rankings for each method

### 4. Detailed Analysis Report
For each method provide:

**Advantage Analysis:**
- Strengths in continuity, biological reasonableness, technical quality
- Shape reasonableness performance (strip-like, arc-shaped distribution, etc.)
- Particularly outstanding performance and innovations

**Disadvantage Analysis:**
- Existing problems and deficiencies
- **Detailed Shape Problem Analysis**:
  * Whether block-like, cluster-like, circular or square clustering appears
  * Whether boundaries are jagged or irregular
  * Whether it violates biological principles of DLPFC layered structure
- **Cross-Method Consistency Analysis**:
  * Whether this method's domain morphology significantly deviates from other methods
  * Whether specific shape deviations exist (e.g., other methods are strip-like, this method is circular)
  * Degree of specific deviation and impact on scoring
- Continuity and technical quality issues
- Possible improvement directions

**Comprehensive Evaluation:**
- Applicability of this method in domain{domain_num} clustering analysis
- Recommended usage scenarios and precautions

**MANDATORY SCORE CALCULATION FORMAT (For Each Method):**
- **Continuity Score**: X.XX
  * **Scoring Justification**: [REQUIRED] Explain specific reasons for score/deductions:
    - **Multi-Region Analysis** (if applicable): "domain{domain_num} appears in X separate regions - Region A: [analysis], Region B: [analysis], Region C: [analysis]"
    - Line continuity issues (e.g., "line breaks in central domain -0.15 points", "fragmented linear structure -0.2 points")
    - **Regional Fragmentation Impact**: "multi-region distribution causes -0.XX points deduction" (if applicable)
    - Discontinuity locations (e.g., "discontinuity in left domain2 region", "complete line interruption -0.25 points")
    - Which areas/regions show good/poor line continuity and why
    - Specific linear structure observations that influenced the score
- **Biological Reasonableness Score**: X.XX  
  * **Scoring Justification**: [REQUIRED] Explain specific reasons for score/deductions:
    - **Regional Shape Analysis** (if multi-region): "Region A: [shape assessment], Region B: [shape assessment], Region C: [shape assessment]"
    - Shape conformity issues (e.g., "block-like morphology violates layer structure -0.2 points", "strip-like distribution conforms to anatomy")
    - **Cross-Regional Consistency**: "regions show consistent/inconsistent morphology" (if applicable)
    - Cross-method comparison deductions with specific reasons
    - Biological principle violations and their impact
- **Technical Quality Score**: X.XX
  * **Scoring Justification**: [REQUIRED] Explain specific reasons for score/deductions:
    - Noise/precision issues (e.g., "excessive noise points -0.1 points", "clustering imprecision -0.15 points")
    - Linear edge quality issues (e.g., "serrated line patterns -0.2 points", "jagged line segments -0.15 points", "rough linear boundaries -0.12 points", "line discontinuities -0.18 points")
    - Domain adjacency inconsistencies (e.g., "domain{domain_num} lacks adjacent domain3 above while mainstream has it -0.2 points", "connects to domain5 instead of mainstream domain3 -0.15 points")
    - **Multi-region adjacency issues** (if applicable): "Region A adjacency deviates from mainstream -0.XX points", "Region B shows different adjacency pattern -0.XX points"
    - Cross-method splitting issues (e.g., "unique domain splitting not seen in other methods -0.2 points")
    - Spatial organization problems and their severity
    - Technical precision observations
- **Weighted Calculation**: (X.XX×0.3)+(X.XX×0.3)+(X.XX×0.4) = X.XX+X.XX+X.XX = X.XXX
- **Cross-Method Deduction**: -X.XX (if applicable)  
- **Final Score**: X.XXX (after deductions)

**CRITICAL: MULTI-REGION DOMAIN ANALYSIS PROTOCOL:**

**Step 1: Domain Region Identification**
- **Region Detection**: First identify if domain{domain_num} appears as a single continuous region or multiple separate regions
  * If single continuous region: Proceed with standard analysis
  * If multiple separate regions: **MANDATORY** to analyze each region individually
- **Region Naming Convention**: Label each region clearly (e.g., "Region A", "Region B", "Region C")
- **Region Count Documentation**: State total number of distinct regions (e.g., "domain{domain_num} appears in 3 separate regions")

**Step 2: Individual Region Analysis (For Each Separate Region)**
- **Regional Morphology Assessment**: Analyze each region's shape, size, and biological appropriateness independently
  * Region A: Shape characteristics, size, biological plausibility
  * Region B: Shape characteristics, size, biological plausibility  
  * Region C: Shape characteristics, size, biological plausibility
- **Regional Continuity Evaluation**: Assess continuity within each individual region
  * Internal continuity of Region A: Line breaks, fragmentation within the region
  * Internal continuity of Region B: Line breaks, fragmentation within the region
  * Internal continuity of Region C: Line breaks, fragmentation within the region
- **Regional Adjacency Analysis**: Determine neighboring domains for each region separately
  * Region A adjacency: Above=domainX, Below=domainY, Left=domainZ, Right=domainW
  * Region B adjacency: Above=domainX, Below=domainY, Left=domainZ, Right=domainW
  * Region C adjacency: Above=domainX, Below=domainY, Left=domainZ, Right=domainW

**Step 3: Multi-Region Integration Assessment**
- **Cross-Regional Consistency**: Compare morphology and characteristics between different regions of the same domain
  * Are all regions morphologically consistent? (e.g., all strip-like vs mixed shapes)
  * Do regions show similar biological appropriateness?
  * Are regional adjacency patterns logically consistent?
- **Fragmentation Impact Evaluation**: Assess how multi-region distribution affects overall domain quality
  * Minor fragmentation (2-3 well-organized regions): Minor deduction -0.05 to -0.10
  * Moderate fragmentation (4-5 scattered regions): Moderate deduction -0.10 to -0.18
  * Severe fragmentation (6+ disconnected regions): Major deduction -0.18 to -0.25
  * Random scattered distribution: Heavy deduction -0.25 to -0.35

**MANDATORY DOMAIN ADJACENCY ANALYSIS (For Each Method):**
- **Adjacent Domain Identification**: Clearly state which domains are adjacent to domain{domain_num} in this method
  * **For Single Region**: Above: domain_X, Below: domain_Y, Left: domain_Z, Right: domain_W
  * **For Multiple Regions**: List adjacency for each region separately:
    - Region A: Above=domainX, Below=domainY, Left=domainZ, Right=domainW
    - Region B: Above=domainX, Below=domainY, Left=domainZ, Right=domainW
    - Region C: Above=domainX, Below=domainY, Left=domainZ, Right=domainW
- **Cross-Method Adjacency Comparison**: Compare this method's adjacency pattern with other methods
  * State mainstream pattern: "6+ methods show domain{domain_num} adjacent to domain3 above and domain5 below"
  * Identify deviations: "This method shows domain{domain_num} adjacent to domain2 above (deviates from mainstream domain3)"
  * For multi-region cases: Compare each region's adjacency with mainstream single-region pattern
- **Adjacency Scoring Impact**: Apply appropriate deductions for adjacency pattern deviations
  * Consistent with mainstream: "adjacency pattern consistent, no deduction"
  * Deviates from mainstream: "adjacency deviation causes -0.XX points deduction"
  * Multi-region penalty: Additional deduction for fragmentation as specified above

## Output Format Requirements:

Please organize analysis results according to the following structure:

1. **Domain{domain_num} Overall Overview**
2. **Cross-Method Shape Feature Statistics** (Must include)
   - Count main shape features of each method
   - Identify mainstream morphological patterns (≥6 methods' common features)
   - Mark abnormal morphology methods
3. **Detailed Analysis of Each Method** (In ranking order)
   - Each method must include cross-method comparison shape assessment
   - Clearly mark degree of difference from mainstream morphology
   - Detailed explanation of cross-method comparison deduction reasons and values
- **MANDATORY**: End each method analysis with complete scoring justification:
  * **Detailed Score Breakdown with Reasons**:
    - Continuity X.XX: [Specific reason for score/deductions, e.g., "left side discontinuity -0.15 points, right side good continuity +0.05 points"]
    - Bio.Reason X.XX: [Specific reason for score/deductions, e.g., "block-like morphology violates layer structure -0.2 points"]
    - Tech.Quality X.XX: [Specific reason for score/deductions, e.g., "clear boundaries but excessive noise -0.1 points"]
  * Step-by-step weighted calculation: (X.XX×0.3)+(X.XX×0.3)+(X.XX×0.4) = result
  * Any cross-method deductions applied with specific reasons
  * Final score after all calculations
4. **Cross-Method Comparison Deduction Summary** (Mandatory requirement)
   - List all methods subject to cross-method comparison deductions
   - Explain specific deduction reasons and values
   - Verify reasonableness of deductions
5. **Scoring Table and Ranking** (Must include cross-method comparison deduction column)
6. **Recommendations and Conclusions**

**Important: Please clearly list specific scores (0-1 points) for each method in domain{domain_num} in the analysis report, format as follows:**
- BayesSpace: 0.xx
- GraphST: 0.xx
- STAGATE: 0.xx
- stLearn: 0.xx
- SEDR: 0.xx
- BASS: 0.xx
- DR-SC: 0.xx
- IRIS: 0.xx

**And must include a detailed scoring table with WEIGHTED CALCULATION AND DEDUCTION REASONS, format as follows:**

| Method     | Continuity (Reason) | Bio.Reason (Reason) | Tech.Quality (Reason) | Weighted Calc | Cross-Method Deduction | Final Score | Rank |
|------------|---------------------|---------------------|----------------------|---------------|------------------------|-------------|------|
| Method Name| 0.xx (specific deduction reason) | 0.xx (specific deduction reason) | 0.xx (specific deduction reason) | (0.xx×0.3)+(0.xx×0.3)+(0.xx×0.4) | -0.xx | 0.xxx | x |

**CRITICAL**: Each score MUST be accompanied by specific deduction reasons in parentheses.

**CRITICAL TABLE REQUIREMENTS:**
- **Continuity**: Raw score (0-1) for continuity assessment  
- **Bio.Reason**: Raw score (0-1) for biological reasonableness
- **Tech.Quality**: Raw score (0-1) for technical quality
- **Weighted Calc**: SHOW the weighted calculation formula: (Continuity×0.3)+(Bio.Reason×0.3)+(Tech.Quality×0.4)
- **Cross-Method Deduction**: Additional deductions for shape inconsistency with mainstream
- **Final Score**: Weighted total MINUS cross-method deductions
- **Rank**: Final ranking based on Final Score

**Table Description:**
- All raw component scores must be on 0-1 scale
- Weighted Calc column must show the mathematical formula being applied
- Cross-Method Deduction: Shows points deducted due to shape features inconsistent with mainstream
- Final Score: Weighted calculation result after deducting cross-method comparison deductions
- If no cross-method comparison deduction, this column shows "0.00"

**Important Reminders:**
- **Cross-method comparison is a core requirement**: Must first count shape features of all methods, then conduct individual assessments
- **Mandatory deduction mechanism execution**: Methods inconsistent with mainstream morphology must be penalized, no exemptions allowed
- **Shape difference quantification**: Must clearly explain specific degree of difference between each method and mainstream morphology
- **Deduction transparency**: Must display specific numerical values of cross-method comparison deductions in scoring table
- Emphasize biological plausibility; penalize clearly non-biological/geometric shapes; avoid fixed morphology priority
- **Consistency verification**: Ensure methods with similar shape problems receive similar deductions
- **Assessment Process**: 
  1. Count shape feature distribution of 8 methods
  2. Determine mainstream morphological patterns (≥6 methods)
  3. Identify abnormal methods and quantify differences
  4. Execute corresponding deductions and record reasons

Please ensure analysis is objective and professional, conduct scientific assessment based on image content, and every method must have clear numerical scoring."""
        
        return base_prompt

    @staticmethod
    def _is_ocr_capability_error(exc: Exception) -> bool:
        """Detect model capability errors that indicate OCR/vision is unavailable."""
        msg = str(exc).lower()
        return (
            "does not support" in msg
            and ("ocr" in msg or "vision" in msg or "image input" in msg)
        )
    
    def analyze_domain(self, domain_num: int) -> Dict[str, Any]:
        """Analyze clustering effectiveness of specified domain"""
        try:
            # 获取所有图片
            images = self.image_manager.list_images()
            if len(images) < 2:
                raise ValueError(f"需要至少2张聚类图片进行对比，当前只有{len(images)}张")
            
            print(f"🔍 开始分析Domain{domain_num} - 共{len(images)}张聚类图片")

            image_payloads = []
            for img in images:
                image_path = self.image_manager.get_image_path(img["filename"])
                if not image_path:
                    continue
                base64_image = self.azure_client.encode_image(image_path)
                image_payloads.append(
                    {
                        "filename": img["filename"],
                        "image_path": image_path,
                        "base64_image": base64_image,
                    }
                )
            if len(image_payloads) < 2:
                raise ValueError("可用图片不足，无法执行跨方法对比分析")

            # OCR is mandatory by default: if the configured model has no vision/OCR support, fail fast.
            self.azure_client.ensure_ocr_capable(image_payloads[0]["image_path"])
            ocr_lines = []
            for payload in image_payloads:
                ocr_text = self.azure_client.extract_ocr_text(payload["image_path"])
                if not ocr_text:
                    continue
                ocr_lines.append(f"{payload['filename']}: {ocr_text[:800]}")
            ocr_context = "\n".join(ocr_lines) if ocr_lines else "[NO_OCR_TEXT]"
            
            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"{self.domain_prompts[domain_num]}\n\n"
                                f"[OCR Extracted Text Summary]\n{ocr_context}"
                            ),
                        }
                    ]
                }
            ]
            
            # 添加所有图片到消息中
            for payload in image_payloads:
                messages[0]["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{payload['base64_image']}"
                        },
                    }
                )
            
            print(f"⏳ 正在分析Domain{domain_num}，请稍候...")
            
            # 调用GPT-4V模型进行分析
            response = self.azure_client.client.chat.completions.create(
                model=self.azure_client.deployment_name,
                messages=messages,
                max_tokens=4000,
                temperature=0.3
            )
            
            analysis_result = response.choices[0].message.content
            
            # 提取评分
            scores = self._extract_scores(analysis_result)
            
            # 如果评分提取失败，打印调试信息
            if not scores:
                print(f"⚠️  Domain{domain_num}评分提取失败，分析文本预览:")
                print(f"前200字符: {analysis_result[:200]}...")
                print(f"后200字符: ...{analysis_result[-200:]}")
            
            # 构建结果
            result = {
                'domain': domain_num,
                'analysis_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'image_count': len(images),
                'analyzed_images': [img['filename'] for img in images],
                'analysis_report': analysis_result,
                'scores': scores,
                'metadata': {
                    'model': self.azure_client.deployment_name,
                    'ocr_model': self.azure_client.ocr_deployment_name,
                    'prompt_type': f'DLPFC_domain{domain_num}_analysis'
                }
            }
            
            return result
            
        except Exception as e:
            raise Exception(f"Domain{domain_num}分析失败: {str(e)}")
    
    def _extract_scores(self, analysis_text: str) -> Dict[str, float]:
        """从分析文本中提取评分"""
        scores = {}
        
        # 定义方法名称的可能变体
        method_patterns = {
            'BayesSpace': ['BayesSpace', 'bayesspace'],
            'GraphST': ['GraphST', 'graphst'],
            'STAGATE': ['STAGATE', 'stagate'],
            'stLearn': ['stLearn', 'stlearn'],
            'SEDR': ['SEDR', 'sedr'],
            'BASS': ['BASS'],
            'DR-SC': ['DR-SC', 'DRSC', 'dr-sc', 'drsc'],
            'IRIS': ['IRIS', 'iris']
        }
        
        # 尝试多种评分提取模式（按优先级排序）
        patterns = [
            # 新格式：Final Score 优先匹配
            r'(\w+[-\w]*)[^0-9]*Final\s*Score[^0-9]*([0-1]?\.\d+)',
            r'(\w+[-\w]*)[^0-9]*最终得分[^0-9]*([0-1]?\.\d+)',
            r'(\w+[-\w]*)[^0-9]*最终分数[^0-9]*([0-1]?\.\d+)',
            # 新表格格式：匹配Final Score列（第7列）
            r'\|\s*(\w+[-\w]*)\s*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|\s*([0-1]?\.\d+)\s*\|[^|]*\|',
            # 改进表格格式：匹配Final Score列（第6列，如果没有Cross-Method Deduction列）
            r'\|\s*(\w+[-\w]*)\s*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|\s*([0-1]?\.\d+)\s*\|[^|]*\|',
            # 精确表格格式：匹配总分列（第5列）
            r'\|\s*(\w+[-\w]*)\s*\|[^|]*\|[^|]*\|[^|]*\|\s*([0-1]?\.\d+)\s*\|[^|]*\|',
            # 冒号格式
            r'(\w+[-\w]*)\s*[:：]\s*([0-1]?\.\d+)',
            r'(\w+[-\w]*)\s*得分\s*[:：]\s*([0-1]?\.\d+)',
            r'(\w+[-\w]*)\s*评分\s*[:：]\s*([0-1]?\.\d+)',
            r'(\w+[-\w]*)\s*分数\s*[:：]\s*([0-1]?\.\d+)',
            # 括号格式
            r'(\w+[-\w]*)\s*\(\s*([0-1]?\.\d+)\s*\)',
            # 连字符格式
            r'(\w+[-\w]*)\s*-\s*([0-1]?\.\d+)',
            # 空格分隔格式
            r'(\w+[-\w]*)\s+([0-1]?\.\d+)',
            # 总分格式：**总分：0.xxx**
            r'(\w+[-\w]*)[^0-9]*总分[^0-9]*([0-1]?\.\d+)',
            # 旧版表格格式（作为后备）
            r'\|\s*(\w+[-\w]*)\s*\|[^|]*\|[^|]*\|[^|]*\|\s*([0-1]?\.\d+)\s*\|',
        ]
        
        # 调试信息：记录所有匹配
        all_matches = []
        
        for i, pattern in enumerate(patterns):
            matches = re.findall(pattern, analysis_text, re.IGNORECASE)
            if matches:
                all_matches.extend([(i, match) for match in matches])
            
            for match in matches:
                method_name = match[0].strip()
                score_str = match[1].strip()
                
                try:
                    score = float(score_str)
                    if 0 <= score <= 1:
                        # 匹配标准方法名
                        for standard_name, variants in method_patterns.items():
                            if method_name in variants or method_name.lower() in [v.lower() for v in variants]:
                                # 优先级策略：Final Score > 高精度分数 > 其他分数
                                should_update = False
                                
                                if standard_name not in scores:
                                    # 如果还没有这个方法的分数，直接添加
                                    should_update = True
                                elif i <= 3:  # 前4个模式是Final Score相关的，优先级最高
                                    should_update = True
                                elif score > 0.0 and scores[standard_name] == 0.0:
                                    # 如果新分数非0，旧分数是0，更新
                                    should_update = True
                                elif score > scores[standard_name] and i <= 6:
                                    # 如果是表格格式且新分数更高，更新
                                    should_update = True
                                
                                if should_update:
                                    scores[standard_name] = score
                                    print(f"[Debug] 提取到 {standard_name}: {score} (模式 {i})")
                                break
                except ValueError:
                    continue
        
        # 如果没有提取到足够的评分，打印调试信息
        if len(scores) < 8:
            print(f"🔍 评分提取调试信息:")
            print(f"   提取到 {len(scores)}/8 个评分")
            print(f"   找到的匹配: {len(all_matches)} 个")
            if all_matches:
                print(f"   匹配示例: {all_matches[:3]}")
        
        return scores
    
    def save_domain_report(self, result: Dict[str, Any]) -> str:
        """保存单个domain的分析报告"""
        try:
            domain_num = result['domain']
            
            # 保存文本报告（无时间后缀，直接覆盖）
            txt_filename = f"domain{domain_num}_report.txt"
            txt_path = os.path.join(self.output_dir, txt_filename)
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"DLPFC Domain{domain_num} 聚类图片横向对比分析报告\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"分析时间: {result['analysis_time']}\n")
                f.write(f"分析域: Domain{domain_num}\n")
                f.write(f"分析图片数量: {result['image_count']}\n")
                f.write(f"使用模型: {result['metadata']['model']}\n\n")
                
                f.write("分析的图片文件:\n")
                for i, filename in enumerate(result['analyzed_images'], 1):
                    f.write(f"  {i}. {filename}\n")
                f.write("\n")
                
                f.write("=" * 80 + "\n")
                f.write("详细分析报告\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(result['analysis_report'])
                f.write("\n\n")
                
                if result['scores']:
                    f.write("=" * 80 + "\n")
                    f.write("提取的评分结果\n")
                    f.write("=" * 80 + "\n")
                    for method, score in result['scores'].items():
                        f.write(f"{method}: {score:.3f}\n")
                    f.write("\n")
                
                f.write("=" * 80 + "\n")
                f.write("报告生成完成\n")
                f.write("=" * 80 + "\n")
            
            return txt_path
            
        except Exception as e:
            raise Exception(f"保存Domain{result['domain']}报告失败: {str(e)}")
    
    def save_summary_json(self) -> str:
        """保存汇总的JSON评分文件"""
        try:
            json_filename = "all_domains_scores.json"
            json_path = os.path.join(self.output_dir, json_filename)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.all_scores, f, ensure_ascii=False, indent=2)
            
            return json_path
            
        except Exception as e:
            raise Exception(f"保存汇总评分文件失败: {str(e)}")
    
    def run_all_domains_analysis(self, domains: List[int] = None, methods: List[str] = None):
        """运行所有domain的分析
        
        参数:
        - domains: 可选，仅分析指定的 domain 列表（如 [1, 3, 7]）
        - methods: 可选，仅覆盖指定方法的评分（如 ["GraphST", "stLearn"]），其余方法保持历史分数不变
        """
        try:
            print("🚀 启动DLPFC聚类图片自动分析系统")
            print("=" * 60)
            max_domain = 7
            if domains:
                # 过滤非法 domain，并去重排序
                domain_list = sorted({d for d in domains if 1 <= d <= max_domain})
            else:
                domain_list = list(range(1, max_domain + 1))

            print(f"将依次分析以下 Domain 的聚类效果: {domain_list}")
            print("=" * 60)
            
            # 检查图片
            images = self.image_manager.list_images()
            if len(images) < 2:
                print(f"❌ 图片数量不足！当前只有 {len(images)} 张图片")
                print("至少需要2张图片进行对比分析")
                return False
            
            print(f"✅ 找到 {len(images)} 张聚类图片")
            for i, img in enumerate(images, 1):
                method_name = img['filename'].split('_')[0]
                print(f"   {i}. {method_name}")
            print()
            
            # 依次分析每个domain
            for idx, domain_num in enumerate(domain_list, start=1):
                print(f"\n📊 开始分析Domain{domain_num} ({idx}/{len(domain_list)})")
                print("-" * 40)
                
                # 重试直到成功
                attempt = 0
                while True:
                    attempt += 1
                    try:
                        # 分析当前domain
                        result = self.analyze_domain(domain_num)
                        break
                    except Exception as e:
                        if self._is_ocr_capability_error(e):
                            print(f"⚠️ Domain{domain_num} 检测到模型不支持 OCR/视觉输入：{e}")
                            print("⚠️ 已自动降级为无视觉模块模式（仅本次运行），请继续在上游使用 --vlm_off。")
                            return False
                        print(f"⚠️ Domain{domain_num} 分析失败（尝试{attempt}）：{e}")
                        print("⏳ 5秒后重试...")
                        time.sleep(5)
                
                # 保存报告
                txt_path = self.save_domain_report(result)
                print(f"✅ Domain{domain_num}分析完成，报告已保存: {txt_path}")
                    
                # 收集评分（增量写入/覆盖该 domain）
                scores = result.get('scores') or {}
                if scores:
                    for method, score in scores.items():
                        # 如果指定了 methods，只更新这些方法；其他方法保持原有分数不变
                        if methods and method not in methods:
                                continue
                        if method not in self.all_scores:
                            self.all_scores[method] = {}
                        self.all_scores[method][f'domain{domain_num}'] = score
                        
                        print(f"📈 Domain{domain_num}评分:")
                    for method, score in scores.items():
                            print(f"   {method}: {score:.3f}")
                        
                        # 调试信息：显示当前累积的所有评分
                    print("🔍 当前累积评分数据:")
                    for method, domains_dict in self.all_scores.items():
                        domain_count = len(domains_dict)
                        print(f"   {method}: {domain_count} domains")
                    else:
                        print(f"⚠️ 未能从Domain{domain_num}分析中提取到评分")
                        preview = (result.get('analysis_report') or '')[:200]
                    if preview:
                        print(f"🔍 分析文本预览: {preview}...")
            
            # 保存汇总评分
            if self.all_scores:
                json_path = self.save_summary_json()
                print(f"\n🎉 所有分析完成！")
                print("=" * 60)
                print(f"📋 汇总评分文件: {json_path}")
                print("📄 各domain报告已保存到output文件夹")
                print("=" * 60)
                
                # 显示汇总评分
                print("\n📊 汇总评分结果:")
                print("-" * 50)
                for method, domains in self.all_scores.items():
                    print(f"{method}:")
                    for domain, score in domains.items():
                        print(f"  {domain}: {score:.3f}")
                    print()
                
                return True
            else:
                print("\n❌ 未能提取到任何评分数据")
                return False
                
        except Exception as e:
            print(f"❌ 分析系统运行失败: {e}")
            return False

def main():
    """主函数"""
    try:
        # 检查配置
        from config import Config
        if not all([
            Config.AZURE_OPENAI_ENDPOINT,
            Config.AZURE_OPENAI_API_KEY,
            Config.AZURE_OPENAI_DEPLOYMENT_NAME
        ]):
            print("❌ Azure OpenAI配置不完整！")
            print("请检查配置文件中的以下项目:")
            print("  - AZURE_OPENAI_ENDPOINT")
            print("  - AZURE_OPENAI_API_KEY") 
            print("  - AZURE_OPENAI_DEPLOYMENT_NAME")
            return
        
        # 确保必要文件夹存在
        os.makedirs('uploads', exist_ok=True)
        os.makedirs('output', exist_ok=True)
        
        # 启动自动分析
        analyzer = AutoClusteringAnalyzer()

        # 解析命令行：
        #   --domainX       只跑指定 domain（如 --domain1 --domain3）
        #   --method=GraphST 只覆盖指定方法的分数，其它方法保持历史结果
        selected_domains: List[int] = []
        selected_methods: List[str] = []
        for arg in sys.argv[1:]:
            m_domain = re.match(r"--domain(\d+)$", arg)
            m_method = re.match(r"--method=(\w+)$", arg)
            if m_domain:
                try:
                    selected_domains.append(int(m_domain.group(1)))
                except ValueError:
                    continue
            elif m_method:
                selected_methods.append(m_method.group(1))

        if selected_domains:
            domains = sorted(set(selected_domains))
            print(f"🔧 仅分析指定的 domains: {domains}")
        else:
            domains = None

        if selected_methods:
            methods = sorted(set(selected_methods))
            print(f"🔧 仅覆盖指定的方法分数: {methods}")
        else:
            methods = None

        success = analyzer.run_all_domains_analysis(domains, methods)
        
        if success:
            print("\n✅ 全部分析流程完成！")
        else:
            print("\n❌ 分析流程未完全成功，请检查错误信息")
        
    except KeyboardInterrupt:
        print("\n\n👋 程序被用户中断，再见！")
    except Exception as e:
        print(f"❌ 程序启动失败: {e}")

if __name__ == "__main__":
    main()
