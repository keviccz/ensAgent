import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
# from langchain.llms import OpenAI
# from langchain.chains import RetrievalQA
from gpt_scorer import GPTDomainScorer, RUBRIC_TEXT, GLOSSARY_DEFAULT, TASK_DESCRIPTION, DEG_GLOSSARY
from pathway_analyzer import PathwayAnalyzer
from visual_score_integrator import VisualScoreIntegrator
# ... 其他必要的导入

# 主类：DomainEvaluator
class DomainEvaluator:
    def __init__(
        self,
        openai_api_key: str,
        output_format: str = 'json',
        api_provider: str = "",
        api_key: str = "",
        api_endpoint: str = "",
        api_model: str = "",
        api_version: str = "",
        azure_endpoint: str = None,
        azure_deployment: str = None,
        azure_api_version: str = None,
        temperature: float = 0.2,
        max_completion_tokens: int = None,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        top_n_deg: int = 5,
        example_db: str = None,
        use_example_db: bool = False,
        knn_k: int = 5,
        enforce_discrimination: bool = False,
    ):
        """
        初始化DomainEvaluator
        :param openai_api_key: OpenAI API密钥
        :param output_format: 输出格式（json, dataframe, sql）
        :param azure_endpoint: Azure OpenAI endpoint
        :param azure_deployment: Azure OpenAI deployment名称
        :param azure_api_version: Azure OpenAI API版本
        :param temperature: 大模型temperature参数
        :param top_n_deg: 每个domain最多考虑的deg数量
        :param example_db: 示例数据库
        :param use_example_db: 是否使用示例数据库
        :param knn_k: 用于检索示例的KNN参数
        """
        self.openai_api_key = openai_api_key
        self.output_format = output_format
        self.gpt_scorer = GPTDomainScorer(
            openai_api_key=api_key or openai_api_key,
            api_provider=api_provider,
            api_key=api_key or openai_api_key,
            api_endpoint=api_endpoint or azure_endpoint,
            api_model=api_model or azure_deployment,
            api_version=api_version or azure_api_version,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            azure_api_version=azure_api_version,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            enforce_discrimination=enforce_discrimination
        )
        # 初始化Langchain检索链（可根据后续需要扩展）
        self.retriever = None  # 这里后续可接入自定义知识库
        self.top_n_deg = top_n_deg
        self.example_db = example_db
        self.use_example_db = use_example_db
        self.knn_k = knn_k
        
        # 示例库功能已移除，避免ground truth或示例泄漏
        self.example_store = None
        
        # 初始化pathway分析器
        self.pathway_analyzer = PathwayAnalyzer()
        self.pathway_data = None
        self.pathway_scores = None
        
        # 初始化视觉评分集成器
        self.visual_integrator = VisualScoreIntegrator()
        self.visual_scores = None
        self.use_visual_integration = False

    def load_pathway_data(self, input_dir: str, method_name: str):
        """
        加载特定方法的pathway数据
        :param input_dir: 输入目录
        :param method_name: 方法名称
        """
        try:
            # 加载所有pathway数据
            all_pathway_data = self.pathway_analyzer.load_pathway_data(input_dir)
            
            if method_name in all_pathway_data:
                self.pathway_data = {method_name: all_pathway_data[method_name]}
                # 分析pathway富集质量
                self.pathway_scores = self.pathway_analyzer.analyze_pathway_enrichment(self.pathway_data)
                print(f"[Info] 成功加载 {method_name} 的pathway数据")
            else:
                print(f"[Warning] 未找到 {method_name} 的pathway数据")
                self.pathway_data = None
                self.pathway_scores = None
        except Exception as e:
            print(f"[Warning] 加载pathway数据失败: {e}")
            self.pathway_data = None
            self.pathway_scores = None

    def load_visual_scores(self, pic_analyze_dir: str = "pic_analyze", sample_id: str = "") -> bool:
        """
        加载视觉评分数据
        
        Args:
            pic_analyze_dir: pic_analyze组件的目录路径
            
        Returns:
            bool: 是否成功加载
        """
        try:
            self.visual_integrator = VisualScoreIntegrator(pic_analyze_dir, sample_id=sample_id)
            success = self.visual_integrator.load_visual_scores()
            
            if success:
                self.use_visual_integration = True
                print("✅ 视觉评分整合已启用")
                
                # 简化统计信息显示，避免潜在的索引错误
                try:
                    if hasattr(self.visual_integrator, 'visual_scores') and self.visual_integrator.visual_scores:
                        method_count = len(self.visual_integrator.visual_scores)
                        print(f"📊 成功加载 {method_count} 个方法的视觉评分数据")
                except Exception as stats_error:
                    print(f"⚠️ 统计信息显示失败: {stats_error}")
                
                return True
            else:
                self.use_visual_integration = False
                print("❌ 视觉评分数据加载失败")
                return False
                
        except Exception as e:
            print(f"❌ 视觉评分整合初始化失败: {e}")
            self.use_visual_integration = False
            return False

    def process_batch(self, samples_list: List[Dict[str, Any]], return_gpt_response: bool = False,
                     target_domains: Optional[List[int]] = None):
        """
        批量处理所有样本
        :param samples_list: 每个元素包含assignment_df, deg_dict, composition_dict（可选）
        :param return_gpt_response: 是否返回gpt原始response
        :return: (格式化结果, gpt原始response, gpt复查日志) 或 格式化结果
        """
        all_results = []
        all_gpt_raw_responses = []
        gpt_logs = []
        for i, sample in enumerate(samples_list):
            print(f"\n[Info] 正在处理第{i+1}个样本 ...")
            result, gpt_raw, gpt_log = self.process_sample(
                sample,
                return_gpt_response=True,
                target_domains=target_domains
            )
            all_results.append(result)
            all_gpt_raw_responses.append(gpt_raw)
            gpt_logs.append(gpt_log)
        # 跨样本比较，选出每个domain的最佳分数
        best_domains = self.cross_sample_compare(all_results)
        formatted = self.format_output(best_domains)
        if return_gpt_response:
            return formatted, all_gpt_raw_responses, gpt_logs
        else:
            return formatted

    def process_sample(self, sample: Dict[str, Any], return_gpt_response: bool = False,
                      target_domains: Optional[List[int]] = None) -> Any:
        """
        处理单个样本，遍历每个domain，组织输入，批量调用GPTDomainScorer打分
        :param sample: 包含assignment_df, deg_dict, composition_dict（可选）
        :param return_gpt_response: 是否返回gpt原始response
        :return: (结果, gpt原始response, gpt复查日志) 或 结果
        """
        # 创建数据副本以避免修改原始数据
        assignment_df = sample['assignment_df'].copy()
        
        deg_dict = sample['deg_dict']
        deg_df = sample.get('deg_df')
        composition_dict = None
        img_metrics = sample.get('img_metrics', {})
        
        # 确保domain_index列为整数类型
        assignment_df['domain_index'] = assignment_df['domain_index'].astype(int)
        
        # === 计算空间度量（使用 spatial.py 中的 SpatialMetrics） ===
        from spatial import SpatialMetrics  # 轻量级导入，避免循环依赖
        spatial_calc = SpatialMetrics()
        
        # 验证坐标数据
        coord_valid, coord_msg = spatial_calc.validate_coordinates(assignment_df)
        if not coord_valid:
            print(f"[Warning] 坐标验证警告: {coord_msg}")
        
        # 计算空间度量
        spatial_stats = spatial_calc.compute_spatial_stats(assignment_df)
        comp_dict = spatial_stats['compactness']
        adj_dict = spatial_stats['adjacency']
        convexity_dict = spatial_stats.get('convexity', {})
        jaggedness_dict = spatial_stats.get('jaggedness', {})
        conn_metrics_all = spatial_stats.get('connectivity_metrics', {})
        
        # 获取所有domain并确保为整数类型
        domain_indices = sorted(set(assignment_df['domain_index'].astype(int)))
        # 如果指定了 target_domains，仅保留这些 domain
        if target_domains:
            domain_indices = [d for d in domain_indices if d in target_domains]
        
        # Cache global row/col maxima for normalized position (if available)
        if "array_row" in assignment_df.columns and "array_col" in assignment_df.columns:
            self._row_max = assignment_df["array_row"].max()
            self._col_max = assignment_df["array_col"].max()
        else:
            self._row_max = None
            self._col_max = None

        # 重用已初始化的ExampleStore实例
        example_store = None
        domains_input = []
        gpt_raw_responses = []
        
        for d in domain_indices:
            # Get spots for this domain (handle missing spot_id column)
            domain_spots = assignment_df[assignment_df['domain_index'] == d]
            if 'spot_id' in domain_spots.columns:
                spots = domain_spots['spot_id'].tolist()
            else:
                spots = domain_spots.index.astype(str).tolist()
            n_spots = len(spots)
            
            # 使用统一的域标识，不再依赖外部layer_guess
            layer = f"Domain_{d}"
            
            # 统一使用字符串键查找DEG，确保数据类型一致性
            degs = deg_dict.get(str(d), [])
            
            # DEG stats if dataframe available
            if deg_df is not None:
                df_dom = deg_df[deg_df['domain'] == d]
                if not df_dom.empty:
                    # ----- NEW: choose top-N DEGs by significance -----
                    degs = self._get_top_degs(df_dom, n=self.top_n_deg)
                    # stats
                    max_logfc = df_dom['logfoldchanges'].max()
                    mean_logfc = df_dom['logfoldchanges'].mean()
                    median_adjP = df_dom['pvals_adj'].median()
                    mean_score = df_dom['scores'].mean()
                else:
                    degs = []
                    max_logfc = mean_logfc = median_adjP = mean_score = np.nan
            else:
                # 如果没有deg_df，使用deg_dict并限制数量
                if not degs:  # 如果字符串键没找到，尝试整数键
                    degs = deg_dict.get(d, [])
                degs = degs[:self.top_n_deg]
                max_logfc = mean_logfc = median_adjP = mean_score = np.nan
            
            # 空间度量：使用前面计算结果
            compactness = comp_dict.get(d, np.nan)
            adjacency = adj_dict.get(d, np.nan)
            dispersion = spatial_stats['dispersion'].get(d, np.nan)
            
            # 连通性和最近邻距离指标
            connectivity_metrics = spatial_stats.get('connectivity_metrics', {}).get(d, {})
            connected_components = connectivity_metrics.get('connected_components', np.nan)
            fragmentation_index = connectivity_metrics.get('fragmentation_index', np.nan)
            
            nn_metrics = spatial_stats.get('nearest_neighbor_distances', {}).get(d, {})
            mean_nn_dist = nn_metrics.get('mean_nn_dist', np.nan)
            median_nn_dist = nn_metrics.get('median_nn_dist', np.nan)
            sub_df = assignment_df[assignment_df["domain_index"] == d]
            
            # 获取该domain的图片质量指标
            domain_img_quality = {}
            if img_metrics and 'domain_quality' in img_metrics:
                domain_img_quality = img_metrics['domain_quality'].get(layer, {})

            # --- 获取Pathway摘要 (新增) ---
            pathway_summary_str = ""
            if self.pathway_data:
                # 尝试获取当前方法的pathway数据
                # sample中可能有method_name，如果没有，尝试从self.pathway_data的键中推断（如果只有一个）
                method_name = sample.get('method_name')
                
                current_pathway_df = None
                if method_name and method_name in self.pathway_data:
                    current_pathway_df = self.pathway_data[method_name]
                elif len(self.pathway_data) == 1:
                    # 如果只加载了一个方法的数据，直接使用
                    current_pathway_df = list(self.pathway_data.values())[0]
                
                if current_pathway_df is not None:
                    pathway_summary_str = self.pathway_analyzer.get_top_pathways_summary(
                        current_pathway_df, d, top_n=3
                    )

            # shape metrics for this domain
            convexity = convexity_dict.get(d, np.nan)
            jaggedness = jaggedness_dict.get(d, np.nan)
            
            summary = self._make_summary(
                d,
                sub_df=sub_df,
                degs=degs,
                compact=compactness,
                adj=adjacency,
                dispersion=dispersion,
                n_spots=n_spots,
                layer=layer,
                comp_dict=None,
                deg_stats={
                    'max_logfc': max_logfc,
                    'mean_logfc': mean_logfc,
                    'median_adjP': median_adjP,
                    'mean_score': mean_score,
                },
                convexity=convexity,
                jaggedness=jaggedness,
                img_quality=domain_img_quality,
                connectivity_stats={
                    'connected_components': connected_components,
                    'fragmentation_index': fragmentation_index,
                },
                nn_stats={
                    'mean_nn_dist': mean_nn_dist,
                    'median_nn_dist': median_nn_dist,
                },
                pathway_summary=pathway_summary_str
            )

            # --- retrieve examples ---
            examples = []
            if example_store is not None:
                examples = example_store.similar_examples(summary, k=self.knn_k)  # may be []

            prompt_parts = [TASK_DESCRIPTION, GLOSSARY_DEFAULT, DEG_GLOSSARY, RUBRIC_TEXT]
            import re
            for idx, ex in enumerate(examples):
                summ = ex['summary']
                # remove compactness numeric values to avoid leakage
                summ = re.sub(r"compactness=[0-9.]+", "compactness=VALUE", summ)
                prompt_parts.append(f"### Example {idx+1}\n<summary>{summ}</summary>\nScore: {ex.get('total_score',0)}")
            prompt_parts.append(
                (
                    "### New domain\n<summary>" + summary + "</summary>\n"
                    "NOTE: Your justification must be **two sentences**.\n"
                    "1) Sentence 1 – concise biological reasoning (mention ≥2 DEGs, at least one pathway or cell-type evidence).\n"
                    "2) Sentence 2 – evidence list starting with `basis:` and MUST include spatial metrics (compactness, connected_components, fragmentation_index), marker_hits, and at least one data source.\n"
                    "Return **exactly one** JSON object (no extra text) with schema:\n"
                    "{\n"
                    "  \"score\": <0-1 float>,\n"
                    "  \"justification\": \"<2 sentences>\"\n"
                    "}\n"
                    "Rules: Score must be between 0.0 and 1.0. Provide detailed justification citing database sources."
                )
            )

            full_prompt = "\n\n".join(prompt_parts)

            result_json = self.gpt_scorer.score_with_prompt(full_prompt)
            gpt_raw_responses.append(result_json)

            # 获取原始LLM评分
            original_score = result_json.get('score', result_json.get('total', 0))

            # --- Programmatic shape/fragmentation enforcement BEFORE visual integration ---
            # Gather metrics
            conn_metrics = conn_metrics_all.get(d, {})
            connected_components = conn_metrics.get('connected_components', np.nan)
            fragmentation_index = conn_metrics.get('fragmentation_index', np.nan)
            jaggedness = jaggedness_dict.get(d, np.nan)
            convexity = convexity_dict.get(d, np.nan)
            compactness_val = comp_dict.get(d, np.nan)

            rule_deduction = 0.0
            
            # === WHITE MATTER EXEMPTION CHECK ===
            # Check if this domain is oligodendrocyte/white matter - exempt from shape penalties
            is_white_matter = self._is_white_matter_domain(degs, deg_df, d)
            
            if is_white_matter:
                # White matter domains are EXEMPT from jaggedness and over-regularization penalties
                # (white matter boundaries are naturally irregular; white matter can be compact)
                pass  # No shape deductions for white matter
            else:
                # 1) Jaggedness deductions (no caps) - only for non-white-matter domains
                if isinstance(jaggedness, (int, float)) and np.isfinite(jaggedness):
                    if 0.35 <= jaggedness <= 0.45:
                        # linear 0.12..0.20 across 0.35..0.45
                        t = (jaggedness - 0.35) / (0.10 + 1e-9)
                        rule_deduction += 0.12 + 0.08 * float(np.clip(t, 0.0, 1.0))
                    elif jaggedness > 0.45:
                        if jaggedness <= 1.0:
                            # 0.20..0.30 for 0.45..1.0
                            t = (jaggedness - 0.45) / (0.55 + 1e-9)
                            rule_deduction += 0.20 + 0.10 * float(np.clip(t, 0.0, 1.0))
                        elif jaggedness > 3.0:
                            # extreme
                            rule_deduction += 0.30 + 0.30  # base heavy + extra extreme
                        else:
                            # >1.0 and <=3.0
                            rule_deduction += 0.30 + 0.15

                # 3) Over-regularization: compactness≥0.90 and convexity≥0.65 → severe deduction (no cap)
                # Only for non-white-matter domains
                if (isinstance(compactness_val, (int, float)) and np.isfinite(compactness_val) and compactness_val >= 0.90 and
                    isinstance(convexity, (int, float)) and np.isfinite(convexity) and convexity >= 0.65):
                    rule_deduction += 0.30

            # 2) Connected components deductions (no caps) - applies to ALL domains
            if isinstance(connected_components, (int, float)) and np.isfinite(connected_components):
                if connected_components > 50:
                    rule_deduction += 0.40
                elif connected_components > 15:
                    rule_deduction += 0.20

            # Apply rule deduction pre-visual
            rule_adjusted = float(np.clip(original_score - rule_deduction, 0.0, 1.0))
            
            # 应用视觉评分调整（如果启用了视觉整合）
            final_score = rule_adjusted
            visual_adjustment_info = {}
            
            if self.use_visual_integration and hasattr(self, 'visual_integrator') and self.visual_integrator:
                try:
                    # 获取方法名称
                    method_name = sample.get('method_name', 'unknown')
                    
                    # 应用视觉评分调整
                    adjusted_score, adjustment_info = self.visual_integrator.apply_visual_adjustment(
                        original_score=rule_adjusted,
                        method_name=method_name,
                        domain_id=d
                    )
                    
                    final_score = adjusted_score
                    visual_adjustment_info = adjustment_info
                    
                    print(f"[Visual] {method_name} Domain{d}: {rule_adjusted:.3f} -> {final_score:.3f} (调整: {adjustment_info.get('adjustment_factor', 1.0):.3f})")
                    
                except Exception as visual_error:
                    print(f"[Warning] 视觉评分调整失败 {method_name} Domain{d}: {visual_error}")
                    final_score = rule_adjusted
            
            domains_input.append({
                'domain_index': d,
                'score': final_score,
                'original_score': original_score,
                'shape_rule_deduction': round(rule_deduction, 6),
                'metrics': {
                    'compactness': compactness_val,
                    'connected_components': connected_components,
                    'fragmentation_index': fragmentation_index,
                    'convexity': convexity,
                    'jaggedness': jaggedness
                },
                'visual_adjustment_info': visual_adjustment_info,
                'justification': result_json.get('justification', '')
            })

        domains_result = domains_input
        result = {
            'sample_id': sample.get('sample_id', ''),
            'domains': domains_result
        }
        if return_gpt_response:
            return result, gpt_raw_responses, ""
        else:
            return result

    def cross_sample_compare(self, all_results: List[Dict]) -> List[Dict]:
        """
        跨样本稳健聚合：对每个domain按中位数选择代表结果（避免“取最大”偏差）。
        :param all_results: 所有样本的结果
        :return: 每个domain的代表结果
        """
        # 聚合同一domain的所有结果
        idx_to_list: Dict[int, List[Dict]] = {}
        for res in all_results:
            for d in res['domains']:
                idx = int(d['domain_index'])
                idx_to_list.setdefault(idx, []).append(d)
        # 选择最接近中位数的结果作为代表
        best_by_median: List[Dict] = []
        for idx in sorted(idx_to_list.keys()):
            lst = idx_to_list[idx]
            scores = np.array([x.get('score', 0.0) for x in lst], dtype=float)
            if len(scores) == 0:
                continue
            med = float(np.median(scores))
            # 选取与中位数最接近的一个作为代表
            i = int(np.argmin(np.abs(scores - med)))
            rep = dict(lst[i])
            # 附加聚合信息（可选，不影响下游）
            rep['aggregation_info'] = {
                'method': 'median_selector',
                'median_score': round(med, 6),
                'num_samples': len(scores),
                'min_score': round(float(np.min(scores)), 6),
                'max_score': round(float(np.max(scores)), 6)
            }
            best_by_median.append(rep)
        return best_by_median

    def format_output(self, best_domains: List[Dict]) -> Any:
        """
        根据用户指定格式输出结果
        :param best_domains: 最佳domain结果
        :return: json/dataframe/sql等
        """
        if self.output_format == 'json':
            return {"best_domains": best_domains}
        elif self.output_format == 'dataframe':
            return pd.DataFrame(best_domains)
        elif self.output_format == 'sql':
            # TODO: 实现写入SQL/NoSQL
            pass
        else:
            raise ValueError('Unsupported output format')

    # Predefined cell-type marker dictionary (enhanced with neuronal subtypes for DLPFC)
    # References: Allen Brain Atlas, PanglaoDB, CellMarker, Human Cell Atlas
    MARKER_DICT = {
        # Glial cells
        "oligodendrocyte": {"MBP", "PLP1", "MOBP", "MOG", "MAG", "CNP", "CLDN11", "OLIG1", "OLIG2", "SOX10"},
        "OPC": {"PDGFRA", "CSPG4", "VCAN", "GPR17"},  # Oligodendrocyte precursor cells
        "astrocyte": {"GFAP", "AQP4", "SLC1A2", "SLC1A3", "ALDH1L1", "S100B", "GJA1"},
        "microglia": {"CX3CR1", "P2RY12", "TREM2", "TMEM119", "AIF1", "CSF1R"},
        
        # Neuronal subtypes (critical for DLPFC layer identification)
        "excitatory_neuron": {"SLC17A7", "CAMK2A", "GRIN1", "GRIN2B", "NRGN", "SATB2"},
        "inhibitory_neuron": {"GAD1", "GAD2", "SLC32A1"},
        "inhibitory_PV": {"PVALB"},  # Fast-spiking interneurons
        "inhibitory_SST": {"SST", "NPY"},  # Somatostatin interneurons
        "inhibitory_VIP": {"VIP", "CALB2"},  # VIP interneurons
        
        # Layer-specific neuronal markers for DLPFC
        "pyramidal_L2_L3": {"CUX1", "CUX2", "SATB2", "RASGRF2"},  # Upper layer pyramidal
        "pyramidal_L4": {"RORB", "RSPO1"},  # Granular layer (L4)
        "pyramidal_L5": {"BCL11B", "FEZF2", "CRYM", "TOX"},  # Corticospinal projection neurons
        "pyramidal_L6": {"TLE4", "FOXP2", "SEMA3E"},  # Corticothalamic neurons
        
        # General synaptic markers (pan-neuronal, Tier 2)
        "synaptic": {"SNAP25", "SYN1", "SYT1", "VAMP2", "STX1A", "RBFOX3"},
        
        # Vascular cells
        "endothelial": {"PECAM1", "VWF", "CLDN5", "FLT1", "KDR", "CDH5"},
        "pericyte": {"PDGFRB", "RGS5", "ABCC9", "KCNJ8"},
    }

    # House-keeping gene set（expanded; GTEx/HRT-Atlas/common qPCR controls）
    HOUSEKEEPING_GENES = {
        # canonical
        "ACTB", "ACTG1", "B2M", "GAPDH", "HPRT1", "TBP", "RPLP0", "RPL13A", "YWHAZ",
        # protein folding/translation & core machinery
        "EEF1A1", "EEF2", "GNB2L1", "RACK1", "EIF4A2", "NONO",
        # cytoskeleton/tubulins
        "TUBB", "TUBA1A", "TUBB2A",
        # glycolysis/energy
        "PGK1", "ALDOA", "ENO1", "TPI1", "GPI", "LDHA", "PKM", "SDHA",
        # proteasome/ubiquitin
        "PSMB2", "PSMB3", "UBC", "UBB",
        # membrane/transport
        "TFRC", "RPSA",
        # qPCR references
        "GUSB", "HMBS",
        # heat shock/chaperones (treat as non-specific background)
        "HSP90AA1", "HSPA8", "HSPB1", "DNAJB1",
        # selected ribosomal (prefix RPL*/RPS* already filtered; include a few common)
        "RPS18", "RPS27A", "RPL30"
    }

    # Heat-shock/chaperone gene prefixes used to flag negative evidence
    HEAT_SHOCK_PREFIXES = ("HSP", "HSPA", "HSPB", "HSPD", "HSPE", "DNAJ", "DNAJB", "DNAJC")
    HEAT_SHOCK_GENES = {"HSP90AA1", "HSPA8", "HSPB1", "DNAJB1"}

    def _match_markers(self, genes: List[str]) -> List[str]:
        gset = {g.upper() for g in genes}
        hits = [ct for ct, markers in self.MARKER_DICT.items() if gset & markers]
        return hits

    def _count_housekeeping(self, genes: List[str]) -> int:
        """Return # of housekeeping genes present in given gene list"""
        gset = {g.upper() for g in genes}
        return len(gset & self.HOUSEKEEPING_GENES)

    def _count_heatshock(self, genes: List[str]) -> int:
        """Return # of heat-shock/chaperone genes present (negative evidence)."""
        gset = {g.upper() for g in genes}
        heat = {
            "HSP90AA1", "HSP90AB1", "HSP90B1", "HSPA1A", "HSPA1B", "HSPA5", "HSPA8",
            "HSPH1", "HSPD1", "HSPE1", "HSPB1", "HSPB8", "DNAJB1", "DNAJB4"
        }
        return len(gset & heat)

    def _is_white_matter_domain(self, degs: List[str], deg_df: "pd.DataFrame" = None, domain_id: int = None) -> bool:
        """
        Check if domain is white matter (oligodendrocyte-enriched).
        White matter domains are EXEMPT from jaggedness and over-regularization penalties.
        
        Criteria: >=2 oligodendrocyte markers (MBP, PLP1, MOBP, CNP, MOG, MAG) in top DEGs
        """
        OLIGO_MARKERS = {"MBP", "PLP1", "MOBP", "CNP", "MOG", "MAG", "CLDN11", "OLIG1", "OLIG2"}
        
        gset = {g.upper() for g in degs}
        oligo_hits = len(gset & OLIGO_MARKERS)
        
        # If we have DEG dataframe, check expression levels for stronger validation
        if oligo_hits >= 2:
            return True
        
        # Also check if top DEGs by logFC are oligodendrocyte markers
        if deg_df is not None and domain_id is not None:
            df_dom = deg_df[deg_df['domain'] == domain_id]
            if not df_dom.empty:
                top_by_logfc = df_dom.nlargest(3, 'logfoldchanges')['names'].str.upper().tolist()
                top_oligo_hits = len(set(top_by_logfc) & OLIGO_MARKERS)
                if top_oligo_hits >= 1 and oligo_hits >= 1:
                    return True
        
        return False

    def _make_summary(
        self,
        domain_idx: int,
        sub_df: "pd.DataFrame",
        degs: List[str],
        compact: float,
        adj: float,
        dispersion: float,
        n_spots: int,
        layer: str,
        comp_dict,
        deg_stats: dict,
        convexity: float = np.nan,
        jaggedness: float = np.nan,
        img_quality: dict = None,
        connectivity_stats: dict = None,
        nn_stats: dict = None,
        pathway_summary: str = None,
    ):
        # show up to 5 genes for richer context
        top_genes = degs[:5]
        gene_part = ", ".join(top_genes) + (" …" if len(degs) > len(top_genes) else "")

        # --- position & embedding features ---
        # Check if array_row and array_col exist
        if (self._row_max is not None and self._col_max is not None and 
            "array_row" in sub_df.columns and "array_col" in sub_df.columns):
            row_norm = sub_df["array_row"].mean() / self._row_max
            col_norm = sub_df["array_col"].mean() / self._col_max
        else:
            row_norm = None
            col_norm = None

        # expression summary (handle datasets that lack these columns)
        mean_counts = sub_df["total_counts"].mean() if "total_counts" in sub_df.columns else np.nan
        mean_genes = sub_df["n_genes"].mean() if "n_genes" in sub_df.columns else np.nan
        mean_hv = sub_df["hv_mean"].mean() if "hv_mean" in sub_df.columns else np.nan

        # embedding features (check if available)
        pca1 = sub_df["embedding_pca1"].mean() if "embedding_pca1" in sub_df.columns else np.nan
        pca2 = sub_df["embedding_pca2"].mean() if "embedding_pca2" in sub_df.columns else np.nan

        marker_hits_list = self._match_markers(top_genes)
        marker_hits = ",".join(marker_hits_list) if marker_hits_list else "NA"

        hk_hits = self._count_housekeeping(top_genes)
        hs_hits = self._count_heatshock(top_genes)

        # DEG stats string
        deg_stat_str = (
            f"deg_stats: maxLogFC={deg_stats['max_logfc']:.1f} "
            f"meanLogFC={deg_stats['mean_logfc']:.1f} "
            f"medianAdjP={deg_stats['median_adjP']:.1e} "
            f"meanScore={deg_stats['mean_score']:.1f}; "
            if not np.isnan(deg_stats['max_logfc']) else ""
        )

        # Build summary string with only available data
        summary_parts = []
        
        # Basic domain info
        basic_info = f"Domain {domain_idx} (layer={layer}, spots={n_spots}"
        if row_norm is not None and col_norm is not None:
            basic_info += f", pos=({row_norm:.2f},{col_norm:.2f})"
        basic_info += ")"
        summary_parts.append(basic_info)
        
        # DEGs info
        if gene_part:
            summary_parts.append(f"DEGs: {gene_part}")
        
        # Pathway info (NEW)
        if pathway_summary and pathway_summary != "No pathways available":
            summary_parts.append(f"Pathways: {pathway_summary}")
        
        # Spatial metrics (adjacency, Moran's I, and Geary's C removed from scoring but still computed for analysis)
        spatial_parts = []
        if not np.isnan(compact):
            spatial_parts.append(f"compactness={compact:.2f}")
        if not np.isnan(dispersion):
            spatial_parts.append(f"dispersion={dispersion:.2f}")
        
        # 连通性指标
        if connectivity_stats:
            connected_comp = connectivity_stats.get('connected_components', np.nan)
            frag_index = connectivity_stats.get('fragmentation_index', np.nan)
            
            if not np.isnan(connected_comp):
                spatial_parts.append(f"connected_components={int(connected_comp)}")
            if not np.isnan(frag_index):
                spatial_parts.append(f"fragmentation_index={frag_index:.3f}")
        
        # 最近邻距离指标
        if nn_stats:
            mean_nn = nn_stats.get('mean_nn_dist', np.nan)
            median_nn = nn_stats.get('median_nn_dist', np.nan)
            
            if not np.isnan(mean_nn):
                spatial_parts.append(f"mean_nn_dist={mean_nn:.3f}")
            if not np.isnan(median_nn):
                spatial_parts.append(f"median_nn_dist={median_nn:.3f}")
        
        if spatial_parts:
            summary_parts.append(f"spatial metrics: {' '.join(spatial_parts)}")
        
        # Embedding info (if available)
        if not np.isnan(pca1) and not np.isnan(pca2):
            summary_parts.append(f"embedding: ({pca1:.2f},{pca2:.2f})")
        
        # Expression summary (only non-empty values)
        expr_parts = []
        if not np.isnan(mean_counts):
            expr_parts.append(f"counts={mean_counts:.0f}")
        if not np.isnan(mean_genes):
            expr_parts.append(f"genes={mean_genes:.0f}")
        if not np.isnan(mean_hv):
            expr_parts.append(f"hv_mean={mean_hv:.2f}")
        if expr_parts:
            summary_parts.append(f"expr_summary: {' '.join(expr_parts)}")
        
        # DEG stats (if available)
        if deg_stat_str:
            summary_parts.append(deg_stat_str.strip())
        
        # Marker & housekeeping hits
        summary_parts.append(f"marker_hits: {marker_hits}")
        summary_parts.append(f"housekeeping_hits: {hk_hits}/{len(top_genes)}")
        summary_parts.append(f"heatshock_hits: {hs_hits}/{len(top_genes)} (negative evidence)")

        # Shape metrics
        if not np.isnan(convexity) or not np.isnan(jaggedness):
            cv = f"{convexity:.3f}" if not np.isnan(convexity) else "nan"
            jg = f"{jaggedness:.3f}" if not np.isnan(jaggedness) else "nan"
            summary_parts.append(f"shape: convexity={cv} jaggedness={jg}")
        
        # 图片质量指标 (基于标准颜色分析)
        if img_quality and isinstance(img_quality, dict):
            img_parts = []
            if 'color_accuracy' in img_quality:
                img_parts.append(f"color_accuracy={img_quality['color_accuracy']:.3f}")
            if 'color_consistency' in img_quality:
                img_parts.append(f"color_consistency={img_quality['color_consistency']:.3f}")
            if 'spatial_coherence' in img_quality:
                img_parts.append(f"spatial_coherence={img_quality['spatial_coherence']:.3f}")
            if 'boundary_clarity' in img_quality:
                img_parts.append(f"boundary_clarity={img_quality['boundary_clarity']:.3f}")
            if 'cross_method_score' in img_quality:
                img_parts.append(f"cross_method_score={img_quality['cross_method_score']:.3f}")
            if 'overall_quality' in img_quality:
                img_parts.append(f"img_overall={img_quality['overall_quality']:.3f}")
            if img_parts:
                summary_parts.append(f"image_quality: {' '.join(img_parts)}")
        
        return " | ".join(summary_parts) + "; "

    # Biologically relevant mitochondrial genes (not pure housekeeping)
    # These are important in neurodegeneration and metabolic studies
    MITO_RELEVANT_GENES = {
        "MT-ND1", "MT-ND2", "MT-ND3", "MT-ND4", "MT-ND5", "MT-ND6",  # Complex I - NADH dehydrogenase
        "MT-CO1", "MT-CO2", "MT-CO3",  # Complex IV - Cytochrome c oxidase
        "MT-ATP6", "MT-ATP8",  # Complex V - ATP synthase
        "MT-CYB",  # Complex III - Cytochrome b
    }
    
    def _get_top_degs(self, df_dom: "pd.DataFrame", n: int = 5) -> List[str]:
        """Return the top-n DEGs using a robust ranking with weighted filtering.

        Enhanced strategy (biologically informed):
        - Mitochondrial genes (MT-*): Downweight by 0.5x instead of removing
          (important in neurodegeneration studies)
        - Ribosomal genes (RPL*/RPS*): Downweight by 0.3x (mostly technical noise)
        - Housekeeping genes: Downweight by 0.2x (non-specific)
        - Robust rank: rank_score = weight * clip(|logFC|, 0.1, 50) * clip(-log10(padj), 1, 100)
        - Falls back to original ordering if required columns missing.
        """
        required_cols = {"logfoldchanges", "pvals_adj", "names"}
        if not required_cols.issubset(df_dom.columns):
            return df_dom["names"].head(n).tolist()

        df = df_dom.copy()
        # Clean and prepare columns
        df["names"] = df["names"].astype(str)
        names_upper = df["names"].str.upper()
        
        # Replace ±inf with NaN, then compute abs and clip
        logfc = df["logfoldchanges"].replace([np.inf, -np.inf], np.nan)
        padj = df["pvals_adj"].replace(0, np.nan).fillna(1e-300).clip(lower=1e-300)
        logfc_abs = logfc.abs().clip(lower=0.1, upper=50)
        neglogp = (-np.log10(padj)).clip(lower=1, upper=100)
        
        # Base score
        base_score = logfc_abs * neglogp
        
        # Apply biologically-informed weights instead of hard filtering
        # Initialize weights to 1.0
        weights = pd.Series(1.0, index=df.index)
        
        # Mitochondrial genes: downweight but don't remove (relevant in neurodegeneration)
        is_mito = names_upper.str.startswith("MT-")
        # Further distinguish: biologically relevant MT genes get 0.7x, others 0.5x
        is_mito_relevant = names_upper.isin(self.MITO_RELEVANT_GENES)
        weights.loc[is_mito & is_mito_relevant] = 0.7  # Relevant MT genes
        weights.loc[is_mito & ~is_mito_relevant] = 0.5  # Other MT genes
        
        # Ribosomal genes: strong downweight (mostly technical noise)
        is_ribo = names_upper.str.startswith("RPL") | names_upper.str.startswith("RPS")
        weights.loc[is_ribo] = 0.3
        
        # Housekeeping genes: strongest downweight
        is_house = names_upper.isin({g.upper() for g in self.HOUSEKEEPING_GENES})
        weights.loc[is_house] = 0.2
        
        # Compute weighted robust rank
        df["_robust_rank"] = base_score * weights
        
        # Sort by weighted robust rank and pick top-n
        top = df.sort_values("_robust_rank", ascending=False).head(n)
        genes = top["names"].tolist()
        return genes

# 示例用法
if __name__ == '__main__':
    # 这里可以添加命令行参数解析，读取输入文件等
    pass 
