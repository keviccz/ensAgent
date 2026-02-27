#!/usr/bin/env python3
"""
Pathway分析器：用于分析和评估pathway富集结果的生物学质量
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set
import os
from collections import defaultdict
import warnings

class PathwayAnalyzer:
    """Pathway富集分析器
    
    Enhanced for DLPFC spatial transcriptomics analysis.
    References: KEGG, Reactome, MSigDB, Gene Ontology
    """
    
    def __init__(self):
        # 定义高质量的生物学pathway类别及其权重
        # Weights optimized for DLPFC brain tissue analysis
        self.pathway_categories = {
            # 神经系统相关（最高权重）
            'neurological': {
                'keywords': ['neuron', 'synap', 'axon', 'dendrit', 'neural', 'brain', 'nervous', 
                           'dopamin', 'serotonin', 'GABA', 'glutamat', 'acetylcholin', 
                           'neurotransmit', 'postsynap', 'presynap', 'cortex', 'cortical'],
                'weight': 1.0,
                'examples': ['Synaptic vesicle cycle', 'Axon guidance', 'Neurotrophin signaling',
                            'Glutamatergic synapse', 'GABAergic synapse', 'Long-term potentiation']
            },
            
            # 神经退行性疾病（高权重）
            'neurodegenerative': {
                'keywords': ['Alzheimer', 'Parkinson', 'Huntington', 'ALS', 'amyotrophic', 
                           'neurodegeneration', 'dementia', 'prion', 'tauopathy', 'amyloid'],
                'weight': 0.95,
                'examples': ['Alzheimer disease', 'Parkinson disease', 'Pathways of neurodegeneration',
                            'Prion disease', 'Amyloidosis']
            },
            
            # 胶质细胞相关（高权重）
            'glial': {
                'keywords': ['glia', 'astrocyte', 'oligodendrocyte', 'microglia', 'myelin', 
                           'myelination', 'white matter', 'schwann', 'ependym', 'radial glia'],
                'weight': 0.9,
                'examples': ['Myelination', 'Glial cell differentiation', 'Astrocyte development']
            },
            
            # 神经炎症（新增 - 高权重，对DLPFC研究很重要）
            'neuroinflammation': {
                'keywords': ['neuroinflam', 'microglial activation', 'astrogliosis', 
                           'blood-brain barrier', 'BBB', 'reactive astrocyte', 'phagocyt'],
                'weight': 0.85,
                'examples': ['Microglial activation', 'Neuroinflammatory response', 
                            'Blood-brain barrier dysfunction']
            },
            
            # 免疫和炎症（提高权重 - 小胶质细胞功能相关）
            'immune': {
                'keywords': ['immune', 'inflammat', 'cytokine', 'interferon', 'complement', 
                           'antigen', 'innate immun', 'toll-like', 'TLR', 'NF-kB', 'inflammasome',
                           'interleukin', 'TNF', 'chemokine'],
                'weight': 0.75,  # 从0.4提高到0.75
                'examples': ['Complement cascade', 'Cytokine signaling', 'Toll-like receptor signaling',
                            'IL-6 signaling', 'TNF signaling pathway']
            },
            
            # 线粒体和能量代谢（新增 - 神经退行性疾病相关）
            'mitochondrial': {
                'keywords': ['mitochondri', 'oxidative phosphoryl', 'electron transport', 
                           'respiratory chain', 'ATP synthase', 'mitophagy', 'ROS', 
                           'reactive oxygen', 'oxidative stress'],
                'weight': 0.8,
                'examples': ['Oxidative phosphorylation', 'Mitochondrial dysfunction',
                            'Reactive oxygen species pathway']
            },
            
            # 细胞周期和增殖（中等权重）
            'cell_cycle': {
                'keywords': ['cell cycle', 'mitosis', 'meiosis', 'proliferation', 'division', 
                           'G1', 'G2', 'S phase', 'checkpoint'],
                'weight': 0.7,
                'examples': ['Cell cycle', 'p53 signaling pathway', 'DNA replication']
            },
            
            # 信号通路（中等权重）
            'signaling': {
                'keywords': ['signaling', 'pathway', 'cascade', 'PI3K', 'MAPK', 'Wnt', 'Notch', 
                           'TGF', 'mTOR', 'Hippo', 'JAK-STAT', 'AMPK'],
                'weight': 0.65,
                'examples': ['PI3K-Akt signaling pathway', 'MAPK signaling pathway', 'mTOR signaling']
            },
            
            # 代谢相关（中等权重）
            'metabolism': {
                'keywords': ['metabol', 'glycoly', 'fatty acid', 'amino acid', 
                           'glucose', 'lipid', 'cholesterol', 'pentose phosphate'],
                'weight': 0.6,
                'examples': ['Glycolysis', 'Fatty acid metabolism', 'Cholesterol biosynthesis']
            },
            
            # 病毒感染（提高权重 - EBV/HSV与神经退行性疾病有关联）
            'viral': {
                'keywords': ['virus', 'viral', 'Epstein-Barr', 'EBV', 'herpes', 'HSV', 
                           'CMV', 'cytomegalovirus'],
                'weight': 0.55,  # 从0.3提高到0.55（EBV与AD有关联研究）
                'examples': ['Epstein-Barr virus infection', 'Herpes simplex virus infection']
            },
            
            # 一般感染（低权重 - 与脑组织关系不大）
            'general_infection': {
                'keywords': ['HIV', 'influenza', 'hepatitis', 'bacterial', 'parasitic', 
                           'tuberculosis', 'malaria'],
                'weight': 0.3,
                'examples': ['Influenza A', 'Hepatitis B', 'HIV infection']
            }
        }
        
        # 定义低质量pathway（需要降权的）
        # 这些通路对DLPFC空间转录组学分析的特异性较低
        self.low_quality_pathways = {
            # 一般性疾病通路（非神经系统特异）
            'generic_disease', 'systemic lupus', 'rheumatoid',
            # 非神经系统癌症
            'breast cancer', 'lung cancer', 'colorectal cancer', 'gastric cancer',
            'prostate cancer', 'pancreatic cancer', 'renal cell carcinoma',
            # 非相关感染
            'bacterial', 'parasitic', 'tuberculosis', 'malaria', 'leishmaniasis',
            # 药物/化学相关
            'drug metabolism', 'xenobiotic', 'chemical carcinogenesis'
        }
        
        # 高质量神经系统相关pathway（额外加分）
        self.premium_neuro_pathways = {
            'long-term potentiation', 'long-term depression',
            'synaptic vesicle cycle', 'glutamatergic synapse',
            'gabaergic synapse', 'dopaminergic synapse',
            'cholinergic synapse', 'serotonergic synapse',
            'axon guidance', 'neurotrophin signaling'
        }
    
    def load_pathway_data(self, input_dir: str) -> Dict[str, pd.DataFrame]:
        """加载所有方法的pathway数据"""
        pathway_data = {}
        
        for file in os.listdir(input_dir):
            if file.endswith('_PATHWAY.csv'):
                method = file.split('_')[0]
                file_path = os.path.join(input_dir, file)
                
                try:
                    df = pd.read_csv(file_path)
                    # 验证必要的列
                    required_cols = ['Term', 'NES', 'NOM p-val', 'Lead_genes', 'Domain']
                    if all(col in df.columns for col in required_cols):
                        pathway_data[method] = df
                    else:
                        warnings.warn(f"Pathway文件 {file} 缺少必要的列")
                except Exception as e:
                    warnings.warn(f"读取pathway文件 {file} 失败: {e}")
        
        return pathway_data
    
    def categorize_pathway(self, pathway_term: str) -> Tuple[str, float]:
        """
        将pathway分类并返回类别和权重
        :param pathway_term: pathway名称
        :return: (类别名, 权重)
        """
        pathway_lower = pathway_term.lower()
        
        # 检查是否为低质量pathway
        for low_quality in self.low_quality_pathways:
            if low_quality in pathway_lower:
                return 'low_quality', 0.1
        
        # 按权重从高到低检查类别
        best_category = 'other'
        best_weight = 0.2  # 默认权重
        
        for category, info in self.pathway_categories.items():
            for keyword in info['keywords']:
                if keyword.lower() in pathway_lower:
                    if info['weight'] > best_weight:
                        best_category = category
                        best_weight = info['weight']
                    break
        
        return best_category, best_weight
    
    def calculate_pathway_quality_score(self, pathways_df: pd.DataFrame, domain_id: int) -> Dict[str, float]:
        """
        计算特定域的pathway质量分数
        :param pathways_df: pathway数据框
        :param domain_id: 域ID
        :return: 质量分数字典
        """
        domain_pathways = pathways_df[pathways_df['Domain'] == domain_id]
        
        if len(domain_pathways) == 0:
            return {
                'pathway_quality': 0.0,
                'pathway_specificity': 0.0,
                'pathway_significance': 0.0,
                'pathway_diversity': 0.0,
                'pathway_coherence': 0.0
            }
        
        # 1. 计算pathway质量（基于分类权重）
        quality_scores = []
        category_counts = defaultdict(int)
        
        for _, row in domain_pathways.iterrows():
            category, weight = self.categorize_pathway(row['Term'])
            
            # 考虑NES绝对值和p值
            nes_score = min(abs(row['NES']) / 2.0, 1.0)  # 标准化到0-1
            p_val_score = 1.0 - min(row['NOM p-val'], 1.0)  # p值越小越好
            
            pathway_score = weight * nes_score * p_val_score
            quality_scores.append(pathway_score)
            category_counts[category] += 1
        
        pathway_quality = np.mean(quality_scores) if quality_scores else 0.0
        
        # 2. 计算pathway特异性（高NES值的比例）
        high_nes_count = len(domain_pathways[abs(domain_pathways['NES']) >= 1.5])
        pathway_specificity = high_nes_count / len(domain_pathways)
        
        # 3. 计算pathway显著性（低p值的比例）
        significant_count = len(domain_pathways[domain_pathways['NOM p-val'] <= 0.05])
        pathway_significance = significant_count / len(domain_pathways)
        
        # 4. 计算pathway多样性（不同类别的数量）
        pathway_diversity = len(category_counts) / len(self.pathway_categories)
        
        # 5. 计算pathway一致性（主要类别的集中度）
        if category_counts:
            max_category_count = max(category_counts.values())
            pathway_coherence = max_category_count / len(domain_pathways)
        else:
            pathway_coherence = 0.0
        
        return {
            'pathway_quality': pathway_quality,
            'pathway_specificity': pathway_specificity,
            'pathway_significance': pathway_significance,
            'pathway_diversity': pathway_diversity,
            'pathway_coherence': pathway_coherence
        }
    
    def analyze_pathway_enrichment(self, pathway_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[int, Dict[str, float]]]:
        """
        分析所有方法的pathway富集质量
        :param pathway_data: 所有方法的pathway数据
        :return: 嵌套字典 {method: {domain: {metric: score}}}
        """
        results = {}
        
        for method, df in pathway_data.items():
            method_results = {}
            
            # 获取所有域ID
            domains = sorted(df['Domain'].unique())
            
            for domain_id in domains:
                domain_scores = self.calculate_pathway_quality_score(df, domain_id)
                method_results[domain_id] = domain_scores
            
            results[method] = method_results
        
        return results
    
    def get_pathway_enhancement_factor(self, pathway_scores: Dict[str, float]) -> float:
        """
        基于pathway分析结果计算增强因子
        :param pathway_scores: pathway质量分数
        :return: 增强因子（0.8-1.2）
        """
        # 综合各项指标计算增强因子
        quality = pathway_scores.get('pathway_quality', 0.0)
        specificity = pathway_scores.get('pathway_specificity', 0.0)
        significance = pathway_scores.get('pathway_significance', 0.0)
        diversity = pathway_scores.get('pathway_diversity', 0.0)
        coherence = pathway_scores.get('pathway_coherence', 0.0)
        
        # 加权计算综合分数
        composite_score = (
            quality * 0.4 +           # pathway质量权重最高
            specificity * 0.2 +       # 特异性
            significance * 0.2 +      # 显著性
            diversity * 0.1 +         # 多样性
            coherence * 0.1           # 一致性
        )
        
        # 将综合分数转换为增强因子（0.8-1.1范围，降低上限防止过度增强）
        # 高质量pathway增强分数，低质量pathway降低分数
        enhancement_factor = 0.8 + (composite_score * 0.3)  # 从0.4降低到0.3
        
        # 确保在更严格的范围内（上限从1.2降低到1.1）
        enhancement_factor = max(0.8, min(1.1, enhancement_factor))
        
        return enhancement_factor
    
    def get_top_pathways_summary(self, pathways_df: pd.DataFrame, domain_id: int, top_n: int = 5) -> str:
        """
        获取域的top pathway摘要
        :param pathways_df: pathway数据框
        :param domain_id: 域ID
        :param top_n: 返回top N个pathway
        :return: pathway摘要字符串
        """
        domain_pathways = pathways_df[pathways_df['Domain'] == domain_id]
        
        if len(domain_pathways) == 0:
            return "No pathways available"
        
        # 按NES绝对值和p值排序
        domain_pathways = domain_pathways.copy()
        domain_pathways['abs_nes'] = abs(domain_pathways['NES'])
        domain_pathways = domain_pathways.sort_values(['abs_nes', 'NOM p-val'], ascending=[False, True])
        
        top_pathways = domain_pathways.head(top_n)
        
        summary_parts = []
        for _, row in top_pathways.iterrows():
            category, weight = self.categorize_pathway(row['Term'])
            summary_parts.append(f"{row['Term']} (NES={row['NES']:.2f}, p={row['NOM p-val']:.3f}, cat={category})")
        
        return "; ".join(summary_parts)

def main():
    """测试pathway分析器"""
    print("🧬 Pathway分析器测试")
    print("=" * 50)
    
    analyzer = PathwayAnalyzer()
    
    # 加载数据
    input_dir = "input"
    pathway_data = analyzer.load_pathway_data(input_dir)
    
    print(f"加载了 {len(pathway_data)} 个方法的pathway数据")
    
    # 分析pathway富集
    enrichment_results = analyzer.analyze_pathway_enrichment(pathway_data)
    
    # 输出结果
    for method, method_results in enrichment_results.items():
        print(f"\n--- {method} ---")
        for domain_id, scores in method_results.items():
            enhancement_factor = analyzer.get_pathway_enhancement_factor(scores)
            
            print(f"域{domain_id}: 质量={scores['pathway_quality']:.3f}, "
                  f"特异性={scores['pathway_specificity']:.3f}, "
                  f"显著性={scores['pathway_significance']:.3f}, "
                  f"增强因子={enhancement_factor:.3f}")
            
            # 显示top pathways
            if method in pathway_data:
                top_pathways = analyzer.get_top_pathways_summary(pathway_data[method], domain_id, 3)
                print(f"  Top pathways: {top_pathways}")
    
    return enrichment_results

if __name__ == "__main__":
    main()
