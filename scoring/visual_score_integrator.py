#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
视觉评分整合器
将pic_analyze组件的视觉评分整合到主评分系统中
"""

import json
import os
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path


class VisualScoreIntegrator:
    """视觉评分整合器，负责加载和处理pic_analyze的视觉评分数据"""
    
    def __init__(self, pic_analyze_dir: str = "pic_analyze"):
        """
        初始化视觉评分整合器
        
        Args:
            pic_analyze_dir: pic_analyze组件的目录路径
        """
        self.pic_analyze_dir = Path(pic_analyze_dir)
        self.visual_scores = None
        self.integration_config = {
            'visual_weight': 0.35,  # 视觉评分在最终评分中的权重 (中等权重确保视觉分析影响)
            'smoothing_factor': 0.8,  # 平滑因子，避免过度调整
            'min_adjustment': 0.80,   # 最小调整因子 (调整范围扩大)
            'max_adjustment': 1.25,   # 最大调整因子 (调整范围扩大)
        }
        
    def load_visual_scores(self) -> bool:
        """
        加载pic_analyze的视觉评分数据
        
        Returns:
            bool: 是否成功加载
        """
        try:
            scores_file = self.pic_analyze_dir / "output" / "all_domains_scores.json"
            
            if not scores_file.exists():
                print(f"⚠️ 视觉评分文件不存在: {scores_file}")
                return False
            
            with open(scores_file, 'r', encoding='utf-8') as f:
                self.visual_scores = json.load(f)
            
            print(f"✅ 成功加载视觉评分数据，包含 {len(self.visual_scores)} 个方法")
            
            # 验证数据完整性
            self._validate_visual_scores()
            
            return True
            
        except Exception as e:
            print(f"❌ 加载视觉评分失败: {e}")
            return False
    
    def _validate_visual_scores(self):
        """验证视觉评分数据的完整性"""
        if not self.visual_scores:
            return
        
        expected_domains = set(range(1, 8))  # domain1-7
        
        for method, scores in self.visual_scores.items():
            # 检查domain完整性
            actual_domains = set(int(k.replace('domain', '')) for k in scores.keys())
            missing_domains = expected_domains - actual_domains
            
            if missing_domains:
                print(f"⚠️ 方法 {method} 缺少domain评分: {missing_domains}")
            
            # 检查评分范围
            for domain, score in scores.items():
                if not (0 <= score <= 1):
                    print(f"⚠️ 方法 {method} domain {domain} 评分超出范围: {score}")
    
    def get_visual_adjustment_factor(self, method_name: str, domain_id: int) -> float:
        """
        获取指定方法和domain的视觉调整因子
        
        Args:
            method_name: 方法名称
            domain_id: domain ID (1-7)
            
        Returns:
            float: 调整因子 (0.85-1.15)
        """
        if not self.visual_scores or method_name not in self.visual_scores:
            return 1.0  # 默认无调整
        
        domain_key = f"domain{domain_id}"
        if domain_key not in self.visual_scores[method_name]:
            return 1.0
        
        visual_score = self.visual_scores[method_name][domain_key]
        
        # 计算相对于所有方法在该domain上的表现
        all_domain_scores = []
        for method_scores in self.visual_scores.values():
            if domain_key in method_scores:
                all_domain_scores.append(method_scores[domain_key])
        
        if not all_domain_scores:
            return 1.0
        
        # 计算相对排名 (0-1)
        relative_rank = self._calculate_relative_rank(visual_score, all_domain_scores)
        
        # 转换为调整因子
        adjustment_factor = self._rank_to_adjustment_factor(relative_rank)
        
        return adjustment_factor
    
    def _calculate_relative_rank(self, score: float, all_scores: list) -> float:
        """
        计算相对排名
        
        Args:
            score: 当前分数
            all_scores: 所有分数列表
            
        Returns:
            float: 相对排名 (0-1)
        """
        if len(all_scores) <= 1:
            return 0.5
        
        # 计算有多少分数低于当前分数
        lower_count = sum(1 for s in all_scores if s < score)
        
        # 相对排名
        relative_rank = lower_count / (len(all_scores) - 1)
        
        return relative_rank
    
    def _rank_to_adjustment_factor(self, relative_rank: float) -> float:
        """
        将相对排名转换为调整因子
        
        Args:
            relative_rank: 相对排名 (0-1)
            
        Returns:
            float: 调整因子
        """
        # 使用平滑的sigmoid函数避免极端调整
        smoothing = self.integration_config['smoothing_factor']
        
        # 将排名转换为调整因子
        # 排名高的方法获得正向调整，排名低的获得负向调整
        normalized_rank = (relative_rank - 0.5) * 2  # 转换到 [-1, 1]
        
        # 应用平滑因子
        smooth_adjustment = normalized_rank * smoothing
        
        # 转换为调整因子 [min_adjustment, max_adjustment]
        min_adj = self.integration_config['min_adjustment']
        max_adj = self.integration_config['max_adjustment']
        
        if smooth_adjustment >= 0:
            # 正向调整
            adjustment_factor = 1.0 + smooth_adjustment * (max_adj - 1.0)
        else:
            # 负向调整
            adjustment_factor = 1.0 + smooth_adjustment * (1.0 - min_adj)
        
        # 确保在范围内
        adjustment_factor = np.clip(adjustment_factor, min_adj, max_adj)
        
        return adjustment_factor
    
    def get_method_visual_summary(self, method_name: str) -> Dict[str, Any]:
        """
        获取指定方法的视觉评分摘要
        
        Args:
            method_name: 方法名称
            
        Returns:
            dict: 视觉评分摘要
        """
        if not self.visual_scores or method_name not in self.visual_scores:
            return {}
        
        method_scores = self.visual_scores[method_name]
        scores_list = list(method_scores.values())
        
        summary = {
            'average_visual_score': np.mean(scores_list),
            'min_visual_score': min(scores_list),
            'max_visual_score': max(scores_list),
            'std_visual_score': np.std(scores_list),
            'domain_scores': method_scores.copy()
        }
        
        # 计算在所有方法中的排名
        all_averages = []
        for method, scores in self.visual_scores.items():
            avg_score = np.mean(list(scores.values()))
            all_averages.append((method, avg_score))
        
        all_averages.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (method, _) in enumerate(all_averages, 1):
            if method == method_name:
                summary['visual_rank'] = rank
                summary['visual_rank_total'] = len(all_averages)
                break
        
        return summary
    
    def apply_visual_adjustment(self, original_score: float, method_name: str, 
                              domain_id: int, integration_strength: float = None) -> Tuple[float, Dict[str, Any]]:
        """
        应用视觉调整到原始评分
        
        Args:
            original_score: 原始评分
            method_name: 方法名称
            domain_id: domain ID
            integration_strength: 整合强度 (0-1)，None使用默认值
            
        Returns:
            tuple: (调整后评分, 调整信息)
        """
        if integration_strength is None:
            integration_strength = self.integration_config['visual_weight']
        
        # 获取视觉调整因子
        visual_factor = self.get_visual_adjustment_factor(method_name, domain_id)
        
        # 应用渐进式整合
        adjustment_factor = 1.0 + integration_strength * (visual_factor - 1.0)
        
        # 计算调整后的评分
        adjusted_score = original_score * adjustment_factor
        
        # 确保评分在合理范围内
        adjusted_score = np.clip(adjusted_score, 0.0, 1.0)
        
        # 记录调整信息
        adjustment_info = {
            'original_score': original_score,
            'visual_factor': visual_factor,
            'adjustment_factor': adjustment_factor,
            'adjusted_score': adjusted_score,
            'score_change': adjusted_score - original_score,
            'integration_strength': integration_strength
        }
        
        return adjusted_score, adjustment_info
    
    def get_integration_statistics(self) -> Dict[str, Any]:
        """
        获取整合统计信息
        
        Returns:
            dict: 统计信息
        """
        if not self.visual_scores:
            return {}
        
        stats = {
            'total_methods': len(self.visual_scores),
            'total_domains': 7,
            'config': self.integration_config.copy()
        }
        
        # 计算视觉评分的整体统计
        all_visual_scores = []
        method_averages = []
        
        for method, scores in self.visual_scores.items():
            method_scores = list(scores.values())
            all_visual_scores.extend(method_scores)
            method_averages.append(np.mean(method_scores))
        
        stats.update({
            'visual_score_range': [min(all_visual_scores), max(all_visual_scores)],
            'visual_score_mean': np.mean(all_visual_scores),
            'visual_score_std': np.std(all_visual_scores),
            'method_average_range': [min(method_averages), max(method_averages)],
            'method_discrimination': max(method_averages) - min(method_averages)
        })
        
        return stats
    
    def update_integration_config(self, **kwargs):
        """
        更新整合配置
        
        Args:
            **kwargs: 配置参数
        """
        for key, value in kwargs.items():
            if key in self.integration_config:
                self.integration_config[key] = value
                print(f"✅ 更新配置 {key}: {value}")
            else:
                print(f"⚠️ 未知配置参数: {key}")


def test_visual_integrator():
    """测试视觉评分整合器"""
    print("🧪 测试视觉评分整合器...")
    
    integrator = VisualScoreIntegrator()
    
    # 加载视觉评分
    if not integrator.load_visual_scores():
        print("❌ 无法加载视觉评分数据")
        return
    
    # 测试调整因子计算
    test_cases = [
        ('STAGATE', 1),
        ('BayesSpace', 4),
        ('SEDR', 6),
        ('DR-SC', 2)
    ]
    
    print("\n📊 测试调整因子计算:")
    for method, domain in test_cases:
        factor = integrator.get_visual_adjustment_factor(method, domain)
        summary = integrator.get_method_visual_summary(method)
        
        print(f"  {method} Domain{domain}: 调整因子={factor:.3f}, "
              f"平均视觉评分={summary.get('average_visual_score', 0):.3f}")
    
    # 测试评分调整
    print("\n🔧 测试评分调整:")
    original_score = 0.75
    for method, domain in test_cases:
        adjusted_score, info = integrator.apply_visual_adjustment(
            original_score, method, domain
        )
        
        print(f"  {method} Domain{domain}: {original_score:.3f} → {adjusted_score:.3f} "
              f"(变化: {info['score_change']:+.3f})")
    
    # 显示统计信息
    stats = integrator.get_integration_statistics()
    print(f"\n📈 整合统计:")
    print(f"  视觉评分范围: {stats['visual_score_range'][0]:.3f} - {stats['visual_score_range'][1]:.3f}")
    print(f"  方法区分度: {stats['method_discrimination']:.3f}")
    print(f"  视觉权重: {stats['config']['visual_weight']}")


if __name__ == "__main__":
    test_visual_integrator()
