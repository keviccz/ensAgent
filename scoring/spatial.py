import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import warnings
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree, Delaunay, ConvexHull
from scipy.spatial import Delaunay, ConvexHull

# 空间度量工具类
class SpatialMetrics:
    def __init__(self):
        pass
    
    def _validate_and_standardize_coords(self, assignment_df: pd.DataFrame) -> pd.DataFrame:
        """
        验证并标准化坐标列，支持多种坐标列命名
        :param assignment_df: 输入DataFrame
        :return: 标准化后的DataFrame（包含x, y列）
        """
        df = assignment_df.copy()
        
        # 检查可能的坐标列命名
        coord_mappings = [
            ('x', 'y'),           # 标准命名
            ('coord_x', 'coord_y'), # 常见命名
            ('X', 'Y'),           # 大写命名
            ('pos_x', 'pos_y'),   # 位置命名
            ('spatial_x', 'spatial_y'), # 空间命名
        ]
        
        found_coords = False
        for x_col, y_col in coord_mappings:
            if x_col in df.columns and y_col in df.columns:
                if x_col != 'x' or y_col != 'y':
                    # 重命名为标准列名
                    df = df.rename(columns={x_col: 'x', y_col: 'y'})
                    print(f"[Info] 坐标列已重命名: {x_col}->'x', {y_col}->'y'")
                found_coords = True
                break
        
        if not found_coords:
            raise ValueError("未找到有效的坐标列。支持的坐标列命名: x/y, coord_x/coord_y, X/Y, pos_x/pos_y, spatial_x/spatial_y")
        
        # 检查并标准化domain列命名
        domain_mappings = ['domain_index', 'spatial_domain', 'domain', 'cluster', 'label']
        found_domain = False
        
        for domain_col in domain_mappings:
            if domain_col in df.columns:
                if domain_col != 'domain_index':
                    df = df.rename(columns={domain_col: 'domain_index'})
                    print(f"[Info] 域列已重命名: {domain_col}->'domain_index'")
                found_domain = True
                break
        
        if not found_domain:
            raise ValueError("未找到有效的域列。支持的域列命名: domain_index, spatial_domain, domain, cluster, label")
        
        # 验证坐标数据类型和范围
        for col in ['x', 'y']:
            if col not in df.columns:
                continue
                
            # 转换为数值类型
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                raise ValueError(f"坐标列 {col} 无法转换为数值类型: {e}")
            
            # 检查缺失值
            if df[col].isna().any():
                na_count = df[col].isna().sum()
                warnings.warn(f"坐标列 {col} 包含 {na_count} 个缺失值，将被排除在空间计算之外")
            
            # 检查坐标范围合理性
            if df[col].min() < 0:
                warnings.warn(f"坐标列 {col} 包含负值，请确认坐标系统正确")
        
        # 确保domain_index为整数类型
        df['domain_index'] = df['domain_index'].astype(int)
        
        return df
    
    def _compute_distance_matrix(self, coords: np.ndarray) -> np.ndarray:
        """
        计算距离矩阵，优化大数据集性能
        :param coords: 坐标数组 (n_points, 2)
        :return: 距离矩阵 (n_points, n_points)
        """
        if len(coords) == 0:
            return np.array([])
        
        # 使用广播计算欧几里得距离
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
        return dist_matrix
    
    def _compute_spatial_weights_matrix(self, coords: np.ndarray, method: str = 'knn', k: int = 8, distance_threshold: float = None) -> np.ndarray:
        """
        计算空间权重矩阵，支持多种权重策略
        :param coords: 坐标数组 (n_points, 2)
        :param method: 权重计算方法 ('knn', 'distance_threshold', 'inverse_distance')
        :param k: KNN方法的邻居数量
        :param distance_threshold: 距离阈值方法的阈值
        :return: 空间权重矩阵 (n_points, n_points)
        """
        n_points = len(coords)
        if n_points < 2:
            return np.zeros((n_points, n_points))
        
        # 计算距离矩阵
        dist_matrix = self._compute_distance_matrix(coords)
        weights_matrix = np.zeros((n_points, n_points))
        
        if method == 'knn':
            # K最近邻权重矩阵
            for i in range(n_points):
                # 获取k个最近邻（排除自身）
                distances = dist_matrix[i]
                # 排除自身距离（距离为0）
                valid_distances = distances[distances > 0]
                if len(valid_distances) > 0:
                    threshold = np.partition(valid_distances, min(k-1, len(valid_distances)-1))[min(k-1, len(valid_distances)-1)]
                    neighbors = (distances <= threshold) & (distances > 0)
                    weights_matrix[i, neighbors] = 1.0
        
        elif method == 'distance_threshold':
            # 距离阈值权重矩阵
            if distance_threshold is None:
                # 自动计算阈值：平均最近邻距离的2倍
                nn_distances = []
                for i in range(n_points):
                    distances = dist_matrix[i]
                    min_dist = np.min(distances[distances > 0]) if np.any(distances > 0) else 0
                    if min_dist > 0:
                        nn_distances.append(min_dist)
                distance_threshold = np.mean(nn_distances) * 2 if nn_distances else 1.0
            
            weights_matrix = (dist_matrix <= distance_threshold) & (dist_matrix > 0)
            weights_matrix = weights_matrix.astype(float)
        
        elif method == 'inverse_distance':
            # 反距离权重矩阵
            with np.errstate(divide='ignore', invalid='ignore'):
                weights_matrix = 1.0 / dist_matrix
            weights_matrix[np.isinf(weights_matrix)] = 0  # 自身权重设为0
            weights_matrix[np.isnan(weights_matrix)] = 0
        
        # 行标准化权重矩阵
        row_sums = np.sum(weights_matrix, axis=1)
        row_sums[row_sums == 0] = 1  # 避免除零
        weights_matrix = weights_matrix / row_sums[:, np.newaxis]
        
        return weights_matrix
    
    def _get_global_scale(self, assignment_df: pd.DataFrame) -> float:
        """
        获取全局坐标尺度，用于归一化
        :param assignment_df: 包含坐标的DataFrame
        :return: 全局最大距离
        """
        if 'x' not in assignment_df.columns or 'y' not in assignment_df.columns:
            return 1.0
        
        # 计算坐标范围
        x_range = assignment_df['x'].max() - assignment_df['x'].min()
        y_range = assignment_df['y'].max() - assignment_df['y'].min()
        
        # 使用对角线距离作为全局尺度
        global_scale = np.sqrt(x_range**2 + y_range**2)
        return max(global_scale, 1.0)  # 避免除零

    def compute_morans_i(self, assignment_df: pd.DataFrame, weight_method: str = 'knn', k: int = 8) -> Dict[int, Dict[str, float]]:
        """
        计算Moran's I空间自相关指数 (权威方法)
        
        理论基础：
        - Moran, P.A.P. (1950). "Notes on Continuous Stochastic Phenomena". Biometrika 37:17-23
        - Anselin, L. (1995). "Local Indicators of Spatial Association". Geographical Analysis 27:93-115
        
        公式：I = (n/S0) * Σ(i,j) w_ij * (x_i - x̄)(x_j - x̄) / Σ_i (x_i - x̄)²
        
        :param assignment_df: 包含spot_id, domain_index, 坐标列的DataFrame
        :param weight_method: 空间权重方法 ('knn', 'distance_threshold', 'inverse_distance')
        :param k: KNN方法的邻居数量
        :return: {domain_index: {'morans_i': value, 'p_value': p_val, 'z_score': z_val}}
        """
        try:
            df = self._validate_and_standardize_coords(assignment_df)
        except ValueError as e:
            warnings.warn(f"坐标验证失败: {e}")
            domains = assignment_df['domain_index'].unique()
            return {int(d): {'morans_i': np.nan, 'p_value': np.nan, 'z_score': np.nan} for d in domains}
        
        results = {}
        
        for d in df['domain_index'].unique():
            domain_idx = int(d)
            sub = df[df['domain_index'] == d].dropna(subset=['x', 'y'])
            
            if len(sub) < 3:  # Moran's I需要至少3个点
                results[domain_idx] = {'morans_i': np.nan, 'p_value': np.nan, 'z_score': np.nan}
                continue
            
            # 获取坐标和域标签
            coords = sub[['x', 'y']].values
            domain_labels = sub['domain_index'].values
            
            # 计算空间权重矩阵
            W = self._compute_spatial_weights_matrix(coords, method=weight_method, k=k)
            S0 = np.sum(W)  # 权重总和
            
            if S0 == 0:
                results[domain_idx] = {'morans_i': np.nan, 'p_value': np.nan, 'z_score': np.nan}
                continue
            
            n = len(coords)
            
            # 计算domain内的一致性（binary变量：1表示同domain，0表示不同domain）
            # 对于单一domain，所有点都是1，这里我们计算坐标的连续性
            x_values = coords[:, 0]  # 使用x坐标作为连续变量
            x_mean = np.mean(x_values)
            x_centered = x_values - x_mean
            
            # 计算Moran's I
            numerator = 0
            for i in range(n):
                for j in range(n):
                    numerator += W[i, j] * x_centered[i] * x_centered[j]
            
            denominator = np.sum(x_centered**2)
            
            if denominator == 0:
                morans_i = 0
            else:
                morans_i = (n / S0) * (numerator / denominator)
            
            # 计算期望值和方差（在零假设下）
            expected_i = -1 / (n - 1)
            
            # 简化的方差计算（完整计算需要更复杂的公式）
            S1 = 0.5 * np.sum((W + W.T)**2)
            S2 = np.sum(np.sum(W, axis=1)**2)
            
            variance_i = ((n * ((n**2 - 3*n + 3) * S1 - n*S2 + 3*S0**2)) - 
                         (expected_i * (n * (n-1) * S1 - 2*n*S2 + 6*S0**2))) / ((n-1) * (n-2) * (n-3) * S0**2)
            
            # 计算Z得分和p值
            if variance_i > 0:
                z_score = (morans_i - expected_i) / np.sqrt(variance_i)
                # 近似p值（双尾检验）
                from scipy.stats import norm
                p_value = 2 * (1 - norm.cdf(abs(z_score)))
            else:
                z_score = np.nan
                p_value = np.nan
            
            results[domain_idx] = {
                'morans_i': float(morans_i),
                'p_value': float(p_value) if not np.isnan(p_value) else np.nan,
                'z_score': float(z_score) if not np.isnan(z_score) else np.nan
            }
        
        return results

    def compute_gearys_c(self, assignment_df: pd.DataFrame, weight_method: str = 'knn', k: int = 8) -> Dict[int, Dict[str, float]]:
        """
        计算Geary's C空间连续性指数 (权威方法)
        
        理论基础：
        - Geary, R.C. (1954). "The Contiguity Ratio and Statistical Mapping". The Incorporated Statistician 5:115-146
        - Cliff, A.D. and Ord, J.K. (1973). "Spatial Autocorrelation". Pion Limited, London
        
        公式：C = ((n-1)/2S0) * Σ(i,j) w_ij * (x_i - x_j)² / Σ_i (x_i - x̄)²
        
        :param assignment_df: 包含spot_id, domain_index, 坐标列的DataFrame
        :param weight_method: 空间权重方法 ('knn', 'distance_threshold', 'inverse_distance')
        :param k: KNN方法的邻居数量
        :return: {domain_index: {'gearys_c': value, 'p_value': p_val, 'z_score': z_val}}
        """
        try:
            df = self._validate_and_standardize_coords(assignment_df)
        except ValueError as e:
            warnings.warn(f"坐标验证失败: {e}")
            domains = assignment_df['domain_index'].unique()
            return {int(d): {'gearys_c': np.nan, 'p_value': np.nan, 'z_score': np.nan} for d in domains}
        
        results = {}
        
        for d in df['domain_index'].unique():
            domain_idx = int(d)
            sub = df[df['domain_index'] == d].dropna(subset=['x', 'y'])
            
            if len(sub) < 3:  # Geary's C需要至少3个点
                results[domain_idx] = {'gearys_c': np.nan, 'p_value': np.nan, 'z_score': np.nan}
                continue
            
            # 获取坐标
            coords = sub[['x', 'y']].values
            
            # 计算空间权重矩阵
            W = self._compute_spatial_weights_matrix(coords, method=weight_method, k=k)
            S0 = np.sum(W)  # 权重总和
            
            if S0 == 0:
                results[domain_idx] = {'gearys_c': np.nan, 'p_value': np.nan, 'z_score': np.nan}
                continue
            
            n = len(coords)
            
            # 使用x坐标作为连续变量计算Geary's C
            x_values = coords[:, 0]
            x_mean = np.mean(x_values)
            
            # 计算Geary's C
            numerator = 0
            for i in range(n):
                for j in range(n):
                    numerator += W[i, j] * (x_values[i] - x_values[j])**2
            
            denominator = 2 * S0 * np.sum((x_values - x_mean)**2)
            
            if denominator == 0:
                gearys_c = 1.0  # 期望值
            else:
                gearys_c = ((n - 1) / denominator) * numerator
            
            # 计算期望值和方差（在零假设下）
            expected_c = 1.0
            
            # 简化的方差计算
            S1 = 0.5 * np.sum((W + W.T)**2)
            S2 = np.sum(np.sum(W, axis=1)**2)
            
            variance_c = ((2*S1 + S2) * (n-1) - 4*S0**2) / (2 * (n+1) * (n-3) * S0**2)
            
            # 计算Z得分和p值
            if variance_c > 0:
                z_score = (gearys_c - expected_c) / np.sqrt(variance_c)
                # 近似p值（双尾检验）
                from scipy.stats import norm
                p_value = 2 * (1 - norm.cdf(abs(z_score)))
            else:
                z_score = np.nan
                p_value = np.nan
            
            results[domain_idx] = {
                'gearys_c': float(gearys_c),
                'p_value': float(p_value) if not np.isnan(p_value) else np.nan,
                'z_score': float(z_score) if not np.isnan(z_score) else np.nan
            }
        
        return results

    def compute_compactness(self, assignment_df: pd.DataFrame) -> Dict[int, float]:
        """
        计算每个domain的紧凑度（compactness）
        :param assignment_df: 包含spot_id, domain_index, 坐标列的DataFrame
        :return: {domain_index: compactness_score}
        """
        try:
            # 验证并标准化坐标
            df = self._validate_and_standardize_coords(assignment_df)
        except ValueError as e:
            warnings.warn(f"坐标验证失败: {e}")
            # 返回所有domain的NaN值
            domains = assignment_df['domain_index'].unique()
            return {int(d): np.nan for d in domains}
        
        compactness = {}
        global_scale = self._get_global_scale(df)
        
        for d in df['domain_index'].unique():
            domain_idx = int(d)
            sub = df[df['domain_index'] == d]
            
            # 过滤掉坐标缺失的点
            sub = sub.dropna(subset=['x', 'y'])
            
            if len(sub) < 2:
                compactness[domain_idx] = 1.0 if len(sub) == 1 else 0.0
                continue
            
            # 获取坐标
            coords = sub[['x', 'y']].values
            
            # 计算距离矩阵
            dist_matrix = self._compute_distance_matrix(coords)
            
            # 计算平均距离（排除对角线）
            n_points = len(coords)
            total_dist = np.sum(dist_matrix) - np.sum(np.diag(dist_matrix))
            mean_dist = total_dist / (n_points * (n_points - 1))
            
            # 归一化并计算紧凑度（距离越小越紧凑）
            norm_score = 1.0 - (mean_dist / global_scale)
            compactness[domain_idx] = float(np.clip(norm_score, 0, 1))
        
        return compactness

    def compute_adjacency(self, assignment_df: pd.DataFrame) -> Dict[int, float]:
        """
        计算每个domain的邻接性（adjacency），用平均最近邻距离的倒数归一化
        :param assignment_df: 包含spot_id, domain_index, 坐标列的DataFrame
        :return: {domain_index: adjacency_score}
        """
        try:
            # 验证并标准化坐标
            df = self._validate_and_standardize_coords(assignment_df)
        except ValueError as e:
            warnings.warn(f"坐标验证失败: {e}")
            # 返回所有domain的NaN值
            domains = assignment_df['domain_index'].unique()
            return {int(d): np.nan for d in domains}
        
        adjacency = {}
        global_scale = self._get_global_scale(df)
        
        for d in df['domain_index'].unique():
            domain_idx = int(d)
            sub = df[df['domain_index'] == d]
            
            # 过滤掉坐标缺失的点
            sub = sub.dropna(subset=['x', 'y'])
            
            if len(sub) < 2:
                adjacency[domain_idx] = 1.0 if len(sub) == 1 else 0.0
                continue
            
            # 获取坐标
            coords = sub[['x', 'y']].values
            
            # 计算每个点到同domain其他点的最近邻距离
            nn_distances = []
            for i in range(len(coords)):
                # 排除自身
                other_coords = np.delete(coords, i, axis=0)
                # 计算到所有其他点的距离
                distances = np.sqrt(np.sum((coords[i] - other_coords)**2, axis=1))
                # 取最小距离
                nn_distances.append(np.min(distances))
            
            # 计算平均最近邻距离
            mean_nn_dist = np.mean(nn_distances)
            
            # 归一化并计算邻接性（距离越小邻接性越高）
            norm_score = 1.0 - (mean_nn_dist / global_scale)
            adjacency[domain_idx] = float(np.clip(norm_score, 0, 1))
        
        return adjacency
    
    def compute_multi_scale_compactness(self, assignment_df: pd.DataFrame, k_values: list = [6, 8, 12, 16]) -> Dict[int, Dict[str, float]]:
        """
        多尺度紧致度计算：在不同k值下计算紧致度
        :param assignment_df: 输入DataFrame
        :param k_values: 不同的k值列表
        :return: 每个域的多尺度紧致度指标
        """
        try:
            df = self._validate_and_standardize_coords(assignment_df)
        except ValueError:
            domains = assignment_df['domain_index'].unique()
            return {int(d): {f'compactness_k{k}': np.nan for k in k_values} for d in domains}
        
        results = {}
        
        for domain in df['domain_index'].unique():
            domain_idx = int(domain)
            domain_points = df[df['domain_index'] == domain][['x', 'y']].values
            if len(domain_points) < 3:
                results[domain_idx] = {f'compactness_k{k}': np.nan for k in k_values}
                results[domain_idx].update({
                    'compactness_mean': np.nan,
                    'compactness_var': np.nan,
                    'compactness_max': np.nan
                })
                continue
                
            k_compactness = []
            for k in k_values:
                if len(domain_points) <= k:
                    k_comp = np.mean(pdist(domain_points)) if len(domain_points) > 1 else 0
                else:
                    # 计算每个点到其k近邻的平均距离
                    tree = cKDTree(domain_points)
                    distances, _ = tree.query(domain_points, k=min(k+1, len(domain_points)))
                    k_comp = np.mean(distances[:, 1:])  # 排除自身
                
                k_compactness.append(k_comp)
                results.setdefault(domain_idx, {})[f'compactness_k{k}'] = k_comp
            
            # 聚合统计
            results[domain_idx]['compactness_mean'] = np.mean(k_compactness)
            results[domain_idx]['compactness_var'] = np.var(k_compactness)
            results[domain_idx]['compactness_max'] = np.max(k_compactness)
            
        return results

    def compute_boundary_metrics(self, assignment_df: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        """
        计算边界相关度量：凸包面积、周长、跨域邻居率等
        :param assignment_df: 输入DataFrame
        :return: 每个域的边界度量
        """
        try:
            df = self._validate_and_standardize_coords(assignment_df)
        except ValueError:
            domains = assignment_df['domain_index'].unique()
            return {int(d): {'convex_hull_area': np.nan, 'perimeter_area_ratio': np.nan, 
                           'cross_domain_neighbor_rate': np.nan, 'boundary_compactness': np.nan} for d in domains}
        
        results = {}
        all_coords = df[['x', 'y']].values
        tree = cKDTree(all_coords)
        
        for domain in df['domain_index'].unique():
            domain_idx = int(domain)
            domain_points = df[df['domain_index'] == domain][['x', 'y']].values
            if len(domain_points) < 3:
                results[domain_idx] = {
                    'convex_hull_area': np.nan,
                    'perimeter_area_ratio': np.nan,
                    'cross_domain_neighbor_rate': np.nan,
                    'boundary_compactness': np.nan
                }
                continue
            
            # 计算凸包面积和周长
            try:
                hull = ConvexHull(domain_points)
                hull_area = hull.volume  # 2D中volume就是面积
                hull_perimeter = hull.area  # 2D中area就是周长
                perimeter_area_ratio = hull_perimeter / hull_area if hull_area > 0 else np.inf
            except Exception:
                hull_area = np.nan
                perimeter_area_ratio = np.nan
            
            # 计算跨域邻居率
            domain_indices = df[df['domain_index'] == domain].index
            cross_domain_neighbors = 0
            total_neighbors = 0
            
            for idx in domain_indices:
                point = all_coords[idx]
                # 找最近的8个邻居
                distances, neighbor_indices = tree.query(point, k=min(9, len(all_coords)))
                neighbor_indices = neighbor_indices[1:]  # 排除自身
                
                for neighbor_idx in neighbor_indices:
                    if neighbor_idx < len(df):
                        neighbor_domain = df.iloc[neighbor_idx]['domain_index']
                        if neighbor_domain != domain:
                            cross_domain_neighbors += 1
                        total_neighbors += 1
            
            cross_domain_rate = cross_domain_neighbors / total_neighbors if total_neighbors > 0 else 0
            
            # 边界紧致度（域内点到域中心的距离变异系数）
            domain_center = np.mean(domain_points, axis=0)
            boundary_distances = np.linalg.norm(domain_points - domain_center, axis=1)
            boundary_compactness = np.std(boundary_distances) / np.mean(boundary_distances) if np.mean(boundary_distances) > 0 else 0
            
            results[domain_idx] = {
                'convex_hull_area': hull_area,
                'perimeter_area_ratio': perimeter_area_ratio,
                'cross_domain_neighbor_rate': cross_domain_rate,
                'boundary_compactness': boundary_compactness
            }
            
        return results

    # --- New shape quality metrics ---
    def _alpha_shape_area(self, coords: np.ndarray, edge_factor: float = 1.5) -> float:
        """Approximate concave (alpha-shape) area via filtered Delaunay triangles."""
        if coords is None or len(coords) < 3:
            return 0.0
        try:
            tri = Delaunay(coords)
        except Exception:
            return 0.0
        simplices = tri.simplices
        if len(simplices) == 0:
            return 0.0
        def tri_edges(pts):
            return [np.linalg.norm(pts[0]-pts[1]), np.linalg.norm(pts[1]-pts[2]), np.linalg.norm(pts[2]-pts[0])]
        edge_lengths = []
        for s in simplices:
            p = coords[s]
            edge_lengths.extend(tri_edges(p))
        med_edge = np.median(edge_lengths) if edge_lengths else np.nan
        max_edge = edge_factor * med_edge if np.isfinite(med_edge) and med_edge > 0 else np.inf
        area = 0.0
        for s in simplices:
            p = coords[s]
            e = tri_edges(p)
            if all(edge <= max_edge for edge in e):
                x1,y1 = p[0]; x2,y2 = p[1]; x3,y3 = p[2]
                area += abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) / 2.0
        return float(area)

    def compute_convexity(self, assignment_df: pd.DataFrame) -> Dict[int, float]:
        """convexity ≈ area(alpha-shape) / area(convex hull) in [0,1]."""
        try:
            df = self._validate_and_standardize_coords(assignment_df)
        except ValueError:
            domains = assignment_df['domain_index'].unique()
            return {int(d): np.nan for d in domains}
        results: Dict[int, float] = {}
        for d in df['domain_index'].unique():
            domain_idx = int(d)
            sub = df[df['domain_index'] == d].dropna(subset=['x','y'])
            coords = sub[['x','y']].values
            if len(coords) < 3:
                results[domain_idx] = np.nan
                continue
            try:
                hull = ConvexHull(coords)
                a_hull = float(hull.volume)
            except Exception:
                a_hull = np.nan
            a_shape = self._alpha_shape_area(coords, edge_factor=1.5)
            if not np.isfinite(a_hull) or a_hull <= 0:
                results[domain_idx] = np.nan
            else:
                results[domain_idx] = float(np.clip(a_shape / a_hull, 0.0, 1.0))
        return results

    def _alpha_shape_boundary(self, coords: np.ndarray, edge_factor: float = 1.5) -> Optional[np.ndarray]:
        if coords is None or len(coords) < 4:
            return None
        try:
            tri = Delaunay(coords)
        except Exception:
            return None
        simplices = tri.simplices
        if len(simplices) == 0:
            return None
        from collections import Counter, defaultdict
        def tri_edges_idx(s):
            a,b,c = int(s[0]), int(s[1]), int(s[2])
            return [(min(a,b), max(a,b)), (min(b,c), max(b,c)), (min(c,a), max(c,a))]
        # threshold by median edge
        def tri_edges_len(pts):
            return [np.linalg.norm(pts[0]-pts[1]), np.linalg.norm(pts[1]-pts[2]), np.linalg.norm(pts[2]-pts[0])]
        all_edge_lens = []
        keep = []
        for s in simplices:
            p = coords[s]
            lens = tri_edges_len(p)
            all_edge_lens.extend(lens)
        med_edge = np.median(all_edge_lens) if all_edge_lens else np.nan
        max_edge = edge_factor * med_edge if np.isfinite(med_edge) and med_edge > 0 else np.inf
        for s in simplices:
            p = coords[s]
            lens = tri_edges_len(p)
            if all(L <= max_edge for L in lens):
                keep.append(s)
        edge_counter = Counter()
        for s in keep:
            for e in tri_edges_idx(s):
                edge_counter[e] += 1
        boundary_edges = [e for e,cnt in edge_counter.items() if cnt == 1]
        if not boundary_edges:
            return None
        adj = defaultdict(list)
        for a,b in boundary_edges:
            adj[a].append(b)
            adj[b].append(a)
        used = set()
        best_loop = None
        best_len = -1
        for start in list(adj.keys()):
            if start in used:
                continue
            loop = [start]
            prev = None
            curr = start
            steps = 0
            while True:
                nbrs = adj[curr]
                nxt = None
                for v in nbrs:
                    if v != prev:
                        nxt = v
                        break
                if nxt is None:
                    break
                if nxt == start and len(loop) > 2:
                    loop.append(nxt)
                    break
                prev = curr
                curr = nxt
                loop.append(curr)
                steps += 1
                if steps > 100000:
                    break
            for v in loop:
                used.add(v)
            if loop and loop[-1] == start and len(loop) > best_len:
                best_loop = loop[:-1]
                best_len = len(best_loop)
        if best_loop is None or len(best_loop) < 3:
            return None
        return coords[np.array(best_loop)]

    def _perimeter(self, poly: Optional[np.ndarray]) -> float:
        if poly is None or len(poly) < 2:
            return np.nan
        return float(np.sqrt(((np.roll(poly, -1, axis=0) - poly)**2).sum(axis=1)).sum())

    def _smooth_closed(self, poly: Optional[np.ndarray], window: int = 7) -> Optional[np.ndarray]:
        if poly is None or len(poly) < 3:
            return None
        k = max(3, window)
        if k % 2 == 0:
            k += 1
        pad = k // 2
        ext = np.vstack([poly[-pad:], poly, poly[:pad]])
        ker = np.ones((k,)) / k
        xs = np.convolve(ext[:, 0], ker, mode='valid')
        ys = np.convolve(ext[:, 1], ker, mode='valid')
        sm = np.vstack([xs, ys]).T
        sm = sm[:len(poly)]
        return sm

    def compute_jaggedness(self, assignment_df: pd.DataFrame) -> Dict[int, float]:
        """Jaggedness index = max(Perimeter(original)/Perimeter(smoothed) - 1, 0)."""
        try:
            df = self._validate_and_standardize_coords(assignment_df)
        except ValueError:
            domains = assignment_df['domain_index'].unique()
            return {int(d): np.nan for d in domains}
        results: Dict[int, float] = {}
        for d in df['domain_index'].unique():
            domain_idx = int(d)
            sub = df[df['domain_index'] == d].dropna(subset=['x','y'])
            coords = sub[['x','y']].values
            boundary = self._alpha_shape_boundary(coords, edge_factor=1.5)
            if boundary is None:
                results[domain_idx] = np.nan
                continue
            p_orig = self._perimeter(boundary)
            sm = self._smooth_closed(boundary, window=7)
            p_smooth = self._perimeter(sm)
            if not (np.isfinite(p_orig) and np.isfinite(p_smooth)) or p_smooth <= 0:
                results[domain_idx] = np.nan
            else:
                results[domain_idx] = float(max(p_orig / p_smooth - 1.0, 0.0))
        return results

    def compute_connectivity_metrics(self, assignment_df: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        """
        计算连通性度量：连通分量数、碎片化指数等
        :param assignment_df: 输入DataFrame
        :return: 每个域的连通性度量
        """
        try:
            df = self._validate_and_standardize_coords(assignment_df)
        except ValueError:
            domains = assignment_df['domain_index'].unique()
            return {int(d): {'connected_components': 1, 'fragmentation_index': 0.0} for d in domains}
        
        results = {}
        
        for domain in df['domain_index'].unique():
            domain_idx = int(domain)
            domain_points = df[df['domain_index'] == domain][['x', 'y']].values
            
            if len(domain_points) < 2:
                results[domain_idx] = {
                    'connected_components': 1,
                    'fragmentation_index': 0.0
                }
                continue
            
            # 构建域内邻接图
            tree = cKDTree(domain_points)
            adjacency_matrix = np.zeros((len(domain_points), len(domain_points)))
            
            # 计算动态距离阈值
            sample_size = min(len(domain_points), 50)
            sample_indices = np.random.choice(len(domain_points), sample_size, replace=False)
            sample_distances = []
            for i in sample_indices:
                distances, _ = tree.query(domain_points[i], k=min(4, len(domain_points)))
                sample_distances.extend(distances[1:])  # 排除自身
            threshold = np.median(sample_distances) * 1.5 if sample_distances else 1.0
            
            for i, point in enumerate(domain_points):
                distances, neighbor_indices = tree.query(point, k=min(9, len(domain_points)))
                for j, neighbor_idx in enumerate(neighbor_indices[1:]):
                    if distances[j+1] <= threshold:
                        adjacency_matrix[i, neighbor_idx] = 1
                        adjacency_matrix[neighbor_idx, i] = 1
            
            # 计算连通分量
            visited = np.zeros(len(domain_points), dtype=bool)
            components = []
            
            def dfs(node, component):
                visited[node] = True
                component.append(node)
                for neighbor in np.where(adjacency_matrix[node])[0]:
                    if not visited[neighbor]:
                        dfs(neighbor, component)
            
            for i in range(len(domain_points)):
                if not visited[i]:
                    component = []
                    dfs(i, component)
                    components.append(component)
            
            num_components = len(components)
            fragmentation_index = (num_components - 1) / len(domain_points) if len(domain_points) > 1 else 0
            
            results[domain_idx] = {
                'connected_components': num_components,
                'fragmentation_index': fragmentation_index
            }
            
        return results

    def compute_spatial_stats(self, assignment_df: pd.DataFrame) -> Dict[str, Dict[int, float]]:
        """
        计算所有空间统计指标（增强版）
        :param assignment_df: 包含spot_id, domain_index, 坐标列的DataFrame
        :return: {'compactness': {...}, 'adjacency': {...}, 'morans_i': {...},
                  'gearys_c': {...}, 'dispersion': {...}, 'multi_scale_compactness': {...},
                  'boundary_metrics': {...}, 'connectivity_metrics': {...}}
        """
        # 计算基础指标
        basic_stats = {
            'compactness': self.compute_compactness(assignment_df),
            'adjacency': self.compute_adjacency(assignment_df),
            'morans_i': self.compute_morans_i(assignment_df),
            'gearys_c': self.compute_gearys_c(assignment_df),
            'dispersion': self.compute_dispersion(assignment_df)
        }
        
        # 计算增强指标
        enhanced_stats = {
            'multi_scale_compactness': self.compute_multi_scale_compactness(assignment_df),
            'boundary_metrics': self.compute_boundary_metrics(assignment_df),
            'connectivity_metrics': self.compute_connectivity_metrics(assignment_df),
            'nearest_neighbor_distances': self.compute_nearest_neighbor_distances(assignment_df),
            'convexity': self.compute_convexity(assignment_df),
            'jaggedness': self.compute_jaggedness(assignment_df)
        }
        
        # 合并所有指标
        basic_stats.update(enhanced_stats)
        return basic_stats

    # --- NEW heterogeneity metric ---
    def compute_dispersion(self, assignment_df: pd.DataFrame) -> Dict[int, float]:
        """计算域内点分散程度（CV 距离中心），0=集中，1=高度分散"""
        try:
            df = self._validate_and_standardize_coords(assignment_df)
        except ValueError:
            domains = assignment_df['domain_index'].unique()
            return {int(d): np.nan for d in domains}

        dispersion = {}
        global_scale = self._get_global_scale(df)

        for d in df['domain_index'].unique():
            domain_idx = int(d)
            sub = df[df['domain_index'] == d].dropna(subset=['x', 'y'])
            if len(sub) < 3:
                dispersion[domain_idx] = 0.0
                continue
            coords = sub[['x', 'y']].values
            centroid = coords.mean(axis=0)
            dists = np.sqrt(((coords - centroid)**2).sum(axis=1))
            cv = np.std(dists) / (np.mean(dists)+1e-6)
            # 归一化到 0-1 （经验上 cv<=0.3 视为集中，>=1 视为极分散）
            norm_disp = np.clip((cv - 0.3) / (1.0 - 0.3), 0, 1)
            dispersion[domain_idx] = float(norm_disp)

        return dispersion
    
    def compute_nearest_neighbor_distances(self, assignment_df: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        """
        计算每个domain的最近邻距离分布统计
        :param assignment_df: 包含spot_id, domain_index, 坐标列的DataFrame
        :return: {domain_index: {'mean_nn_dist': float, 'median_nn_dist': float}}
        """
        try:
            df = self._validate_and_standardize_coords(assignment_df)
        except ValueError:
            domains = assignment_df['domain_index'].unique()
            return {int(d): {'mean_nn_dist': np.nan, 'median_nn_dist': np.nan} for d in domains}
        
        from scipy.spatial import cKDTree
        results = {}
        global_scale = self._get_global_scale(df)
        
        for d in df['domain_index'].unique():
            domain_idx = int(d)
            sub = df[df['domain_index'] == d].dropna(subset=['x', 'y'])
            
            if len(sub) < 2:
                results[domain_idx] = {
                    'mean_nn_dist': 0.0 if len(sub) == 1 else np.nan,
                    'median_nn_dist': 0.0 if len(sub) == 1 else np.nan
                }
                continue
            
            coords = sub[['x', 'y']].values
            tree = cKDTree(coords)
            
            # 计算每个点的最近邻距离
            nn_distances = []
            for i, point in enumerate(coords):
                # 查找最近的2个点（包括自身），取第二个作为最近邻
                distances, _ = tree.query(point, k=min(2, len(coords)))
                if len(distances) > 1:
                    nn_distances.append(distances[1])  # 排除自身
                else:
                    nn_distances.append(0.0)
            
            # 计算统计指标
            mean_nn = np.mean(nn_distances)
            median_nn = np.median(nn_distances)
            
            # 归一化到全局尺度
            results[domain_idx] = {
                'mean_nn_dist': float(mean_nn / global_scale),  # 归一化平均最近邻距离
                'median_nn_dist': float(median_nn / global_scale)  # 归一化中位数距离
            }
        
        return results
    
    def validate_coordinates(self, assignment_df: pd.DataFrame) -> Tuple[bool, str]:
        """
        验证坐标数据的完整性和合理性
        :param assignment_df: 输入DataFrame
        :return: (是否有效, 错误信息)
        """
        try:
            df = self._validate_and_standardize_coords(assignment_df)
            
            # 检查坐标覆盖率
            total_spots = len(df)
            valid_coords = len(df.dropna(subset=['x', 'y']))
            coverage = valid_coords / total_spots if total_spots > 0 else 0
            
            if coverage < 0.8:
                return False, f"坐标覆盖率过低: {coverage:.2%} (< 80%)"
            
            # 检查坐标分布
            if df['x'].nunique() < 2 or df['y'].nunique() < 2:
                return False, "坐标分布异常：所有点位于同一直线上"
            
            return True, "坐标验证通过"
            
        except Exception as e:
            return False, f"坐标验证失败: {e}" 