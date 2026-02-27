import os
import pandas as pd
import json
import numpy as np
from typing import List, Tuple, Dict, Callable
from tqdm import tqdm

def validate_data_quality(assignment_df: pd.DataFrame, deg_df: pd.DataFrame, sample_id: str, log_func: Callable = print) -> List[str]:
    """
    增强的数据质量检查
    :param assignment_df: 空间分配数据框
    :param deg_df: DEG数据框
    :param sample_id: 样本ID
    :param log_func: 日志函数
    :return: 问题列表
    """
    issues = []
    
    # === 1. 基本数据完整性检查 ===
    if assignment_df.empty:
        issues.append("空间分配数据为空")
        return issues
    
    if deg_df.empty:
        issues.append("DEG数据为空")
        return issues
    
    # === 2. 坐标数据检查 ===
    coord_cols = assignment_df.columns.tolist()
    has_coords = False
    
    # 检查各种坐标命名约定
    coord_pairs = [
        ('x', 'y'),
        ('coord_x', 'coord_y'),
        ('X', 'Y'),
        ('pos_x', 'pos_y'),
        ('spatial_x', 'spatial_y')
    ]
    
    for x_col, y_col in coord_pairs:
        if x_col in coord_cols and y_col in coord_cols:
            has_coords = True
            # 检查坐标缺失值
            if assignment_df[x_col].isna().any() or assignment_df[y_col].isna().any():
                issues.append(f"坐标数据存在缺失值 ({x_col}, {y_col})")
            
            # 检查坐标范围合理性
            x_range = assignment_df[x_col].max() - assignment_df[x_col].min()
            y_range = assignment_df[y_col].max() - assignment_df[y_col].min()
            if x_range == 0 or y_range == 0:
                issues.append(f"坐标范围异常: x_range={x_range}, y_range={y_range}")
            
            # 检查坐标是否为数值类型
            if not pd.api.types.is_numeric_dtype(assignment_df[x_col]):
                issues.append(f"坐标列 {x_col} 不是数值类型")
            if not pd.api.types.is_numeric_dtype(assignment_df[y_col]):
                issues.append(f"坐标列 {y_col} 不是数值类型")
            
            # 检查坐标重复度（过多重复可能有问题）
            coord_pairs_count = len(assignment_df[[x_col, y_col]].drop_duplicates())
            total_spots = len(assignment_df)
            duplicate_ratio = 1 - (coord_pairs_count / total_spots)
            if duplicate_ratio > 0.5:
                issues.append(f"坐标重复率过高: {duplicate_ratio:.2%}")
            
            break
    
    if not has_coords:
        issues.append("未找到有效的坐标列")
    
    # === 3. 域标签检查 ===
    if 'domain_index' in assignment_df.columns:
        domains = sorted(assignment_df['domain_index'].unique())
        
        # 检查域标签是否为数值
        if not pd.api.types.is_numeric_dtype(assignment_df['domain_index']):
            issues.append("域标签不是数值类型")
        
        # 检查域标签缺失值
        if assignment_df['domain_index'].isna().any():
            issues.append("域标签存在缺失值")
        
        # 检查域标签连续性
        if domains:
            expected_domains = list(range(min(domains), max(domains) + 1))
            if domains != expected_domains:
                issues.append(f"域标签不连续: 实际={domains}, 期望={expected_domains}")
            
            # 检查域标签分布合理性
            domain_counts = assignment_df['domain_index'].value_counts()
            min_spots = domain_counts.min()
            max_spots = domain_counts.max()
            if min_spots < 5:
                issues.append(f"存在过小的域 (最小域仅{min_spots}个spot)")
            if max_spots / min_spots > 50:
                issues.append(f"域大小差异过大: 最大{max_spots}个spot, 最小{min_spots}个spot")
    
    # === 4. DEG数据质量检查 ===
    required_deg_cols = ['domain', 'names']
    optional_deg_cols = ['logfoldchanges', 'pvals_adj', 'scores', 'pvals']
    
    # 检查必需列
    missing_cols = [col for col in required_deg_cols if col not in deg_df.columns]
    if missing_cols:
        issues.append(f"DEG数据缺少必需列: {missing_cols}")
    
    # 检查DEG数据完整性
    if 'names' in deg_df.columns:
        if deg_df['names'].isna().any():
            issues.append("DEG数据存在缺失的基因名")
        
        # 检查基因名格式
        invalid_genes = deg_df[deg_df['names'].str.len() < 2]['names'].tolist()
        if invalid_genes:
            issues.append(f"存在过短的基因名: {invalid_genes[:5]}...")
    
    # 检查统计值
    if 'pvals_adj' in deg_df.columns:
        if deg_df['pvals_adj'].isna().any():
            issues.append("DEG数据存在缺失的调整p值")
        
        # 检查p值范围
        invalid_pvals = deg_df[(deg_df['pvals_adj'] < 0) | (deg_df['pvals_adj'] > 1)]['pvals_adj']
        if not invalid_pvals.empty:
            issues.append(f"存在无效的p值范围: {len(invalid_pvals)}个值不在[0,1]范围内")
    
    if 'logfoldchanges' in deg_df.columns:
        if deg_df['logfoldchanges'].isna().any():
            issues.append("DEG数据存在缺失的logFC值")
        
        # 检查logFC合理性
        extreme_logfc = deg_df[abs(deg_df['logfoldchanges']) > 10]['logfoldchanges']
        if not extreme_logfc.empty:
            issues.append(f"存在极端的logFC值: {len(extreme_logfc)}个值的绝对值>10")
    
    # === 5. 数据一致性检查 ===
    if 'domain_index' in assignment_df.columns and 'domain' in deg_df.columns:
        assignment_domains = set(assignment_df['domain_index'].unique())
        deg_domains = set(deg_df['domain'].unique())
        
        # 检查域标签一致性
        missing_in_deg = assignment_domains - deg_domains
        missing_in_assignment = deg_domains - assignment_domains
        
        if missing_in_deg:
            issues.append(f"以下域在DEG数据中缺失: {sorted(missing_in_deg)}")
        if missing_in_assignment:
            issues.append(f"以下域在空间数据中缺失: {sorted(missing_in_assignment)}")
    
    # === 6. 数据规模检查 ===
    if len(assignment_df) < 100:
        issues.append(f"样本规模过小: 仅{len(assignment_df)}个spot")
    
    if len(deg_df) < 50:
        issues.append(f"DEG数量过少: 仅{len(deg_df)}个基因")
    
    # === 7. 其他列检查 ===
    # 移除layer_guess字段检查，该字段不再使用
    
    if 'array_row' in assignment_df.columns and 'array_col' in assignment_df.columns:
        if assignment_df['array_row'].isna().any() or assignment_df['array_col'].isna().any():
            issues.append("数组行列坐标存在缺失值")
    
    # 记录检查结果
    if issues:
        log_func(f"[Warning] 样本 {sample_id} 数据质量问题: {len(issues)}个")
        for issue in issues:
            log_func(f"  - {issue}")
    else:
        log_func(f"[Info] 样本 {sample_id} 数据质量检查通过")
    
    return issues

def validate_coordinate_consistency(assignment_df: pd.DataFrame, log_func: Callable = print) -> bool:
    """
    验证坐标一致性和空间合理性
    :param assignment_df: 空间分配数据框
    :param log_func: 日志函数
    :return: 是否通过验证
    """
    coord_cols = assignment_df.columns.tolist()
    
    # 查找坐标列
    coord_pairs = [
        ('x', 'y'),
        ('coord_x', 'coord_y'),
        ('X', 'Y'),
        ('pos_x', 'pos_y'),
        ('spatial_x', 'spatial_y')
    ]
    
    for x_col, y_col in coord_pairs:
        if x_col in coord_cols and y_col in coord_cols:
            # 计算基本距离统计（不使用scipy）
            coords = assignment_df[[x_col, y_col]].values
            if len(coords) > 1:
                # 计算所有点对的距离（采样以避免内存问题）
                n_coords = len(coords)
                if n_coords > 1000:
                    # 对于大数据集，随机采样计算距离
                    sample_indices = np.random.choice(n_coords, 1000, replace=False)
                    sample_coords = coords[sample_indices]
                else:
                    sample_coords = coords
                
                # 计算距离矩阵的上三角部分
                distances = []
                for i in range(len(sample_coords)):
                    for j in range(i + 1, len(sample_coords)):
                        dist = np.sqrt((sample_coords[i][0] - sample_coords[j][0])**2 + 
                                     (sample_coords[i][1] - sample_coords[j][1])**2)
                        distances.append(dist)
                
                if distances:
                    min_dist = np.min(distances)
                    median_dist = np.median(distances)
                    
                    # 检查是否有重叠点
                    if min_dist == 0:
                        log_func("[Warning] 存在重叠的坐标点")
                        return False
                    
                    # 检查距离分布是否合理
                    if median_dist / min_dist > 100:
                        log_func(f"[Warning] 坐标距离分布异常: 中位数距离/最小距离 = {median_dist/min_dist:.2f}")
                        return False
            
            return True
    
    log_func("[Warning] 未找到有效的坐标列进行一致性检查")
    return False

def load_and_validate_inputs(
    input_dir: str,
    log_func: Callable = print,
    strict_validation: bool = False,
) -> Tuple[list, list, list]:
    """
    加载并校验输入目录下所有样本，支持csv+degs.json(+composition.json)三件套。
    :param input_dir: 输入文件夹路径
    :param log_func: 日志函数
    :param strict_validation: 是否启用严格验证（有质量问题时拒绝样本）
    :return: (样本列表, 样本名列表, 有效样本名列表)
    """
    samples_list = []
    sample_names = []
    valid_samples = []
    files = os.listdir(input_dir)
    # 只保留spot.csv文件（新的命名格式）
    csv_files = [f for f in files if f.endswith('_spot.csv')]
    
    # 图片处理功能已移除

    for csv_file in tqdm(csv_files, desc="Input Check"):
        # 从 *_spot.csv 提取前缀
        prefix = csv_file[:-9]  # 移除 '_spot.csv' (9个字符)
        csv_deg = os.path.join(input_dir, f'{prefix}_DEGs.csv')
        comp_file = os.path.join(input_dir, f'{prefix}_composition.json')
        
        try:
            # 加载csv
            assignment_df = pd.read_csv(os.path.join(input_dir, csv_file))
            # 若列名是 spatial_domain -> 重命名为 domain_index
            if 'domain_index' not in assignment_df.columns and 'spatial_domain' in assignment_df.columns:
                assignment_df.rename(columns={'spatial_domain': 'domain_index'}, inplace=True)

            # 若缺少 spot_id, 则使用行索引作为 spot_id
            if 'spot_id' not in assignment_df.columns:
                assignment_df.insert(0, 'spot_id', assignment_df.index.astype(str))

            # 再次检查必要列
            for col in ['spot_id', 'domain_index']:
                if col not in assignment_df.columns:
                    log_func(f"[Error] {csv_file} 缺少必要列: {col}")
                    raise ValueError(f"缺少必要列: {col}")
                    
            # 加载 DEGs (CSV)
            if not os.path.exists(csv_deg):
                log_func(f"[Error] 缺少DEGs文件: {csv_deg}")
                raise FileNotFoundError(f"缺少DEGs文件: {csv_deg}")
            df_deg = pd.read_csv(csv_deg)
            if 'domain' not in df_deg.columns or 'names' not in df_deg.columns:
                log_func(f"[Error] {csv_deg} 缺少 domain 或 names 列")
                raise ValueError('DEGs csv format error')
            
            # 确保domain列为整数类型
            df_deg['domain'] = df_deg['domain'].astype(int)
            deg_df = df_deg.copy()

            # 先按实际存在的域构建deg_dict
            deg_dict = df_deg.groupby('domain')['names'].apply(list).to_dict()

            # 处理缺失DEG的域：按“该域无DEG”处理，避免后续报错
            if 'domain_index' in assignment_df.columns:
                assignment_domains = set(assignment_df['domain_index'].astype(int).unique())
                deg_domains = set(df_deg['domain'].unique())
                missing_in_deg = assignment_domains - deg_domains
                if missing_in_deg:
                    log_func(f"[Info] 以下域在DEG数据中缺失，将按无DEG域处理，不影响后续打分: {sorted(missing_in_deg)}")
                    for d in missing_in_deg:
                        deg_dict.setdefault(int(d), [])

            # 将键转换为字符串以确保数据类型一致性
            deg_dict = {str(k): v for k, v in deg_dict.items()}
            
            # === 新增：数据质量检查 ===
            quality_issues = validate_data_quality(assignment_df, deg_df, prefix, log_func)
            
            # 如果启用严格验证且存在质量问题，跳过该样本
            if strict_validation and quality_issues:
                log_func(f"[Error] 样本 {prefix} 因数据质量问题被跳过")
                continue
            
            # 坐标一致性检查
            coord_valid = validate_coordinate_consistency(assignment_df, log_func)
            if strict_validation and not coord_valid:
                log_func(f"[Error] 样本 {prefix} 因坐标一致性问题被跳过")
                continue
            
            # 不再加载 composition_dict
            composition_dict = None

            # 图片处理功能已移除
            img_path = None
            img_metrics = {}
            
            # 记录样本
            sample = {
                'sample_id': prefix,
                'assignment_df': assignment_df,
                'deg_dict': deg_dict,
                'deg_df': deg_df,
                'quality_issues': quality_issues,  # 记录质量问题供后续参考
                'image_path': img_path,
                'img_metrics': img_metrics,
            }
            samples_list.append(sample)
            sample_names.append(prefix)
            valid_samples.append(prefix)
            log_func(f"[Info] 样本 {prefix} 加载成功")
            
        except Exception as e:
            log_func(f"[Error] 样本 {prefix} 加载失败: {e}")
            
    return samples_list, sample_names, valid_samples 