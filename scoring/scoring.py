import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import logging
import re
import json
import subprocess
import sys
import traceback
logging.getLogger("langchain").setLevel(logging.ERROR)

import os
import pandas as pd
import argparse
from tqdm import tqdm
from domain_evaluator import DomainEvaluator
from input_handler import load_and_validate_inputs
from output_handler import write_output
from logger import Logger
from cli_selectors import parse_selector_overrides
# 导入构建矩阵的功能
from typing import List, Dict
from sklearn.preprocessing import MinMaxScaler
from decimal import Decimal, getcontext
from pathlib import Path
try:
    from visual_score_paths import visual_scores_path
except Exception:
    from scoring.visual_score_paths import visual_scores_path  # type: ignore

# 安全地导入配置，兼容有无config.py文件的情况
from config_loader import get_legacy_config
try:
    from config import load_config as load_scoring_config
except Exception:
    load_scoring_config = None
try:
    from provider_runtime import resolve_provider_config
except Exception:
    from scoring.provider_runtime import resolve_provider_config  # type: ignore

AZURE_OPENAI_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT, AZURE_API_VERSION = get_legacy_config()
DEFAULT_API_PROVIDER = ""
DEFAULT_API_KEY = ""
DEFAULT_API_ENDPOINT = ""
DEFAULT_API_VERSION = ""
DEFAULT_API_MODEL = ""
if load_scoring_config is not None:
    try:
        _cfg_defaults = load_scoring_config()
        DEFAULT_API_PROVIDER = str(getattr(_cfg_defaults, "api_provider", "") or "")
        DEFAULT_API_KEY = str(getattr(_cfg_defaults, "api_key", "") or "")
        DEFAULT_API_ENDPOINT = str(getattr(_cfg_defaults, "api_endpoint", "") or "")
        DEFAULT_API_VERSION = str(getattr(_cfg_defaults, "api_version", "") or "")
        DEFAULT_API_MODEL = str(getattr(_cfg_defaults, "api_model", "") or "")
        AZURE_OPENAI_KEY = str(getattr(_cfg_defaults, "azure_openai_key", "") or AZURE_OPENAI_KEY)
        AZURE_ENDPOINT = str(getattr(_cfg_defaults, "azure_endpoint", "") or AZURE_ENDPOINT)
        AZURE_DEPLOYMENT = str(getattr(_cfg_defaults, "azure_deployment", "") or AZURE_DEPLOYMENT)
        AZURE_API_VERSION = str(getattr(_cfg_defaults, "azure_api_version", "") or AZURE_API_VERSION)
    except Exception:
        pass
DEFAULT_OPENAI_KEY = DEFAULT_API_KEY or AZURE_OPENAI_KEY
DEFAULT_AZURE_ENDPOINT = DEFAULT_API_ENDPOINT or AZURE_ENDPOINT
DEFAULT_AZURE_DEPLOYMENT = DEFAULT_API_MODEL or AZURE_DEPLOYMENT
DEFAULT_AZURE_API_VERSION = DEFAULT_API_VERSION or AZURE_API_VERSION

# 解析命令行参数
parser = argparse.ArgumentParser(description='DomainEvaluator Batch Runner')
parser.add_argument('--output_format', type=str, default='json', choices=['json', 'dataframe', 'excel', 'sql', 'markdown'], help='输出格式，可选：json, dataframe(csv), excel(xlsx), sql, markdown(md)')
parser.add_argument('--api_provider', type=str, default=DEFAULT_API_PROVIDER, help='通用provider（如 azure/openai/anthropic/openrouter）')
parser.add_argument('--api_key', type=str, default=DEFAULT_API_KEY, help='通用provider API密钥')
parser.add_argument('--api_endpoint', type=str, default=DEFAULT_API_ENDPOINT, help='通用provider endpoint/base URL')
parser.add_argument('--api_version', type=str, default=DEFAULT_API_VERSION, help='通用provider API版本（Azure常用）')
parser.add_argument('--api_model', type=str, default=DEFAULT_API_MODEL, help='通用provider 模型或部署名')
parser.add_argument('--openai_key', type=str, default=DEFAULT_OPENAI_KEY, help='OpenAI或Azure OpenAI的API密钥')
parser.add_argument('--azure_endpoint', type=str, default=DEFAULT_AZURE_ENDPOINT, help='Azure OpenAI endpoint')
parser.add_argument('--azure_deployment', type=str, default=DEFAULT_AZURE_DEPLOYMENT, help='Azure OpenAI deployment名称')
parser.add_argument('--azure_api_version', type=str, default=DEFAULT_AZURE_API_VERSION, help='Azure OpenAI API版本')
parser.add_argument('--sql_db', type=str, default='output/result.db', help='SQL输出时的sqlite数据库文件路径')
parser.add_argument('--top_n_deg', type=int, default=5, help='每个domain取前N个DEG进行摘要')

# 新增的LLM参数
parser.add_argument('--temperature', type=float, default=0.0, help='大模型采样温度 (0.0-2.0)')
parser.add_argument('--max_completion_tokens', type=int, default=30000, help='最大生成token数（None表示不限制）')
parser.add_argument('--top_p', type=float, default=1.0, help='核采样概率 (0.0-1.0)')
parser.add_argument('--frequency_penalty', type=float, default=0.0, help='频率惩罚 (-2.0-2.0)')
parser.add_argument('--presence_penalty', type=float, default=0.0, help='存在惩罚 (-2.0-2.0)')

# 图片分析参数
# 图片处理功能已移除

parser.add_argument('--use_example_db', action='store_true', help='是否启用向量数据库Few-shot检索')
parser.add_argument('--knn_k', type=int, default=4, help='Few-shot检索返回K个示例')
parser.add_argument('--build_matrices', action='store_true', default=True, help='是否自动构建分数和标签矩阵')
parser.add_argument('--no_build_matrices', action='store_true', help='禁用自动矩阵构建')
parser.add_argument('--enforce_discrimination', action='store_true', help='让GPT打分强制保持较大的区分度(默认关闭)')
parser.add_argument('--vlm_off', action='store_true', help='关闭视觉评分整合（VLM），仅使用文本与空间/DEG指标')
# ARI计算功能已移除
# parser.add_argument('--calculate_ari', action='store_true', default=True, help='是否计算ARI并生成可视化')
# parser.add_argument('--no_calculate_ari', action='store_true', help='禁用ARI计算和可视化')

parser.add_argument('--skip_existing', action='store_true', help='跳过已存在的所有文件（包括主程序结果）')
parser.add_argument('--clean_output', action='store_true', help='运行前清理所有输出文件')

# I/O overrides & Tool-runner integration (Phase B)
parser.add_argument('--input_dir', type=str, default=None, help='输入目录（默认使用脚本目录下的 input/）')
parser.add_argument('--output_dir', type=str, default=None, help='输出目录（默认使用脚本目录下的 output/）')
parser.add_argument('--toolrunner_output_dir', type=str, default=None, help='Tool-runner 输出目录（包含 spot/ DEGs/ PATHWAY/ 子目录）。提供后将自动 stage 到 scoring input/')
parser.add_argument('--toolrunner_sample_id', type=str, default=None, help='用于从 Tool-runner 输出中过滤文件的 sample_id（例如 "DLPFC_151507"）。与 --toolrunner_output_dir 一起使用')
parser.add_argument('--toolrunner_overwrite', action='store_true', help='stage Tool-runner 输出时覆盖 scoring input/ 中已存在的文件')
parser.add_argument('--sample_id', type=str, default=None, help='当前组织样本ID，用于选择 sample-aware 的视觉评分缓存')

# Annotation 功能参数
parser.add_argument('--annotation_multiagent', action='store_true', help='是否运行 Multi-Agent Domain Annotation 模式（含VLM/Peer/Critic/Loop/日志）')
parser.add_argument('--domain', type=str, help='指定要 Annotate 的 domain ID (逗号分隔，例如 "1,3")，仅在 --annotation_multiagent 模式下有效')
parser.add_argument('--annotation_data_dir', type=str, default=None, help='Multi-Agent Annotation 输入目录（包含 BEST_<sample_id>_{spot,DEGs,PATHWAY}.csv）')
parser.add_argument('--annotation_sample_id', type=str, default=None, help='Multi-Agent Annotation 的 sample_id（用于定位 BEST_* 文件与 result.png）')

args, unknown_args = parser.parse_known_args()

effective_api = resolve_provider_config(
    api_provider=args.api_provider,
    api_key=args.api_key or args.openai_key,
    api_endpoint=args.api_endpoint or args.azure_endpoint,
    api_version=args.api_version or args.azure_api_version,
    api_model=args.api_model or args.azure_deployment,
    api_deployment=args.azure_deployment,
    azure_openai_key=args.openai_key,
    azure_endpoint=args.azure_endpoint,
    azure_api_version=args.azure_api_version,
    azure_deployment=args.azure_deployment,
)
args.api_provider = effective_api.provider
args.api_key = effective_api.api_key
args.api_endpoint = effective_api.endpoint
args.api_version = effective_api.api_version
args.api_model = effective_api.model

# Keep legacy aliases for backward compatibility in downstream code.
if args.api_provider == "azure":
    args.openai_key = args.api_key
    args.azure_endpoint = args.api_endpoint
    args.azure_api_version = args.api_version
    args.azure_deployment = args.api_model

# 获取脚本所在目录，确保路径相对于脚本文件而不是工作目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# Default I/O dirs (can be overridden by CLI)
input_dir = args.input_dir or os.path.join(script_dir, 'input')
output_dir = args.output_dir or os.path.join(script_dir, 'output')
# 图片处理功能已移除


def stage_toolrunner_outputs_to_scoring_input(
    *,
    tool_output_dir: str,
    scoring_input_dir: str,
    sample_id: str,
    overwrite: bool,
):
    """Stage Tool-runner outputs into scoring input/ directory using expected filenames.

    Tool-runner typically outputs:
      <tool_output_dir>/spot/*_spot.csv
      <tool_output_dir>/DEGs/*_DEGs.csv
      <tool_output_dir>/PATHWAY/*_PATHWAY.csv

    Tool-runner file prefixes are usually like:
      IRIS_domain_DLPFC_151507_spot.csv
    while scoring expects:
      IRIS_DLPFC_151507_spot.csv
    """
    import shutil
    import re

    spot_dir = os.path.join(tool_output_dir, "spot")
    deg_dir = os.path.join(tool_output_dir, "DEGs")
    pathway_dir = os.path.join(tool_output_dir, "PATHWAY")

    if not os.path.exists(spot_dir):
        raise FileNotFoundError(f"Not found: {spot_dir}")
    if not os.path.exists(deg_dir):
        raise FileNotFoundError(f"Not found: {deg_dir}")
    if not os.path.exists(pathway_dir):
        raise FileNotFoundError(f"Not found: {pathway_dir}")

    os.makedirs(scoring_input_dir, exist_ok=True)

    domain_prefix_re = re.compile(r"^(?P<method>[^_]+)_domain_(?P<rest>.+)$")

    def rewrite_prefix(prefix: str) -> str:
        m = domain_prefix_re.match(prefix)
        if not m:
            return prefix
        return f"{m.group('method')}_{m.group('rest')}"

    def copy_group(src_dir: str, suffix: str) -> int:
        n = 0
        for name in os.listdir(src_dir):
            if not name.endswith(f"_{suffix}.csv"):
                continue
            prefix = name[: -(len(f"_{suffix}.csv"))]
            if sample_id and (sample_id not in prefix):
                continue
            new_prefix = rewrite_prefix(prefix)
            src = os.path.join(src_dir, name)
            dst = os.path.join(scoring_input_dir, f"{new_prefix}_{suffix}.csv")
            if os.path.exists(dst) and not overwrite:
                continue
            shutil.copy2(src, dst)
            n += 1
        return n

    n_spot = copy_group(spot_dir, "spot")
    n_deg = copy_group(deg_dir, "DEGs")
    n_pw = copy_group(pathway_dir, "PATHWAY")

    if n_spot == 0:
        raise RuntimeError(f"No spot files staged from {spot_dir} (sample_id filter: {sample_id})")
    if n_deg == 0:
        raise RuntimeError(f"No DEG files staged from {deg_dir} (sample_id filter: {sample_id})")
    if n_pw == 0:
        print(f"[Warning] No PATHWAY files staged from {pathway_dir} (sample_id filter: {sample_id})")

    print(f"[Info] Tool-runner outputs staged into: {scoring_input_dir}")
    print(f"  - spot: {n_spot}")
    print(f"  - DEGs: {n_deg}")
    print(f"  - PATHWAY: {n_pw}")


# Phase B: optionally stage tool-runner outputs into scoring input/ before scoring
if args.toolrunner_output_dir:
    if not args.toolrunner_sample_id:
        print("[Error] --toolrunner_sample_id is required when --toolrunner_output_dir is provided.")
        exit(1)
    try:
        stage_toolrunner_outputs_to_scoring_input(
            tool_output_dir=args.toolrunner_output_dir,
            scoring_input_dir=input_dir,
            sample_id=args.toolrunner_sample_id,
            overwrite=bool(args.toolrunner_overwrite),
        )
    except Exception as e:
        print(f"[Error] Failed to stage Tool-runner outputs: {e}")
        traceback.print_exc()
        exit(1)

# 清理输出文件功能
def clean_output_files(output_dir: str, log_func):
    """清理输出文件"""
    files_to_clean = []
    
    # 主程序结果文件
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.endswith('_result.json') or file.endswith('_result.csv') or file.endswith('_result.xlsx') or file.endswith('_result.md'):
                file_path = os.path.join(output_dir, file)
                files_to_clean.append(file_path)
    
    # 矩阵文件
    consensus_dir = os.path.join(output_dir, 'consensus')
    if os.path.exists(consensus_dir):
        consensus_files = [
            'scores_matrix.csv',
            'labels_matrix.csv'
        ]
        for file in consensus_files:
            file_path = os.path.join(consensus_dir, file)
            if os.path.exists(file_path):
                files_to_clean.append(file_path)
    
    if files_to_clean:
        print(f"[Info] 正在清理 {len(files_to_clean)} 个输出文件...")
        log_func(f"正在清理 {len(files_to_clean)} 个输出文件...")
        
        for file_path in files_to_clean:
            try:
                os.remove(file_path)
                log_func(f"已删除: {file_path}")
            except Exception as e:
                log_func(f"删除失败: {file_path} - {e}")
        
        print(f"[Info] 清理完成")
        log_func("输出文件清理完成")
    else:
        print("[Info] 没有找到需要清理的输出文件")
        log_func("没有找到需要清理的输出文件")

# 解析形如 --domain2 / --method=IRIS 的参数，只对指定 domain / 方法 进行GPT打分
selected_domains, selected_methods = parse_selector_overrides(unknown_args)

if selected_domains:
    TARGET_DOMAINS = sorted(set(selected_domains))
    print(f"[Info] 仅对以下 domain 进行 GPT 打分: {TARGET_DOMAINS}")
else:
    TARGET_DOMAINS = None

if selected_methods:
    TARGET_METHODS = sorted(set(selected_methods))
    print(f"[Info] 仅对以下方法进行 GPT 打分: {TARGET_METHODS}")
else:
    TARGET_METHODS = None


def merge_domain_results_incremental(prefix: str,
                                     new_result: Dict,
                                     output_dir: str,
                                     target_domains: List[int] | None):
    """
    增量合并结果：
    - 如果指定了 TARGET_DOMAINS，则从已有的 {prefix}_result.json 中读取旧结果，
      只用 new_result 中对应 domain_index 的条目覆盖旧结果，其余 domain 保持不变。
    - 如果没有旧文件或未指定 target_domains，则直接返回 new_result。
    """
    if not target_domains:
        return new_result

    result_path = os.path.join(output_dir, f"{prefix}_result.json")
    if not os.path.exists(result_path):
        # 没有旧文件，直接使用新结果
        return new_result

    try:
        with open(result_path, "r", encoding="utf-8") as f:
            old_result = json.load(f)
    except Exception:
        # 旧文件损坏或无法解析，避免阻塞流程，直接使用新结果
        return new_result

    # 统一获取旧/新 domain 列表（兼容旧版 "domains" 字段）
    old_list = old_result.get("best_domains", old_result.get("domains", [])) or []
    new_list = new_result.get("best_domains", new_result.get("domains", [])) or []

    # 以 domain_index 为键构建映射
    domain_map: Dict[int, Dict] = {}
    for d in old_list:
        idx = d.get("domain_index")
        if idx is None:
            continue
        try:
            domain_map[int(idx)] = d
        except Exception:
            continue

    # 用新结果覆盖对应 domain
    for d in new_list:
        idx = d.get("domain_index")
        if idx is None:
            continue
        try:
            domain_map[int(idx)] = d
        except Exception:
            continue

    # 合并后按 domain_index 排序
    merged_list = sorted(
        domain_map.values(),
        key=lambda x: int(x.get("domain_index", 0))
    )

    # 在旧结果结构上替换 domain 列表，保留 sample_id 等其它字段
    old_result["best_domains"] = merged_list
    return old_result

# 日志对象
logger = Logger()
log = logger.log

# --- Annotation 独立运行模式 (新增) ---
if args.annotation_multiagent:
    print("[Info] Enter Multi-Agent Domain Annotation mode ...")
    log("[Info] Enter Multi-Agent Domain Annotation mode ...")

    # Domain filter (same semantics as --annotation)
    target_domains = None
    if args.domain:
        try:
            target_domains = [int(d.strip()) for d in args.domain.split(',')]
            print(f"[Info] 指定运行 Domains(来自 --domain): {target_domains}")
            log(f"[Info] 指定运行 Domains(来自 --domain): {target_domains}")
        except ValueError:
            print("[Error] --domain 参数格式错误，应为逗号分隔的整数 (例如 '1,3')")
            log("[Error] --domain 参数格式错误，应为逗号分隔的整数 (例如 '1,3')")
            exit(1)
    elif TARGET_DOMAINS:
        target_domains = TARGET_DOMAINS
        print(f"[Info] 指定运行 Domains(来自 --domainX): {target_domains}")
        log(f"[Info] 指定运行 Domains(来自 --domainX): {target_domains}")

    try:
        from config import load_config, ScoreRagConfig
        from annotation.annotation_multiagent.orchestrator import run_annotation_multiagent

        try:
            cfg = load_config()
        except Exception:
            cfg = ScoreRagConfig.from_env()
        # Override credentials with resolved provider args (legacy azure fields kept for compatibility).
        cfg = cfg.update(
            api_provider=args.api_provider,
            api_key=args.api_key,
            api_endpoint=args.api_endpoint,
            api_version=args.api_version,
            api_model=args.api_model,
            azure_openai_key=args.openai_key if args.api_provider == "azure" else cfg.azure_openai_key,
            azure_endpoint=args.azure_endpoint if args.api_provider == "azure" else cfg.azure_endpoint,
            azure_deployment=args.azure_deployment if args.api_provider == "azure" else cfg.azure_deployment,
            azure_api_version=args.azure_api_version if args.api_provider == "azure" else cfg.azure_api_version,
        )

        # Determine sample_id + data_dir (generalized; keeps legacy fallbacks).
        annotation_sample_id = args.sample_id or args.annotation_sample_id or args.toolrunner_sample_id
        data_dir = args.annotation_data_dir
        potential_dirs = []
        if data_dir:
            potential_dirs.append(os.path.abspath(data_dir))
        potential_dirs.extend(
            [
                os.path.abspath(os.path.join(os.getcwd(), "../ARI&Picture_DLPFC_151507/output")),  # legacy
                os.path.abspath("ARI&Picture_DLPFC_151507/output"),  # legacy
                output_dir,  # scoring output dir
                os.path.abspath(os.path.join(os.path.dirname(output_dir), "ensemble")),  # optional
                os.path.abspath(os.path.join(os.path.dirname(output_dir), "ensemble_output")),  # optional
            ]
        )

        def _exists_best_bundle(d: str, sid: str) -> bool:
            if not sid:
                return False
            candidates = [
                os.path.join(d, f"BEST_{sid}_DEGs.csv"),
                os.path.join(d, f"BEST_DLPFC_{sid}_DEGs.csv"),  # legacy
            ]
            return any(os.path.exists(p) for p in candidates)

        if not annotation_sample_id:
            # try infer from files in potential dirs (BEST_*_spot.csv)
            for d in potential_dirs:
                if not os.path.exists(d):
                    continue
                try:
                    for fn in os.listdir(d):
                        if fn.startswith("BEST_") and fn.endswith("_spot.csv"):
                            # BEST_<sample_id>_spot.csv
                            annotation_sample_id = fn[len("BEST_") : -len("_spot.csv")]
                            break
                except Exception:
                    continue
                if annotation_sample_id:
                    break

        picked = None
        for d in potential_dirs:
            if os.path.exists(d) and annotation_sample_id and _exists_best_bundle(d, annotation_sample_id):
                picked = d
                break
        if not picked:
            print(f"[Error] 未找到 Multi-Agent Annotation 所需的数据文件 (BEST spot/DEGs/PATHWAY). 搜索路径: {potential_dirs}")
            print(f"[Error] 请提供 --annotation_data_dir 与 --annotation_sample_id (或先运行 Phase C 生成 BEST_* 文件).")
            log("[Error] 未找到 Multi-Agent Annotation 所需的数据文件 (BEST spot/DEGs/PATHWAY).")
            exit(1)
        data_dir = picked

        print(f"[Info] 加载数据源: {data_dir}")
        log(f"[Info] 加载数据源: {data_dir}")

        output_base = os.path.dirname(data_dir)
        annot_out_dir = os.path.join(output_base, "annotation_output")
        run_annotation_multiagent(
            data_dir=data_dir,
            sample_id=str(annotation_sample_id),
            target_domains=target_domains,
            output_dir=annot_out_dir,
            config=cfg,
        )

        print(f"[Success] Multi-Agent Annotation finished. Results saved to: {annot_out_dir}")
        log("[Success] Multi-Agent Annotation finished.")
        exit(0)
    except Exception as e:
        print(f"[Error] Multi-Agent Annotation 运行失败: {e}")
        log(f"[Error] Multi-Agent Annotation 运行失败: {e}")
        traceback.print_exc()
        exit(1)

# 如果需要清理输出文件
if args.clean_output:
    clean_output_files(output_dir, log)

print("[Info] 正在校验输入数据...")
log("开始校验输入数据...")
# 加载并校验所有输入样本，返回样本列表、样本名、有效样本名
samples_list, sample_names, valid_samples = load_and_validate_inputs(
    input_dir,
    log_func=log,
    # 图片处理功能已移除
)

if not samples_list:
    print("[Error] 没有可用的有效样本，程序终止。请查看output_log获取详细信息。"); log("没有可用的有效样本，程序终止。")
    exit(1)

print(f"[Info] 共通过校验的样本数: {len(samples_list)}\n")
log(f"共通过校验的样本数: {len(samples_list)}")

# 初始化DomainEvaluator，负责大模型打分

evaluator = DomainEvaluator(
    openai_api_key=args.api_key or args.openai_key,
    output_format=args.output_format,
    api_provider=args.api_provider,
    api_key=args.api_key,
    api_endpoint=args.api_endpoint,
    api_model=args.api_model,
    api_version=args.api_version,
    azure_endpoint=args.azure_endpoint,
    azure_deployment=args.azure_deployment,
    azure_api_version=args.azure_api_version,
    temperature=args.temperature,
    max_completion_tokens=args.max_completion_tokens,
    top_p=args.top_p,
    frequency_penalty=args.frequency_penalty,
    presence_penalty=args.presence_penalty,
    top_n_deg=args.top_n_deg,
    use_example_db=args.use_example_db,
    knn_k=args.knn_k,
    enforce_discrimination=args.enforce_discrimination
)

# 批量处理，带进度条和异常处理
max_retries = 2
all_results = []
all_gpt_raw_responses = []
all_gpt_logs = []
output_files = []
# 仅记录成功处理的样本与结果配对，避免索引问题
successful_results: List[tuple[str, Dict]] = []
print("[Info] 正在批量处理样本...")
log("开始批量处理样本...")

# 为每个样本加载pathway数据
print("[Info] 正在加载Pathway数据...")
all_methods = set()
for idx, sample in enumerate(samples_list):
    sample_id = valid_samples[idx]
    method_name = sample_id.split('_')[0]  # 从样本ID提取方法名
    sample['method_name'] = method_name
    all_methods.add(method_name)

# 一次性加载所有方法的pathway数据
evaluator.pathway_data = evaluator.pathway_analyzer.load_pathway_data(input_dir)
evaluator.pathway_scores = evaluator.pathway_analyzer.analyze_pathway_enrichment(evaluator.pathway_data)
print(f"[Info] 成功加载 {len(evaluator.pathway_data)} 个方法的pathway数据: {list(evaluator.pathway_data.keys())}")

# 加载视觉评分数据（可通过 --vlm_off 关闭）
if args.vlm_off:
    evaluator.use_visual_integration = False
    print(f"[Info] 已按 --vlm_off 关闭视觉评分整合，使用传统评分方式")
else:
    visual_sample_id = str(args.sample_id or args.annotation_sample_id or args.toolrunner_sample_id or "").strip()
    script_dir = Path(__file__).resolve().parent
    pic_analyze_dir = script_dir / "pic_analyze"
    visual_scores_file = visual_scores_path(pic_analyze_dir, visual_sample_id)
    if not visual_scores_file.exists():
        pic_script = pic_analyze_dir / "run.py"
        if not pic_script.exists():
            print(f"[Error] 视觉评分文件不存在且未找到预处理脚本: {pic_script}")
            log(f"视觉评分文件不存在且未找到预处理脚本: {pic_script}")
            exit(1)

        print(f"[Info] 视觉评分文件缺失，先运行 pic_analyze: {pic_script}")
        log(f"视觉评分文件缺失，先运行 pic_analyze: {pic_script}")
        try:
            pic_cmd = [sys.executable, str(pic_script)]
            if visual_sample_id:
                pic_cmd.extend(["--sample_id", visual_sample_id])
            pic_proc = subprocess.run(
                pic_cmd,
                cwd=str(pic_analyze_dir),
                check=False,
            )
        except Exception as pic_err:
            print(f"[Error] 运行 pic_analyze 失败: {pic_err}")
            log(f"运行 pic_analyze 失败: {pic_err}")
            exit(1)

        if int(pic_proc.returncode) != 0:
            print(f"[Error] pic_analyze 运行失败，退出码: {pic_proc.returncode}")
            log(f"pic_analyze 运行失败，退出码: {pic_proc.returncode}")
            exit(1)

        if not visual_scores_file.exists():
            print(f"[Error] pic_analyze 已运行但仍未生成视觉评分文件: {visual_scores_file}")
            log(f"pic_analyze 已运行但仍未生成视觉评分文件: {visual_scores_file}")
            exit(1)

    print(f"[Info] 尝试加载视觉评分数据...")
    visual_loaded = evaluator.load_visual_scores(str(pic_analyze_dir), sample_id=visual_sample_id)
    if visual_loaded:
        print(f"[Info] 视觉评分整合已启用，将影响最终评分")
    else:
        print(f"[Info] 视觉评分整合未启用，使用传统评分方式")

try:
    for idx, sample in enumerate(tqdm(samples_list, desc="Sample Batch")):
        # 方法过滤：如果指定了 TARGET_METHODS，则只处理这些方法
        method_name = sample.get('method_name') or valid_samples[idx].split('_')[0]
        if TARGET_METHODS and method_name not in TARGET_METHODS:
            print(f"[Info] 跳过样本 {valid_samples[idx]} (方法 {method_name} 不在指定列表中)")
            log(f"跳过样本 {valid_samples[idx]} (方法 {method_name} 不在指定列表中)")
            continue

        for attempt in range(max_retries + 1):
            try:
                print(f"[Info] 正在处理样本: {valid_samples[idx]} (第{attempt+1}次尝试)")
                log(f"正在处理样本: {valid_samples[idx]} (第{attempt+1}次尝试)")
                # 调用大模型批量打分，返回结果、原始response、复查日志
                res, gpt_raw, gpt_log = evaluator.process_batch(
                    [sample],
                    return_gpt_response=True,
                    target_domains=TARGET_DOMAINS
                )
                all_results.append(res)
                all_gpt_raw_responses.append(gpt_raw[0])
                all_gpt_logs.append(gpt_log)
                # 记录成功样本与结果
                successful_results.append((sample_names[idx], res))
                break
            except RecursionError as e:
                # 专门捕获递归错误，打印完整堆栈并跳过该样本
                print(f"[Fatal] 处理样本 {valid_samples[idx]} 时出现 RecursionError: {e}")
                log(f"处理样本 {valid_samples[idx]} 时出现 RecursionError: {e}")
                traceback.print_exc()
                print("[Error] 遇到递归深度问题，已跳过该样本。若方便，请将完整 traceback 复制出来用于进一步定位。")
                log("遇到递归深度问题，已跳过该样本。")
                break
            except Exception as e:
                print(f"[Error] 处理样本 {valid_samples[idx]} 失败: {e}")
                log(f"处理样本 {valid_samples[idx]} 失败: {e}")
                if attempt == max_retries:
                    print(f"[Error] 已重试{max_retries+1}次，跳过该样本。请查看output_log获取详细信息。")
                    log(f"已重试{max_retries+1}次，跳过该样本。")
                else:
                    print("[Info] 正在重试...")
                    log("正在重试...")
except Exception as e:
    print(f"[Fatal] 批量处理过程中发生致命错误: {e}，请查看output_log获取详细信息。")
    log(f"批量处理过程中发生致命错误: {e}")
    exit(1)

# 打印大模型原始response
print("\n===== GPT Raw Response (English) =====")
for idx, resp in enumerate(all_gpt_raw_responses):
    print(f"\n[Sample {idx+1} GPT Response]:\n{resp}\n")
print("===== End of GPT Raw Response =====\n")
log("全部样本处理完成。")

# --- 输出文件名以输入csv名为前缀 ---
if len(successful_results) == 1:
    prefix, result = successful_results[0]
    # 如果是单/少数 domain 重跑，进行增量合并：只覆盖指定 domain，其余 domain 使用历史得分
    merged_result = merge_domain_results_incremental(prefix, result, output_dir, TARGET_DOMAINS)
    df_result = pd.DataFrame(merged_result['best_domains']) if 'best_domains' in merged_result else pd.DataFrame(merged_result)
    write_output(merged_result, df_result, prefix, output_dir, args.output_format, log_func=log, sql_db=args.sql_db, skip_existing=args.skip_existing)
    ext = {'json': 'json', 'dataframe': 'csv', 'excel': 'xlsx', 'sql': 'db', 'markdown': 'md'}[args.output_format]
    output_file = os.path.join(output_dir, f'{prefix}_result.{ext}')
    output_files.append(output_file)
elif len(successful_results) > 1:
    for prefix, result in successful_results:
        merged_result = merge_domain_results_incremental(prefix, result, output_dir, TARGET_DOMAINS)
        df_result = pd.DataFrame(merged_result['best_domains']) if 'best_domains' in merged_result else pd.DataFrame(merged_result)
        write_output(merged_result, df_result, prefix, output_dir, args.output_format, log_func=log, sql_db=args.sql_db, skip_existing=args.skip_existing)
        ext = {'json': 'json', 'dataframe': 'csv', 'excel': 'xlsx', 'sql': 'db', 'markdown': 'md'}[args.output_format]
        output_file = os.path.join(output_dir, f'{prefix}_result.{ext}')
        output_files.append(output_file)
else:
    print('[Warning] 所有样本均处理失败，未生成任何主程序输出文件。')
    log('所有样本均处理失败，未生成任何主程序输出文件。')

if output_files:
    for f in output_files:
        print(f'已生成 {f} 到output文件中')
else:
    print('未生成任何输出文件，请查看output_log获取详细信息。')

# 添加构建矩阵的功能函数
def load_domain_scores(result_json: str) -> Dict[int, float]:
    """return mapping domain_index -> total score (0-1)."""
    with open(result_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    doms = data.get("best_domains", data.get("domains", []))
    m = {}
    for d in doms:
        # 优先使用sub_scores的精确总和，提供更好的分数区分度（增强版）
        if "sub_scores" in d and isinstance(d["sub_scores"], list) and len(d["sub_scores"]) > 0:
            # 使用高精度计算避免浮点误差
            getcontext().prec = 12

            # 将子分数转换为高精度Decimal进行计算
            sub_scores = [Decimal(str(s)) for s in d["sub_scores"]]
            val = float(sum(sub_scores))

            # 保留更高精度减少同分：保留8位小数
            val = round(val, 8)
            
            # 添加确定性微扰避免完全同分
            domain_idx = int(d["domain_index"])
            perturbation = ((domain_idx * 2654435761) % 997) * 1e-6  # 确定性微扰，增大到1e-6
            val += perturbation
            
        else:
            # 回退到预设的score或total字段
            val = d.get("score", d.get("total", 0))
            val = float(val)
            
            # 为回退分数也添加微扰
            domain_idx = int(d["domain_index"])
            perturbation = ((domain_idx * 2654435761) % 997) * 1e-6  # 确定性微扰，增大到1e-6
            val += perturbation

        # 统一使用0-1分制，如果分数>1，则报错提示需要转换
        if val > 1:
            raise ValueError(f"Score {val} is greater than 1. Please ensure all scores are in 0-1 range.")

        m[domain_idx] = val
    return m


def build_matrices(methods: List[str], spot_csvs: List[str], result_jsons: List[str]) -> (pd.DataFrame, pd.DataFrame):
    # Collect all spot_ids
    all_spots = set()
    per_method_scores = {}
    per_method_labels = {}

    for m, csv_path, res_path in zip(methods, spot_csvs, result_jsons):
        # 检查CSV文件结构
        header_df = pd.read_csv(csv_path, nrows=0)
        
        if "spot_id" in header_df.columns:
            # 如果有明确的spot_id列
            spot_df = pd.read_csv(csv_path)
            spot_ids = spot_df["spot_id"].astype(str)
        elif "Unnamed: 0" in header_df.columns:
            # 第一列包含spot_id但名为"Unnamed: 0"
            spot_df = pd.read_csv(csv_path)
            spot_ids = spot_df["Unnamed: 0"].astype(str)
            # 重命名第一列为spot_id
            spot_df = spot_df.rename(columns={"Unnamed: 0": "spot_id"})
        else:
            # 第一列是索引，直接设为spot_id
            spot_df = pd.read_csv(csv_path, index_col=0)
            spot_ids = spot_df.index.astype(str)
            spot_df = spot_df.reset_index()
            spot_df = spot_df.rename(columns={spot_df.columns[0]: 'spot_id'})
            
        # 标准化列名
        if "domain_index" not in spot_df.columns and "spatial_domain" in spot_df.columns:
            spot_df = spot_df.rename(columns={"spatial_domain": "domain_index"})

        # ensure domain_index is integer so it matches JSON keys
        spot_df["domain_index"] = spot_df["domain_index"].astype(int)

        all_spots.update(spot_ids)

        # 创建spot_id到domain_index的映射
        spot_to_domain = dict(zip(spot_ids, spot_df["domain_index"]))
        per_method_labels[m] = pd.Series(spot_to_domain)

        # Map domain score
        score_map = load_domain_scores(res_path)
        per_method_scores[m] = pd.Series(spot_to_domain).map(score_map)

    # 正确的spot排序：对于barcode使用字典序
    def sort_spot_ids(spot_ids):
        """正确排序spot_id：barcode使用字典序，数字使用数值序"""
        spot_list = list(spot_ids)
        if not spot_list:
            return spot_list
            
        # 检查是否是barcode格式（包含字母和连字符）
        sample_spot = spot_list[0]
        if '-' in sample_spot and any(c.isalpha() for c in sample_spot):
            # barcode格式，使用字典序
            return sorted(spot_list)
        else:
            # 数字格式，尝试数值排序
            try:
                return sorted(spot_list, key=lambda x: int(x))
            except ValueError:
                # 如果转换失败，回退到字典序
                return sorted(spot_list)

    all_spots = sort_spot_ids(all_spots)

    # Build DataFrames with all spots
    scores_df = pd.DataFrame(index=all_spots)
    labels_df = pd.DataFrame(index=all_spots)
    for m in methods:
        scores_df[m] = per_method_scores[m].reindex(all_spots)
        labels_df[m] = per_method_labels[m].reindex(all_spots)

    # 填补缺失域：如果某方法没有特定domain，用 -1 占位，避免列整体缺失
    # 注意：只转换列数据类型，不影响索引
    for col in labels_df.columns:
        labels_df[col] = labels_df[col].fillna(-1).astype(int)

    # 所有分数应该已经是0-1范围，不需要额外的缩放处理
    
    # 保持原始的spot_id作为索引，确保数据一致性
    # 不再重新索引为数字，这样可以保持spot_id与spatial_domain的正确对应关系

    return scores_df, labels_df


def generate_domain_choice_record(scores_df: pd.DataFrame, labels_df: pd.DataFrame, output_dir: str, log_func):
    """
    生成domain选择记录文件，记录每个domain的最高分数和对应方法
    
    Args:
        scores_df: 分数矩阵DataFrame
        labels_df: 标签矩阵DataFrame  
        output_dir: 输出目录
        log_func: 日志函数
    """
    try:
        domain_choices = {}
        
        # 遍历每个spot，找到最高分数和对应的方法
        for spot_id in scores_df.index:
            spot_scores = scores_df.loc[spot_id]
            spot_labels = labels_df.loc[spot_id]
            
            # 跳过全为NaN的行
            if spot_scores.isna().all():
                continue
                
            # 找到最高分数的方法
            max_score_method = spot_scores.idxmax()
            max_score = spot_scores.max()
            domain_label = spot_labels[max_score_method]
            
            # 跳过NaN值
            if pd.isna(max_score) or pd.isna(domain_label):
                continue
                
            domain_key = f"Domain_{int(domain_label)}"
            
            # 如果这个domain还没有记录，或者当前分数更高，则更新记录
            if domain_key not in domain_choices or max_score > domain_choices[domain_key]["score"]:
                domain_choices[domain_key] = {
                    "score": float(max_score),
                    "method": max_score_method,
                    "domain_index": int(domain_label),
                    "representative_spot": spot_id
                }
        
        # 按domain编号排序
        sorted_choices = dict(sorted(domain_choices.items(), 
                                   key=lambda x: x[1]["domain_index"]))
        
        # 添加汇总信息
        summary = {
            "total_domains": len(sorted_choices),
            "methods_used": list(set(choice["method"] for choice in sorted_choices.values())),
            "average_score": sum(choice["score"] for choice in sorted_choices.values()) / len(sorted_choices) if sorted_choices else 0,
            "generation_info": {
                "total_spots_analyzed": len(scores_df),
                "methods_available": list(scores_df.columns),
                "timestamp": pd.Timestamp.now().isoformat()
            }
        }
        
        # 构建最终结果
        result = {
            "summary": summary,
            "domain_choices": sorted_choices
        }
        
        # 保存到output目录下的domain_choice.json
        choice_file_path = os.path.join(output_dir, "domain_choice.json")
        with open(choice_file_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        log_func(f"Domain选择记录已生成: {choice_file_path}")
        log_func(f"  - 总域数: {summary['total_domains']}")
        log_func(f"  - 使用的方法: {', '.join(summary['methods_used'])}")
        log_func(f"  - 平均分数: {summary['average_score']:.3f}")
        
        print(f"[Info] Domain选择记录已生成: {choice_file_path}")
        print(f"  - 总域数: {summary['total_domains']}")
        print(f"  - 使用的方法: {', '.join(summary['methods_used'])}")
        print(f"  - 平均分数: {summary['average_score']:.3f}")
        
        return True
        
    except Exception as e:
        log_func(f"生成domain选择记录时发生错误: {e}")
        print(f"[Warning] 生成domain选择记录时发生错误: {e}")
        return False


def auto_build_matrices(input_dir: str, output_dir: str, valid_samples: List[str], log_func, skip_existing: bool = False):
    """自动构建矩阵，基于已有的输入文件和输出结果"""
    try:
        # 保存矩阵文件路径（与主输出目录对齐）
        # 统一输出到: output/consensus/
        matrix_output_dir = os.path.join(output_dir, "consensus")
        os.makedirs(matrix_output_dir, exist_ok=True)
        
        scores_path = os.path.join(matrix_output_dir, "scores_matrix.csv")
        labels_path = os.path.join(matrix_output_dir, "labels_matrix.csv")
        
        # 检查是否已存在矩阵文件（只有在skip_existing=True时才跳过）
        if skip_existing and os.path.exists(scores_path) and os.path.exists(labels_path):
            log_func(f"矩阵文件已存在，跳过构建：")
            log_func(f"  - 分数矩阵: {scores_path}")
            log_func(f"  - 标签矩阵: {labels_path}")
            print(f"[Info] 矩阵文件已存在，跳过构建")
            return True, matrix_output_dir
        
        # 查找所有可用的方法（基于输入文件）
        methods = []
        spot_csvs = []
        result_jsons = []

        for sample in valid_samples:
            # 构建输入CSV文件路径
            spot_csv = os.path.join(input_dir, f"{sample}_spot.csv")
            result_json = os.path.join(output_dir, f"{sample}_result.json")

            if os.path.exists(spot_csv) and os.path.exists(result_json):
                # 从sample名称中提取方法名（假设格式为 METHOD_DATASET_ID）
                method_name = sample.split('_')[0]  # 例如从 "GraphST_DLPFC_151507" 提取 "GraphST"
                methods.append(method_name)
                spot_csvs.append(spot_csv)
                result_jsons.append(result_json)

        if len(methods) < 2:
            log_func(f"需要至少2个方法才能构建矩阵，当前只有{len(methods)}个有效方法")
            return False, None

        log_func(f"正在为以下方法构建矩阵: {', '.join(methods)}")

        # 构建矩阵
        scores_df, labels_df = build_matrices(methods, spot_csvs, result_jsons)

        # 保存矩阵文件（默认覆盖）
        scores_df.to_csv(scores_path)
        labels_df.to_csv(labels_path)

        log_func(f"矩阵构建完成:")
        log_func(f"  - 分数矩阵: {scores_path}")
        log_func(f"  - 标签矩阵: {labels_path}")
        
        print(f"[Info] 矩阵构建完成:")
        print(f"  - 分数矩阵: {scores_path}")
        print(f"  - 标签矩阵: {labels_path}")
        
        # 生成domain选择记录
        choice_success = generate_domain_choice_record(scores_df, labels_df, output_dir, log_func)
        if choice_success:
            log_func("Domain选择记录生成成功")
        else:
            log_func("Domain选择记录生成失败")
        
        return True, matrix_output_dir

    except Exception as e:
        log_func(f"构建矩阵时发生错误: {e}")
        print(f"[Warning] 构建矩阵时发生错误: {e}")
        return False, None

# ARI计算函数已移除
# 原来的calculate_ari_and_visualize, select_best_labels, 
# get_neighbor_indices, smooth_spatial_domains等函数已删除



# 自动构建矩阵（如果启用了该选项且有多个方法）
if args.build_matrices and not args.no_build_matrices:
    if len(valid_samples) >= 2:
        print("[Info] 正在尝试构建矩阵...")
        log("正在尝试构建矩阵...")
        
        # 构建矩阵
        matrix_success, matrix_output_dir = auto_build_matrices(input_dir, output_dir, valid_samples, log, args.skip_existing)
        
        # ARI计算功能已移除
        if matrix_success:
            print("[Info] 矩阵构建成功")
            log("矩阵构建成功")
        else:
            print("[Warning] 矩阵构建失败")
            log("矩阵构建失败")
    else:
        print("[Info] 只有一个样本，跳过矩阵构建")
        log("只有一个样本，跳过矩阵构建")
else:
    print("[Info] 矩阵构建被禁用")
    log("矩阵构建被禁用")

print("\n" + "="*60)
print("🎉 程序执行完成！")
print("="*60)
print("📁 文件处理说明:")
if args.skip_existing:
    print("  ✅ 跳过模式：已存在的文件将被跳过")
elif args.clean_output:
    print("  🔄 清理模式：所有文件已重新生成")
else:
    print("  🔄 覆盖模式：矩阵文件已更新（主程序结果文件如已存在则跳过）")


print("\n💡 常用命令参考:")
print("  python scoring.py                     # 完整流程（评分→矩阵生成）")
print("  python scoring.py --skip_existing     # 跳过所有已存在文件")
print("  python scoring.py --clean_output      # 清理后重新生成所有文件")
print("  python scoring.py --no_build_matrices # 仅运行主程序，不构建矩阵")
print("="*60)

 
