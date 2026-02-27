from __future__ import annotations
"""Central configuration for Azure OpenAI credentials and Score Rag settings.

Usage
-----
# 传统方式（向后兼容）
from config import (
    AZURE_OPENAI_KEY,
    AZURE_ENDPOINT,
    AZURE_DEPLOYMENT,
    AZURE_API_VERSION,
)

# 新的配置类方式
from config import ScoreRagConfig, load_config

config = load_config()
evaluator = DomainEvaluator(config)

• Values are first read from environment variables so that users can override
  them without modifying the repository.
• If relevant environment variables are absent, the hard-coded defaults below
  are used (provided by the user during set-up).
"""
import os
import json
import warnings
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
import yaml

# ---------------------------------------------------------------------------
# Hard-coded defaults
# ---------------------------------------------------------------------------
# SECURITY NOTE:
# - Do NOT hard-code real credentials in the repository.
# - Set these via environment variables instead (recommended), or load from a
#   config JSON using `load_config()`.
_DEFAULT_KEY = ""
_DEFAULT_ENDPOINT = ""
_DEFAULT_DEPLOYMENT = "gpt-4o"
_DEFAULT_API_VERSION = "2024-12-01-preview"


# ---------------------------------------------------------------------------
# Public constants: pull from env or fall back to defaults (向后兼容)
# ---------------------------------------------------------------------------
try:
    import api_config as _api_cfg  # type: ignore
    warnings.warn(
        "api_config.py is deprecated; prefer pipeline_config.yaml or environment variables.",
        DeprecationWarning,
        stacklevel=2,
    )
except Exception:
    _api_cfg = None


def _load_pipeline_raw() -> Dict[str, Any]:
    try:
        repo_root = Path(__file__).resolve().parent.parent
        cfg_path = repo_root / "pipeline_config.yaml"
        if not cfg_path.exists():
            return {}
        return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


_pipeline_cfg = _load_pipeline_raw()
_pipeline_endpoint = str(_pipeline_cfg.get("api_endpoint") or _pipeline_cfg.get("azure_endpoint") or "")
_pipeline_provider = str(_pipeline_cfg.get("api_provider") or "").strip().lower()
_pipeline_is_azure = (
    _pipeline_provider == "azure"
    or "openai.azure.com" in _pipeline_endpoint.lower()
    or "cognitiveservices.azure.com" in _pipeline_endpoint.lower()
)

AZURE_OPENAI_KEY: str = os.getenv(
    "AZURE_OPENAI_KEY",
    str(
        (_pipeline_cfg.get("api_key") if _pipeline_is_azure else "")
        or _pipeline_cfg.get("azure_openai_key")
        or (getattr(_api_cfg, "AZURE_OPENAI_KEY", _DEFAULT_KEY) if _api_cfg else _DEFAULT_KEY)
    ),
)
AZURE_ENDPOINT: str = os.getenv(
    "AZURE_OPENAI_ENDPOINT",
    str(
        (_pipeline_cfg.get("api_endpoint") if _pipeline_is_azure else "")
        or _pipeline_cfg.get("azure_endpoint")
        or (getattr(_api_cfg, "AZURE_OPENAI_ENDPOINT", _DEFAULT_ENDPOINT) if _api_cfg else _DEFAULT_ENDPOINT)
    ),
)
AZURE_DEPLOYMENT: str = os.getenv(
    "AZURE_OPENAI_DEPLOYMENT",
    str(
        (_pipeline_cfg.get("api_model") if _pipeline_is_azure else "")
        or (_pipeline_cfg.get("api_deployment") if _pipeline_is_azure else "")
        or _pipeline_cfg.get("azure_deployment")
        or (getattr(_api_cfg, "AZURE_OPENAI_DEPLOYMENT", _DEFAULT_DEPLOYMENT) if _api_cfg else _DEFAULT_DEPLOYMENT)
    ),
)
AZURE_API_VERSION: str = os.getenv(
    "AZURE_OPENAI_API_VERSION",
    str(
        (_pipeline_cfg.get("api_version") if _pipeline_is_azure else "")
        or _pipeline_cfg.get("azure_api_version")
        or (getattr(_api_cfg, "AZURE_OPENAI_API_VERSION", _DEFAULT_API_VERSION) if _api_cfg else _DEFAULT_API_VERSION)
    ),
)

# ---------------------------------------------------------------------------
# 新的统一配置类
# ---------------------------------------------------------------------------

@dataclass
class ScoreRagConfig:
    """Score Rag 系统的统一配置类"""
    
    # === API配置 ===
    azure_openai_key: str = field(default_factory=lambda: AZURE_OPENAI_KEY)
    azure_endpoint: str = field(default_factory=lambda: AZURE_ENDPOINT)
    azure_deployment: str = field(default_factory=lambda: AZURE_DEPLOYMENT)
    azure_api_version: str = field(default_factory=lambda: AZURE_API_VERSION)
    
    # === 评分配置 ===
    top_n_deg: int = 5
    temperature: float = 1.0
    max_completion_tokens: Optional[int] = 30000  # 最大生成token数，None表示不限制
    top_p: float = 1.0  # 核采样概率
    frequency_penalty: float = 0.0  # 频率惩罚
    presence_penalty: float = 0.0  # 存在惩罚
    max_retries: int = 3
    timeout_seconds: int = 60
    
    # === 验证配置 ===
    min_justification_length: int = 15
    sub_scores_tolerance: float = 0.02
    max_sub_score: float = 0.20
    strict_validation: bool = False
    
    # === 空间度量配置 ===
    compactness_method: str = "mean_pairwise"  # "mean_pairwise", "centroid_based"
    adjacency_method: str = "nearest_neighbor"  # "nearest_neighbor", "threshold_based"
    spatial_distance_threshold: float = 2.0
    
    # === 数据处理配置 ===
    chunk_size: int = 100
    max_memory_usage_mb: int = 1024
    enable_caching: bool = True
    cache_dir: str = "cache"
    
    # === 输出配置 ===
    output_format: str = "json"  # "json", "dataframe", "sql"
    save_intermediate_results: bool = True
    log_level: str = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
    
    # === 示例数据库配置 ===
    use_example_db: bool = False
    example_db_dir: str = "vector_db"
    knn_k: int = 5
    
    # === 错误恢复配置 ===
    enable_fallback: bool = True
    fallback_score: float = 0.1
    max_fallback_attempts: int = 2
    
    # === Annotation (Multi-Agent) 配置 ===
    # NOTE: For paper-grade runs, keep these values in config (not hard-coded in code).
    annotation_enable_multiagent: bool = False
    annotation_standard_score: float = 0.80
    annotation_max_rounds: int = 5
    annotation_language: str = "en"
    annotation_log_dir: str = "output/annotation_runs"

    # Label space for annotation (dataset/task-specific). Default matches the original DLPFC layered cortex task.
    annotation_label_space: List[str] = field(
        default_factory=lambda: [
            "Layer 1",
            "Layer 2",
            "Layer 3",
            "Layer 4",
            "Layer 5",
            "Layer 6",
            "White Matter",
            "Mixed L6/White Matter",
            "Mixed L1/L2",
            "Mixed L2/L3",
            "Mixed L3/L4",
            "Mixed L4/L5",
            "Mixed L5/L6",
        ]
    )

    # Optional knowledge base path for critic guardrails (can be empty/None)
    annotation_kb_path: str = ""

    # Expert weights (must sum to 1.0)
    annotation_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "Marker": 0.55,
            "Pathway": 0.10,
            "Spatial": 0.10,
            "VLM": 0.25,
        }
    )

    # VLM settings (VLM is required in the accepted design)
    annotation_vlm_required: bool = True
    annotation_vlm_min_score: float = 0.60
    # Default: empty -> resolved relative to data_dir as "<sample_id>_result.png"
    annotation_vlm_image_path: str = ""

    # Spatial improvements (optional / experimental)
    # Use domain adjacency graph (from spot kNN) to smooth per-domain depth for layer-order consistency.
    annotation_spatial_adjacency_smoothing: bool = False
    annotation_spatial_knn_k: int = 8
    annotation_spatial_smoothing_alpha: float = 0.35
    annotation_spatial_smoothing_iters: int = 3
    annotation_spatial_adjacency_min_edges: int = 30

    # White Matter guardrail (optional)
    # This guardrail is meant to prevent false WM due to a single marker (e.g., MBP),
    # but should not over-reject true WM domains. Keep conservative defaults.
    annotation_wm_guard_enabled: bool = True
    
    def __post_init__(self):
        """配置验证和后处理"""
        # 验证配置参数
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError(f"temperature must be between 0 and 2, got {self.temperature}")
        
        if self.top_n_deg < 1:
            raise ValueError(f"top_n_deg must be positive, got {self.top_n_deg}")
        
        if self.max_sub_score <= 0 or self.max_sub_score > 1:
            raise ValueError(f"max_sub_score must be between 0 and 1, got {self.max_sub_score}")
        
        # 创建缓存目录
        if self.enable_caching:
            Path(self.cache_dir).mkdir(exist_ok=True)

        # Create annotation log dir lazily (safe even if not used)
        try:
            Path(self.annotation_log_dir).mkdir(parents=True, exist_ok=True)
        except Exception:
            # Avoid blocking the whole system due to logging directory issues
            pass
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ScoreRagConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> 'ScoreRagConfig':
        """从JSON文件加载配置"""
        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_env(cls) -> 'ScoreRagConfig':
        """从环境变量创建配置"""
        config = cls()
        
        # API配置
        config.azure_openai_key = os.getenv("AZURE_OPENAI_KEY", config.azure_openai_key)
        config.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", config.azure_endpoint)
        config.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", config.azure_deployment)
        config.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", config.azure_api_version)
        
        # 评分配置
        config.top_n_deg = int(os.getenv("SCORE_RAG_TOP_N_DEG", config.top_n_deg))
        config.temperature = float(os.getenv("SCORE_RAG_TEMPERATURE", config.temperature))
        
        # 新增的LLM参数
        max_tokens_env = os.getenv("SCORE_RAG_MAX_COMPLETION_TOKENS")
        if max_tokens_env:
            config.max_completion_tokens = int(max_tokens_env)
        else:
            config.max_completion_tokens = 30000  # 新的默认值
        
        config.top_p = float(os.getenv("SCORE_RAG_TOP_P", config.top_p))
        config.frequency_penalty = float(os.getenv("SCORE_RAG_FREQUENCY_PENALTY", config.frequency_penalty))
        config.presence_penalty = float(os.getenv("SCORE_RAG_PRESENCE_PENALTY", config.presence_penalty))
        
        config.max_retries = int(os.getenv("SCORE_RAG_MAX_RETRIES", config.max_retries))
        
        # 验证配置
        config.strict_validation = os.getenv("SCORE_RAG_STRICT_VALIDATION", "false").lower() == "true"
        
        # 数据处理配置
        config.chunk_size = int(os.getenv("SCORE_RAG_CHUNK_SIZE", config.chunk_size))
        config.enable_caching = os.getenv("SCORE_RAG_ENABLE_CACHING", "true").lower() == "true"
        
        # 输出配置
        config.output_format = os.getenv("SCORE_RAG_OUTPUT_FORMAT", config.output_format)
        config.log_level = os.getenv("SCORE_RAG_LOG_LEVEL", config.log_level)

        # Annotation multi-agent config
        config.annotation_enable_multiagent = os.getenv("SCORE_RAG_ANNOTATION_MULTIAGENT", "false").lower() == "true"
        config.annotation_standard_score = float(os.getenv("SCORE_RAG_ANNOTATION_STANDARD_SCORE", str(config.annotation_standard_score)))
        config.annotation_max_rounds = int(os.getenv("SCORE_RAG_ANNOTATION_MAX_ROUNDS", str(config.annotation_max_rounds)))
        config.annotation_language = os.getenv("SCORE_RAG_ANNOTATION_LANGUAGE", config.annotation_language)
        config.annotation_log_dir = os.getenv("SCORE_RAG_ANNOTATION_LOG_DIR", config.annotation_log_dir)
        config.annotation_vlm_required = os.getenv("SCORE_RAG_ANNOTATION_VLM_REQUIRED", "true").lower() == "true"
        config.annotation_vlm_min_score = float(os.getenv("SCORE_RAG_ANNOTATION_VLM_MIN_SCORE", str(config.annotation_vlm_min_score)))
        config.annotation_vlm_image_path = os.getenv("SCORE_RAG_ANNOTATION_VLM_IMAGE_PATH", config.annotation_vlm_image_path)
        config.annotation_spatial_adjacency_smoothing = (
            os.getenv("SCORE_RAG_ANNOTATION_SPATIAL_ADJ_SMOOTHING", "false").lower() == "true"
        )
        config.annotation_spatial_knn_k = int(os.getenv("SCORE_RAG_ANNOTATION_SPATIAL_KNN_K", str(config.annotation_spatial_knn_k)))
        config.annotation_spatial_smoothing_alpha = float(
            os.getenv("SCORE_RAG_ANNOTATION_SPATIAL_SMOOTH_ALPHA", str(config.annotation_spatial_smoothing_alpha))
        )
        config.annotation_spatial_smoothing_iters = int(
            os.getenv("SCORE_RAG_ANNOTATION_SPATIAL_SMOOTH_ITERS", str(config.annotation_spatial_smoothing_iters))
        )
        config.annotation_spatial_adjacency_min_edges = int(
            os.getenv(
                "SCORE_RAG_ANNOTATION_SPATIAL_ADJ_MIN_EDGES", str(config.annotation_spatial_adjacency_min_edges)
            )
        )
        config.annotation_wm_guard_enabled = os.getenv("SCORE_RAG_ANNOTATION_WM_GUARD", "true").lower() == "true"

        # Optional: override weights via env (kept simple: JSON string)
        weights_json = os.getenv("SCORE_RAG_ANNOTATION_WEIGHTS")
        if weights_json:
            try:
                parsed = json.loads(weights_json)
                if isinstance(parsed, dict):
                    config.annotation_weights = {str(k): float(v) for k, v in parsed.items()}
            except Exception:
                # ignore malformed override
                pass
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def to_json(self, json_path: str) -> None:
        """保存到JSON文件"""
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def update(self, **kwargs) -> 'ScoreRagConfig':
        """更新配置参数，返回新的配置对象"""
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return self.from_dict(config_dict)
    
    def validate(self) -> None:
        """验证配置有效性"""
        errors = []
        
        # API配置验证
        if not self.azure_openai_key:
            errors.append("azure_openai_key is required")
        if not self.azure_endpoint:
            errors.append("azure_endpoint is required")
        if not self.azure_deployment:
            errors.append("azure_deployment is required")
        
        # 数值范围验证
        if self.temperature < 0 or self.temperature > 2:
            errors.append(f"temperature must be between 0 and 2, got {self.temperature}")
        
        if self.top_n_deg < 1:
            errors.append(f"top_n_deg must be positive, got {self.top_n_deg}")
        
        if self.max_sub_score <= 0 or self.max_sub_score > 1:
            errors.append(f"max_sub_score must be between 0 and 1, got {self.max_sub_score}")
        
        if self.chunk_size < 1:
            errors.append(f"chunk_size must be positive, got {self.chunk_size}")
        
        # 方法验证
        valid_compactness_methods = ["mean_pairwise", "centroid_based"]
        if self.compactness_method not in valid_compactness_methods:
            errors.append(f"compactness_method must be one of {valid_compactness_methods}")
        
        valid_adjacency_methods = ["nearest_neighbor", "threshold_based"]
        if self.adjacency_method not in valid_adjacency_methods:
            errors.append(f"adjacency_method must be one of {valid_adjacency_methods}")
        
        valid_output_formats = ["json", "dataframe", "sql"]
        if self.output_format not in valid_output_formats:
            errors.append(f"output_format must be one of {valid_output_formats}")
        
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        if self.log_level not in valid_log_levels:
            errors.append(f"log_level must be one of {valid_log_levels}")

        # Annotation validation
        if self.annotation_language not in ["en", "zh", "auto"]:
            errors.append("annotation_language must be one of ['en','zh','auto']")
        if not (0.0 <= self.annotation_standard_score <= 1.0):
            errors.append(f"annotation_standard_score must be in [0,1], got {self.annotation_standard_score}")
        if self.annotation_max_rounds < 1 or self.annotation_max_rounds > 20:
            errors.append(f"annotation_max_rounds must be in [1,20], got {self.annotation_max_rounds}")
        if not (0.0 <= self.annotation_vlm_min_score <= 1.0):
            errors.append(f"annotation_vlm_min_score must be in [0,1], got {self.annotation_vlm_min_score}")

        # weights sum check (tolerate slight float error)
        w = self.annotation_weights or {}
        if w:
            s = sum(float(v) for v in w.values() if isinstance(v, (int, float)))
            if not (0.99 <= s <= 1.01):
                errors.append(f"annotation_weights must sum to 1.0 (±0.01), got {s:.4f}")
            if self.annotation_vlm_required and float(w.get("VLM", 0.0)) <= 0.0:
                errors.append("annotation_vlm_required=True but annotation_weights['VLM'] is 0")

        # Spatial adjacency smoothing validation (optional)
        if self.annotation_spatial_adjacency_smoothing:
            if self.annotation_spatial_knn_k < 3 or self.annotation_spatial_knn_k > 50:
                errors.append(f"annotation_spatial_knn_k must be in [3,50], got {self.annotation_spatial_knn_k}")
            if not (0.0 <= float(self.annotation_spatial_smoothing_alpha) <= 1.0):
                errors.append(
                    f"annotation_spatial_smoothing_alpha must be in [0,1], got {self.annotation_spatial_smoothing_alpha}"
                )
            if self.annotation_spatial_smoothing_iters < 1 or self.annotation_spatial_smoothing_iters > 20:
                errors.append(
                    f"annotation_spatial_smoothing_iters must be in [1,20], got {self.annotation_spatial_smoothing_iters}"
                )
            if self.annotation_spatial_adjacency_min_edges < 0:
                errors.append(
                    f"annotation_spatial_adjacency_min_edges must be >=0, got {self.annotation_spatial_adjacency_min_edges}"
                )
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors))

# ---------------------------------------------------------------------------
# 配置加载函数
# ---------------------------------------------------------------------------

def load_config(config_path: Optional[str] = None) -> ScoreRagConfig:
    """
    加载配置，优先级：命令行参数 > 配置文件 > 环境变量 > 默认值
    
    :param config_path: 配置文件路径（可选）
    :return: 配置对象
    """
    # 从环境变量开始
    config = ScoreRagConfig.from_env()
    
    # 如果提供了配置文件，覆盖环境变量配置
    if config_path and os.path.exists(config_path):
        try:
            file_config = ScoreRagConfig.from_json(config_path)
            # 合并配置（文件配置覆盖环境变量配置）
            config_dict = config.to_dict()
            config_dict.update(file_config.to_dict())
            config = ScoreRagConfig.from_dict(config_dict)
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
    
    # 验证最终配置
    config.validate()
    
    return config

def create_default_config_file(config_path: str = "score_rag_config.json") -> None:
    """创建默认配置文件"""
    config = ScoreRagConfig()
    config.to_json(config_path)
    print(f"Created default configuration file: {config_path}")

# ---------------------------------------------------------------------------
# 向后兼容性
# ---------------------------------------------------------------------------

__all__ = [
    # 传统常量（向后兼容）
    "AZURE_OPENAI_KEY",
    "AZURE_ENDPOINT", 
    "AZURE_DEPLOYMENT",
    "AZURE_API_VERSION",
    # 新的配置类
    "ScoreRagConfig",
    "load_config",
    "create_default_config_file",
] 
