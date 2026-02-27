"""
安全的配置加载器
兼容有无config.py文件的情况，自动从环境变量获取配置
"""
import os
from typing import Optional

def load_config_safely():
    """
    安全地加载配置，兼容有无config.py文件的情况
    优先级：config.py > 环境变量 > 默认值
    """
    try:
        # 避免循环导入，这里不要导入自己
        import config as config_module
        return config_module.load_config()
    except ImportError:
        # 如果config.py不存在，从环境变量创建配置
        print("[Warning] config.py not found, using environment variables...")
        return _create_config_from_env()

def _create_config_from_env():
    """从环境变量创建配置"""
    # 导入必要的类
    import json
    from dataclasses import dataclass, field
    from typing import Dict, Any
    from pathlib import Path
    
    @dataclass
    class ScoreRagConfig:
        """Score Rag 系统的统一配置类（简化版）"""
        
        # === API配置 ===
        azure_openai_key: str = ""
        azure_endpoint: str = ""
        azure_deployment: str = "gpt-4"
        azure_api_version: str = "2024-12-01-preview"
        
        # === 评分配置 ===
        top_n_deg: int = 5
        temperature: float = 0.0
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
        compactness_method: str = "mean_pairwise"
        adjacency_method: str = "nearest_neighbor"
        spatial_distance_threshold: float = 2.0
        
        # === 数据处理配置 ===
        chunk_size: int = 100
        max_memory_usage_mb: int = 1024
        enable_caching: bool = True
        cache_dir: str = "cache"
        
        # === 输出配置 ===
        output_format: str = "json"
        save_intermediate_results: bool = True
        log_level: str = "INFO"
        
        # === 示例数据库配置 ===
        use_example_db: bool = False
        example_db_dir: str = "vector_db"
        knn_k: int = 5
        
        # === 错误恢复配置 ===
        enable_fallback: bool = True
        fallback_score: float = 0.1
        max_fallback_attempts: int = 2
        
        def __post_init__(self):
            """配置验证和后处理"""
            # 创建缓存目录
            if self.enable_caching:
                Path(self.cache_dir).mkdir(exist_ok=True)
    
    # 从环境变量创建配置
    config = ScoreRagConfig()
    
    # API配置
    config.azure_openai_key = os.getenv("AZURE_OPENAI_KEY", "")
    config.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    config.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
    config.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    
    # 评分配置
    config.top_n_deg = int(os.getenv("SCORE_RAG_TOP_N_DEG", "5"))
    config.temperature = float(os.getenv("SCORE_RAG_TEMPERATURE", "0.0"))
    
    # 新增的LLM参数
    max_tokens_env = os.getenv("SCORE_RAG_MAX_COMPLETION_TOKENS")
    if max_tokens_env:
        config.max_completion_tokens = int(max_tokens_env)
    else:
        config.max_completion_tokens = 30000
    
    config.top_p = float(os.getenv("SCORE_RAG_TOP_P", "1.0"))
    config.frequency_penalty = float(os.getenv("SCORE_RAG_FREQUENCY_PENALTY", "0.0"))
    config.presence_penalty = float(os.getenv("SCORE_RAG_PRESENCE_PENALTY", "0.0"))
    
    config.max_retries = int(os.getenv("SCORE_RAG_MAX_RETRIES", "3"))
    
    # 验证配置
    config.strict_validation = os.getenv("SCORE_RAG_STRICT_VALIDATION", "false").lower() == "true"
    
    # 数据处理配置
    config.chunk_size = int(os.getenv("SCORE_RAG_CHUNK_SIZE", "100"))
    config.enable_caching = os.getenv("SCORE_RAG_ENABLE_CACHING", "true").lower() == "true"
    
    # 输出配置
    config.output_format = os.getenv("SCORE_RAG_OUTPUT_FORMAT", "json")
    config.log_level = os.getenv("SCORE_RAG_LOG_LEVEL", "INFO")
    
    # 验证必要的配置
    if not config.azure_openai_key:
        raise ValueError("AZURE_OPENAI_KEY is required but not found in environment variables!")
    if not config.azure_endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT is required but not found in environment variables!")
    
    return config

def get_legacy_config():
    """
    获取传统的配置变量，兼容旧代码
    """
    try:
        # 避免循环导入
        import config as config_module
        key = getattr(config_module, "AZURE_OPENAI_KEY", "") or ""
        endpoint = getattr(config_module, "AZURE_ENDPOINT", "") or ""
        deployment = getattr(config_module, "AZURE_DEPLOYMENT", "gpt-4") or "gpt-4"
        version = getattr(config_module, "AZURE_API_VERSION", "2024-12-01-preview") or "2024-12-01-preview"

        # Even when config.py exists, require non-empty credentials.
        if not key:
            raise ValueError("AZURE_OPENAI_KEY is required but missing/empty in config.py and environment variables!")
        if not endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT is required but missing/empty in config.py and environment variables!")

        return key, endpoint, deployment, version
    except ImportError:
        key = os.getenv("AZURE_OPENAI_KEY", "")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
        version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        
        if not key:
            raise ValueError("AZURE_OPENAI_KEY is required but not found in environment variables!")
        if not endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT is required but not found in environment variables!")
        
        return key, endpoint, deployment, version 