from pathlib import Path
from typing import Any

from ensagent_tools.config_manager import (
    PipelineConfig,
    load_config as load_pipeline_config_file,
    save_config as save_pipeline_config,
)

CONFIG_PATH = Path(__file__).parent.parent / "pipeline_config.yaml"


def load_pipeline_config() -> PipelineConfig:
    return load_pipeline_config_file(path=CONFIG_PATH)


def _coerce_value(field_name: str, value: Any, current: Any) -> Any:
    if isinstance(current, bool):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, list):
        if isinstance(value, list):
            return value
        return [item.strip() for item in str(value).split(",") if item.strip()]
    if isinstance(current, dict):
        return dict(value) if isinstance(value, dict) else current
    return "" if value is None else str(value)


def _sync_api_aliases(cfg: PipelineConfig) -> None:
    if not cfg.api_model and cfg.api_deployment:
        cfg.api_model = cfg.api_deployment
    if not cfg.api_deployment and cfg.api_model:
        cfg.api_deployment = cfg.api_model

    endpoint = (cfg.api_endpoint or cfg.azure_endpoint or "").lower()
    is_azure = (cfg.api_provider or "").strip().lower() == "azure"
    is_azure = is_azure or ("openai.azure.com" in endpoint) or ("cognitiveservices.azure.com" in endpoint)

    if not cfg.api_key and cfg.azure_openai_key:
        cfg.api_key = cfg.azure_openai_key
    if not cfg.api_endpoint and cfg.azure_endpoint:
        cfg.api_endpoint = cfg.azure_endpoint
    if not cfg.api_version and cfg.azure_api_version:
        cfg.api_version = cfg.azure_api_version
    if not cfg.api_model and cfg.azure_deployment:
        cfg.api_model = cfg.azure_deployment
        cfg.api_deployment = cfg.azure_deployment

    if is_azure:
        if cfg.api_key:
            cfg.azure_openai_key = cfg.api_key
        if cfg.api_endpoint:
            cfg.azure_endpoint = cfg.api_endpoint
        if cfg.api_version:
            cfg.azure_api_version = cfg.api_version
        if cfg.api_model:
            cfg.azure_deployment = cfg.api_model
            cfg.api_deployment = cfg.api_model


def load_config() -> dict:
    cfg = load_pipeline_config()
    _sync_api_aliases(cfg)
    return cfg.to_dict()


def save_config(data: dict[str, Any]) -> None:
    cfg = load_pipeline_config()
    fields = PipelineConfig.__dataclass_fields__
    alias_map = {
        "azure_openai_endpoint": "azure_endpoint",
    }

    for raw_key, raw_value in data.items():
        field_name = alias_map.get(raw_key, raw_key)
        if field_name not in fields:
            continue
        current = getattr(cfg, field_name)
        setattr(cfg, field_name, _coerce_value(field_name, raw_value, current))

    _sync_api_aliases(cfg)
    save_pipeline_config(cfg, path=CONFIG_PATH)
