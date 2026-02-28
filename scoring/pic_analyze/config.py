import os
import warnings
from pathlib import Path
from typing import Any, Dict, Mapping
try:
    from provider_runtime import resolve_provider_config
except Exception:
    from scoring.provider_runtime import resolve_provider_config  # type: ignore

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency fallback
    def load_dotenv(*_args, **_kwargs):
        return False

    warnings.warn(
        "python-dotenv is not installed; .env autoload is disabled for pic_analyze.",
        RuntimeWarning,
        stacklevel=2,
    )

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency fallback
    yaml = None
    warnings.warn(
        "PyYAML is not installed; pipeline_config.yaml will not be read by pic_analyze.",
        RuntimeWarning,
        stacklevel=2,
    )

# Load environment variables from local .env when present.
load_dotenv()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_pipeline_config(repo_root: Path | None = None) -> Dict[str, Any]:
    if yaml is None:
        return {}
    root = repo_root or _repo_root()
    cfg_path = root / "pipeline_config.yaml"
    if not cfg_path.exists():
        return {}
    try:
        return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _normalize_provider(provider: str | None) -> str:
    normalized = str(provider or "").strip().lower()
    if normalized in {"", "auto", "none"}:
        return ""
    aliases = {
        "together": "together_ai",
    }
    return aliases.get(normalized, normalized)


def _detect_provider_from_endpoint(endpoint: str | None) -> str:
    ep = str(endpoint or "").strip().lower()
    if not ep:
        return ""
    if "openai.azure.com" in ep or "cognitiveservices.azure.com" in ep:
        return "azure"
    if "api.openai.com" in ep:
        return "openai"
    if "api.anthropic.com" in ep:
        return "anthropic"
    return ""


def _first_non_empty(*values: str) -> str:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _resolve_azure_openai_settings(
    *,
    env: Mapping[str, str] | None = None,
    pipeline_raw: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    env_map = dict(os.environ) if env is None else dict(env)
    raw = _load_pipeline_config() if pipeline_raw is None else dict(pipeline_raw)

    resolved = resolve_provider_config(
        api_provider=str(raw.get("api_provider") or env_map.get("ENSAGENT_API_PROVIDER", "")),
        api_key=_first_non_empty(
            env_map.get("ENSAGENT_API_KEY", ""),
            str(raw.get("api_key") or ""),
        ),
        api_endpoint=_first_non_empty(
            env_map.get("ENSAGENT_API_ENDPOINT", ""),
            str(raw.get("api_endpoint") or ""),
        ),
        api_version=_first_non_empty(
            env_map.get("ENSAGENT_API_VERSION", ""),
            str(raw.get("api_version") or ""),
        ),
        api_model=_first_non_empty(
            env_map.get("ENSAGENT_API_MODEL", ""),
            str(raw.get("api_model") or raw.get("api_deployment") or ""),
        ),
        azure_openai_key=_first_non_empty(
            env_map.get("AZURE_OPENAI_API_KEY", ""),
            env_map.get("AZURE_OPENAI_KEY", ""),
            str(raw.get("azure_openai_key") or ""),
        ),
        azure_endpoint=_first_non_empty(
            env_map.get("AZURE_OPENAI_ENDPOINT", ""),
            env_map.get("AZURE_ENDPOINT", ""),
            str(raw.get("azure_endpoint") or ""),
        ),
        azure_api_version=_first_non_empty(
            env_map.get("AZURE_OPENAI_API_VERSION", ""),
            env_map.get("AZURE_API_VERSION", ""),
            str(raw.get("azure_api_version") or ""),
        ),
        azure_deployment=_first_non_empty(
            env_map.get("AZURE_OPENAI_DEPLOYMENT_NAME", ""),
            env_map.get("AZURE_OPENAI_DEPLOYMENT", ""),
            env_map.get("AZURE_DEPLOYMENT", ""),
            str(raw.get("azure_deployment") or ""),
        ),
        env=env_map,
    )
    effective_provider = resolved.provider
    is_azure = effective_provider == "azure"
    endpoint = resolved.endpoint
    api_key = resolved.api_key
    api_version = resolved.api_version
    deployment_name = resolved.model
    ocr_deployment_name = _first_non_empty(
        env_map.get("ENSAGENT_OCR_MODEL", ""),
        env_map.get("AZURE_OPENAI_OCR_DEPLOYMENT_NAME", ""),
        deployment_name,
    )

    return {
        "provider": effective_provider,
        "is_azure_provider": is_azure,
        "endpoint": endpoint,
        "api_key": api_key,
        "api_version": api_version,
        "deployment_name": deployment_name,
        "ocr_deployment_name": ocr_deployment_name,
    }


_RESOLVED_API = _resolve_azure_openai_settings()


class Config:
    """Configuration class for pic_analyze."""

    API_PROVIDER = _RESOLVED_API["provider"]
    IS_AZURE_PROVIDER = bool(_RESOLVED_API["is_azure_provider"])
    API_ENDPOINT = _RESOLVED_API["endpoint"]
    API_KEY = _RESOLVED_API["api_key"]
    API_VERSION = _RESOLVED_API["api_version"]
    API_MODEL = _RESOLVED_API["deployment_name"]

    # Azure OpenAI settings
    AZURE_OPENAI_ENDPOINT = _RESOLVED_API["endpoint"]
    AZURE_OPENAI_API_KEY = _RESOLVED_API["api_key"]
    AZURE_OPENAI_API_VERSION = _RESOLVED_API["api_version"]
    AZURE_OPENAI_DEPLOYMENT_NAME = _RESOLVED_API["deployment_name"]
    AZURE_OPENAI_OCR_DEPLOYMENT_NAME = _RESOLVED_API["ocr_deployment_name"]

    # OCR is required by default in pic_analyze.
    OCR_REQUIRED = os.getenv("PIC_ANALYZE_OCR_REQUIRED", "true").strip().lower() not in {"0", "false", "no"}

    # Image storage settings
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}

    # Flask settings
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"

    # Conversation settings
    MAX_CONVERSATION_HISTORY = 10
    DEFAULT_SYSTEM_MESSAGE = """你是一个智能助手，能够分析图片内容并与用户进行对话。
    当用户询问关于图片的问题时，你可以：
    1. 识别图片中的物体、人物、场景
    2. 提取图片中的文字内容
    3. 分析图片的情感和氛围
    4. 回答关于图片内容的具体问题

    请用中文回答用户的问题。"""

    @staticmethod
    def init_app(app):
        """Initialize application settings."""
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        app.config["SECRET_KEY"] = Config.SECRET_KEY
        app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH
        app.config["UPLOAD_FOLDER"] = Config.UPLOAD_FOLDER
