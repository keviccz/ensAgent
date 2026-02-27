"""
Settings page for EnsAgent.
Houses API configuration, pipeline parameters, and preferences.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import streamlit as st

from streamlit_app.utils.state import (
    get_state,
    set_state,
)
from streamlit_app.components.sidebar import _save_config
from streamlit_app.utils.api import _detect_provider
from streamlit_app.utils.config_bridge import load_pipeline_fields, save_pipeline_fields
from streamlit_app.utils.data_discovery import discover_data_defaults


PROVIDER_CATALOG: list[tuple[str, str | None]] = [
    ("Auto-detect", None),
    ("OpenAI", "openai"),
    ("Azure OpenAI", "azure"),
    ("Anthropic", "anthropic"),
    ("Google Gemini", "gemini"),
    ("OpenRouter", "openrouter"),
    ("DeepSeek", "deepseek"),
    ("Groq", "groq"),
    ("Together AI", "together_ai"),
    ("Mistral", "mistral"),
    ("Cohere", "cohere"),
    ("xAI", "xai"),
    ("Perplexity", "perplexity"),
    ("OpenAI-compatible (Custom)", "openai_compatible"),
    ("Others", "others"),
]

PROVIDER_ENDPOINT_PLACEHOLDERS = {
    "openai": "https://api.openai.com/v1",
    "azure": "https://<resource>.openai.azure.com/",
    "openai_compatible": "https://your-provider.example.com/v1",
    "anthropic": "https://api.anthropic.com",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "together_ai": "https://api.together.xyz/v1",
    "mistral": "https://api.mistral.ai/v1",
    "cohere": "https://api.cohere.com/v1",
    "xai": "https://api.x.ai/v1",
    "perplexity": "https://api.perplexity.ai",
    "others": "https://your-provider.example.com/v1",
}

PROVIDER_LABELS = {key: label for label, key in PROVIDER_CATALOG if key is not None}


def get_provider_catalog() -> list[tuple[str, str | None]]:
    """Return available provider options for the Settings provider selector."""
    return list(PROVIDER_CATALOG)


def _get_default_endpoint(provider: str | None) -> str:
    return PROVIDER_ENDPOINT_PLACEHOLDERS.get(provider or "", "")


def _normalize_provider(provider: str | None) -> str | None:
    legacy_map = {
        "openai_compatible": "openai_compatible",
        "openai": "openai",
        "azure": "azure",
        "anthropic": "anthropic",
        "gemini": "gemini",
        "openrouter": "openrouter",
        "deepseek": "deepseek",
        "groq": "groq",
        "together": "together_ai",
        "together_ai": "together_ai",
        "mistral": "mistral",
        "cohere": "cohere",
        "xai": "xai",
        "perplexity": "perplexity",
        "others": "others",
    }
    if provider is None:
        return None
    provider = str(provider).strip().lower()
    if provider in {"", "auto", "none"}:
        return None
    return legacy_map.get(provider, provider)


def _open_folder(path: Path) -> None:
    """Open a folder in the local file explorer."""
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
        return

    if sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def render_settings() -> None:
    """Render the settings page with API and pipeline configuration."""
    if not get_state("_settings_defaults_loaded", False):
        defaults = load_pipeline_fields()
        for key in [
            "api_provider",
            "api_key",
            "api_endpoint",
            "api_version",
            "api_model",
            "api_deployment",
            "data_path",
            "sample_id",
            "n_clusters",
        ]:
            incoming = defaults.get(key, "")
            current = get_state(key)
            if key == "n_clusters":
                if incoming is not None and incoming != current:
                    set_state("n_clusters", int(incoming))
                continue
            if incoming and not current:
                set_state(key, incoming)

        detection = discover_data_defaults()
        if not get_state("data_path") and detection.get("data_path"):
            set_state("data_path", detection["data_path"])
            save_pipeline_fields(data_path=detection["data_path"])
        if not get_state("sample_id") and detection.get("sample_id"):
            set_state("sample_id", detection["sample_id"])
            save_pipeline_fields(sample_id=detection["sample_id"])
        set_state("_data_auto_source", detection.get("source", "none"))
        set_state("_data_auto_source_detail", detection.get("source_detail", ""))
        set_state("_settings_defaults_loaded", True)

    st.caption("Configure model provider credentials and runtime sampling parameters.")

    provider_catalog = get_provider_catalog()
    provider_options = [label for label, _ in provider_catalog]
    provider_keys = [key for _, key in provider_catalog]
    current_provider = _normalize_provider(get_state("api_provider"))

    try:
        provider_index = provider_keys.index(current_provider)
    except ValueError:
        provider_index = 0

    def _on_provider_change() -> None:
        idx = provider_options.index(st.session_state._provider_select)
        selected = provider_keys[idx]
        set_state("api_provider", selected)
        save_pipeline_fields(api_provider=selected or "")

    st.selectbox(
        "Provider",
        options=provider_options,
        index=provider_index,
        key="_provider_select",
        on_change=_on_provider_change,
    )

    selected_provider = _normalize_provider(get_state("api_provider"))

    def _on_api_key_change() -> None:
        new_key = st.session_state._api_key_input
        set_state("api_key", new_key)
        save_pipeline_fields(api_key=new_key)

    st.text_input(
        "API Key",
        value=get_state("api_key", ""),
        type="password",
        placeholder="sk-... or your API key",
        key="_api_key_input",
        on_change=_on_api_key_change,
    )

    current_endpoint = get_state("api_endpoint", "")
    detected_provider = _normalize_provider(_detect_provider(current_endpoint))
    effective_provider = selected_provider or detected_provider or "openai"
    placeholder = _get_default_endpoint(effective_provider) or "API Endpoint URL"

    def _on_endpoint_change() -> None:
        new_endpoint = st.session_state._endpoint_input
        set_state("api_endpoint", new_endpoint)
        save_pipeline_fields(api_endpoint=new_endpoint)

    st.text_input(
        "API Endpoint",
        value=current_endpoint,
        placeholder=placeholder,
        key="_endpoint_input",
        on_change=_on_endpoint_change,
    )

    col_ver, col_dep = st.columns(2)

    with col_ver:

        def _on_version_change() -> None:
            new_ver = st.session_state._api_version_input
            set_state("api_version", new_ver if new_ver else None)
            save_pipeline_fields(api_version=new_ver)

        st.text_input(
            "API Version",
            value=get_state("api_version", "") or "",
            placeholder="e.g. 2024-12-01-preview",
            key="_api_version_input",
            on_change=_on_version_change,
        )

    with col_dep:

        def _on_deployment_change() -> None:
            new_dep = st.session_state._api_deployment_input
            set_state("api_model", new_dep if new_dep else None)
            set_state("api_deployment", new_dep if new_dep else None)
            save_pipeline_fields(api_model=new_dep, api_deployment=new_dep)

        st.text_input(
            "Model / Deployment",
            value=(get_state("api_model") or get_state("api_deployment", "") or ""),
            placeholder="e.g. gpt-4o, claude-3-5-sonnet, deepseek-chat",
            key="_api_deployment_input",
            on_change=_on_deployment_change,
        )

    final_provider = selected_provider or _normalize_provider(_detect_provider(get_state("api_endpoint", "")))
    if final_provider:
        label = PROVIDER_LABELS.get(final_provider, final_provider.replace("_", " ").title())
        st.markdown(
            f"""
            <div class="ens-provider-badge" data-provider="{final_provider}">
                <span class="ens-provider-dot"></span>
                <span class="ens-provider-label">{label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col_temp, col_top_p, col_n_clusters = st.columns(3)

    with col_temp:

        def _on_temperature_change() -> None:
            new_temp = float(st.session_state._temp_input)
            set_state("temperature", new_temp)
            _save_config(temperature=new_temp)

        st.number_input(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=float(get_state("temperature", 0.7)),
            step=0.1,
            format="%.2f",
            key="_temp_input",
            on_change=_on_temperature_change,
        )

    with col_top_p:

        def _on_top_p_change() -> None:
            new_top_p = float(st.session_state._top_p_input)
            set_state("top_p", new_top_p)
            _save_config(top_p=new_top_p)

        st.number_input(
            "Top-p",
            min_value=0.0,
            max_value=1.0,
            value=float(get_state("top_p", 1.0)),
            step=0.05,
            format="%.2f",
            key="_top_p_input",
            on_change=_on_top_p_change,
        )

    with col_n_clusters:

        def _on_n_clusters_change() -> None:
            new_n_clusters = int(st.session_state._n_clusters_input)
            set_state("n_clusters", new_n_clusters)
            save_pipeline_fields(n_clusters=new_n_clusters)

        st.number_input(
            "n_clusters",
            min_value=2,
            max_value=30,
            value=int(get_state("n_clusters", 7) or 7),
            step=1,
            key="_n_clusters_input",
            on_change=_on_n_clusters_change,
        )

    st.markdown('<div class="ens-spacer-md"></div>', unsafe_allow_html=True)
    st.caption("Set your data path and sample identifier for analysis.")
    source = get_state("_data_auto_source", "none")
    source_detail = get_state("_data_auto_source_detail", "")
    if source and source != "none":
        source_text = source_detail or source
        st.caption(f"Auto-detected source: {source_text}")

    def _on_data_path_change() -> None:
        val = st.session_state._data_path_input
        set_state("data_path", val if val else None)
        save_pipeline_fields(data_path=val or "")

    st.text_input(
        "Data Path",
        value=get_state("data_path", "") or "",
        placeholder="/path/to/your/spatial/data",
        key="_data_path_input",
        on_change=_on_data_path_change,
    )

    def _on_sample_id_change() -> None:
        val = st.session_state._sample_id_input
        set_state("sample_id", val if val else None)
        save_pipeline_fields(sample_id=val or "")

    st.text_input(
        "Sample ID",
        value=get_state("sample_id", "") or "",
        placeholder="e.g. DLPFC_151507",
        key="_sample_id_input",
        on_change=_on_sample_id_change,
    )

    st.markdown('<div class="ens-spacer-md"></div>', unsafe_allow_html=True)
    st.caption("Import chat history (JSON).")

    if st.button("Open chat_history folder", use_container_width=True):
        try:
            folder = Path(__file__).resolve().parent.parent.parent / "chat_history"
            folder.mkdir(parents=True, exist_ok=True)
            _open_folder(folder)
        except Exception as exc:
            st.error(f"Could not open folder: {exc}")
