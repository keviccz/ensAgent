import base64
from types import SimpleNamespace
from typing import Any, Dict, List

try:
    from .config import Config
except Exception:  # pragma: no cover - script-mode fallback
    from config import Config

try:
    from provider_runtime import resolve_provider_config, completion_text
except Exception:
    from scoring.provider_runtime import resolve_provider_config, completion_text  # type: ignore


class _CompletionAdapter:
    def __init__(self, runtime: "AzureOpenAIClient"):
        self._runtime = runtime

    def create(self, *, model: str, messages: List[Dict[str, Any]], max_tokens: int = 1200, temperature: float = 0.2, **_kwargs):
        return self._runtime._create_completion(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )


class _ChatAdapter:
    def __init__(self, runtime: "AzureOpenAIClient"):
        self.completions = _CompletionAdapter(runtime)


class _ClientAdapter:
    def __init__(self, runtime: "AzureOpenAIClient"):
        self.chat = _ChatAdapter(runtime)


class AzureOpenAIClient:
    """Backward-compatible name; now uses unified multi-provider runtime."""

    def __init__(self):
        self._provider_cfg = resolve_provider_config(
            api_provider=Config.API_PROVIDER,
            api_key=Config.API_KEY,
            api_endpoint=Config.API_ENDPOINT,
            api_version=Config.API_VERSION,
            api_model=Config.API_MODEL,
            azure_openai_key=Config.AZURE_OPENAI_API_KEY,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            azure_api_version=Config.AZURE_OPENAI_API_VERSION,
            azure_deployment=Config.AZURE_OPENAI_DEPLOYMENT_NAME,
        )
        if not self._provider_cfg.api_key:
            raise ValueError("Provider API key is missing for pic_analyze.")
        if self._provider_cfg.provider in {"openai_compatible", "others"} and not self._provider_cfg.endpoint:
            raise ValueError("Custom OpenAI-compatible provider requires api_endpoint.")

        self.client = _ClientAdapter(self)
        self.deployment_name = Config.API_MODEL
        self.ocr_deployment_name = Config.AZURE_OPENAI_OCR_DEPLOYMENT_NAME
        self._ocr_capability_checked = False
        self._ocr_cache: Dict[str, str] = {}

    def _create_completion(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1200,
        temperature: float = 0.2,
    ) -> Any:
        text = completion_text(
            config=self._provider_cfg,
            model=model or self.deployment_name,
            messages=messages,
            temperature=float(temperature),
            top_p=1.0,
            max_tokens=int(max_tokens),
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
        )

    def encode_image(self, image_path: str) -> str:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as exc:
            raise Exception(f"Failed to encode image: {exc}")

    def _looks_like_ocr_capability_error(self, err: Exception) -> bool:
        msg = str(err).lower()
        markers = [
            "image_url",
            "vision",
            "image input",
            "multimodal",
            "does not support",
            "unsupported",
            "invalid content",
        ]
        return any(marker in msg for marker in markers)

    def ensure_ocr_capable(self, image_path: str) -> None:
        """Check once that the configured OCR model supports image/OCR input."""
        if self._ocr_capability_checked or not Config.OCR_REQUIRED:
            return

        base64_image = self.encode_image(image_path)
        try:
            self.client.chat.completions.create(
                model=self.ocr_deployment_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Return only OCR_OK."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        ],
                    }
                ],
                max_tokens=16,
                temperature=0.0,
            )
            self._ocr_capability_checked = True
        except Exception as exc:
            if self._looks_like_ocr_capability_error(exc):
                raise RuntimeError(
                    "Configured model does not support OCR/vision image input. "
                    "Please set an OCR-capable model, or disable visual module with --vlm_off."
                ) from exc
            raise

    def extract_ocr_text(self, image_path: str) -> str:
        """Extract OCR text from one image (cached by path)."""
        if not Config.OCR_REQUIRED:
            return ""
        if image_path in self._ocr_cache:
            return self._ocr_cache[image_path]

        self.ensure_ocr_capable(image_path)
        base64_image = self.encode_image(image_path)
        try:
            response = self.client.chat.completions.create(
                model=self.ocr_deployment_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Extract all readable text from this figure. "
                                    "Return plain text only; if no text, return [NO_TEXT]."
                                ),
                            },
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        ],
                    }
                ],
                max_tokens=1200,
                temperature=0.0,
            )
            text = str(response.choices[0].message.content or "").strip()
            self._ocr_cache[image_path] = text
            return text
        except Exception as exc:
            if self._looks_like_ocr_capability_error(exc):
                raise RuntimeError(
                    "Configured model does not support OCR/vision image input. "
                    "Please set an OCR-capable model, or disable visual module with --vlm_off."
                ) from exc
            raise Exception(f"OCR extraction failed: {exc}")

    def analyze_image(self, image_path: str, prompt: str = "请详细分析这张图片的内容") -> str:
        try:
            base64_image = self.encode_image(image_path)
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        ],
                    }
                ],
                max_tokens=1000,
                temperature=0.7,
            )
            return str(response.choices[0].message.content or "")
        except Exception as exc:
            raise Exception(f"Image analysis failed: {exc}")

    def chat_with_image_context(self, user_message: str, image_analysis: str | None = None) -> str:
        try:
            messages: List[Dict[str, Any]] = [{"role": "system", "content": Config.DEFAULT_SYSTEM_MESSAGE}]
            if image_analysis:
                messages.append({"role": "assistant", "content": f"图片分析结果：{image_analysis}"})
            messages.append({"role": "user", "content": user_message})
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
            )
            return str(response.choices[0].message.content or "")
        except Exception as exc:
            raise Exception(f"Chat failed: {exc}")

    def chat(self, messages: List[Dict[str, str]]) -> str:
        try:
            system_message = {"role": "system", "content": Config.DEFAULT_SYSTEM_MESSAGE}
            if len(messages) > Config.MAX_CONVERSATION_HISTORY:
                messages = messages[-Config.MAX_CONVERSATION_HISTORY :]
            full_messages = [system_message] + messages
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=full_messages,
                max_tokens=1000,
                temperature=0.7,
            )
            return str(response.choices[0].message.content or "")
        except Exception as exc:
            raise Exception(f"Chat failed: {exc}")

