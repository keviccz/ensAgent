import base64
from typing import Any, Dict, List

from openai import AzureOpenAI

from config import Config


class AzureOpenAIClient:
    """Azure OpenAI client for conversation + image analysis + OCR checks."""

    def __init__(self):
        if not Config.IS_AZURE_PROVIDER:
            raise ValueError(
                "pic_analyze currently supports Azure OpenAI only. "
                "Set api_provider=azure in pipeline_config.yaml, or use --vlm_off to disable visual module."
            )
        if not all(
            [
                Config.AZURE_OPENAI_ENDPOINT,
                Config.AZURE_OPENAI_API_KEY,
                Config.AZURE_OPENAI_DEPLOYMENT_NAME,
            ]
        ):
            raise ValueError("Azure OpenAI configuration is incomplete.")

        self.client = AzureOpenAI(
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            api_key=Config.AZURE_OPENAI_API_KEY,
            api_version=Config.AZURE_OPENAI_API_VERSION,
        )
        self.deployment_name = Config.AZURE_OPENAI_DEPLOYMENT_NAME
        self.ocr_deployment_name = Config.AZURE_OPENAI_OCR_DEPLOYMENT_NAME
        self._ocr_capability_checked = False
        self._ocr_cache: Dict[str, str] = {}

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
                    "Please set an OCR-capable Azure deployment, or disable visual module with --vlm_off."
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
                    "Please set an OCR-capable Azure deployment, or disable visual module with --vlm_off."
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
