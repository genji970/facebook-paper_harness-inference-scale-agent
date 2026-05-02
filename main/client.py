import os
from dataclasses import dataclass
from typing import Any, Optional

from main.args import args


@dataclass
class LLMConfig:
    provider: str
    api_key: str
    base_url: str
    model_name: str

    @classmethod
    def from_args(cls) -> "LLMConfig":
        provider = args.provider

        if provider == "gemini":
            api_key = args.gemini_api_key or os.getenv("GEMINI_API_KEY", "") or args.api_key
            base_url = ""
        else:
            api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
            base_url = args.endpoint_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        return cls(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model_name=args.model_name,
        )


class LLMClient:
    """
    provider에 따라 native Gemini API 또는 OpenAI API를 호출하는 client.

    provider == "gemini":
        google-genai SDK 사용

    provider == "openai":
        OpenAI-compatible chat completions 사용
    """

    def __init__(self):
        self.config = LLMConfig.from_args()

        if self.config.provider == "gemini":
            self.client = self._build_gemini_client()
        elif self.config.provider == "openai":
            self.client = self._build_openai_client()
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    def _build_gemini_client(self) -> Any:
        if not self.config.api_key:
            raise ValueError(
                "Gemini API key is missing. Set GEMINI_API_KEY or pass --gemini_api_key."
            )

        from google import genai

        return genai.Client(api_key=self.config.api_key)

    def _build_openai_client(self) -> Any:
        if not self.config.api_key:
            raise ValueError(
                "OpenAI API key is missing. Set OPENAI_API_KEY or pass --api_key."
            )

        from openai import OpenAI

        return OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        if self.config.provider == "gemini":
            return self._generate_gemini(
                prompt=prompt,
                system_prompt=system_prompt,
            )

        return self._generate_openai(
            prompt=prompt,
            system_prompt=system_prompt,
        )

    def _generate_gemini(self, prompt: str, system_prompt: str = "") -> str:
        from google.genai import types

        config = None

        if system_prompt:
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.0,
            )
        else:
            config = types.GenerateContentConfig(
                temperature=0.0,
            )

        response = self.client.models.generate_content(
            model=self.config.model_name,
            contents=prompt,
            config=config,
        )

        text = getattr(response, "text", None)

        if text:
            return text

        return self._extract_gemini_text(response)

    def _generate_openai(self, prompt: str, system_prompt: str = "") -> str:
        messages = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=0.0,
        )

        return response.choices[0].message.content

    @staticmethod
    def _extract_gemini_text(response: Any) -> str:
        """
        response.text가 비어 있을 때 candidates 구조에서 텍스트를 최대한 복구한다.
        """

        chunks = []

        candidates = getattr(response, "candidates", None) or []

        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) if content else None

            if not parts:
                continue

            for part in parts:
                text = getattr(part, "text", None)
                if text:
                    chunks.append(text)

        if chunks:
            return "\n".join(chunks)

        return ""


if __name__ == "__main__":
    llm_client = LLMClient()

    response = llm_client.generate(
        prompt="Say hello in one sentence.",
        system_prompt="You are a helpful assistant.",
    )

    print(response)