import os
import time
import json
from typing import List, Dict, Any, Optional, AsyncIterator
import httpx
from .base import (
    BaseProvider,
    ProviderResponse,
    UsageStats,
    ProviderRateLimitError,
    ProviderQuotaExhaustedError,
    ProviderError,
)


class GeminiGemmaProvider(BaseProvider):
    """
    Google AI Studio Provider:
    Offers generous free tier access to Gemini 1.5 Pro, Gemini 1.5 Flash,
    and official Google Gemma 2 (gemma-2-27b-it, gemma-2-9b-it) endpoints.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai",
        timeout_seconds: float = 60.0,
    ):
        super().__init__(provider_id="gemini_gemma_cloud", is_local=False, is_free_tier=True)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_seconds

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 5)

    async def is_healthy(self) -> bool:
        if not self.is_configured:
            return False
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                res = await client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                return res.status_code == 200
        except Exception:
            return False

    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ProviderResponse:
        if not self.is_configured:
            raise ProviderError(
                "Gemini/Gemma API key is not configured. Set GEMINI_API_KEY in .env.",
                status_code=401,
                provider_id=self.provider_id,
            )

        model_name = model or "gemini-1.5-flash"
        start_time = time.time()

        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )

                if res.status_code == 200:
                    data = res.json()
                    choice = data.get("choices", [{}])[0]
                    content = choice.get("message", {}).get("content", "")
                    usage_data = data.get("usage", {})
                    latency = round((time.time() - start_time) * 1000, 2)
                    return ProviderResponse(
                        content=content,
                        model=model_name,
                        provider_id=self.provider_id,
                        usage=UsageStats(
                            prompt_tokens=usage_data.get("prompt_tokens", 0),
                            completion_tokens=usage_data.get("completion_tokens", 0),
                            total_tokens=usage_data.get("total_tokens", 0),
                        ),
                        latency_ms=latency,
                    )
                elif res.status_code == 429:
                    raise ProviderRateLimitError(
                        f"Google AI Studio rate limit exceeded (HTTP 429): {res.text}",
                        status_code=429,
                        provider_id=self.provider_id,
                    )
                elif res.status_code in (402, 403):
                    raise ProviderQuotaExhaustedError(
                        f"Google AI Studio quota exhausted: {res.text}",
                        status_code=res.status_code,
                        provider_id=self.provider_id,
                    )
                else:
                    raise ProviderError(
                        f"Google AI Studio error HTTP {res.status_code}: {res.text}",
                        status_code=res.status_code,
                        provider_id=self.provider_id,
                    )
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise ProviderError(f"Connection to Google AI Studio failed: {e}", provider_id=self.provider_id)

    async def stream_complete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        if not self.is_configured:
            raise ProviderError("Gemini API key not configured.", provider_id=self.provider_id)

        model_name = model or "gemini-1.5-flash"
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", f"{self.base_url}/chat/completions", json=payload, headers=headers) as response:
                if response.status_code == 429:
                    raise ProviderRateLimitError("Google AI Studio rate limit (429)", status_code=429, provider_id=self.provider_id)
                elif response.status_code != 200:
                    text = await response.aread()
                    raise ProviderError(f"HTTP {response.status_code}: {text.decode('utf-8')}", status_code=response.status_code, provider_id=self.provider_id)
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except Exception:
                            continue
