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


class OpenAICompatibleProvider(BaseProvider):
    """Generic high-performance OpenAI-compatible gateway adapter."""

    def __init__(
        self,
        provider_id: str,
        base_url: str,
        api_key: Optional[str] = None,
        env_key_name: Optional[str] = None,
        is_free_tier: bool = False,
        default_model: str = "gpt-4o",
        timeout_seconds: float = 60.0,
    ):
        super().__init__(provider_id=provider_id, is_local=False, is_free_tier=is_free_tier)
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout_seconds
        self.api_key = api_key or (os.getenv(env_key_name, "") if env_key_name else "")

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 3)

    async def is_healthy(self) -> bool:
        if not self.is_configured:
            return False
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
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
                f"Provider {self.provider_id} is missing an API key.",
                status_code=401,
                provider_id=self.provider_id,
            )

        model_name = model or self.default_model
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
                        f"Rate limit on {self.provider_id}: {res.text}",
                        status_code=429,
                        provider_id=self.provider_id,
                    )
                elif res.status_code in (402, 403):
                    raise ProviderQuotaExhaustedError(
                        f"Quota or credit exhausted on {self.provider_id}: {res.text}",
                        status_code=res.status_code,
                        provider_id=self.provider_id,
                    )
                else:
                    raise ProviderError(
                        f"{self.provider_id} error HTTP {res.status_code}: {res.text}",
                        status_code=res.status_code,
                        provider_id=self.provider_id,
                    )
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise ProviderError(f"Connection failed for {self.provider_id}: {e}", provider_id=self.provider_id)

    async def stream_complete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        if not self.is_configured:
            raise ProviderError(f"{self.provider_id} not configured.", provider_id=self.provider_id)

        model_name = model or self.default_model
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
                    raise ProviderRateLimitError(f"{self.provider_id} rate limit (429)", status_code=429, provider_id=self.provider_id)
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
