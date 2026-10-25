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


class AnthropicProvider(BaseProvider):
    """Anthropic Claude API Provider (Claude 3.5 Sonnet / Claude 3 Opus)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.anthropic.com/v1",
        timeout_seconds: float = 60.0,
    ):
        super().__init__(provider_id="anthropic_claude", is_local=False, is_free_tier=False)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_seconds

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 5)

    async def is_healthy(self) -> bool:
        return self.is_configured

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
                "Anthropic API key is not configured. Set ANTHROPIC_API_KEY in .env.",
                status_code=401,
                provider_id=self.provider_id,
            )

        model_name = model or "claude-3-5-sonnet-20241022"
        start_time = time.time()

        # Separate system messages from user/assistant messages
        system_content = ""
        claude_messages: List[Dict[str, str]] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                system_content += f"\n{content}"
            else:
                claude_messages.append({"role": role, "content": content})

        if not claude_messages:
            claude_messages = [{"role": "user", "content": "Hello"}]

        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": claude_messages,
            "max_tokens": max_tokens or 4096,
            "temperature": temperature,
        }
        if system_content.strip():
            payload["system"] = system_content.strip()

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(
                    f"{self.base_url}/messages",
                    json=payload,
                    headers=headers,
                )

                if res.status_code == 200:
                    data = res.json()
                    content_blocks = data.get("content", [])
                    content_text = "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
                    usage_data = data.get("usage", {})
                    latency = round((time.time() - start_time) * 1000, 2)
                    return ProviderResponse(
                        content=content_text,
                        model=model_name,
                        provider_id=self.provider_id,
                        usage=UsageStats(
                            prompt_tokens=usage_data.get("input_tokens", 0),
                            completion_tokens=usage_data.get("output_tokens", 0),
                            total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),
                        ),
                        latency_ms=latency,
                    )
                elif res.status_code == 429:
                    raise ProviderRateLimitError(
                        f"Anthropic rate limit exceeded: {res.text}",
                        status_code=429,
                        provider_id=self.provider_id,
                    )
                elif res.status_code in (400, 402, 403) and "credit" in res.text.lower():
                    raise ProviderQuotaExhaustedError(
                        f"Anthropic credit exhausted: {res.text}",
                        status_code=res.status_code,
                        provider_id=self.provider_id,
                    )
                else:
                    raise ProviderError(
                        f"Anthropic error HTTP {res.status_code}: {res.text}",
                        status_code=res.status_code,
                        provider_id=self.provider_id,
                    )
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise ProviderError(f"Connection to Anthropic failed: {e}", provider_id=self.provider_id)

    async def stream_complete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        # For simplicity and robustness, execute complete and stream text
        res = await self.complete(messages, model=model, temperature=temperature, max_tokens=max_tokens, **kwargs)
        yield res.content
