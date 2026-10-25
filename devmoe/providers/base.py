from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, AsyncIterator


class ProviderError(Exception):
    """Base exception for provider calls."""
    def __init__(self, message: str, status_code: Optional[int] = None, provider_id: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.provider_id = provider_id


class ProviderRateLimitError(ProviderError):
    """Raised when provider returns HTTP 429 or quota limit."""
    pass


class ProviderQuotaExhaustedError(ProviderError):
    """Raised when provider credit or free tier quota is depleted (HTTP 402/403)."""
    pass


class ProviderOfflineError(ProviderError):
    """Raised when local service (e.g., Ollama) or upstream is unreachable."""
    pass


@dataclass
class UsageStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ProviderResponse:
    content: str
    model: str
    provider_id: str
    usage: UsageStats = field(default_factory=UsageStats)
    finish_reason: str = "stop"
    latency_ms: float = 0.0
    was_fallback: bool = False
    fallback_reason: Optional[str] = None


class BaseProvider(ABC):
    """Abstract interface for all LLM providers (Local and Cloud)."""

    def __init__(self, provider_id: str, is_local: bool = False, is_free_tier: bool = False):
        self.provider_id = provider_id
        self.is_local = is_local
        self.is_free_tier = is_free_tier

    @abstractmethod
    async def is_healthy(self) -> bool:
        """Check if provider endpoint is reachable and responsive."""
        pass

    @abstractmethod
    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ProviderResponse:
        """Execute chat completion synchronously or await completion."""
        pass

    @abstractmethod
    async def stream_complete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream chunks of chat completion."""
        pass
