import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List


@dataclass
class ProviderUsage:
    provider_id: str
    total_requests: int = 0
    successful_requests: int = 0
    rate_limited_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    last_error_code: Optional[int] = None
    last_error_message: Optional[str] = None
    rate_limited_until: float = 0.0
    is_exhausted: bool = False

    @property
    def is_available(self) -> bool:
        if self.is_exhausted:
            return False
        return time.time() >= self.rate_limited_until


class QuotaTracker:
    """Monitors token usage, rate limits, and circuit breakers across providers."""

    def __init__(self, default_cooldown_seconds: int = 60):
        self.default_cooldown_seconds = default_cooldown_seconds
        self.providers: Dict[str, ProviderUsage] = {}
        self.global_fallback_events: List[Dict] = []
        self._all_cloud_exhausted: bool = False

    def get_or_create(self, provider_id: str) -> ProviderUsage:
        if provider_id not in self.providers:
            self.providers[provider_id] = ProviderUsage(provider_id=provider_id)
        return self.providers[provider_id]

    def record_success(self, provider_id: str, prompt_tokens: int = 0, completion_tokens: int = 0):
        usage = self.get_or_create(provider_id)
        usage.total_requests += 1
        usage.successful_requests += 1
        usage.prompt_tokens += prompt_tokens
        usage.completion_tokens += completion_tokens
        usage.last_error_code = None

    def record_rate_limit(
        self,
        provider_id: str,
        status_code: int = 429,
        error_msg: str = "Rate limit exceeded",
        cooldown_seconds: Optional[int] = None,
    ):
        usage = self.get_or_create(provider_id)
        usage.total_requests += 1
        usage.rate_limited_count += 1
        usage.last_error_code = status_code
        usage.last_error_message = error_msg
        
        cooldown = cooldown_seconds or self.default_cooldown_seconds
        usage.rate_limited_until = time.time() + cooldown

        if status_code in (402, 403) or "quota" in error_msg.lower():
            # Quota hard-exhausted for the billing cycle
            usage.is_exhausted = True

        event = {
            "timestamp": time.time(),
            "provider_id": provider_id,
            "status_code": status_code,
            "error_msg": error_msg,
            "cooldown_seconds": cooldown,
        }
        self.global_fallback_events.append(event)

    def is_provider_available(self, provider_id: str) -> bool:
        usage = self.get_or_create(provider_id)
        return usage.is_available

    def are_cloud_providers_exhausted(self, cloud_provider_ids: List[str]) -> bool:
        """Returns True if every configured cloud provider is currently rate limited or exhausted."""
        if not cloud_provider_ids:
            return True
        for pid in cloud_provider_ids:
            if self.is_provider_available(pid):
                return False
        return True

    def get_status_summary(self) -> Dict:
        return {
            "providers": {
                pid: {
                    "requests": u.total_requests,
                    "success": u.successful_requests,
                    "rate_limits": u.rate_limited_count,
                    "prompt_tokens": u.prompt_tokens,
                    "completion_tokens": u.completion_tokens,
                    "is_available": u.is_available,
                    "is_exhausted": u.is_exhausted,
                    "cooldown_remaining_sec": max(0, int(u.rate_limited_until - time.time())),
                    "last_error": u.last_error_message,
                }
                for pid, u in self.providers.items()
            },
            "recent_events": self.global_fallback_events[-10:],
        }
