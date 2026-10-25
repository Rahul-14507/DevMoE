"""Live model discovery and ranking.

Nothing here knows specific model names. Providers are asked what they serve
(`GET /models`), non-chat models are filtered out by capability keywords, and
the remainder is ranked by tier (fast / strong) using version and size hints
parsed from the model id itself.
"""
import re
from typing import List

NON_CHAT_KEYWORDS = (
    "tts", "image", "embedding", "embed", "lyria", "robotics", "deep-research",
    "computer-use", "antigravity", "aqa", "transcribe", "whisper", "orpheus",
    "guard", "safeguard", "live", "audio", "native-audio", "translate",
    "veo", "imagen", "omni", "nano", "customtools", "moderation", "dall-e",
)


def is_chat_model(model_id: str) -> bool:
    mid = model_id.lower()
    return not any(k in mid for k in NON_CHAT_KEYWORDS)


def _version(model_id: str) -> float:
    """Highest dotted version number in the id, e.g. 'gemini-3.5-flash' -> 3.5."""
    nums = re.findall(r"(?<![\d.])(\d+(?:\.\d+)?)(?![\d.]*b\b)", model_id.lower())
    vals = []
    for n in nums:
        try:
            vals.append(float(n))
        except ValueError:
            pass
    return max(vals) if vals else 0.0


def _size_b(model_id: str) -> float:
    """Parameter count in billions if the id advertises one, e.g. '27b' or 'a4b'."""
    m = re.findall(r"(\d+(?:\.\d+)?)b\b", model_id.lower())
    return max((float(x) for x in m), default=0.0)


def _is_preview_or_alias(model_id: str) -> bool:
    mid = model_id.lower()
    return "preview" in mid or mid.endswith("-latest") or "exp" in mid


def rank_models(model_ids: List[str], tier: str) -> List[str]:
    """Return chat models best-first for a tier.

    tier == "fast":   prefer lite/flash/small models, newest first
    tier == "strong": prefer pro/large models, newest first
    Stable ids are preferred over previews/aliases when otherwise equal.
    """
    chat = [m.replace("models/", "") for m in model_ids if is_chat_model(m)]

    def fast_key(m: str):
        mid = m.lower()
        speed = 2 if "lite" in mid else 1 if ("flash" in mid or "mini" in mid or "instant" in mid) else 0
        return (speed, not _is_preview_or_alias(m), _version(m), -_size_b(m))

    def strong_key(m: str):
        mid = m.lower()
        power = 2 if "pro" in mid or "opus" in mid else 1 if ("flash" in mid and "lite" not in mid) else 0
        return (power, _size_b(m), not _is_preview_or_alias(m), _version(m))

    return sorted(chat, key=fast_key if tier == "fast" else strong_key, reverse=True)
