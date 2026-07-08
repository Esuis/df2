"""Prompt templates for memory update and injection."""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from typing import Any, cast

logger = logging.getLogger(__name__)

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Prompt template for updating memory based on conversation
MEMORY_UPDATE_PROMPT = """你是一个记忆管理系统。你的任务是分析对话并更新用户的记忆档案。

当前记忆状态：
{current_memory}

新的对话：
{conversation}

{correction_hint}

输出格式（JSON）：
{{
  "memory": {{"summary": "...", "shouldUpdate": true/false}}
}}

重要规则：
- 仅当有有意义的新信息时才设置shouldUpdate=true
- 摘要应为1-2句话，简洁概括用户的关键信息
- 聚焦于对未来交互和个性化有用的信息
- 重要：不要在记忆中记录文件上传事件。上传的文件是会话特定的且临时的——它们在未来的会话中不可访问。记录上传事件会在后续对话中造成混淆。

只返回有效的JSON，不要解释或markdown。"""


# Prompt template for extracting facts from a single message
FACT_EXTRACTION_PROMPT = """Extract factual information about the user from this message.

Message:
{message}

Extract facts in this JSON format:
{{
  "facts": [
    {{ "content": "...", "category": "preference|knowledge|context|behavior|goal|correction", "confidence": 0.0-1.0 }}
  ]
}}

Categories:
- preference: User preferences (likes/dislikes, styles, tools)
- knowledge: User's expertise or knowledge areas
- context: Background context (location, job, projects)
- behavior: Behavioral patterns
- goal: User's goals or objectives
- correction: Explicit corrections or mistakes to avoid repeating

Rules:
- Only extract clear, specific facts
- Confidence should reflect certainty (explicit statement = 0.9+, implied = 0.6-0.8)
- Skip vague or temporary information

Return ONLY valid JSON."""


# Module-level tiktoken encoding cache.  Populated lazily on first use;
# subsequent calls are a dict lookup (no network I/O).  Pre-warming at
# startup via :func:`warm_tiktoken_cache` avoids blocking a request on the
# (potentially slow) first ``get_encoding`` call.
#
# A *failed* load is cached as a ``(None, monotonic_timestamp)`` tuple so that
# a network-restricted environment does not re-attempt the blocking BPE
# download on every subsequent call.  After ``_TIKTOKEN_RETRY_COOLDOWN_S`` the
# failure is allowed to expire so a transient network outage can self-heal back
# to accurate tiktoken counting without a process restart.  A load already in
# progress is cached as ``_TIKTOKEN_ENCODING_LOADING`` so concurrent callers
# fall back immediately instead of spawning more blocking
# ``tiktoken.get_encoding`` threads.  Use the ``memory.token_counting: char``
# config to skip tiktoken entirely.
_TIKTOKEN_ENCODING_MISSING = object()
_TIKTOKEN_ENCODING_LOADING = object()
# Cooldown before a *failed* tiktoken load is re-attempted. This is an internal
# tuning constant rather than a user-facing config: it only affects how quickly
# the default ``tiktoken`` mode self-heals after a transient network outage.
# Deployments that want to avoid tiktoken's network dependency entirely should
# set ``memory.token_counting: char`` instead of tuning this value.
_TIKTOKEN_RETRY_COOLDOWN_S = 600.0
_tiktoken_encoding_cache: dict[str, Any] = {}
_tiktoken_encoding_cache_lock = threading.Lock()


def _get_tiktoken_encoding(encoding_name: str = "cl100k_base") -> tiktoken.Encoding | None:
    """Return a cached tiktoken encoding, or ``None`` on failure / unavailability.

    On the very first call for a given *encoding_name*, tiktoken may need to
    download the BPE data from ``openaipublic.blob.core.windows.net``.  In
    network-restricted environments (e.g. deployments behind the GFW) this
    download can block for tens of minutes before the OS TCP timeout kicks in.
    The caller must therefore be prepared for this to block and should run it
    off the event loop (e.g. via ``asyncio.to_thread``).

    A failed load is remembered (with a timestamp) so subsequent calls fall
    back immediately to character-based estimation instead of re-triggering the
    blocking download. The failure expires after ``_TIKTOKEN_RETRY_COOLDOWN_S``
    so a transient outage can self-heal without a restart. A load already in
    progress is also remembered so that a timed-out caller does not leave a
    window where later requests start more blocking ``get_encoding`` calls.
    """
    if not TIKTOKEN_AVAILABLE:
        return None

    with _tiktoken_encoding_cache_lock:
        cached = _tiktoken_encoding_cache.get(encoding_name, _TIKTOKEN_ENCODING_MISSING)
        if cached is _TIKTOKEN_ENCODING_LOADING:
            return None
        if isinstance(cached, tuple):
            # Cached failure: (None, failed_at). Retry only after cooldown.
            _, failed_at = cached
            if time.monotonic() - failed_at < _TIKTOKEN_RETRY_COOLDOWN_S:
                return None
            cached = _TIKTOKEN_ENCODING_MISSING
        if cached is not _TIKTOKEN_ENCODING_MISSING:
            return cast("tiktoken.Encoding", cached)
        _tiktoken_encoding_cache[encoding_name] = _TIKTOKEN_ENCODING_LOADING

    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception:
        logger.warning("Failed to load tiktoken encoding %r; falling back to char-based estimation", encoding_name, exc_info=True)
        with _tiktoken_encoding_cache_lock:
            _tiktoken_encoding_cache[encoding_name] = (None, time.monotonic())
        return None

    with _tiktoken_encoding_cache_lock:
        _tiktoken_encoding_cache[encoding_name] = encoding
    return encoding


def _char_based_token_estimate(text: str) -> int:
    """Network-free token estimate that accounts for CJK density.

    The plain ``len(text) // 4`` heuristic is reasonable for English/code
    (~4 chars per token) but significantly under-estimates token counts for
    Chinese, Japanese, and Korean text, where the ratio is closer to 1.5-2
    characters per token. Counting CJK characters separately (~2 chars per
    token) avoids over-filling the injection budget for CJK-heavy memory
    content.
    """
    cjk = sum(
        1
        for ch in text
        if "\u4e00" <= ch <= "\u9fff"  # CJK Unified Ideographs
        or "\u3040" <= ch <= "\u30ff"  # Hiragana + Katakana
        or "\uac00" <= ch <= "\ud7a3"  # Hangul syllables
    )
    return (len(text) - cjk) // 4 + cjk // 2


def _count_tokens(text: str, encoding_name: str = "cl100k_base", *, use_tiktoken: bool = True) -> int:
    """Count tokens in text using tiktoken.

    Args:
        text: The text to count tokens for.
        encoding_name: The encoding to use (default: cl100k_base for GPT-4/3.5).
        use_tiktoken: When ``False``, skip tiktoken entirely and use the
            network-free character-based estimate. This guarantees no BPE
            download is attempted (see ``memory.token_counting`` config).

    Returns:
        The number of tokens in the text.
    """
    if not use_tiktoken:
        return _char_based_token_estimate(text)

    encoding = _get_tiktoken_encoding(encoding_name)
    if encoding is None:
        # Fallback to CJK-aware character estimation if tiktoken is not
        # available or the encoding failed to load.
        return _char_based_token_estimate(text)

    try:
        return len(encoding.encode(text))
    except Exception:
        # Fallback to CJK-aware character estimation on error.
        return _char_based_token_estimate(text)


def warm_tiktoken_cache() -> bool:
    """Pre-warm the tiktoken encoding cache.

    Call at startup (off the event loop) so the first request never blocks
    on the BPE download.  Returns ``True`` if the encoding was loaded
    successfully (or was already cached), ``False`` if tiktoken is
    unavailable or the download failed.
    """
    return _get_tiktoken_encoding("cl100k_base") is not None


def _coerce_confidence(value: Any, default: float = 0.0) -> float:
    """Coerce a confidence-like value to a bounded float in [0, 1].

    Non-finite values (NaN, inf, -inf) are treated as invalid and fall back
    to the default before clamping, preventing them from dominating ranking.
    The ``default`` parameter is assumed to be a finite value.
    """
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return max(0.0, min(1.0, default))
    if not math.isfinite(confidence):
        return max(0.0, min(1.0, default))
    return max(0.0, min(1.0, confidence))


def format_memory_for_injection(memory_data: dict[str, Any], max_tokens: int = 2000, *, use_tiktoken: bool = True) -> str:
    """Format memory data for injection into system prompt.

    Args:
        memory_data: The memory data dictionary.
        max_tokens: Maximum tokens to use (counted via tiktoken for accuracy).
        use_tiktoken: When ``False``, all token counting uses the network-free
            character-based estimate instead of tiktoken (see
            ``memory.token_counting`` config). Defaults to ``True``.

    Returns:
        Formatted memory string for system prompt injection.
    """
    if not memory_data:
        return ""

    mem = memory_data.get("memory")
    if not isinstance(mem, dict):
        return ""

    summary = mem.get("summary", "")
    if not isinstance(summary, str) or not summary.strip():
        return ""

    result = f"场景记忆：\n- {summary.strip()}"

    token_count = _count_tokens(result, use_tiktoken=use_tiktoken)
    if token_count > max_tokens:
        char_per_token = len(result) / token_count
        target_chars = int(max_tokens * char_per_token * 0.95)
        result = result[:target_chars] + "\n..."

    return result


def format_conversation_for_update(messages: list[Any]) -> str:
    """Format conversation messages for memory update prompt.

    Args:
        messages: List of conversation messages.

    Returns:
        Formatted conversation string.
    """
    lines = []
    for msg in messages:
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", str(msg))

        # Handle content that might be a list (multimodal)
        if isinstance(content, list):
            text_parts = []
            for p in content:
                if isinstance(p, str):
                    text_parts.append(p)
                elif isinstance(p, dict):
                    text_val = p.get("text")
                    if isinstance(text_val, str):
                        text_parts.append(text_val)
            content = " ".join(text_parts) if text_parts else str(content)

        # Strip uploaded_files tags from human messages to avoid persisting
        # ephemeral file path info into long-term memory.  Skip the turn entirely
        # when nothing remains after stripping (upload-only message).
        if role == "human":
            content = re.sub(r"<uploaded_files>[\s\S]*?</uploaded_files>\n*", "", str(content)).strip()
            if not content:
                continue

        # Truncate very long messages
        if len(str(content)) > 1000:
            content = str(content)[:1000] + "..."

        if role == "human":
            lines.append(f"User: {content}")
        elif role == "ai":
            lines.append(f"Assistant: {content}")

    return "\n\n".join(lines)
