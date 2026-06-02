"""
Voice persona loader for the hermes-voice plugin.

The voice LLM gets a different system prompt than text chat. Text chat uses
the full Hermes context (memex8, skills, tools, conversation history). Voice
uses a tight, conversation-focused prompt so the LLM responds in 1-3 seconds.

Resolution order:
1. `HERMES_VOICE_PROMPT_FILE` env var (path to a custom prompt file)
2. `~/.hermes/VOICE.md` (the recommended location for hermes-voice users)
3. `~/.hermes/SOUL.md` (back-compat with Hermes Voice Shell)
4. Generic Hermes persona (under 100 tokens)

Optionally tacks on:
- `~/.hermes/USER.md` (user context — name, preferences, projects)
- Recent voice memory (last N turns from `voice_memory.md`)
- Most recent memex8 memories (if memex8 is available)

The result is cached in module state so we only read the file once per process.
"""
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger("hermes-voice.persona")

# We import memory lazily to avoid an import cycle (memory doesn't depend on persona,
# but persona reads memory at request time, and tests reload both modules).
_memory_module = None

def _get_memory():
    global _memory_module
    if _memory_module is None:
        from hermes_voice import memory
        _memory_module = memory
    return _memory_module

_VOICE_PROMPT_CACHE: Optional[str] = None
_VOICE_PROMPT_CACHE_VERSION = (-1, -1)  # (mtime, size) for cache invalidation when memory changes


def _load_voice_prompt() -> str:
    """Return the voice system prompt (cached after first call, invalidated
    on memory change via mtime + file size)."""
    global _VOICE_PROMPT_CACHE, _VOICE_PROMPT_CACHE_VERSION
    memory = _get_memory()

    # Bump version when the memory file changes (mtime + size, since atomic
    # rename can preserve mtime across writes)
    memory_file = memory.VOICE_MEMORY_FILE
    try:
        if memory_file.exists():
            stat = memory_file.stat()
            current_version = (int(stat.st_mtime), stat.st_size)
        else:
            current_version = (0, 0)
    except OSError:
        current_version = (0, 0)

    if _VOICE_PROMPT_CACHE is not None and _VOICE_PROMPT_CACHE_VERSION == current_version:
        return _VOICE_PROMPT_CACHE

    # Option 1: explicit env var override
    prompt_file = os.getenv("HERMES_VOICE_PROMPT_FILE", "")
    if prompt_file and Path(prompt_file).exists():
        _VOICE_PROMPT_CACHE = Path(prompt_file).read_text(encoding="utf-8")
        _VOICE_PROMPT_CACHE_VERSION = current_version
        logger.info(f"Voice prompt loaded from {prompt_file} ({len(_VOICE_PROMPT_CACHE)} chars)")
        return _VOICE_PROMPT_CACHE

    # Option 2: ~/.hermes/VOICE.md (recommended) — single source of truth for voice
    # Option 3: ~/.hermes/SOUL.md (back-compat fallback only — used if VOICE.md is absent)
    parts = []
    voice_md = Path.home() / ".hermes" / "VOICE.md"
    soul_md = Path.home() / ".hermes" / "SOUL.md"
    if voice_md.exists():
        # VOICE.md is authoritative — don't double up with SOUL.md
        parts.append(voice_md.read_text(encoding="utf-8"))
        logger.info(f"Voice prompt: using {voice_md} (authoritative)")
    elif soul_md.exists():
        parts.append(soul_md.read_text(encoding="utf-8"))
        logger.info(f"Voice prompt: using {soul_md} (no VOICE.md found, falling back)")

    # USER.md (optional user context — always additive)
    user_md = Path.home() / ".hermes" / "USER.md"
    if user_md.exists():
        parts.append("\n\n# User Context\n" + user_md.read_text(encoding="utf-8"))
        logger.info(f"Voice prompt: using {user_md}")

    # Recent voice memory (last N turns, for conversational continuity)
    memory_block = memory.recent_as_prompt()
    if memory_block:
        parts.append("\n\n" + memory_block)
        logger.info(f"Voice prompt: included {len(memory.recent())} recent memory entries")

    # memex8 recall (optional, non-fatal if memex8 unavailable)
    memex_block = _try_memex8_recall()
    if memex_block:
        parts.append("\n\n# Recent Memory (from memex8)\n" + memex_block)
        logger.info("Voice prompt: included memex8 recall")

    # Skill registry — LAZY LOADED. The voice LLM gets a tiny stub here so
    # it knows the registry exists, but the full ~11K manifest is NOT
    # included in the system prompt. It gets injected on first tool call
    # in `_run_tool_loop` (gated by a once-per-process flag), so most
    # conversational turns never pay the cost.
    try:
        from hermes_voice.skills_registry import has_cached_registry
        if not has_cached_registry():
            parts.append(
                "\n\n# Skills (lazy-loaded)\n\n"
                "Wired tools are listed under `## Tools` below — call them with\n"
                "`[[TOOL:name arg=\"value\"...]]` and the gateway will run them.\n\n"
                "For broader awareness of the hundreds of registered skills installed\n"
                "on this system, the gateway will load a manifest automatically the\n"
                "first time you invoke a tool. You don't need to do anything; the\n"
                "follow-up turn will see the full list. The first tool call is the\n"
                "only one that has a slightly longer filler phrase (\"one second\")."
            )
    except Exception:
        logger.exception("Voice prompt: skills stub failed (non-fatal)")

    from hermes_voice.naming import get_assistant_name
    name = get_assistant_name()

    if parts:
        _VOICE_PROMPT_CACHE = (
            "\n\n".join(parts)
            + f"\n\n---\nYou are {name}, a concise voice assistant. Keep responses SHORT — under 30 words. "
            "Conversational, direct, no filler. You are speaking aloud, not typing. "
            "No markdown, no bullet points, no lists. Plain spoken sentences only."
        )
    else:
        # Option 4: generic persona (last-resort fallback)
        _VOICE_PROMPT_CACHE = (
            f"You are {name}, a concise voice assistant. Keep responses under 25 words. "
            "Speak conversationally, as if out loud. No markdown, no lists, no filler phrases. "
            "Be direct and helpful."
        )
    _VOICE_PROMPT_CACHE_VERSION = current_version
    logger.info(f"Voice prompt total: {len(_VOICE_PROMPT_CACHE)} chars")
    return _VOICE_PROMPT_CACHE


def _try_memex8_recall(limit: int = 5) -> str:
    """Try to pull the N most recent memex8 memories. Returns empty string on failure."""
    try:
        # Memex8 is exposed via Hermes' own memory system, not a direct import.
        # The exact API is still being designed (Phase 4 of the refactor).
        # For now, return empty — the gateway works fine without it.
        return ""
    except Exception:
        return ""


def reload() -> None:
    """Clear the cache so the next call re-reads the file (useful for tests)."""
    global _VOICE_PROMPT_CACHE, _VOICE_PROMPT_CACHE_VERSION
    _VOICE_PROMPT_CACHE = None
    _VOICE_PROMPT_CACHE_VERSION = (-1, -1)
