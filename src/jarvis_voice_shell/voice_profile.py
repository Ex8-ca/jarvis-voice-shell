"""Voice-session helpers for concise, speakable JARVIS output.

These helpers are intentionally local to the voice shell. They do not alter the
main Hermes profile; they only wrap prompts sent from voice mode and clean text
before it is handed to TTS.
"""

from __future__ import annotations

import re


VOICE_MODE_PREFIX = """
You are answering through a spoken JARVIS voice session.
Voice rules:
- Be brief: one to three short spoken sentences unless the user asks for detail.
- Use natural speech only. Do not use markdown, bullets, tables, JSON, code blocks, raw logs, or stack traces.
- Do not read punctuation, quotes, slashes, backticks, brackets, file paths, or command syntax aloud.
- Summarize tool results and failures in plain English.
- If a tool/model/action failed, say so plainly and briefly.
- Keep the main Hermes personality and memory, but format this turn for speech.

User said:
""".strip()

_STATUS_PATTERNS = (
    "status",
    "are you online",
    "are you still there",
    "you alive",
    "are we online",
    "system status",
)

_SYMBOL_REPLACEMENTS = {
    "&": " and ",
    "@": " at ",
    "%": " percent ",
    "+": " plus ",
    "=": " equals ",
}


def wrap_for_voice(transcript: str) -> str:
    """Wrap a user transcript with voice-only response instructions."""
    return f"{VOICE_MODE_PREFIX}\n{transcript.strip()}"


def is_status_query(transcript: str) -> bool:
    """Return True when the user is asking whether the voice session is alive."""
    text = re.sub(r"[^a-z0-9 ]+", " ", transcript.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return any(pattern in text for pattern in _STATUS_PATTERNS)


def sanitize_for_speech(text: str, *, max_chars: int = 700) -> str:
    """Convert model output into text suitable for TTS.

    Keeps a readable console answer possible upstream, but strips markdown,
    code/log noise, URLs/paths, and punctuation that TTS engines tend to read
    unnaturally.
    """
    if not text:
        return ""

    cleaned = text.strip()

    # Remove fenced code blocks entirely; a spoken summary should not read code.
    cleaned = re.sub(r"```.*?```", " ", cleaned, flags=re.DOTALL)
    # Drop inline code markers but keep simple contents if they are words.
    cleaned = cleaned.replace("`", "")
    # Remove markdown emphasis/headings/list markers.
    cleaned = re.sub(r"^\s{0,3}#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"[*_~]{1,3}", "", cleaned)
    # Replace common paths and URLs with a natural placeholder.
    cleaned = re.sub(r"https?://\S+", " the link ", cleaned)
    cleaned = re.sub(r"[A-Za-z]:[\\/][^\s,;]+", " the file path ", cleaned)
    cleaned = re.sub(r"(?:[./~]?[^\s,;]+[\\/]){1,}[^\s,;]+", " the file path ", cleaned)
    # Remove table separators and quote/bracket punctuation.
    cleaned = cleaned.replace("|", " ")
    cleaned = cleaned.translate(str.maketrans({
        "\"": " ", "'": " ", "“": " ", "”": " ", "‘": " ", "’": " ",
        "*": " ",
        "(": " ", ")": " ", "[": " ", "]": " ", "{": " ", "}": " ",
        "/": " ", "\\": " ", "-": " ", "—": " ", "–": " ", ":": " ", ";": " ",
    }))
    for symbol, replacement in _SYMBOL_REPLACEMENTS.items():
        cleaned = cleaned.replace(symbol, replacement)
    # Collapse repeated punctuation/whitespace.
    cleaned = re.sub(r"[<>#]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if max_chars and len(cleaned) > max_chars:
        clipped = cleaned[:max_chars].rsplit(" ", 1)[0].strip()
        cleaned = f"{clipped}. I have more detail on screen if needed."

    return cleaned


def spoken_failure_message(error: Exception | str) -> str:
    """Short, non-technical failure alert for voice mode."""
    text = str(error).lower()
    if "timeout" in text or "timed out" in text:
        return "Model timeout, sir. I am still listening."
    if "hermes" in text or "bridge" in text:
        return "Hermes bridge failure, sir. I am still listening."
    if "audio" in text or "device" in text:
        return "Audio device failure, sir."
    return "Tool failure, sir. I am still listening."
