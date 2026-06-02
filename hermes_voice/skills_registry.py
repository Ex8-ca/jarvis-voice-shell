"""
Skill registry scanner for the hermes-voice plugin.

Discovers skills installed under `~/.hermes/skills/` and injects a compact
manifest into the LLM's system prompt so it knows:

  1. Which "wired" skills it can call directly via `[[TOOL:name ...]]`
  2. Which "registered" skills exist on disk but aren't wired to the voice
     gateway (and so the LLM should acknowledge their existence honestly
     instead of hallucinating a tool call)

We deliberately do NOT auto-execute registered skill scripts from the
voice gateway. That's a security hole (arbitrary code from a directory
that the LLM can influence). Adding wired support for a new skill
means writing a `hermes_voice/skills/<name>.py` wrapper.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger("hermes-voice.skills_registry")

# Roots we scan. openclaw-imports is checked first so its versions shadow
# any duplicates in the native skills/ tree.
SKILL_ROOTS: Tuple[Path, ...] = (
    Path.home() / ".hermes" / "skills" / "openclaw-imports",
    Path.home() / ".hermes" / "skills",
)

# Skip these noisy paths.
SKIP_PATH_PARTS = (
    "node_modules",
    ".git",
    "__pycache__",
)


@dataclass(frozen=True)
class SkillEntry:
    """One discovered skill."""
    name: str            # the skill's `name` from frontmatter (basename fallback)
    category: str        # the parent category dir, or "openclaw-imports" / "uncategorized"
    description: str     # one-line description (first sentence, max ~120 chars)
    source: str          # "openclaw-imports" | "native"
    wired: bool = False  # True if a wired Tool subclass exists in hermes_voice.skills

    def compact(self) -> str:
        """One-line, prompt-friendly representation."""
        # Strip multi-line, collapse whitespace
        d = " ".join(self.description.split())
        if len(d) > 120:
            d = d[:117] + "..."
        prefix = "★" if self.wired else " "
        return f"  {prefix} {self.name}: {d}"


# ── Frontmatter parsing ─────────────────────────────────────────────────────
# Lightweight: we don't need full YAML, just the first `---` block.

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_frontmatter(text: str) -> dict:
    """Return the first YAML-ish frontmatter block as a dict. Tolerant of
    `description: >-` block scalars, lists, and multi-line values. We only
    need scalar fields for the registry (name, category, description)."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}

    block = m.group(1)
    out: dict = {}
    current_key: Optional[str] = None
    current_value: List[str] = []

    def _flush():
        nonlocal current_key, current_value
        if current_key is not None:
            val = " ".join(s.strip() for s in current_value if s.strip())
            if val:
                out[current_key] = val
        current_key = None
        current_value = []

    for raw_line in block.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            if current_key:
                current_value.append("")
            continue
        # Indented continuation of a block scalar
        if current_key and (line.startswith("  ") or line.startswith("\t")):
            current_value.append(line.strip())
            continue
        # New key
        _flush()
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        current_key = key
        if value in (">", ">-", "|", "|-"):
            # Block scalar — next lines are the value
            current_value = []
        else:
            # Strip surrounding quotes if any
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]
            current_value = [value]

    _flush()
    return out


def _first_sentence(text: str, max_len: int = 200) -> str:
    """Pull the first sentence of a description. Falls back to first N chars."""
    text = " ".join(text.split())
    if not text:
        return ""
    # Take first sentence (split on . ! ? followed by space/end)
    m = re.match(r"(.+?[.!?])(?:\s|$)", text)
    if m:
        out = m.group(1).strip()
    else:
        out = text
    if len(out) > max_len:
        out = out[: max_len - 1].rstrip() + "..."
    return out


# ── Discovery ───────────────────────────────────────────────────────────────

def discover_skills() -> List[SkillEntry]:
    """Walk SKILL_ROOTS, return all discovered skills, deduped by name
    (openclaw-imports wins on collision)."""
    seen: dict[str, SkillEntry] = {}

    for root in SKILL_ROOTS:
        if not root.exists():
            continue

        # Source label
        source = "openclaw-imports" if "openclaw-imports" in root.parts else "native"

        for skill_md in root.rglob("SKILL.md"):
            # Skip node_modules / .git / __pycache__
            if any(p in SKIP_PATH_PARTS for p in skill_md.parts):
                continue

            try:
                text = skill_md.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                logger.debug(f"Skipping {skill_md}: {e}")
                continue

            fm = _parse_frontmatter(text)
            name = fm.get("name") or skill_md.parent.name
            # Skip if the name is something weird (starts with dot, contains spaces)
            if not name or not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$", name):
                continue

            # Category: explicit in frontmatter, else from path (parent dir name)
            category = fm.get("category", "").strip()
            if not category:
                # /home/marc/.hermes/skills/devops/local-stt-tts/SKILL.md → "devops"
                rel = skill_md.parent.relative_to(root)
                if len(rel.parts) >= 2:
                    category = rel.parts[0]
                elif source == "openclaw-imports":
                    category = "openclaw-imports"
                else:
                    category = "uncategorized"

            description = _first_sentence(fm.get("description", ""))

            entry = SkillEntry(
                name=name,
                category=category,
                description=description,
                source=source,
            )

            # First-wins: openclaw-imports is in SKILL_ROOTS first, so it
            # shadows native duplicates.
            if name not in seen:
                seen[name] = entry

    return list(seen.values())


# Wired-skills set: names of Tool subclasses registered in hermes_voice.skills.
# We compute this lazily so we don't trigger an import cycle.

def _wired_skill_names() -> set[str]:
    """Return the set of skill names that have a wired Tool wrapper.

    We pull from the live REGISTRY (the actual source of truth) rather than
    a list constant — that way the wired check is always in sync with
    whatever tools self-registered at import time.
    """
    try:
        from hermes_voice.tools import REGISTRY  # type: ignore
        return set(REGISTRY.names())
    except ImportError:
        return set()


def mark_wired(entries: List[SkillEntry]) -> List[SkillEntry]:
    """Return a new list with `wired` flagged for entries that have a Tool wrapper."""
    wired = _wired_skill_names()
    return [
        SkillEntry(
            name=e.name, category=e.category, description=e.description,
            source=e.source, wired=(e.name in wired),
        )
        for e in entries
    ]


# ── Prompt rendering ───────────────────────────────────────────────────────

def render_for_prompt(
    entries: Optional[List[SkillEntry]] = None,
    show_unwired_descriptions: bool = False,
) -> str:
    """Build the system-prompt section. Compact, categorized, ≤ ~12KB.

    By default, wired skills show full descriptions (the LLM needs to know
    when to call them), but unwired skills are name-only (the LLM only needs
    to know the name exists so it doesn't hallucinate). Set
    `show_unwired_descriptions=True` to get the full ~46KB block (useful
    for testing or for prompts that aren't voice-cost-sensitive).
    """
    if entries is None:
        entries = mark_wired(discover_skills())

    if not entries:
        return ""

    # Bucket by category
    by_cat: dict[str, list[SkillEntry]] = {}
    for e in entries:
        by_cat.setdefault(e.category, []).append(e)

    # Stable order: wired skills first within each category, then alpha
    for cat in by_cat:
        by_cat[cat].sort(key=lambda e: (not e.wired, e.name.lower()))

    # Stable category order: by entry count desc, then alpha
    cat_order = sorted(by_cat.keys(), key=lambda c: (-len(by_cat[c]), c.lower()))

    lines: list[str] = [
        "\n\n# Available Skills",
        "",
        "Two kinds of skills are available:",
        "",
        "- ★ WIRED skills can be invoked directly with `[[TOOL:name arg=\"value\"...]]`. "
        "The gateway runs them and feeds the result back to you as a follow-up turn.",
        "- Unmarked (registered) skills exist on disk but are NOT wired into the voice gateway. "
        "If a user asks for one, tell them honestly that the skill exists but isn't reachable "
        "from voice yet, and suggest they enable it. Do NOT invent a tool call for an unwired skill.",
        "",
    ]

    for cat in cat_order:
        items = by_cat[cat]
        lines.append(f"## {cat} ({len(items)})")
        for e in items:
            if e.wired or show_unwired_descriptions:
                lines.append(e.compact())
            else:
                # Name only, no description — saves ~75% of the block size
                marker = "★" if e.wired else " "
                lines.append(f"  {marker} {e.name}")
        lines.append("")

    total_wired = sum(1 for e in entries if e.wired)
    lines.append(
        f"---\nTotal: {len(entries)} skills registered across {len(by_cat)} categories; "
        f"{total_wired} wired for direct invocation."
    )

    block = "\n".join(lines)
    logger.debug(
        f"Skill registry block: {len(block)} chars, {len(entries)} entries, "
        f"{total_wired} wired, full_descriptions={show_unwired_descriptions}"
    )
    return block


# ── CLI smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
    entries = mark_wired(discover_skills())
    block = render_for_prompt(entries)
    print(block)
    print(f"\n[stdin: {len(block)} chars]", file=sys.stderr)
