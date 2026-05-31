"""Public-sharing safety scan for this repository."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SKIP_DIRS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "__pycache__",
    "build",
    "dist",
    ".venv",
    "venv",
    "env",
    "htmlcov",
}

SKIP_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".wav",
    ".mp3",
    ".m4a",
    ".ogg",
    ".flac",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
}

ALLOW_EMPTY_SECRET = re.compile(r"^\s*(API_SERVER_KEY|OPENAI_API_KEY)\s*=\s*$")

SAFE_WINDOWS_USERS = (
    "All Users",
    "Default",
    "Default User",
    "Public",
    "Example",
    "example-user",
    "YourName",
    "YourUser",
    "username",
)

RULES = [
    (
        "personal Windows user path",
        re.compile(
            r"C:\\Users\\(?!"
            + "|".join(re.escape(user) for user in SAFE_WINDOWS_USERS)
            + r")([^\\\s]+)",
            re.I,
        ),
    ),
    (
        "MSYS personal user path",
        re.compile(
            r"/c/Users/(?!"
            + "|".join(re.escape(user) for user in SAFE_WINDOWS_USERS)
            + r")([^/\s]+)",
            re.I,
        ),
    ),
    ("hardcoded local gateway demo key", re.compile(r"jarvis-local-voice", re.I)),
    (
        "machine-specific CLI audio device",
        re.compile(r"--(?:input|output)-device\s+\d+", re.I),
    ),
    (
        "committed private key block",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PRIVATE )?KEY-----"),
    ),
]

SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|secret|token|password|passwd)\b"
    r"\s*[=:]\s*['\"]?([^'\"\s#]+)"
)

PLACEHOLDER_VALUES = {
    "",
    "your-local-gateway-key",
    "your-local-dev-key",
    "example",
    "placeholder",
    "changeme",
    "change-me",
    "dummy",
    "test",
    "voice-key",
    "sk-test",
}

SAFE_SECRET_CODE_FRAGMENTS = (
    "os.environ.get",
    "getattr(",
    "_read_env_key",
    "api_key=api_key",
)


def iter_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if path.is_dir():
            continue
        rel_parts = set(path.relative_to(ROOT).parts)
        if rel_parts & SKIP_DIRS:
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        files.append(path)
    return files


def is_binary(path: Path) -> bool:
    try:
        chunk = path.read_bytes()[:2048]
    except OSError:
        return True
    return b"\x00" in chunk


def scan_line(rel: Path, lineno: int, line: str) -> list[str]:
    findings: list[str] = []
    if ALLOW_EMPTY_SECRET.match(line):
        return findings

    for label, regex in RULES:
        if regex.search(line):
            findings.append(f"{rel}:{lineno}: {label}: {line.strip()[:160]}")

    match = SECRET_ASSIGNMENT.search(line)
    if match:
        if any(fragment in line for fragment in SAFE_SECRET_CODE_FRAGMENTS):
            return findings
        value = match.group(2).strip().strip('"\'')
        if value.lower() not in PLACEHOLDER_VALUES and len(value) >= 6:
            findings.append(
                f"{rel}:{lineno}: possible hardcoded secret assignment: "
                f"{line.strip()[:160]}"
            )
    return findings


def main() -> int:
    findings: list[str] = []

    for path in iter_files():
        rel = path.relative_to(ROOT)
        if rel.as_posix() == "scripts/public_safety_scan.py":
            continue
        if path.name == "nul":
            findings.append(f"{rel}: generated Windows nul file should not be committed")
            continue
        if is_binary(path):
            findings.append(f"{rel}: binary file should not be committed unless intentional")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(f"{rel}: non-UTF8 text/binary file")
            continue

        for lineno, line in enumerate(text.splitlines(), start=1):
            findings.extend(scan_line(rel, lineno, line))

    if findings:
        print("PUBLIC SAFETY SCAN FAILED")
        for item in findings:
            print("-", item)
        return 1

    print("PUBLIC SAFETY SCAN PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
