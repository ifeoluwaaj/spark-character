"""Persona spec loading.

The persona is a markdown artifact under artifacts/. The harness mutates
that file to evolve voice; everything else reads from it.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .prompt_guard import sanitize_prompt_text

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
OVERLAYS_DIR = ARTIFACTS_DIR / "overlays"
LATEST_POINTER = ARTIFACTS_DIR / "persona.latest.txt"
POINTER_CHANGE_LOG = ARTIFACTS_DIR / "persona.pointer.log"
DEFAULT_PERSONA_VERSION = "v1"
PERSONA_VERSION_PATTERN = re.compile(r"^v[1-9][0-9]*$")


@dataclass(frozen=True)
class PersonaSpec:
    version: str
    text: str
    overlay_kind: str | None = None

    @property
    def system_prompt(self) -> str:
        return self.text.strip()


def load_overlay(provider_kind: str | None) -> str:
    """Return the overlay markdown for a given backend kind, or '' if
    none is configured. Provider kinds: 'zai', 'minimax', 'codex',
    'openai', 'ollama'. Unknown kinds return ''."""
    if not provider_kind:
        return ""
    path = OVERLAYS_DIR / f"{provider_kind.lower().strip()}.md"
    if not path.exists():
        return ""
    return sanitize_prompt_text(path.read_text(encoding="utf-8")).strip()


def load_surface_overlay(surface: str | None) -> str:
    """Return the overlay markdown for a given surface, or '' if none
    is configured. Surfaces: 'voice', 'browser_extension', 'telegram',
    'tui', 'cli'. Unknown surfaces return ''."""
    if not surface:
        return ""
    path = OVERLAYS_DIR / "surface" / f"{surface.lower().strip()}.md"
    if not path.exists():
        return ""
    return sanitize_prompt_text(path.read_text(encoding="utf-8")).strip()


def detect_provider_kind(provider) -> str | None:
    """Heuristic: classify a ProviderSpec or CodexSpec into a kind
    string for overlay lookup. Returns None when uncertain."""
    base = (getattr(provider, "base_url", "") or "").lower()
    if "codex-cli" in base:
        return "codex"
    if "z.ai" in base or "zhipuai" in base or "bigmodel" in base:
        return "zai"
    if "minimax" in base:
        return "minimax"
    if "openai.com" in base:
        return "openai"
    if "localhost" in base or "127.0.0.1" in base or "ollama" in base:
        return "ollama"
    return None


def validate_persona_version(version: str) -> str:
    value = str(version or "").strip()
    if not PERSONA_VERSION_PATTERN.fullmatch(value):
        raise ValueError("Persona version must match vN, for example v8.")
    return value


def protect_latest_pointer(path: Path = LATEST_POINTER) -> None:
    if path.exists():
        os.chmod(path, 0o444)


def set_latest_persona_version(
    version: str,
    *,
    actor: str = "spark-character",
    reason: str = "",
    pointer_path: Path = LATEST_POINTER,
    log_path: Path = POINTER_CHANGE_LOG,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> None:
    """Update persona.latest.txt through a validated, logged write path."""
    resolved = validate_persona_version(version)
    artifact_path = artifacts_dir / f"persona.{resolved}.md"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Persona artifact not found: {artifact_path}")

    previous = pointer_path.read_text(encoding="utf-8").strip() if pointer_path.exists() else ""
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    if pointer_path.exists():
    temp_path = pointer_path.with_name(f".{pointer_path.name}.{os.getpid()}.tmp")
    temp_path.write_text(f"{resolved}\n", encoding="utf-8")
    os.replace(temp_path, pointer_path)
    protect_latest_pointer(pointer_path)

    record = {
        "changed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "actor": actor,
        "reason": reason,
        "previous": previous or None,
        "current": resolved,
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def resolve_latest_persona_version() -> str:
    """Resolve which persona version is currently active.

    Priority:
      1. persona.latest.txt pointer (written by the evolution loop on
         promotion).
      2. Highest-numbered persona.vN.md file on disk.
      3. DEFAULT_PERSONA_VERSION as the final fallback.
    """
    if LATEST_POINTER.exists():
        text = LATEST_POINTER.read_text(encoding="utf-8").strip()
        if text:
            return validate_persona_version(text)
    versions: list[int] = []
    for path in ARTIFACTS_DIR.glob("persona.v*.md"):
        try:
            n = int(path.stem.split(".v", 1)[-1])
            versions.append(n)
        except ValueError:
            continue
    if versions:
        return f"v{max(versions)}"
    return DEFAULT_PERSONA_VERSION


def load_persona(
    version: str | None = None,
    *,
    provider_kind: str | None = None,
    surface: str | None = None,
) -> PersonaSpec:
    """Load a persona artifact with optional provider and surface overlays.

    Composition order:
      1. Base persona (chip-rendered or flat MD).
      2. Provider overlay if provider_kind is given (artifacts/overlays/<kind>.md).
      3. Surface overlay if surface is given (artifacts/overlays/surface/<surface>.md).

    Provider overlays target backend-specific drift (zai chatty, minimax
    helper register, codex hallucinated context).

    Surface overlays target surface-specific format constraints: voice
    needs short declarative sentences with no markdown, browser_extension
    needs short replies that fit a popup, etc.

    Both axes compose orthogonally. The full chain reaches the model
    in a single system prompt.
    """
    resolved = version or resolve_latest_persona_version()
    path = ARTIFACTS_DIR / f"persona.{resolved}.md"
    if not path.exists():
        raise FileNotFoundError(f"Persona artifact not found: {path}")
    base_text = sanitize_prompt_text(path.read_text(encoding="utf-8"))
    parts = [base_text.rstrip()]
    overlay_text = load_overlay(provider_kind)
    if overlay_text:
        parts.append(overlay_text)
    surface_text = load_surface_overlay(surface)
    if surface_text:
        parts.append(surface_text)
    combined = "\n\n---\n\n".join(parts)
    return PersonaSpec(
        version=resolved,
        text=combined,
        overlay_kind=provider_kind if overlay_text else None,
    )


def load_persona_from_path(path: str | Path) -> PersonaSpec:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Persona artifact not found: {p}")
    version = p.stem.split(".", 1)[-1] if "." in p.stem else "custom"
    return PersonaSpec(version=version, text=sanitize_prompt_text(p.read_text(encoding="utf-8")))
