"""Persona critic-rewriter.

Takes a draft reply, runs it through a critic LLM with the persona spec
as the rule set, returns either the original (if it passes) or a
rewritten version.

Cheap, provider-agnostic, evolvable. The critic prompt itself lives in
artifacts/critic.{version}.md so the harness can mutate it the same way
it mutates the persona.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .persona import ARTIFACTS_DIR, PersonaSpec
from .prompt_guard import sanitize_prompt_text
from .provider import ProviderSpec, call_provider, call_provider_async

DEFAULT_CRITIC_VERSION = "v1"
PASS_TOKEN = "PASS"


@dataclass(frozen=True)
class CriticSpec:
    version: str
    text: str

    @property
    def system_prompt(self) -> str:
        return self.text.strip()


@dataclass(frozen=True)
class CritiqueResult:
    final: str
    rewritten: bool
    draft: str


def load_critic(version: str = DEFAULT_CRITIC_VERSION) -> CriticSpec:
    path = ARTIFACTS_DIR / f"critic.{version}.md"
    if not path.exists():
        raise FileNotFoundError(f"Critic artifact not found: {path}")
    return CriticSpec(version=version, text=sanitize_prompt_text(path.read_text(encoding="utf-8")))


def _build_critic_user_prompt(persona: PersonaSpec, draft: str) -> str:
    return (
        "[Persona spec]\n"
        f"{persona.system_prompt}\n\n"
        "[Draft reply]\n"
        f"{draft}\n\n"
        "Apply the rules. Return PASS or the rewritten reply only."
    )


def critique(
    *,
    provider: ProviderSpec,
    persona: PersonaSpec,
    critic: CriticSpec,
    draft: str,
    temperature: float = 0.2,
    max_tokens: int = 600,
) -> CritiqueResult:
    user_prompt = _build_critic_user_prompt(persona, draft)
    response = call_provider(
        provider=provider,
        system_prompt=critic.system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return _interpret(draft, response)


async def critique_async(
    *,
    provider: ProviderSpec,
    persona: PersonaSpec,
    critic: CriticSpec,
    draft: str,
    temperature: float = 0.2,
    max_tokens: int = 600,
) -> CritiqueResult:
    user_prompt = _build_critic_user_prompt(persona, draft)
    response = await call_provider_async(
        provider=provider,
        system_prompt=critic.system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return _interpret(draft, response)


def _interpret(draft: str, response: str) -> CritiqueResult:
    cleaned = response.strip()
    if not cleaned:
        return CritiqueResult(final=draft, rewritten=False, draft=draft)
    if cleaned.upper().startswith(PASS_TOKEN) and len(cleaned) <= 8:
        return CritiqueResult(final=draft, rewritten=False, draft=draft)
    return CritiqueResult(final=cleaned, rewritten=True, draft=draft)
