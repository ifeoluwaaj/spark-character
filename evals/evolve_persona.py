"""Multi-tier persona evolution.

Like evals/evolve.py, but the fitness function combines T1 + T2 + T3
instead of T1 alone. Targets the axis where the most real headroom
lives (currently T2 distinctiveness at ~0.78).

Workflow per cycle:
  1. Score baseline persona on T1 (8 prompts) + T2 (judge per reply) +
     T3 (5 sycophancy/curiosity/care/identity/honesty/initiative
     probes from src/spark_character/probes.py).
  2. Compute composite = w1*T1 + w2*T2 + w3*T3 (default 0.2/0.5/0.3).
  3. Diagnose weaknesses across all three tiers.
  4. Generate N candidate persona mutations targeting those weaknesses.
  5. Score each candidate the same way.
  6. Promote the best candidate to persona.v(N+1).md only if its
     composite strictly beats the baseline and no individual tier
     regresses by more than tier_floor_drop.

Usage:
  python evals/evolve_persona.py
  python evals/evolve_persona.py --candidates 4 --weights 0.2,0.5,0.3
  python evals/evolve_persona.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean as mean_

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from spark_character import (  # noqa: E402
    PROBES,
    T6_EMOTIONAL_ATTUNEMENT_PROBES,
    T7_MEMORY_COHERENCE_PROBES,
    T8_INITIATIVE_PROBES,
    AuditMiner,
    PersonaSpec,
    ProviderSpec,
    attach_chip_context,
    call_provider,
    generate,
    run_deep_probe,
    run_probe,
    score_distinctiveness,
    score_persona,
)
from spark_character.persona import ARTIFACTS_DIR, set_latest_persona_version  # noqa: E402

PROMPTS = [
    "List three things I should focus on as a founder this week.",
    "Should I prioritize fundraising or shipping the product first?",
    "The web search just failed. What happened?",
    "What can you actually help me with right now?",
    "Yeah let's keep going on the launch plan from earlier.",
    "hey",
    "where are we",
    "Should I raise now or wait six months?",
    "What does TVL mean in DeFi?",
    "I'm anxious about the launch tomorrow.",
    "Quick gut check: ship the redesign or hold a week?",
]

MUTATOR_SYSTEM = (
    "You are a persona spec evolution engine for the Spark agent. "
    "Given a baseline persona spec written in markdown, produce a single "
    "improved variant that addresses the listed weaknesses across the "
    "T1 (mechanics), T2 (voice signature), and T3 (behavioral traits) "
    "axes. Keep the same structure (a one-line title, then a short "
    "intro, then a 'Voice rules' bulleted list). Stay under 1700 "
    "characters.\n\n"
    "Hard constraints in every variant: never use em dashes. Never "
    "tell the agent to use em dashes. Never name internal subsystems. "
    "The first sentence rule must remain. The no-greeting-reset rule "
    "must remain. The no-plumbing-leak rule must remain.\n\n"
    "Important: T2 distinctiveness is the headroom axis. The variant "
    "should tighten what makes Spark distinctive: lead with a real "
    "call when evidence is good enough, push back when warranted, "
    "admit limits cleanly, engage with the user's actual emotional "
    "state when present, ask one specific curious follow-up when "
    "warranted. Cut hedging. Cut filler. Add spine.\n\n"
    "Output the markdown spec only. No preamble, no explanation, no "
    "code fences."
)


def _format_score_line(label: str, scores: dict, comp: float, elapsed: float | None = None) -> str:
    parts = [
        f"T1={scores['t1_mean']}",
        f"T2={scores['t2_mean']}",
        f"T3={scores['t3_mean']}",
    ]
    if scores.get("deeper_included"):
        parts.extend([
            f"T6={scores['t6_mean']}",
            f"T7={scores['t7_mean']}",
            f"T8={scores['t8_mean']}",
        ])
    parts.append(f"composite={comp}")
    suffix = f" in {elapsed:.1f}s" if elapsed is not None else ""
    return f"[{label}] {' '.join(parts)}{suffix}\n"


def find_latest_persona() -> tuple[int, PersonaSpec]:
    versions = []
    for path in ARTIFACTS_DIR.glob("persona.v*.md"):
        match = re.match(r"persona\.v(\d+)\.md", path.name)
        if match:
            versions.append((int(match.group(1)), path))
    if not versions:
        raise FileNotFoundError(
            f"No persona.vN.md artifacts found in {ARTIFACTS_DIR}. "
            f"Seed persona.v1.md in that directory before running evolve_persona, "
            f"or override the location via ARTIFACTS_DIR before launching."
        )
    versions.sort(key=lambda x: x[0])
    n, path = versions[-1]
    text = path.read_text(encoding="utf-8")
    return n, PersonaSpec(version=f"v{n}", text=text)


def score_all_tiers(
    provider: ProviderSpec,
    persona: PersonaSpec,
    *,
    include_deeper: bool = False,
    chip_load: list[str] | None = None,
) -> dict:
    """Run T1+T2 over PROMPTS and T3 over PROBES. Returns aggregated dict.

    When include_deeper=True, also runs T6 emotional attunement,
    T7 memory coherence, and T8 initiative probes. ~3x more LLM
    calls per persona scored.

    When chip_load is a list of chip keys (e.g. ["xcontent", "startup-yc"]),
    each prompt is wrapped with that chip context before generation, so
    candidates are scored on the path users actually experience in
    production rather than the bare-LLM path.
    """
    t1_means: list[float] = []
    t2_scores: list[float] = []
    t3_scores: list[float] = []
    t6_scores: list[float] = []
    t7_scores: list[float] = []
    t8_scores: list[float] = []
    failures_t1: list[str] = []
    failures_t2: list[tuple[str, float, str]] = []
    failures_t3: list[tuple[str, float]] = []
    failures_t6: list[tuple[str, float]] = []
    failures_t7: list[tuple[str, float]] = []
    failures_t8: list[tuple[str, float]] = []

    chip_keys = list(chip_load or [])
    for prompt in PROMPTS:
        try:
            wrapped = attach_chip_context(prompt, chip_keys) if chip_keys else prompt
            result = generate(wrapped, provider=provider, persona=persona)
            t1 = score_persona(result.final)
            t1_means.append(t1.mean)
            if t1.mean < 1.0:
                tags = []
                if t1.p1_em_dash < 1.0: tags.append("em_dash")
                if t1.p2_hits: tags.append("plumbing=" + ",".join(t1.p2_hits))
                if t1.p3_reset < 1.0: tags.append("reset")
                if t1.p4_lead < 1.0: tags.append("hedge")
                if t1.p5_voice < 1.0: tags.append(f"voice={t1.p5_voice}")
                failures_t1.append(f"on prompt {prompt!r}: {' '.join(tags)}")
            try:
                t2 = score_distinctiveness(result.final, provider=provider)
                t2_scores.append(t2.score)
                if t2.score < 0.7:
                    failures_t2.append((prompt, t2.score, result.final[:200]))
            except Exception:
                pass
        except Exception:
            pass

    for probe in PROBES:
        try:
            r = run_probe(probe, provider=provider, persona=persona)
            t3_scores.append(r.score)
            if r.score < 0.8:
                failures_t3.append((probe.id, r.score))
        except Exception:
            pass

    if include_deeper:
        for probe in T6_EMOTIONAL_ATTUNEMENT_PROBES:
            try:
                r = run_deep_probe(probe, provider=provider, persona=persona)
                t6_scores.append(r.score)
                if r.score < 0.8:
                    failures_t6.append((probe.id, r.score))
            except Exception:
                pass
        for probe in T7_MEMORY_COHERENCE_PROBES:
            try:
                r = run_deep_probe(probe, provider=provider, persona=persona)
                t7_scores.append(r.score)
                if r.score < 0.8:
                    failures_t7.append((probe.id, r.score))
            except Exception:
                pass
        for probe in T8_INITIATIVE_PROBES:
            try:
                r = run_deep_probe(probe, provider=provider, persona=persona)
                t8_scores.append(r.score)
                if r.score < 0.8:
                    failures_t8.append((probe.id, r.score))
            except Exception:
                pass

    return {
        "t1_mean": round(mean_(t1_means), 3) if t1_means else None,
        "t2_mean": round(mean_(t2_scores), 3) if t2_scores else None,
        "t3_mean": round(mean_(t3_scores), 3) if t3_scores else None,
        "t6_mean": round(mean_(t6_scores), 3) if t6_scores else None,
        "t7_mean": round(mean_(t7_scores), 3) if t7_scores else None,
        "t8_mean": round(mean_(t8_scores), 3) if t8_scores else None,
        "failures_t1": failures_t1,
        "failures_t2": failures_t2,
        "failures_t3": failures_t3,
        "failures_t6": failures_t6,
        "failures_t7": failures_t7,
        "failures_t8": failures_t8,
        "deeper_included": bool(include_deeper),
    }


def composite(scores: dict, weights: tuple[float, ...]) -> float:
    """Composite fitness. weights is a 3- or 6-tuple:
    (T1, T2, T3) or (T1, T2, T3, T6, T7, T8).
    Skips tiers with None scores (unscored) and renormalizes weights."""
    w = list(weights) + [0.0] * 6
    keys = ("t1_mean", "t2_mean", "t3_mean", "t6_mean", "t7_mean", "t8_mean")
    scored = [(i, scores.get(keys[i])) for i in range(6)]
    scored = [(i, v) for i, v in scored if v is not None]
    if not scored:
        return 0.0
    total_w = sum(w[i] for i, _ in scored)
    if total_w == 0:
        return 0.0
    return round(sum(w[i] * v for i, v in scored) / total_w, 3)


def diagnose(scores: dict) -> list[str]:
    out: list[str] = []
    for f in scores.get("failures_t1", [])[:3]:
        out.append(f"T1 mechanics: {f}")
    for prompt, t2, _preview in scores.get("failures_t2", [])[:4]:
        out.append(
            f"T2 distinctiveness only {t2} on prompt {prompt!r}. "
            "Make the voice sharper, more decisive, less hedged, and more Spark-specific."
        )
    for probe_id, t3 in scores.get("failures_t3", [])[:3]:
        out.append(f"T3 trait probe {probe_id} only {t3}. Strengthen the behavioral cue in the spec.")
    for probe_id, t6 in scores.get("failures_t6", [])[:3]:
        out.append(
            f"T6 emotional attunement probe {probe_id} only {t6}. "
            f"The spec is not pulling Spark to engage with the emotional state strongly enough."
        )
    for probe_id, t7 in scores.get("failures_t7", [])[:3]:
        out.append(
            f"T7 memory coherence probe {probe_id} only {t7}. "
            f"Spark is not acting on facts the user stated earlier in the conversation."
        )
    for probe_id, t8 in scores.get("failures_t8", [])[:3]:
        out.append(
            f"T8 initiative probe {probe_id} only {t8}. "
            f"Spark is answering the literal question but missing the buried implicit signal the user did not ask about."
        )
    if not out:
        out.append("No specific failures. Push for sharper warmth, more memorable phrasing, less filler.")
    return out


def mutate_persona(provider: ProviderSpec, baseline: PersonaSpec, weaknesses: list[str]) -> str:
    user_prompt = (
        "[Baseline persona spec]\n"
        f"{baseline.system_prompt}\n\n"
        "[Observed weaknesses on real model output]\n"
        + "\n".join(f"- {w}" for w in weaknesses)
        + "\n\n[Task]\nProduce one improved variant of the persona spec markdown that addresses these weaknesses while keeping all hard constraints. Output ONLY the final markdown spec. Do not include analysis steps, planning, code fences, or any other preamble or commentary. The first line of your output must be a heading like '# Spark persona vN'."
    )
    raw = call_provider(
        provider=provider,
        system_prompt=MUTATOR_SYSTEM,
        user_prompt=user_prompt,
        max_tokens=1400,
        temperature=0.9,
    )
    return _sanitize_mutator_output(raw)


_CODE_FENCE = re.compile(r"```(?:markdown|md)?\s*\n(.*?)```", re.DOTALL)
_FIRST_HEADING = re.compile(r"^#\s+Spark persona", re.MULTILINE)


def _sanitize_mutator_output(text: str) -> str:
    """Extract the actual persona spec markdown from a mutator response.

    The mutator system prompt asks for raw markdown only, but reasoning
    models sometimes emit chain-of-thought analysis followed by the spec
    inside a code fence. Strip those wrappers so persona.vN.md is a
    clean spec, not a transcript of the model thinking through the task.
    """
    raw = (text or "").strip()
    if not raw:
        return ""
    # First, prefer a fenced markdown block if one exists
    fence = _CODE_FENCE.search(raw)
    if fence:
        candidate = fence.group(1).strip()
        if _FIRST_HEADING.search(candidate):
            return candidate
    # Otherwise, find the first '# Spark persona' heading and return from there
    match = _FIRST_HEADING.search(raw)
    if match:
        return raw[match.start():].strip()
    # Last resort: return the raw text and let the scorer / reviewer catch it
    return raw


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {parsed}")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--weights", default="0.2,0.5,0.3", help="Comma-separated weights. 3 floats T1,T2,T3 or 6 floats T1,T2,T3,T6,T7,T8.")
    parser.add_argument(
        "--include-deeper",
        action="store_true",
        help="Also score T6 emotional attunement, T7 memory coherence, and "
        "T8 initiative probes during evolution. Triples LLM cost per cycle "
        "but lets the mutator target deeper trait gaps.",
    )
    parser.add_argument("--floor-drop", type=float, default=0.05, help="Max allowed regression on any single axis")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", default="evals/_evolve_persona_last.json")
    parser.add_argument(
        "--sib-home",
        default=None,
        help="Path to a Spark Intelligence Builder home directory. When set, "
        "the audit miner reads its gateway-outbound.jsonl and feeds the most "
        "frequent summarized T1 failure patterns into the diagnose step so "
        "the mutator targets observed failure classes, not just synthetic "
        "probe failures.",
    )
    parser.add_argument(
        "--audit-limit",
        type=_positive_int,
        default=200,
        help="How many recent SIB outbound rows to mine when --sib-home is set (must be a positive integer).",
    )
    parser.add_argument(
        "--chip-load",
        default="",
        help="Comma-separated chip keys whose context to inject into eval "
        "prompts (e.g. 'xcontent,startup-yc'). When set, candidates are "
        "scored on the chip-loaded path users actually experience in "
        "production, not the bare-LLM path. Closes the 88-em-dash gap "
        "the audit miner found.",
    )
    args = parser.parse_args()

    weights = tuple(float(x) for x in args.weights.split(","))
    if len(weights) not in (3, 6):
        raise SystemExit("weights must be 3 floats (T1,T2,T3) or 6 floats (T1,T2,T3,T6,T7,T8)")

    provider = ProviderSpec.from_env()
    n, baseline = find_latest_persona()
    print(f"\n=== persona evolution (multi-tier) | baseline=v{n} | candidates={args.candidates} | weights={weights} ===\n")

    tier_label = "T1+T2+T3+T6+T7+T8" if args.include_deeper else "T1+T2+T3"
    print(f"[baseline v{n}] scoring {tier_label}...")
    t0 = time.time()
    chip_load = [c.strip() for c in (args.chip_load or "").split(",") if c.strip()]
    baseline_scores = score_all_tiers(provider, baseline, include_deeper=args.include_deeper, chip_load=chip_load)
    baseline_composite = composite(baseline_scores, weights)
    print(_format_score_line(f"baseline v{n}", baseline_scores, baseline_composite, time.time() - t0))

    weaknesses = diagnose(baseline_scores)

    if args.sib_home:
        miner = AuditMiner.from_sib_home(args.sib_home)
        live = miner.recent_findings(limit=args.audit_limit)
        print(live.summary())
        print()
        production_lines = live.diagnose_lines(max_per_kind=2)
        if production_lines:
            weaknesses = production_lines + weaknesses
            print(
                f"Augmented diagnose with {len(production_lines)} production failure "
                f"line(s) from real Telegram conversations."
            )
            print()

    print("Diagnosed weaknesses:")
    for w in weaknesses:
        print(f"  - {w}")
    print()

    candidates = []
    for i in range(args.candidates):
        print(f"[candidate {i + 1}] mutating + scoring...")
        t0 = time.time()
        try:
            text = mutate_persona(provider, baseline, weaknesses)
            cand = PersonaSpec(version=f"cand-{i + 1}", text=text)
            scores = score_all_tiers(provider, cand, include_deeper=args.include_deeper, chip_load=chip_load)
            comp = composite(scores, weights)
            candidates.append({"index": i + 1, "text": text, "scores": scores, "composite": comp})
            print(_format_score_line(f"candidate {i + 1}", scores, comp, time.time() - t0))
        except Exception as exc:
            print(f"[candidate {i + 1}] ERROR: {exc}\n")

    candidates.sort(key=lambda c: c["composite"], reverse=True)
    print("=== verdict ===")
    print(_format_score_line(f"baseline v{n}", baseline_scores, baseline_composite))
    for c in candidates:
        print(_format_score_line(f"candidate {c['index']}", c["scores"], c["composite"]))

    winner = candidates[0] if candidates else None
    promote = False
    promote_reason = ""
    if winner is not None and winner["composite"] > baseline_composite:
        regression_axes = ["t1_mean", "t2_mean", "t3_mean"]
        if baseline_scores.get("deeper_included"):
            regression_axes.extend(["t6_mean", "t7_mean", "t8_mean"])
        regressions = []
        for axis in regression_axes:
            drop = baseline_scores.get(axis, 0.0) - winner["scores"].get(axis, 0.0)
            if drop > args.floor_drop:
                regressions.append(f"{axis}: -{drop:.3f}")
        if regressions:
            promote = False
            promote_reason = (
                f"composite improved but axis regression too large: {', '.join(regressions)}"
            )
        else:
            promote = True
            promote_reason = (
                f"composite {winner['composite']} > {baseline_composite} with no axis regression > {args.floor_drop}"
            )

    if promote and not args.dry_run:
        new_n = n + 1
        new_path = ARTIFACTS_DIR / f"persona.v{new_n}.md"
        new_path.write_text(winner["text"], encoding="utf-8")
        set_latest_persona_version(
            f"v{new_n}",
            actor="evals/evolve_persona.py",
            reason=promote_reason,
        )
        print(f"\nPROMOTED: candidate {winner['index']} -> {new_path.name}")
        print(f"  reason: {promote_reason}")
        # Sidecar promotion to spark-personality-chip-labs registry
        try:
            from spark_character import promote_evolved_persona_to_chip_lab
            chip_lab_path = promote_evolved_persona_to_chip_lab(
                base_chip_id="founder-operator",
                base_persona_version=f"v{n}",
                new_persona_version=f"v{new_n}",
                persona_markdown=winner["text"],
                composite_score=winner["composite"],
            )
            if chip_lab_path is not None:
                print(f"  registry: also wrote {chip_lab_path}")
            else:
                print("  registry: chip lab not found locally, skipped sidecar promotion")
        except Exception as exc:
            print(f"  registry: sidecar promotion failed: {exc}")
    elif promote:
        print(f"\nWOULD PROMOTE candidate {winner['index']} (dry-run)")
    else:
        print("\nNO PROMOTION.")
        if winner is not None:
            print(f"  reason: {promote_reason or 'baseline composite still wins'}")

    Path(args.out).write_text(
        json.dumps({
            "baseline_version": n, "weights": list(weights),
            "baseline_scores": baseline_scores, "baseline_composite": baseline_composite,
            "candidates": [
                {"index": c["index"], "scores": c["scores"], "composite": c["composite"]}
                for c in candidates
            ],
            "winner_text": winner["text"] if winner else None,
            "promoted": bool(promote and not args.dry_run),
        }, indent=2)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
