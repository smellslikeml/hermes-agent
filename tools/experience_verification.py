"""Independent Verify gate for self-improvement experience writes.

Adapted from *"Escaping the Self-Confirmation Trap: An Execute-Distill-Verify
Paradigm for Agentic Experience Learning"* (arXiv:2606.24428).

The background self-improvement review fork (``agent.background_review``) both
**distills** a candidate experience out of the conversation *and* **decides**
whether to commit it to persistent memory. EDV names this failure mode the
*Self-Confirmation Trap*: the agent that produced (and is convinced by) a
trajectory is a biased judge of it, so wrong-but-self-consistent content slips
into memory and is later retrieved and reused, compounding the error.

EDV's fix is to *decouple* the Verify stage from the executor/distiller. This
module is a small, rule-based realisation of that Verify stage. It inspects the
candidate write produced under the ``background_review`` origin and rejects
content matching the anti-patterns the review prompt warns against but cannot
itself enforce — negative claims about tools/features, environment/setup
failures, and transient errors. The distilling fork no longer gets to talk its
way past those rules: an independent check sees the proposed content and decides.

Scope decisions (deliberately narrow, to deliver the *result* not the method):
  * Only the autonomous review fork is gated (``write_approval.is_background()``).
    Foreground, user-driven writes are never second-guessed — if the user says
    "remember the browser tool is broken on my box", that is a fact about their
    environment, not a self-confirmed experience.
  * The verifier is rule-based rather than a separate heterogeneous agent +
    consensus vote. It captures EDV's insight (decoupled verification before
    insertion) without the multi-agent execution group, which the repo's
    single-fork harness does not host.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, NamedTuple, Optional


class ExperienceVerdict(NamedTuple):
    """Outcome of verifying one candidate experience.

    ``approved`` is True when the candidate may be written. When False,
    ``category`` names the anti-pattern that tripped and ``reason`` is a short,
    actionable message surfaced back to the review fork as the tool error.
    """

    approved: bool
    category: str = ""
    reason: str = ""


# ---------------------------------------------------------------------------
# Anti-pattern detectors.
#
# Each pattern targets one of the "Do NOT capture" classes the review prompt
# enumerates (see agent/background_review.py). These harden into persistent
# self-imposed constraints — exactly the cumulative-error mode EDV filters.
# Patterns are intentionally specific (capability noun adjacent to a negation,
# or a concrete setup-failure phrase) to keep false positives low.
# ---------------------------------------------------------------------------

# A capability/feature asserted to be broken or unusable — e.g.
# "the browser tool doesn't work", "web search is broken", "cannot use
# execute_code". These become refusals the agent later cites against itself.
_NEGATIVE_TOOL_CLAIM = re.compile(
    r"\b(tool|tools|browser|feature|command|function|api|search|terminal|"
    r"plugin|integration|backend|capability)\b[^.\n]{0,48}?\b("
    r"do(?:es)?\s*n[o']?t\s+work|did\s*n[o']?t\s+work|not\s+working|"
    r"is\s+broken|are\s+broken|is\s+unavailable|are\s+unavailable|"
    r"can\s*n[o']?t\s+be\s+used|cannot\s+be\s+used)\b",
    re.IGNORECASE,
)
# "cannot use X from Y" / "can't use the browser" phrasing the prompt quotes.
_CANNOT_USE = re.compile(
    r"\bcan\s*n[o']?t\s+use\b[^.\n]{0,40}\b(tool|browser|terminal|execute_code|"
    r"web_extract|search|plugin|from\b)",
    re.IGNORECASE,
)

# Environment / setup failures the user can fix — not durable rules.
_ENVIRONMENT_FAILURE = re.compile(
    r"\b(command not found|not installed|isn['’]?t installed|"
    r"no such file or directory|permission denied|"
    r"missing (?:binary|binaries|dependency|dependencies|package|packages|module)|"
    r"module ?not ?found(?:error)?|no module named|importerror|"
    r"unconfigured|fresh[- ]install)\b",
    re.IGNORECASE,
)

# Transient errors that resolved within the session — the lesson is the retry,
# not the failure. Conservative: requires an explicit transient signal.
_TRANSIENT_ERROR = re.compile(
    r"\b(timed out|timeout|rate[- ]?limit(?:ed|ing)?|"
    r"temporarily unavailable|503|502|try again later|"
    r"connection reset|connection refused)\b",
    re.IGNORECASE,
)

_DETECTORS = (
    (
        "negative_tool_claim",
        _NEGATIVE_TOOL_CLAIM,
        "reads as a negative claim that a tool/feature is broken; such claims "
        "harden into refusals the agent cites against itself long after the "
        "problem is fixed. Capture the working pattern instead, or nothing.",
    ),
    (
        "negative_tool_claim",
        _CANNOT_USE,
        "asserts a capability cannot be used; this becomes a self-imposed "
        "constraint. Record the fix or the working approach instead.",
    ),
    (
        "environment_failure",
        _ENVIRONMENT_FAILURE,
        "describes an environment/setup failure (missing binary, bad path, "
        "unconfigured creds) the user can fix — not a durable experience. If a "
        "fix exists, capture the fix, not the failure.",
    ),
    (
        "transient_error",
        _TRANSIENT_ERROR,
        "describes a transient error (timeout/rate-limit/network) that does not "
        "generalise. If retrying worked, the lesson is the retry pattern.",
    ),
)


def verify_experience_candidate(
    content: Optional[str],
    *,
    action: str = "add",
    target: str = "memory",
) -> ExperienceVerdict:
    """Independently verify one candidate experience before insertion.

    Removals carry no new content and approve unconditionally. Content-bearing
    writes (``add``/``replace``) are checked against the anti-pattern detectors;
    the first match rejects with its category and reason.
    """
    if action == "remove":
        return ExperienceVerdict(True)
    text = (content or "").strip()
    if not text:
        # Empty content is the store's concern, not the verifier's.
        return ExperienceVerdict(True)
    for category, pattern, reason in _DETECTORS:
        if pattern.search(text):
            return ExperienceVerdict(False, category=category, reason=reason)
    return ExperienceVerdict(True)


def verify_memory_write(
    action: Optional[str],
    target: str,
    content: Optional[str],
    operations: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """EDV Verify gate for the memory write path.

    Returns a JSON tool-error string when the candidate must be blocked, or
    ``None`` when the write may proceed. Active ONLY under the
    ``background_review`` write origin — foreground, user-driven writes pass
    through untouched (see module docstring).

    Mirrors ``memory_tool``'s own gate contract: the caller treats a non-None
    return as a terminal tool result and a None return as "proceed".
    """
    try:
        from tools.write_approval import is_background

        if not is_background():
            return None
    except Exception:
        # Can't determine origin -> fail open, exactly like the write gate.
        return None

    # Batch: verify every content-bearing op; reject the whole batch on the
    # first candidate that fails (the batch applies atomically).
    if operations:
        for op in operations:
            op = op or {}
            verdict = verify_experience_candidate(
                op.get("content"),
                action=op.get("action", "add"),
                target=target,
            )
            if not verdict.approved:
                return _rejection(verdict)
        return None

    verdict = verify_experience_candidate(
        content, action=action or "add", target=target
    )
    if not verdict.approved:
        return _rejection(verdict)
    return None


def _rejection(verdict: ExperienceVerdict) -> str:
    """Render a rejected verdict as a memory-tool error result."""
    return json.dumps(
        {
            "error": (
                f"Experience-verification gate rejected this write: {verdict.reason}"
            ),
            "success": False,
            "verified": False,
            "verification_category": verdict.category,
        },
        ensure_ascii=False,
    )


__all__ = [
    "ExperienceVerdict",
    "verify_experience_candidate",
    "verify_memory_write",
]
