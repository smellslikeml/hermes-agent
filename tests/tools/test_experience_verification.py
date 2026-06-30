"""Tests for the EDV Verify gate (tools/experience_verification.py) and its
wiring into the memory write path (tools/memory_tool.py).

Adapted from "Escaping the Self-Confirmation Trap: An Execute-Distill-Verify
Paradigm for Agentic Experience Learning" (arXiv:2606.24428). The gate is only
active under the background_review write origin — the self-improvement fork that
both distills and decides insertion — so the integration tests drive
``memory_tool`` through that origin to prove the wiring, not just the verifier
in isolation.
"""

import json

import pytest

from tools.experience_verification import (
    ExperienceVerdict,
    verify_experience_candidate,
    verify_memory_write,
)
from tools.memory_tool import MemoryStore, memory_tool
from tools.skill_provenance import (
    set_current_write_origin,
    reset_current_write_origin,
)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
    s = MemoryStore(memory_char_limit=2000, user_char_limit=1000)
    s.load_from_disk()
    return s


@pytest.fixture()
def background_origin():
    """Run the body under the background_review write origin, like the fork."""
    token = set_current_write_origin("background_review")
    try:
        yield
    finally:
        reset_current_write_origin(token)


# =========================================================================
# Verifier unit behaviour
# =========================================================================


class TestVerifyExperienceCandidate:
    def test_clean_experience_approved(self):
        v = verify_experience_candidate("User prefers terse answers and dark mode.")
        assert v.approved is True

    def test_negative_tool_claim_rejected(self):
        v = verify_experience_candidate("The browser tool doesn't work in this repo.")
        assert v.approved is False
        assert v.category == "negative_tool_claim"
        assert v.reason

    def test_cannot_use_claim_rejected(self):
        v = verify_experience_candidate("Cannot use web_extract from execute_code.")
        assert v.approved is False
        assert v.category == "negative_tool_claim"

    def test_environment_failure_rejected(self):
        v = verify_experience_candidate(
            "ffmpeg: command not found, so video skills fail."
        )
        assert v.approved is False
        assert v.category == "environment_failure"

    def test_transient_error_rejected(self):
        v = verify_experience_candidate("The API timed out, so saving never works.")
        assert v.approved is False
        assert v.category == "transient_error"

    def test_remove_action_always_approved(self):
        # Removing a stale entry carries no new (possibly wrong) experience.
        v = verify_experience_candidate("anything is broken", action="remove")
        assert v.approved is True

    def test_empty_content_left_to_store(self):
        assert verify_experience_candidate("", action="add").approved is True


# =========================================================================
# Origin gating — verify_memory_write is a no-op in the foreground
# =========================================================================


class TestVerifyMemoryWriteOriginGate:
    def test_foreground_writes_not_gated(self):
        # Default origin is foreground -> even a flagged candidate passes.
        assert verify_memory_write("add", "memory", "the search tool is broken") is None

    def test_background_clean_passes(self, background_origin):
        assert verify_memory_write("add", "memory", "User ships on Fridays.") is None

    def test_background_flagged_rejected(self, background_origin):
        rejected = verify_memory_write("add", "memory", "the terminal tool is broken")
        assert rejected is not None
        payload = json.loads(rejected)
        assert payload["success"] is False
        assert payload["verified"] is False
        assert payload["verification_category"] == "negative_tool_claim"

    def test_background_batch_rejected_on_any_bad_op(self, background_origin):
        ops = [
            {"action": "add", "content": "User likes concise output."},
            {"action": "add", "content": "no module named pandas, skill unusable"},
        ]
        rejected = verify_memory_write("batch", "memory", None, operations=ops)
        assert rejected is not None
        assert json.loads(rejected)["verification_category"] == "environment_failure"


# =========================================================================
# End-to-end through memory_tool — proves the gate is wired into the store
# =========================================================================


class TestMemoryToolIntegration:
    def test_background_review_negative_claim_blocked(self, store, background_origin):
        before = list(store.memory_entries)
        result = memory_tool(
            action="add",
            target="memory",
            content="The browser tool does not work here.",
            store=store,
        )
        payload = json.loads(result)
        assert payload["success"] is False
        assert payload.get("verification_category") == "negative_tool_claim"
        # The candidate never reached the store.
        assert store.memory_entries == before

    def test_background_review_clean_write_lands(self, store, background_origin):
        result = memory_tool(
            action="add",
            target="memory",
            content="User prefers uv over pip for this project.",
            store=store,
        )
        assert json.loads(result)["success"] is True
        assert any("uv over pip" in e for e in store.memory_entries)

    def test_foreground_negative_claim_allowed(self, store):
        # No background origin set: a user explicitly recording an environment
        # fact must NOT be second-guessed by the EDV gate.
        result = memory_tool(
            action="add",
            target="memory",
            content="On my laptop the browser tool is broken until I run xvfb.",
            store=store,
        )
        assert json.loads(result)["success"] is True
        assert any("browser tool is broken" in e for e in store.memory_entries)

    def test_background_review_batch_blocked(self, store, background_origin):
        before = list(store.memory_entries)
        result = memory_tool(
            target="memory",
            operations=[
                {"action": "add", "content": "User is in Berlin (CET)."},
                {"action": "add", "content": "git push timed out, so deploys fail."},
            ],
            store=store,
        )
        payload = json.loads(result)
        assert payload["success"] is False
        assert payload.get("verification_category") == "transient_error"
        # Atomic batch: nothing landed, including the good op.
        assert store.memory_entries == before


def test_verdict_is_namedtuple():
    v = ExperienceVerdict(False, category="x", reason="y")
    assert (v.approved, v.category, v.reason) == (False, "x", "y")
