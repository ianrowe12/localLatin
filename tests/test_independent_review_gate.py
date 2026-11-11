"""Behavioural tests for the Independent Review gate (issue #267).

The gate script, .github/scripts/independent_review_gate.py, is loaded by path
and its main() is run against a fake GitHub REST API (urlopen is patched), the
same way the workflow runs it. Parser cases are ported from the Aria repo's gate
tests; the trusted/untrusted author cases are new, because this repo is public.
"""

from __future__ import annotations

import importlib.util
import io
import json
import re
import socket
import sys
import textwrap
import urllib.error
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / ".github" / "scripts" / "independent_review_gate.py"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "independent-review-gate.yml"
RELAY = REPO_ROOT / ".github" / "workflows" / "independent-review-events.yml"
PROTOCOL = REPO_ROOT / "docs" / "agents" / "independent-review.md"

REPO = "ianrowe12/localLatin"
PR = 4349
HEAD_SHA = "a17d08e" + "0" * 33
OLD_HEAD_SHA = "0423f73" + "0" * 33
OWNER = "ianrowe12"
T0 = "2026-07-16T13:05:00Z"
T1 = "2026-07-16T13:06:00Z"

APPROVED = ("success", "Independent review approved for current head")
REQUESTED = ("failure", "Independent review requested changes for current head")
NONE = ("failure", "No independent review verdict on this PR yet")
UNBOUND = ("failure", "Verdict found; reviewer must add exactly one 'Reviewed head: <40-hex SHA>' line")
LATER_UNBOUND = ("failure", "A later verdict lacks exactly one 'Reviewed head:' line; it blocks the approval")
CONFLICTING = ("failure", "A verdict comment has marker lines that disagree; quote recaps with '>'")
MISPLACED = ("failure", "A verdict marker is not its comment's final line; it cannot approve")
TIE = ("failure", "Conflicting independent review verdicts at latest timestamp")
STALE = ("failure", "No verdict names the current head SHA; re-review required")
ERROR = ("failure", "Gate could not finish evaluating; re-run the Independent Review Gate workflow")


def verdict(direction: str, sha: str = HEAD_SHA) -> str:
    return f"Reviewed head: {sha}\nIndependent review verdict: {direction}"


def load_gate() -> ModuleType:
    """Load the script fresh by path. dataclasses needs it in sys.modules while it runs."""
    name = "independent_review_gate"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def load_workflow(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def triggers(workflow: dict) -> dict:
    # PyYAML reads the bare key `on` as boolean True.
    return workflow.get("on", workflow.get(True))


class FakeResponse(io.BytesIO):
    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def http_error(url: str, code: int, msg: str = "error") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, msg, {}, io.BytesIO(b"{}"))


def _user(author: object) -> dict | None:
    if author is None or isinstance(author, dict):
        return author
    return {"login": author, "type": "Bot" if str(author).endswith("[bot]") else "User"}


class FakeGitHub:
    """Serves the handful of REST paths the gate reads, and records what it does."""

    def __init__(self, *, comments=(), reviews=(), review_states=None, labels=(),
                 permissions=None, pr_state="open", relay_run=None, relay_candidates=None,
                 mutate_after_first_comment_read=None, failures=None,
                 post_status_failures=None):
        self.comments = [tuple(c) for c in comments]
        self.reviews = [tuple(r) for r in reviews]
        self.review_states = review_states or ["COMMENTED"] * len(self.reviews)
        self.labels = list(labels)
        self.permissions = {OWNER: "admin", **(permissions or {})}
        self.pr_state = pr_state
        self.relay_run = relay_run
        self.relay_candidates = relay_candidates
        self.mutate = mutate_after_first_comment_read
        self.failures = failures or {}
        self.post_status_failures = list(post_status_failures or [])
        self.statuses: list[dict] = []
        self.trace: list[str] = []
        self.comment_reads = 0

    @staticmethod
    def _entry(row: tuple) -> tuple[str, str, dict | None]:
        timestamp, body, *rest = row
        return timestamp, body, _user(rest[0] if rest else OWNER)

    def urlopen(self, request: object, timeout: int = 30) -> FakeResponse:
        del timeout
        url = request.full_url
        method = request.method
        path = urlparse(url).path
        key = f"{method} {path}"
        self.trace.append(key)
        if self.failures.get(key):
            raise self.failures[key].pop(0)

        base = f"/repos/{REPO}"
        if method == "POST" and path == f"{base}/statuses/{HEAD_SHA}":
            self.statuses.append(json.loads(request.data.decode("utf-8")))
            if self.post_status_failures:
                raise self.post_status_failures.pop(0)
            payload = {}
        elif path == f"{base}/actions/runs/456":
            payload = self.relay_run
        elif path == f"{base}/pulls":
            query = parse_qs(urlparse(url).query)
            head_owner = self.relay_run["head_repository"]["full_name"].split("/")[0]
            expected_head = f"{head_owner}:{self.relay_run['head_branch']}"
            if query.get("base") != ["main"] or query.get("head") != [expected_head]:
                raise AssertionError(f"unscoped review relay query: {url}")
            payload = self.relay_candidates if self.relay_candidates is not None else [{
                "number": PR, "state": "open", "base": {"ref": "main"},
                "head": {"ref": "review-branch", "repo": {"full_name": REPO}},
            }]
        elif path == f"{base}/pulls/{PR}":
            payload = {
                "state": self.pr_state,
                "labels": [{"name": label} for label in self.labels],
                "head": {"sha": HEAD_SHA},
            }
        elif path == f"{base}/issues/{PR}/comments":
            self.comment_reads += 1
            if self.comment_reads == 2 and self.mutate is not None:
                self.comments.append(tuple(self.mutate))
            payload = []
            for index, row in enumerate(self.comments, start=1):
                timestamp, body, user = self._entry(row)
                payload.append({"id": index, "created_at": timestamp, "updated_at": timestamp,
                                "body": body, "user": user})
        elif path == f"{base}/pulls/{PR}/reviews":
            payload = []
            for index, row in enumerate(self.reviews, start=1):
                timestamp, body, user = self._entry(row)
                payload.append({"id": index, "submitted_at": timestamp, "body": body,
                                "user": user, "state": self.review_states[index - 1]})
        elif path.startswith(f"{base}/collaborators/") and path.endswith("/permission"):
            login = path.split("/")[-2]
            permission = self.permissions.get(login, "read")
            if permission == 404:
                raise http_error(url, 404, "Not Found")
            payload = {"permission": permission}
        else:
            raise AssertionError(f"unexpected gate request: {method} {url}")
        return FakeResponse(json.dumps(payload).encode("utf-8"))


def run_gate(fake: FakeGitHub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps({"workflow_run": {"id": 456}}))
    monkeypatch.setenv("GH_TOKEN", "test-token")
    monkeypatch.setenv("REPO", REPO)
    monkeypatch.setenv("PR_NUMBER", "" if fake.relay_run else str(PR))
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_run" if fake.relay_run else "issue_comment")
    gate = load_gate()
    with patch("urllib.request.urlopen", side_effect=fake.urlopen), patch("time.sleep"):
        gate.main()
    return fake.statuses


@pytest.fixture
def gate(tmp_path, monkeypatch):
    """Run the gate once and return its final status as (state, description)."""

    def _run(**kwargs) -> tuple[str, str]:
        statuses = run_gate(FakeGitHub(**kwargs), tmp_path, monkeypatch)
        assert statuses, "expected at least one status update"
        return statuses[-1]["state"], statuses[-1]["description"]

    return _run


def sources(body: str, timestamp: str = T0, author: object = OWNER):
    """The same body posted as an issue comment and as a review body."""
    row = (timestamp, body, author)
    return [pytest.param({"comments": [row]}, id="comment"),
            pytest.param({"reviews": [row]}, id="review")]


# --- Workflow wiring --------------------------------------------------------


def test_gate_triggers_and_relay_are_wired_as_designed():
    workflow = load_workflow(WORKFLOW)
    on = triggers(workflow)
    assert set(on) == {"pull_request_target", "issue_comment", "workflow_run"}
    assert on["pull_request_target"]["branches"] == ["main"]
    assert on["pull_request_target"]["types"] == [
        "opened", "reopened", "synchronize", "labeled", "unlabeled"]
    assert on["issue_comment"]["types"] == ["created", "edited", "deleted"]
    assert on["workflow_run"] == {"workflows": ["Independent Review Events"], "types": ["completed"]}

    relay = load_workflow(RELAY)
    assert relay["name"] == "Independent Review Events"
    assert triggers(relay) == {"pull_request_review": {"types": ["submitted", "edited", "dismissed"]}}
    assert relay["permissions"] == {}
    relay_steps = [step for job in relay["jobs"].values() for step in job["steps"]]
    assert all("uses" not in step for step in relay_steps), "the relay must not check anything out"


def test_every_pr_comment_event_reaches_the_gate():
    gate_job = load_workflow(WORKFLOW)["jobs"]["gate"]
    condition = gate_job["if"]
    assert "github.event.issue.pull_request" in condition
    assert "github.event_name == 'workflow_run'" in condition
    assert "contains(" not in condition


def test_gate_jobs_are_serialised_without_cancelling_pending_events():
    concurrency = load_workflow(WORKFLOW)["jobs"]["gate"]["concurrency"]
    assert concurrency["group"] == "independent-review-statuses"
    assert concurrency["cancel-in-progress"] is False
    assert concurrency["queue"] == "max"


def test_gate_checks_out_only_its_script_from_the_default_branch():
    workflow = load_workflow(WORKFLOW)
    steps = workflow["jobs"]["gate"]["steps"]
    checkouts = [step for step in steps if str(step.get("uses", "")).startswith("actions/checkout")]
    assert len(checkouts) == 1
    options = checkouts[0]["with"]
    assert options["ref"] == "${{ github.event.repository.default_branch }}"
    assert options["persist-credentials"] is False
    assert options["sparse-checkout-cone-mode"] is False
    assert options["sparse-checkout"].strip() == ".github/scripts/independent_review_gate.py"
    source = WORKFLOW.read_text(encoding="utf-8")
    for unsafe in ("pull_request.head", "head_sha", "head_ref", "refs/pull"):
        assert unsafe not in source
    assert steps[-1]["run"].strip() == "python3 -I .github/scripts/independent_review_gate.py"
    assert workflow["jobs"]["gate"]["runs-on"] == "ubuntu-latest"


def test_gate_script_uses_only_the_standard_library():
    tree = SCRIPT.read_text(encoding="utf-8")
    imported = set(re.findall(r"^(?:from|import) ([a-zA-Z_]+)", tree, re.MULTILINE))
    assert imported <= {"__future__", "json", "os", "re", "socket", "sys", "time", "urllib",
                        "dataclasses"}


# --- Basic decisions --------------------------------------------------------


@pytest.mark.parametrize("posted", sources(verdict("APPROVE")))
def test_current_approve_verdict_sets_success(gate, posted):
    assert gate(**posted) == APPROVED


def test_current_request_changes_verdict_sets_failure(gate):
    assert gate(comments=[(T0, verdict("REQUEST CHANGES"))]) == REQUESTED


def test_missing_verdict_sets_failure(gate):
    assert gate() == NONE


def test_stale_verdict_sets_failure(gate):
    assert gate(comments=[(T0, verdict("APPROVE", OLD_HEAD_SHA))]) == STALE


def test_exemption_sets_success(gate):
    assert gate(labels=["independent-review-exempt"]) == (
        "success", "Exempt via independent-review-exempt label")


def test_closed_pr_gets_no_status(tmp_path, monkeypatch):
    fake = FakeGitHub(pr_state="closed", comments=[(T0, verdict("APPROVE"))])
    assert run_gate(fake, tmp_path, monkeypatch) == []


def test_status_targets_the_protocol_and_the_run(tmp_path, monkeypatch):
    statuses = run_gate(FakeGitHub(comments=[(T0, verdict("APPROVE"))]), tmp_path, monkeypatch)
    assert statuses[-1]["context"] == "Independent Review"
    assert statuses[-1]["target_url"] == (
        f"https://github.com/{REPO}/blob/main/docs/agents/independent-review.md?pr={PR}&run=123")


def test_protocol_closing_blocks_produce_head_bound_verdicts(gate):
    protocol = PROTOCOL.read_text(encoding="utf-8")
    rule = protocol.split("## The rule\n", 1)[1].split("\n## ", 1)[0]
    blocks = re.findall(r"(?ms)^   ```\n(.*?)^   ```$", rule)
    assert len(blocks) == 2
    for block, expected in zip(blocks, (APPROVED, REQUESTED), strict=True):
        body = textwrap.dedent(block).replace("<full 40-character SHA>", HEAD_SHA)
        assert gate(comments=[(T0, body)]) == expected
        assert gate(reviews=[(T0, body)]) == expected


# --- Head binding -----------------------------------------------------------


@pytest.mark.parametrize("source", ["comments", "reviews"])
def test_old_head_approval_after_current_commit_still_requires_review(gate, source):
    assert gate(**{source: [(T0, verdict("APPROVE", OLD_HEAD_SHA))]})[0] == "failure"


@pytest.mark.parametrize("source", ["comments", "reviews"])
def test_current_head_approval_does_not_depend_on_commit_timestamp(gate, source):
    assert gate(**{source: [("2026-07-16T12:59:00Z", verdict("APPROVE"))]})[0] == "success"


@pytest.mark.parametrize("source", ["comments", "reviews"])
@pytest.mark.parametrize("head", [
    "",
    f"I reviewed {HEAD_SHA[:7]} in full at {HEAD_SHA}.\n",
    f"Reviewed head: {HEAD_SHA[:7]}\n",
    f"Reviewed head: {HEAD_SHA}0\n",
    f"Reviewed head: {'g' * 40}\n",
    f"> Reviewed head: {HEAD_SHA}\n",
    f"Previous Reviewed head: {HEAD_SHA}\n",
    f"Reviewed head: {HEAD_SHA} extra text\n",
    f"Reviewed head: {HEAD_SHA}\nReviewed head: {OLD_HEAD_SHA}\n",
    f"Reviewed head: {HEAD_SHA}\nReviewed head: {HEAD_SHA}\n",
])
def test_unbound_or_ambiguous_approval_requires_review(gate, source, head):
    assert gate(**{source: [(T0, head + "Independent review verdict: APPROVE")]}) == UNBOUND


def test_current_head_review_accepts_crlf_and_uppercase_sha(gate):
    body = verdict("APPROVE", HEAD_SHA.upper()).replace("\n", "\r\n")
    assert gate(reviews=[(T0, body)])[0] == "success"


@pytest.mark.parametrize("direction, expected", [("APPROVE", "success"), ("REQUEST CHANGES", "failure")])
def test_later_other_head_verdict_cannot_override_current_head(gate, direction, expected):
    other = "REQUEST CHANGES" if direction == "APPROVE" else "APPROVE"
    status = gate(comments=[("2026-07-16T13:01:00Z", verdict(direction))],
                  reviews=[("2026-07-16T13:02:00Z", verdict(other, OLD_HEAD_SHA))])
    assert status[0] == expected


def test_head_lines_count_anywhere_but_only_one_binds(gate):
    fenced_head = f"```\nReviewed head: {HEAD_SHA}\n```\nIndependent review verdict: APPROVE"
    assert gate(comments=[(T0, fenced_head)]) == APPROVED
    second_fenced = verdict("APPROVE").replace(
        "Independent review verdict",
        f"```\nReviewed head: {OLD_HEAD_SHA}\n```\nIndependent review verdict")
    assert gate(comments=[(T0, second_fenced)]) == UNBOUND


# --- Marker parsing ---------------------------------------------------------


@pytest.mark.parametrize("source", ["comments", "reviews"])
@pytest.mark.parametrize("extra, closing, above", [
    ("Independent review verdict: APPROVE", "REQUEST CHANGES", True),
    ("Verdict: approve on the earlier head", "REQUEST CHANGES", True),
    ("  Verdict approve was the round-1 call", "REQUEST CHANGES", True),
    ("Independent review verdict: APPROVE", "REQUEST CHANGES", False),
    ("Verdict: approve", "REQUEST CHANGES", False),
    ("Independent review verdict: REQUEST CHANGES", "APPROVE", True),
    ("Verdict: request-changes", "APPROVE", False),
])
def test_disagreeing_marker_lines_in_one_comment_fail_closed(gate, source, extra, closing, above):
    block = verdict(closing)
    body = f"Findings.\n{extra}\n\n{block}" if above else f"{block}\n\n{extra}\n"
    assert gate(**{source: [(T0, body)]}) == CONFLICTING


@pytest.mark.parametrize("source", ["comments", "reviews"])
@pytest.mark.parametrize("body, expected", [
    (f"Verdict: approve\n\n{verdict('APPROVE')}", APPROVED),
    (f"Verdict: requestchanges\n\n{verdict('REQUEST-CHANGES')}", REQUESTED),
    (f"Verdict: request changes\n\n{verdict('RequestChanges')}", REQUESTED),
    (f"> Independent review verdict: REQUEST CHANGES\n\n{verdict('APPROVE')}", APPROVED),
])
def test_agreeing_marker_lines_and_spellings_keep_their_direction(gate, source, body, expected):
    assert gate(**{source: [(T0, body)]}) == expected


def test_disagreeing_comment_blocks_a_separate_approval_at_the_same_head(gate):
    status = gate(comments=[(T0, verdict("APPROVE")),
                            (T1, f"{verdict('REQUEST CHANGES')}\nVerdict: approve")])
    assert status[0] == "failure"


def test_later_clean_verdict_supersedes_a_disagreeing_comment(gate):
    assert gate(comments=[(T0, f"Verdict: approve\n\n{verdict('REQUEST CHANGES')}"),
                          (T1, verdict("APPROVE"))]) == APPROVED


@pytest.mark.parametrize("source", ["comments", "reviews"])
@pytest.mark.parametrize("direction", [
    "APPROVED", "approves", "REQUEST CHANGESET",
    "APPROVE_pending_fixes", "APPROVE__x", "APPROVEé", "approve*s", "REQUEST CHANGES_later",
])
def test_direction_word_continuations_are_not_verdicts(gate, source, direction):
    assert gate(**{source: [(T0, verdict(direction))]}) == NONE


@pytest.mark.parametrize("source", ["comments", "reviews"])
@pytest.mark.parametrize("body, expected", [
    (f"Reviewed head: {HEAD_SHA}\n**Independent review verdict: APPROVE**", APPROVED),
    (f"Reviewed head: {HEAD_SHA}\n__Independent review verdict: APPROVE__", APPROVED),
    (f"Reviewed head: {HEAD_SHA}\n*Verdict:* approve", APPROVED),
    (f"Reviewed head: {HEAD_SHA}\n**Independent review verdict: REQUEST CHANGES**", REQUESTED),
    (f"Reviewed head: {HEAD_SHA}\n_Independent review verdict: REQUEST CHANGES_", REQUESTED),
    (f"Reviewed head: {HEAD_SHA}\n**Independent review verdict**: REQUEST CHANGES", REQUESTED),
    (f"Reviewed head: {HEAD_SHA}\nIndependent review verdict: **REQUEST CHANGES**", REQUESTED),
    (f"> **Independent review verdict: REQUEST CHANGES**\n\n{verdict('APPROVE')}", APPROVED),
])
def test_emphasised_markers_keep_their_direction(gate, source, body, expected):
    assert gate(**{source: [(T0, body)]}) == expected


@pytest.mark.parametrize("earlier", [
    "**Verdict: request changes**",
    "__Independent review verdict__: REQUEST CHANGES",
    "Independent review verdict: *request changes*",
])
def test_emphasised_earlier_marker_makes_the_comment_conflicting(gate, earlier):
    assert gate(comments=[(T0, f"Findings.\n{earlier}\n\n{verdict('APPROVE')}")]) == CONFLICTING


@pytest.mark.parametrize("suffix", ["\n", "\n\n   \n", "\r\n\r\n", "  \t"])
def test_final_marker_ignores_trailing_blank_lines(gate, suffix):
    assert gate(comments=[(T0, verdict("APPROVE") + suffix)]) == APPROVED


@pytest.mark.parametrize("source", ["comments", "reviews"])
@pytest.mark.parametrize("body", [
    f"Reviewed head: {HEAD_SHA}\nIndependent review verdict: APPROVE\n\nThanks.",
    f"Reviewed head: {HEAD_SHA}\n```\nIndependent review verdict: APPROVE\n```",
    f"Reviewed head: {HEAD_SHA}\n**Independent review verdict: APPROVE**\n\nThanks.",
    f"Reviewed head: {HEAD_SHA}\nIndependent review verdict: APPROVE pending fixes",
    f"Reviewed head: {HEAD_SHA}\nIndependent review verdict: APPROVE.",
    f"Reviewed head: {HEAD_SHA}\n> Independent review verdict: APPROVE\nVerdict: approve\nDone.",
])
def test_a_marker_that_is_not_the_final_line_cannot_approve(gate, source, body):
    assert gate(**{source: [(T0, body)]}) == MISPLACED


@pytest.mark.parametrize("later", [
    f"Reviewed head: {HEAD_SHA}\nIndependent review verdict: REQUEST CHANGES\n\nSorry, one more finding.",
    f"Reviewed head: {HEAD_SHA}\n_Independent review verdict: REQUEST CHANGES_\n\nSorry.",
])
def test_a_misplaced_bound_rejection_blocks_an_earlier_approval(gate, later):
    assert gate(comments=[(T0, verdict("APPROVE")), (T1, later)]) == MISPLACED


# --- Ordering between verdicts ----------------------------------------------


def test_latest_approve_supersedes_request_changes(gate):
    assert gate(comments=[("2026-07-16T13:01:00Z", verdict("REQUEST CHANGES")),
                          ("2026-07-16T13:02:00Z", verdict("APPROVE"))]) == APPROVED


def test_latest_request_changes_supersedes_approve_across_sources(gate):
    assert gate(comments=[("2026-07-16T13:01:00Z", verdict("APPROVE"))],
                reviews=[("2026-07-16T13:02:00Z", verdict("REQUEST CHANGES"))]) == REQUESTED


def test_dismissed_review_verdict_is_ignored(gate):
    assert gate(comments=[("2026-07-16T13:01:00Z", verdict("APPROVE"))],
                reviews=[("2026-07-16T13:02:00Z", verdict("REQUEST CHANGES"))],
                review_states=["DISMISSED"]) == APPROVED


def test_conflicting_equal_timestamp_verdicts_fail_closed(gate):
    assert gate(comments=[(T0, verdict("REQUEST CHANGES")), (T0, verdict("APPROVE"))]) == TIE
    assert gate(comments=[(T0, verdict("REQUEST CHANGES"))],
                reviews=[(T0, verdict("APPROVE"))]) == TIE


@pytest.mark.parametrize("source", ["comments", "reviews"])
@pytest.mark.parametrize("head", [
    "",
    f"Reviewed head: {HEAD_SHA}\nReviewed head: {OLD_HEAD_SHA}\n",
    f"Reviewed head: {HEAD_SHA}\nReviewed head: {HEAD_SHA}\n",
])
@pytest.mark.parametrize("closing", [
    "Independent review verdict: REQUEST CHANGES",
    "Verdict: approve\nIndependent review verdict: REQUEST CHANGES",
])
def test_later_unbound_non_approval_blocks_an_earlier_approval(gate, source, head, closing):
    approval = (T0, verdict("APPROVE"))
    later = (T1, head + closing)
    if source == "comments":
        status = gate(comments=[approval, later])
    else:
        status = gate(comments=[approval], reviews=[later])
    assert status == LATER_UNBOUND


def test_unbound_request_changes_at_the_approval_timestamp_blocks_it(gate):
    assert gate(comments=[(T0, verdict("APPROVE")),
                          (T0, "Independent review verdict: REQUEST CHANGES")]) == LATER_UNBOUND


@pytest.mark.parametrize("other", [
    (T1, "Independent review verdict: APPROVE"),
    ("2026-07-16T13:04:00Z", "Independent review verdict: REQUEST CHANGES"),
])
def test_unbound_approval_or_earlier_unbound_rejection_keeps_the_approval(gate, other):
    assert gate(comments=[(T0, verdict("APPROVE")), other]) == APPROVED


@pytest.mark.parametrize("later, expected", [
    (f"Reviewed head: {HEAD_SHA}\n**Independent review verdict: REQUEST CHANGES**", REQUESTED),
    ("**Independent review verdict: REQUEST CHANGES**", LATER_UNBOUND),
])
def test_later_emphasised_request_changes_overrides_an_earlier_approval(gate, later, expected):
    assert gate(comments=[(T0, verdict("APPROVE")), (T1, later)]) == expected


REJECTION = "Independent review verdict: REQUEST CHANGES"


@pytest.mark.parametrize("source", ["comments", "reviews"])
@pytest.mark.parametrize("recap", [
    f"Disposition.\n\n```\n{REJECTION}\n```\n",
    f"Disposition.\n\n```text\nReviewed head: {HEAD_SHA}\nReviewed head: {OLD_HEAD_SHA}\n{REJECTION}\n```",
    f"Disposition.\n\n~~~\n{REJECTION}\n~~~\n",
    f"Disposition.\n\n````\n```\n{REJECTION}\n````\n",
    f"Disposition.\n\n    {REJECTION}\n\nFixed in the next push.",
    f"{REJECTION}\n\nThat was round 1; this round only replies.",
    f"Disposition.\n\n> {REJECTION}\n",
])
def test_recaps_before_the_final_line_do_not_count(gate, source, recap):
    approval = (T0, verdict("APPROVE"))
    later = (T1, recap)
    if source == "comments":
        status = gate(comments=[approval, later])
    else:
        status = gate(comments=[approval], reviews=[later])
    assert status == APPROVED


@pytest.mark.parametrize("later", [
    f"Findings.\n\n```\n{REJECTION}",
    f"Findings.\n\n````\n```\n{REJECTION}",
    f"- Findings.\n\n    {REJECTION}",
    f"Findings.\n    {REJECTION}",
    f"Disposition.\n\n    {REJECTION}\n",
    f"    {REJECTION}\n",
])
def test_markers_that_render_as_text_still_block_a_later_approval(gate, later):
    # The gate does not parse Markdown: a final-line marker counts even inside an
    # unclosed fence, a list continuation or an indented code block.
    assert gate(comments=[(T0, verdict("APPROVE")), (T1, later)]) == LATER_UNBOUND


@pytest.mark.parametrize("body", [
    f"```this is `not` a fence\n{REJECTION}\n```",
    f"<!--\n```\n-->\n{REJECTION}\n\n```",
    f"<details>\n```\n</details>\n\n{REJECTION}\n```",
])
def test_non_final_marker_blocks_only_when_bound_to_one_head(gate, body):
    approval = (T0, verdict("APPROVE"))
    assert gate(comments=[approval, (T1, f"Reviewed head: {HEAD_SHA}\n{body}")]) == MISPLACED
    assert gate(comments=[approval, (T1, body)]) == APPROVED


# --- Public-repo trust filter -----------------------------------------------


@pytest.mark.parametrize("permission", ["admin", "write"])
def test_verdicts_from_write_access_accounts_count(gate, permission):
    status = gate(comments=[(T0, verdict("APPROVE"), "agent-account")],
                  permissions={"agent-account": permission})
    assert status == APPROVED


UNTRUSTED_AUTHORS = [
    pytest.param("stranger", id="read-permission"),
    pytest.param("triager", id="triage-folds-to-read"),
    pytest.param("ghost-user", id="not-a-user-404"),
    pytest.param("none-perm", id="permission-none"),
    pytest.param({"login": "github-actions[bot]", "type": "Bot"}, id="actions-bot"),
    pytest.param({"login": "helper[bot]", "type": "User"}, id="bot-suffix"),
    pytest.param({"login": "app-account", "type": "Bot"}, id="bot-type"),
    pytest.param(None, id="deleted-account"),
]
UNTRUSTED_PERMISSIONS = {"stranger": "read", "triager": "read", "ghost-user": 404,
                         "none-perm": "none", "app-account": "admin"}


@pytest.mark.parametrize("author", UNTRUSTED_AUTHORS)
@pytest.mark.parametrize("source", ["comments", "reviews"])
def test_untrusted_approval_does_not_approve(gate, author, source):
    status = gate(**{source: [(T0, verdict("APPROVE"), author)]},
                  permissions=UNTRUSTED_PERMISSIONS)
    assert status == NONE


@pytest.mark.parametrize("author", UNTRUSTED_AUTHORS)
@pytest.mark.parametrize("later", [
    verdict("REQUEST CHANGES"),
    REJECTION,
    f"Verdict: approve\n\n{verdict('REQUEST CHANGES')}",
    f"Reviewed head: {HEAD_SHA}\n{REJECTION}\n\nmore text",
])
def test_untrusted_verdicts_do_not_block_a_trusted_approval(gate, author, later):
    status = gate(comments=[(T0, verdict("APPROVE")), (T1, later, author)],
                  permissions=UNTRUSTED_PERMISSIONS)
    assert status == APPROVED


def test_untrusted_approval_does_not_supersede_a_trusted_rejection(gate):
    status = gate(comments=[(T0, verdict("REQUEST CHANGES")), (T1, verdict("APPROVE"), "stranger")])
    assert status == REQUESTED


def test_untrusted_verdict_does_not_tie_a_trusted_one(gate):
    status = gate(comments=[(T0, verdict("APPROVE")), (T0, verdict("REQUEST CHANGES"), "stranger")])
    assert status == APPROVED


def test_permission_is_looked_up_once_per_author_and_only_for_verdicts(tmp_path, monkeypatch):
    fake = FakeGitHub(comments=[
        (T0, verdict("APPROVE")),
        ("2026-07-16T13:05:30Z", "Looks fine to me", "chatty"),
        (T1, verdict("REQUEST CHANGES"), "stranger"),
        ("2026-07-16T13:07:00Z", REJECTION, "stranger"),
    ])
    statuses = run_gate(fake, tmp_path, monkeypatch)
    assert (statuses[-1]["state"], statuses[-1]["description"]) == APPROVED
    lookups = [key for key in fake.trace if key.endswith("/permission")]
    assert sorted(lookups) == [
        f"GET /repos/{REPO}/collaborators/{OWNER}/permission",
        f"GET /repos/{REPO}/collaborators/stranger/permission",
    ]


def test_logs_name_no_accounts(tmp_path, monkeypatch, capsys):
    fake = FakeGitHub(comments=[(T0, verdict("APPROVE")), (T1, REJECTION, "stranger")])
    run_gate(fake, tmp_path, monkeypatch)
    out = capsys.readouterr().out
    assert "1 ignored from accounts without write access" in out
    assert "stranger" not in out and OWNER not in out


def test_permission_lookup_error_fails_closed(tmp_path, monkeypatch):
    url = f"https://api.github.com/repos/{REPO}/collaborators/{OWNER}/permission"
    fake = FakeGitHub(comments=[(T0, verdict("APPROVE"))], failures={
        f"GET /repos/{REPO}/collaborators/{OWNER}/permission": [http_error(url, 500)]})
    with pytest.raises(urllib.error.HTTPError):
        run_gate(fake, tmp_path, monkeypatch)
    assert [(s["state"], s["description"]) for s in fake.statuses] == [ERROR]


# --- Re-read before write, retries ------------------------------------------


def test_stale_success_is_not_posted_when_verdict_changes_before_write(tmp_path, monkeypatch):
    fake = FakeGitHub(comments=[(T0, verdict("APPROVE"))],
                      mutate_after_first_comment_read=(T1, verdict("REQUEST CHANGES")))
    statuses = run_gate(fake, tmp_path, monkeypatch)
    assert statuses
    assert "success" not in [status["state"] for status in statuses]
    assert (statuses[-1]["state"], statuses[-1]["description"]) == REQUESTED


def test_untrusted_comment_during_evaluation_does_not_force_churn(tmp_path, monkeypatch):
    fake = FakeGitHub(comments=[(T0, verdict("APPROVE"))],
                      mutate_after_first_comment_read=(T1, REJECTION, "stranger"))
    statuses = run_gate(fake, tmp_path, monkeypatch)
    assert [(s["state"], s["description"]) for s in statuses] == [APPROVED]


PULL = f"GET /repos/{REPO}/pulls/{PR}"
PULL_URL = f"https://api.github.com/repos/{REPO}/pulls/{PR}"


def test_transient_api_failure_recovers_without_duplicate_status_write(tmp_path, monkeypatch):
    fake = FakeGitHub(comments=[(T0, verdict("APPROVE"))], failures={
        PULL: [http_error(PULL_URL, 502), http_error(PULL_URL, 503), http_error(PULL_URL, 504)]})
    statuses = run_gate(fake, tmp_path, monkeypatch)
    assert fake.trace.count(PULL) == 5
    assert [(s["state"], s["description"]) for s in statuses] == [APPROVED]


def test_connection_failure_recovers_within_the_request_bound(tmp_path, monkeypatch):
    fake = FakeGitHub(comments=[(T0, verdict("APPROVE"))], failures={PULL: [
        urllib.error.URLError(ConnectionError("connection reset")),
        TimeoutError("request timed out"),
        urllib.error.URLError(socket.gaierror(socket.EAI_AGAIN, "temporary DNS failure")),
    ]})
    statuses = run_gate(fake, tmp_path, monkeypatch)
    assert fake.trace.count(PULL) == 5
    assert len(statuses) == 1


@pytest.mark.parametrize("failure, attempts", [
    pytest.param([http_error(PULL_URL, 504) for _ in range(4)], 4, id="bounded-transient"),
    pytest.param([http_error(PULL_URL, 422)], 1, id="non-transient-http"),
    pytest.param([urllib.error.URLError(ValueError("unsupported URL scheme"))], 1, id="non-connection"),
    pytest.param([urllib.error.URLError(socket.gaierror(socket.EAI_NONAME, "name not known"))], 1,
                 id="permanent-dns"),
])
def test_failures_reading_the_pr_stop_without_a_status(tmp_path, monkeypatch, failure, attempts):
    fake = FakeGitHub(comments=[(T0, verdict("APPROVE"))], failures={PULL: failure})
    with pytest.raises(urllib.error.URLError):
        run_gate(fake, tmp_path, monkeypatch)
    assert fake.trace.count(PULL) == attempts
    assert fake.statuses == []


def test_ambiguous_status_write_failure_is_not_retried(tmp_path, monkeypatch):
    fake = FakeGitHub(comments=[(T0, verdict("APPROVE"))],
                      post_status_failures=[TimeoutError("response timed out")])
    with pytest.raises(TimeoutError):
        run_gate(fake, tmp_path, monkeypatch)
    assert fake.trace.count(f"POST /repos/{REPO}/statuses/{HEAD_SHA}") == 1
    assert len(fake.statuses) == 1


# --- Review relay -----------------------------------------------------------

RELAY_RUN = {
    "event": "pull_request_review",
    "path": ".github/workflows/independent-review-events.yml",
    "head_repository": {"full_name": REPO},
    "head_branch": "review-branch",
    "pull_requests": [],
}
CANDIDATE = {"number": PR, "state": "open", "base": {"ref": "main"},
             "head": {"repo": {"full_name": REPO}, "ref": "review-branch"}}


def test_review_relay_resolves_its_source_branch(tmp_path, monkeypatch):
    fake = FakeGitHub(relay_run=RELAY_RUN, comments=[(T0, verdict("APPROVE"))])
    statuses = run_gate(fake, tmp_path, monkeypatch)
    assert (statuses[-1]["state"], statuses[-1]["description"]) == APPROVED
    assert f"?pr={PR}&run=123" in statuses[-1]["target_url"]


def test_review_relay_resolves_a_fork_branch(tmp_path, monkeypatch):
    fork = {"full_name": "someone/localLatin"}
    fake = FakeGitHub(relay_run={**RELAY_RUN, "head_repository": fork},
                      relay_candidates=[{**CANDIDATE, "head": {"repo": fork, "ref": "review-branch"}}],
                      comments=[(T0, verdict("APPROVE"))])
    statuses = run_gate(fake, tmp_path, monkeypatch)
    assert (statuses[-1]["state"], statuses[-1]["description"]) == APPROVED


@pytest.mark.parametrize("field, value", [
    ("event", "push"),
    ("path", ".github/workflows/untrusted.yml"),
    ("head_repository", None),
    ("head_branch", ""),
])
def test_review_relay_rejects_other_runs(tmp_path, monkeypatch, field, value):
    fake = FakeGitHub(relay_run={**RELAY_RUN, field: value})
    with pytest.raises(SystemExit, match="review-event relay"):
        run_gate(fake, tmp_path, monkeypatch)
    assert fake.statuses == []


@pytest.mark.parametrize("candidates", [
    [],
    [CANDIDATE, CANDIDATE],
    [{**CANDIDATE, "state": "closed"}],
    [{**CANDIDATE, "base": {"ref": "other"}}],
    [{**CANDIDATE, "head": {"repo": None, "ref": "review-branch"}}],
    [{**CANDIDATE, "head": {"repo": {"full_name": "fork/localLatin"}, "ref": "review-branch"}}],
    [{**CANDIDATE, "head": {"repo": {"full_name": REPO}, "ref": "other"}}],
])
def test_review_relay_needs_exactly_one_matching_open_pr(tmp_path, monkeypatch, candidates):
    fake = FakeGitHub(relay_run=RELAY_RUN, relay_candidates=candidates)
    with pytest.raises(SystemExit, match="exactly one"):
        run_gate(fake, tmp_path, monkeypatch)
    assert fake.statuses == []
