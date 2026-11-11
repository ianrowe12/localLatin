#!/usr/bin/env python3
"""Decide and post the `Independent Review` commit status for one pull request.

Run by .github/workflows/independent-review-gate.yml from a checkout of the
default branch. It never reads or runs anything from the pull request's branch:
everything it needs comes from the GitHub REST API. Standard library only.

The protocol it enforces is docs/agents/independent-review.md. In short:

* A verdict is a PR comment (or review body) whose last non-blank line is
  `Independent review verdict: APPROVE` or `... REQUEST CHANGES`, and which
  carries exactly one `Reviewed head: <40-hex SHA>` line.
* The status is green only when the newest verdict bound to the current head is
  APPROVE, or when the `independent-review-exempt` label is on the PR.
* Only verdicts written by an account with write, maintain or admin permission
  on the repo count. This repo is public, so anyone can comment; an untrusted
  comment neither approves nor blocks. Bot accounts are ignored outright.
"""

from __future__ import annotations

import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from urllib.parse import quote, urlencode

API = "https://api.github.com"
CONTEXT = "Independent Review"
EXEMPT_LABEL = "independent-review-exempt"
BASE_BRANCH = "main"
RELAY_PATH = ".github/workflows/independent-review-events.yml"
PROTOCOL_DOC = "docs/agents/independent-review.md"

# The legacy `permission` field folds maintain into write and triage into read,
# so this set is exactly "write, maintain or admin".
TRUSTED_PERMISSIONS = frozenset({"admin", "write"})

TRANSIENT_STATUSES = frozenset({502, 503, 504})
REQUEST_ATTEMPTS = 4
DECISION_ATTEMPTS = 5

APPROVE = "approve"
REQUEST_CHANGES = "request changes"
CONFLICTING = "conflicting"
MISPLACED = "misplaced"

DESC_NONE = "No independent review verdict on this PR yet"
DESC_EXEMPT = "Exempt via independent-review-exempt label"
DESC_APPROVED = "Independent review approved for current head"
DESC_REQUESTED = "Independent review requested changes for current head"
DESC_TIE = "Conflicting independent review verdicts at latest timestamp"
DESC_CONFLICTING = "A verdict comment has marker lines that disagree; quote recaps with '>'"
DESC_MISPLACED = "A verdict marker is not its comment's final line; it cannot approve"
DESC_LATER_UNBOUND = "A later verdict lacks exactly one 'Reviewed head:' line; it blocks the approval"
DESC_UNBOUND = "Verdict found; reviewer must add exactly one 'Reviewed head: <40-hex SHA>' line"
DESC_STALE = "No verdict names the current head SHA; re-review required"
DESC_CHURN = "Verdicts changed during evaluation; retry required"
DESC_ERROR = "Gate could not finish evaluating; re-run the Independent Review Gate workflow"

# A comment's verdict is its last non-blank line, which must be the whole marker
# after trimming whitespace and Markdown emphasis. A recap quoted earlier in the
# comment, fenced or not, never becomes that comment's direction.
FINAL_MARKER = re.compile(
    r"[*_]{0,3}(independent review )?verdict[*_]{0,3}:?[*_]{0,3}\s*[*_]{0,3}"
    r"(approve|request[- ]?changes)[*_]{0,3}",
    re.IGNORECASE,
)
# Any line-anchored marker, emphasised or not. Quoted text ("> Independent review
# verdict: ...") and mid-sentence mentions do not match, and neither do
# "approved" or "APPROVE_pending". An earlier marker line that disagrees with
# the final one makes the comment conflicting.
MARKER = re.compile(
    r"^\s*[*_]{0,3}(independent review )?verdict[*_]{0,3}:?[*_]{0,3}\s*[*_]{0,3}"
    r"(approve|request[- ]?changes)(?=[*_]{0,3}(?![\w*]))",
    re.IGNORECASE | re.MULTILINE,
)
# The gate does not parse Markdown: a head line counts wherever it sits, fenced
# or not, and a second one anywhere leaves the verdict unbound.
HEAD_MARKER = re.compile(
    r"^[ \t]*Reviewed head:[ \t]*([0-9a-f]{40})[ \t]*\r?$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class Verdict:
    timestamp: str
    direction: str
    reviewed_head: str | None


@dataclass(frozen=True)
class Decision:
    sha: str
    state: str
    description: str
    generation: tuple


def _direction(word: str) -> str:
    return APPROVE if word.lower() == APPROVE else REQUEST_CHANGES


def parse_verdict(body: str, timestamp: str) -> Verdict | None:
    """Return the verdict a comment body carries, or None if it carries none."""
    directions = {_direction(match.group(2)) for match in MARKER.finditer(body)}
    heads = HEAD_MARKER.findall(body)
    reviewed_head = heads[0].lower() if len(heads) == 1 else None
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    final = FINAL_MARKER.fullmatch(lines[-1]) if lines else None
    if final:
        directions.add(_direction(final.group(2)))
    elif directions and reviewed_head:
        # A marker that is not the final line cannot approve. Bound to one head
        # it fails closed for that head; with no single head line it reads as a
        # recap and is ignored.
        directions = {MISPLACED}
    else:
        return None
    direction = directions.pop() if len(directions) == 1 else CONFLICTING
    return Verdict(timestamp=timestamp, direction=direction, reviewed_head=reviewed_head)


def decide(sha: str, labels: tuple[str, ...], verdicts: list[Verdict]) -> tuple[str, str]:
    """Return (state, description) for the current head. ISO-8601 UTC timestamps
    compare correctly as strings."""
    if EXEMPT_LABEL in labels:
        return "success", DESC_EXEMPT
    fresh = [verdict for verdict in verdicts if verdict.reviewed_head == sha]
    if fresh:
        latest = max(verdict.timestamp for verdict in fresh)
        latest_directions = {v.direction for v in fresh if v.timestamp == latest}
        if len(latest_directions) > 1:
            return "failure", DESC_TIE
        if latest_directions == {CONFLICTING}:
            return "failure", DESC_CONFLICTING
        if latest_directions == {MISPLACED}:
            return "failure", DESC_MISPLACED
        if latest_directions == {APPROVE}:
            # A later non-approval the gate cannot bind to one head still
            # outranks the approval it follows.
            if any(
                v.reviewed_head is None and v.direction != APPROVE and v.timestamp >= latest
                for v in verdicts
            ):
                return "failure", DESC_LATER_UNBOUND
            return "success", DESC_APPROVED
        return "failure", DESC_REQUESTED
    if verdicts:
        if any(verdict.reviewed_head is None for verdict in verdicts):
            return "failure", DESC_UNBOUND
        return "failure", DESC_STALE
    return "failure", DESC_NONE


def _is_transient_connection_error(exc: BaseException) -> bool:
    reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    if isinstance(reason, socket.gaierror):
        return reason.errno == socket.EAI_AGAIN
    return isinstance(reason, (ConnectionError, TimeoutError))


@dataclass
class GitHub:
    token: str
    repo: str
    # login -> trusted, cached for the run so a busy thread costs one lookup per author.
    trust_cache: dict[str, bool] = field(default_factory=dict)

    def request(self, path: str, data: bytes | None = None, method: str = "GET",
                retry_transient: bool = True):
        req = urllib.request.Request(
            API + path,
            data=data,
            method=method,
            headers={
                "Authorization": "Bearer " + self.token,
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
            },
        )
        for attempt in range(REQUEST_ATTEMPTS):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return json.load(resp)
            except urllib.error.HTTPError as exc:
                if (not retry_transient or exc.code not in TRANSIENT_STATUSES
                        or attempt == REQUEST_ATTEMPTS - 1):
                    raise
                exc.close()
            except (urllib.error.URLError, TimeoutError, ConnectionError, socket.gaierror) as exc:
                if (not retry_transient or not _is_transient_connection_error(exc)
                        or attempt == REQUEST_ATTEMPTS - 1):
                    raise
            time.sleep(0.5 * (2 ** attempt))

    def paginated(self, path: str) -> list:
        items, page = [], 1
        while True:
            batch = self.request(f"{path}?per_page=100&page={page}")
            items.extend(batch)
            if len(batch) < 100:
                return items
            page += 1

    def is_trusted(self, user: dict | None) -> bool:
        """True only for a human account with write, maintain or admin access.

        The live permission lookup is the source of truth. author_association is
        not used: COLLABORATOR and MEMBER include read and triage roles.
        """
        if not user or not user.get("login"):
            return False  # deleted ("ghost") accounts
        login = user["login"]
        if user.get("type") == "Bot" or login.endswith("[bot]"):
            return False
        if login not in self.trust_cache:
            try:
                info = self.request(
                    f"/repos/{self.repo}/collaborators/{quote(login, safe='')}/permission")
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    raise
                exc.close()
                info = {}
            self.trust_cache[login] = info.get("permission") in TRUSTED_PERMISSIONS
        return self.trust_cache[login]


def resolve_relay_pr(gh: GitHub, event_path: str) -> str:
    """Map a completed review-relay run to the one open PR it came from.

    Review runs carry no pull_requests list, so resolve the run's own head
    repository and branch. Fork heads are allowed: the relay only chooses which
    PR to re-evaluate, and the evaluation reads nothing from the fork.
    """
    with open(event_path, encoding="utf-8") as source:
        event = json.load(source)
    run = gh.request(f"/repos/{gh.repo}/actions/runs/{int(event['workflow_run']['id'])}")
    head_repo = (run.get("head_repository") or {}).get("full_name") or ""
    if (run.get("event") != "pull_request_review" or run.get("path") != RELAY_PATH
            or "/" not in head_repo or not run.get("head_branch")):
        raise SystemExit("::error::Expected a review-event relay run.")
    query = urlencode({
        "state": "open",
        "base": BASE_BRANCH,
        "head": head_repo.split("/")[0] + ":" + run["head_branch"],
        "per_page": 100,
    })
    candidates = [
        pr for pr in gh.request(f"/repos/{gh.repo}/pulls?{query}")
        if pr["state"] == "open" and pr["base"]["ref"] == BASE_BRANCH
        and pr["head"]["repo"] and pr["head"]["repo"]["full_name"] == head_repo
        and pr["head"]["ref"] == run["head_branch"]
    ]
    if len(candidates) != 1:
        raise SystemExit("::error::Review relay must resolve exactly one open PR into main.")
    return str(candidates[0]["number"])


def evaluate(gh: GitHub, pr_number: str, seen: dict) -> Decision | None:
    pr = gh.request(f"/repos/{gh.repo}/pulls/{pr_number}")
    if pr["state"] != "open":
        return None
    sha = pr["head"]["sha"].lower()
    seen["sha"] = sha
    labels = tuple(sorted(label["name"] for label in pr["labels"]))

    stamped, untrusted = [], 0
    for c in gh.paginated(f"/repos/{gh.repo}/issues/{pr_number}/comments"):
        stamped.append((c.get("user"), c.get("updated_at") or c.get("created_at") or "",
                        c.get("body") or ""))
    for r in gh.paginated(f"/repos/{gh.repo}/pulls/{pr_number}/reviews"):
        if r.get("state") != "DISMISSED":
            stamped.append((r.get("user"), r.get("submitted_at") or "", r.get("body") or ""))

    verdicts = []
    for user, timestamp, body in stamped:
        verdict = parse_verdict(body, timestamp)
        if verdict is None:
            continue
        # Parse first, so only verdict-shaped bodies cost a permission lookup.
        if not gh.is_trusted(user):
            untrusted += 1
            continue
        verdicts.append(verdict)

    state, description = decide(sha, labels, verdicts)
    generation = (
        sha,
        labels,
        tuple(sorted((v.timestamp, v.direction, v.reviewed_head or "") for v in verdicts)),
    )
    # Counts only: logins never go into this public repo's Actions logs.
    print(f"PR #{pr_number}: {len(verdicts)} trusted verdict(s), "
          f"{untrusted} ignored from accounts without write access.")
    return Decision(sha, state, description, generation)


def post_status(gh: GitHub, pr_number: str, sha: str, state: str, description: str,
                seen: dict) -> None:
    seen["posted"] = True
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    payload = {
        "state": state,
        "context": CONTEXT,
        "description": description,
        "target_url": (f"https://github.com/{gh.repo}/blob/{BASE_BRANCH}/{PROTOCOL_DOC}"
                       f"?pr={pr_number}&run={run_id}"),
    }
    # Not retried: a write whose response was lost may already have landed.
    gh.request(f"/repos/{gh.repo}/statuses/{sha}", data=json.dumps(payload).encode(),
               method="POST", retry_transient=False)
    print(f"{CONTEXT} status on {sha}: {state} ({description})")


def run(gh: GitHub, pr_number: str, seen: dict) -> None:
    decision = evaluate(gh, pr_number, seen)
    if decision is None:
        print(f"PR #{pr_number} is closed; skipping status update.")
        return
    # Concurrency already serialises gate jobs. Re-read before writing as well,
    # so a verdict that changed during evaluation replaces the stale decision
    # instead of ever being published.
    for _attempt in range(DECISION_ATTEMPTS):
        current = evaluate(gh, pr_number, seen)
        if current is None:
            return
        if current.generation != decision.generation:
            decision = current
            continue
        post_status(gh, pr_number, current.sha, current.state, current.description, seen)
        return
    post_status(gh, pr_number, decision.sha, "failure", DESC_CHURN, seen)


def main() -> int:
    gh = GitHub(token=os.environ["GH_TOKEN"], repo=os.environ["REPO"])
    if os.environ.get("GITHUB_EVENT_NAME") == "workflow_run":
        pr_number = resolve_relay_pr(gh, os.environ["GITHUB_EVENT_PATH"])
    else:
        pr_number = os.environ.get("PR_NUMBER", "")
    if not pr_number.isdigit():
        raise SystemExit("::error::No pull request number for this event.")

    seen: dict = {}
    try:
        run(gh, pr_number, seen)
    except Exception:
        # Fail closed: a crash after a new verdict arrived must not leave an
        # older green status standing. Skip this when a status write was already
        # attempted, since that write may have landed.
        if seen.get("sha") and not seen.get("posted"):
            try:
                post_status(gh, pr_number, seen["sha"], "failure", DESC_ERROR, seen)
            except Exception as exc:  # best effort; the original error is re-raised
                print(f"::warning::Could not post the fail-closed status: {exc!r}")
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
