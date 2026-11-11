# Independent Review Protocol

Every pull request in this repo is reviewed by a fresh agent that did not write the change
before it merges. The author re-reads its own code as correct, because it carries the reasons
for every decision it made. A cold reviewer re-derives what the change should do from the issue
and the diff alone.

This file is the protocol. The merge gate that enforces part of it is described in
[Enforcement](#enforcement). It was ported from Ian's Aria repo (issue #267) and trimmed for a
small research repo with one orchestrator.

## The rule

1. Once the PR is open, scope-complete and conflict-free, the orchestrator pins its head SHA
   and starts **one fresh reviewer agent** that had no part in writing the change. A subagent
   qualifies if it starts from a clean context and reads the diff cold. The reviewer is
   **Opus**: launch it with the Agent tool and `model: "opus"`. Start the review while CI runs;
   do not wait for CI to go green.
2. Immediately before posting, the reviewer checks that the PR head still matches the pinned
   SHA. If it moved, the reviewer posts nothing for the old head and reviews the new one.
3. The reviewer posts all of a round's findings and its verdict in **one** PR comment, with file
   and line for each finding. The author never summarizes findings away. Every finding starts
   with a severity prefix, `BLOCKING:` or `ADVISORY:` (see [Finding severity](#finding-severity)).
4. The comment ends with a closing block: exactly one `Reviewed head:` line naming the full
   40-character pinned SHA, a `Reviewed by:` line, and the verdict marker as the **final line**.
   Copy one of these blocks as plain text, without the code fences:

   ```
   Reviewed head: <full 40-character SHA>
   Reviewed by: unknown
   Independent review verdict: APPROVE
   ```
   or
   ```
   Reviewed head: <full 40-character SHA>
   Reviewed by: unknown
   Independent review verdict: REQUEST CHANGES
   ```

   Parsing rules the gate applies, so the reviewer does not trip them:
   - The verdict is the comment's last non-blank line, after trimming whitespace and Markdown
     emphasis. `**Independent review verdict: APPROVE**` counts.
   - No line outside the closing block may start with `Independent review verdict`,
     `Verdict:` or `Reviewed head:`. An earlier marker line that disagrees with the final one
     makes the comment conflicting, which keeps the gate red.
   - To recap an earlier verdict, quote it with `>`. Quoted lines are ignored.
   - A marker that is not the final line never approves. With exactly one `Reviewed head:` line
     it blocks that head; without one it reads as a recap and is ignored.
   - A `REQUEST CHANGES` posted after the current head's `APPROVE` keeps the gate red even when
     it lacks exactly one `Reviewed head:` line.
   - The gate does not parse Markdown. A `Reviewed head:` line counts wherever it sits, inside a
     fence or a list included. A SHA in prose does not count.
   - `Reviewed by:` is the reviewer's own claim about its model. A model cannot reliably tell
     which weights serve it, so `unknown` is the correct answer unless the runtime states it.
     The gate ignores this line.
5. The author addresses **every** finding before merge. A blocking finding is fixed, or answered
   with why it is wrong; an answered blocking finding counts as resolved only when a later
   reviewer pass reads the answer and posts `APPROVE`. An advisory finding gets a one-line reply
   and, only if it clears the [filing bar](#advisory-findings-are-not-automatically-issues), an
   issue. Silent dismissal is a protocol violation.
6. The reviewer never pushes fixes and never merges. Review and authorship stay separate.

Merge needs green CI and a fresh `APPROVE` on the current head. They can finish in either order.

## Finding severity

Every finding starts with one of two prefixes. There is no third bucket and no unlabelled finding.

- **`BLOCKING:`** the change must not merge in this state. Any open blocking finding makes the
  verdict `REQUEST CHANGES`.
- **`ADVISORY:`** not merge work for this PR. It is **never** fixed in this PR, not even in a
  push that fixes blocking findings. Advisory edits enlarge the diff the next reviewer must
  re-read, which is the cheapest place for an unreviewed regression to hide.

`APPROVE` means "merge this head as it stands". A reviewer that wants a change in this PR marks
the finding `BLOCKING` and votes `REQUEST CHANGES`. The test for blocking is "I would not ship
this head", not "this change is worth making". Never downgrade a blocking finding to unblock a
PR, and never inflate an advisory one to get it done here.

## Advisory findings are not automatically issues

An advisory finding lives in the review comment. File a follow-up issue only when it is:

- a wrong number, claim or provenance statement that would reach the paper or the webapp,
- a security, privacy or data-loss risk,
- something that breaks CI, a deploy or a scheduled job, or
- a violation of a documented decision (`CLAUDE.md`, `docs/research/benchmark_v1.md`, the
  paper writing rules below), even when the code works.

Everything else stays a note: style and naming, doc phrasing, speculative hardening, "this
could be stronger". An issue is a commitment to do work; the review comment is already the
audit trail. If the same class of finding keeps recurring and a script could decide it, propose
a CI check instead of filing another instance.

The author replies to every advisory finding with one of three dispositions: **filed** (issue
link), **below the bar** (say which part falls short), or **false positive** (say why). None of
them gates merge or buys another review round.

## Test findings need a mutation proof

"This test could be stronger" is not a finding. A test that passes while the behavior it is named
for is broken **is** one, because it reports safety that does not exist.

To raise a test finding, name a concrete mutation to the source that breaks the named behavior,
say exactly where it goes (file, line, and the enclosing condition), run the suite, and report
that it still passes. Show that the mutation actually changed the code, for example with
`git diff`, since a mutation that never applied also leaves the suite green. A demonstrated false
green is `BLOCKING`. Undemonstrated unease about a test is not a finding at all.

## The stop rule: an APPROVE ends the cycle

After an `APPROVE` on the current head, the author files the advisory findings that clear the
bar, replies to the rest, and merges. Push to the branch again only for:

- a CI failure on the approved head (needs a [CI-fix re-verdict](#ci-fix-and-branch-update-re-verdicts)),
- a merge conflict or a required branch update (needs a branch-update confirmation), or
- a genuinely blocking defect found after the verdict.

Polishing an approved head is a protocol violation. It turns the gate red, costs a CI run and a
review, and puts unreviewed code on the branch.

## Review rounds are capped

A PR gets at most **three `REQUEST CHANGES` rounds**, then one closing round, then an arbiter.
Count rounds by counting `REQUEST CHANGES` verdicts on the PR. No step waits on Ian.

1. **Round 1 reads everything.** The reviewer reads the whole diff cold.
2. **Later rounds review what changed.** The reviewer checks that every blocking finding from the
   previous round was fixed or answered, then reviews only `git diff <last-reviewed-sha> <head>`
   (`git range-diff` after a rebase). A new blocking finding is allowed only in code that delta
   touched, including a regression the fix introduced, or for a **critical defect** anywhere in
   the PR. Everything else in already-reviewed code is advisory.
3. **After the third `REQUEST CHANGES`, one closing round.** The closing reviewer checks only that
   the round-3 blocking findings are fixed and that the fix added no critical defect.
4. **If the closing round requests changes, the arbiter decides.** Do not start another round.

A **critical defect** in this repo is one of: a secret, token or private data in a commit, a log or
a public artifact; data loss or corruption (the reviewer feedback DB, `runs/active/`, the frozen
benchmark); a production outage of the webapp; or a paper number or claim that is wrong and would
reach a submission.

The author answers all of a round's findings in one reply. Repeated blocking rounds in the same
place usually mean the change is under-specified; say so and split the PR early.

### CI-fix and branch-update re-verdicts

Neither of these is a round when it approves. Both review only the delta.

- **Get the verdict before chasing CI.** If CI goes red while a review is in flight, do not push a
  fix mid-review. Wait for the verdict, then push the CI fix together with any blocking fixes.
- **CI fix after an `APPROVE`.** The author states the approved SHA, the failing check with a run
  link, and one line on what the fix changes. The reviewer reads only
  `git diff <approved-sha> <head>` and blocks only if the delta does more than the failure needs,
  makes CI green by weakening a test, assertion or timeout, introduces a defect in the lines it
  changes, or adds a critical defect.
- **Branch update after an `APPROVE`.** When the only push is a merge or rebase onto `main`, the
  reviewer runs `git range-diff` to confirm the PR's own changes are unchanged, reads any resolved
  conflict hunks, and re-posts `APPROVE` for the new head. Do not update a branch that is not
  conflicting: the ruleset does not require up-to-date branches.

A re-verdict that posts `REQUEST CHANGES` is a round and counts toward the cap.

### The arbiter closes a capped loop

The arbiter is a fresh **Opus** agent (`model: "opus"`) that wrote none of the PR's code and posted
none of its verdicts. It reads the PR's goal and linked issue, the whole diff against `main`, and
every verdict, finding and reply. It rules on what is open; it does not start a new review.

- **Fix now:** any critical defect, whatever severity a reviewer gave it, and a head that does not
  do what the PR says. The arbiter pushes the smallest fix, touching only those lines.
- **Ship and file:** everything else, filed as one `needs-triage` issue that quotes each finding
  and links its review comment.

If it pushed nothing, it posts one comment starting `Arbiter ruling` with its ruling and a normal
`APPROVE` closing block. If it pushed fixes it cannot approve its own code: it posts the ruling
with `Arbiter base: <sha>` (the head before its first push) and no verdict, and a fresh Opus
reviewer runs a delta check on `git diff <arbiter-base> <head>`. The delta check blocks only on a
critical defect or on a fix that weakens a check. Two blocking delta checks in total end the
arbiter stage: the arbiter converts the PR to draft and files one issue holding the critical
finding, and the work restarts in a fresh PR. The arbiter never merges and never adds the
exemption label.

## Review dimensions

The reviewer hunts, at minimum, for:

- **Logic errors and edge cases:** off-by-ones, empty inputs, partial failures, wrong defaults.
- **Contract mismatches:** does the change do what the issue says, re-derived from the issue
  rather than the PR description?
- **Regressions of documented decisions** in `CLAUDE.md` and `docs/research/`. A working change
  that contradicts one is still a finding.
- **Tests:** do they exercise the behavior they are named for (see the mutation-proof rule)?

**Research and data rules** (from `CLAUDE.md` and `docs/research/benchmark_v1.md`):

- All selection is train-only: layers, thresholds, ABTT `D`, SIF probabilities and the
  `EmbeddingCleaner` fit use train files only (`--split_csv`). Any test-set leak is blocking.
- Embedding caches are never paired with split rows by position; loads go through
  `AlignmentResolver`.
- The labelled corpus is frozen as benchmark v1. New CCL material goes to v2, never into it.
- GPU jobs need Ian's approval outside issue #47, and every sbatch sets a realistic `--time`.

**Paper rules** for anything under `overleaf_drafts/` (`CLAUDE.md`,
`docs/research/paper_writing_guidelines.md`):

- No em-dashes.
- The corpus comes from the Carolingian Canon Law project. Never write "Latin department",
  and never "we annotate": the labels are expert-assigned by the project.
- Never call a directory a "family". "Family" is only for model families.
- No claim that embeddings beat surface (lexical) overlap.
- Every reported number comes from a generator script (for example the tables under
  `overleaf_drafts/tables/`). A number typed by hand into the prose or a table is a finding.
- Page fit: the main text ends on page 8 and Limitations starts on page 9. A change that
  pushes either is blocking.
- Short plain sentences, self-contained captions, no puffery.

**Public-repo and privacy rules.** This repo and its Actions logs are public.

- No secrets, tokens, credentials or private keys in commits, workflow output or artifacts.
- No webapp reviewer identities (names, emails, usernames) in Actions logs, commit messages or
  committed files. Workflows that touch the feedback DB redact them.
- The CCL TEI exports are not committed (see `docs/research/data_derivation.md`).
- A workflow that runs on `pull_request_target`, `issue_comment` or `workflow_run` must never
  check out or execute code from the PR.

## Enforcement

`.github/workflows/independent-review-gate.yml` sets the **`Independent Review`** commit status on
the PR head. The `main` ruleset requires it. It is green only when:

- the newest trusted verdict whose single `Reviewed head:` line names the current head is
  `APPROVE`, and no later unbound `REQUEST CHANGES` follows it, or
- the PR carries the `independent-review-exempt` label.

Everything else is red: no verdict, `REQUEST CHANGES`, a verdict for an older head, conflicting
markers, or two verdicts that disagree at the same timestamp. A push moves the head, so the
status goes red until a fresh verdict names the new head. Creating, editing or deleting a PR
comment, and submitting, editing or dismissing a review, re-runs the gate when the comment or
review was written by an `OWNER`, `MEMBER` or `COLLABORATOR` (see the flood prefilter below), so
removing a trusted verdict cannot leave a stale green. Gate jobs run one at a time and re-read the verdicts just before
writing. If a run fails after reading the head, it writes a red status rather than leave an
older green one standing.

**Trust filter.** Anyone with a GitHub account can comment on a public repo's PRs. The gate counts
a verdict only if its author has write, maintain or admin permission on the repo, checked live
per run (`GET /repos/{repo}/collaborators/{user}/permission`). A verdict from any other account
neither approves nor blocks. Bot accounts and deleted accounts are ignored. The gate's log prints
how many verdicts it ignored, never whose.

**Flood prefilter.** So that outsiders cannot fill the gate's shared queue or spend the token's
API quota, the gate job's `if:` skips a comment event unless the comment's `author_association`
is `OWNER`, `MEMBER` or `COLLABORATOR`, and the review relay skips reviews the same way. The gate
also skips a relay run that concluded `skipped`. The prefilter never applies to
`pull_request_target` events. Skipping is safe because an outsider's comment never counts, an
outsider cannot edit or delete a trusted comment, and edit and delete events carry the comment
author's association rather than the actor's, so a trusted author's edit or delete still
re-evaluates.

**Safety of the workflow itself.** The gate runs from the default branch on every trigger, checks
out only its own script from the default branch, and reads everything else from the REST API.
Nothing from the PR is checked out or executed. Pull request reviews reach it through
`independent-review-events.yml`, a relay with no token and no checkout.

**Honest limits.** The gate checks the verdict and its head binding. It does not check the
review's substance or the reviewer's independence. Reviewer agents post through `gh` as
`ianrowe12`, the same account the author agents use, so the machine cannot tell a self-review
from a real one. An author that posts its own `APPROVE` turns the gate green; that is a protocol
violation that leaves a permanent audit trail, not something the gate prevents. Other limits:

- The gate assumes trusted writers. The `main` ruleset accepts an `Independent Review` status
  from any GitHub Actions workflow, so anyone with write access can push a branch with a
  workflow that requests `statuses: write` and posts its own passing status. Only the repo owner
  and trusted agents have write access.
- The flood prefilter is a cheap first cut, not the trust filter. `author_association` admits
  read and triage collaborators too, so the live permission check in the script stays
  authoritative. The prefilter is also incomplete: on a fork PR the relay is the fork's copy, so
  an outsider can remove it for reviews on their own PR, and pushes to a fork PR
  (`pull_request_target`) are never prefiltered. Either can still fill the queue or spend the
  quota: gate runs then fail red, and an overflowing queue can drop a real event's job, which
  leaves that PR's status out of date until its next event. Neither produces a new green. If
  GitHub ever reported a trusted writer's association outside the three
  values, that writer's comment events would be skipped too, so their status would update only
  on the PR's next qualifying event.
- Commit statuses belong to a SHA, not a PR. A fork PR whose head is the same commit as a real
  PR can turn that commit's status red until the real PR's next event. It can never turn it
  green.
- The trust filter keeps outsiders out. It does not check who applied the exemption label;
  only accounts with triage access or higher can label, and this repo has no triage-only
  collaborators.
- A rejection whose marker is not its comment's final line, and that lacks exactly one
  `Reviewed head:` line, does not block. Without a Markdown parser the gate cannot tell it from a
  recap.
- Editing an old verdict comment refreshes its timestamp and can make it the newest verdict for
  the head it names.

The gate makes skipping review hard to do by accident, not impossible to do on purpose.

**Emergency exemption.** Apply `independent-review-exempt` only with Ian's explicit approval for
that specific PR, and only for a genuine emergency such as a broken production deploy. Say in a PR
comment who approved it and why. The label is an audit trail, not a convenience. The round cap
ends with an arbiter verdict and never needs it.

## Paste-ready reviewer prompt

Launch a fresh Opus agent (Agent tool, `model: "opus"`) that had no part in the change, replace
`<PR_NUMBER>`, and paste:

````text
You are an independent reviewer for the ianrowe12/localLatin repo. You did not write the change
you are about to review. Never push, edit code, merge, or approve through GitHub's review UI.
Review PR #<PR_NUMBER>.

1. Read docs/agents/independent-review.md (the protocol you are executing) and CLAUDE.md.
2. Run `gh pr view <PR_NUMBER> --repo ianrowe12/localLatin --json headRefOid,mergeable,body`
   and pin headRefOid. If the PR is CONFLICTING, post one comment naming the conflict, post no
   verdict, and stop. Read the issue the body closes and re-derive what the change should do
   from it, not from the PR description.
3. Read the full diff: `gh pr diff <PR_NUMBER> --repo ianrowe12/localLatin`, and the
   surrounding code wherever the diff alone is ambiguous. Then read the PR's earlier comments
   (`gh pr view <PR_NUMBER> --repo ianrowe12/localLatin --comments`) and pick your scope from
   "Review rounds are capped":
   - No earlier verdict: round 1, review everything.
   - Earlier verdicts: check every blocking finding from the last round was fixed or answered,
     then review only `git diff <last-reviewed-sha> <pinned-sha>`. Block only on code that delta
     touched or on a critical defect anywhere.
   - Newest verdict is APPROVE and the push is only a merge/rebase onto main or only a CI fix:
     follow "CI-fix and branch-update re-verdicts" and review only the delta.
   - Three REQUEST CHANGES verdicts already exist: this is the closing round. Check only that the
     round-3 blocking findings are fixed and the fix added no critical defect.
   - The closing round already requested changes: post nothing and report that the PR needs the
     arbiter.
4. Hunt across every review dimension in the protocol, including the research, paper and
   public-repo rules. Run the relevant tests where you can.
5. Immediately before posting, rerun
   `gh pr view <PR_NUMBER> --repo ianrowe12/localLatin --json headRefOid`. If it changed, do not
   post; review the new head instead.
6. Post ONE PR comment (`gh pr comment <PR_NUMBER> --repo ianrowe12/localLatin --body-file ...`).
   Every finding has file and line and starts with `BLOCKING:` or `ADVISORY:`. A test finding
   needs a mutation proof (the mutation, where it goes, and that the suite still passed).
   Do not file issues yourself; say in an advisory finding whether you think it clears the
   filing bar.
7. End that comment with the closing block from rule 4, as plain text: `Reviewed head: <pinned
   SHA>`, then `Reviewed by: <model, or unknown>`, then the verdict as the final line. Use
   APPROVE when nothing blocks merge. No other line in the comment may start with
   `Independent review verdict`, `Verdict:` or `Reviewed head:`; quote any recap with `>`.
````

## Paste-ready arbiter prompt

Use this only for a PR past the round cap. Launch a fresh Opus agent (`model: "opus"`) in a
worktree on the PR's head branch, replace `<PR_NUMBER>`, and paste:

````text
You are the arbiter for ianrowe12/localLatin PR #<PR_NUMBER>. It is past the review round cap.
You wrote none of its code and posted none of its verdicts. Your ruling ends the review loop.

1. Read "The arbiter closes a capped loop" and "Review rounds are capped" in
   docs/agents/independent-review.md, then CLAUDE.md.
2. Pin the head with `gh pr view <PR_NUMBER> --repo ianrowe12/localLatin --json headRefOid,body`.
   Read the linked issue, the whole diff against main, and every comment on the PR.
3. If two arbiter-stage delta checks (REQUEST CHANGES verdicts after the first comment starting
   `Arbiter ruling`) already blocked: run `gh pr ready <PR_NUMBER> --undo`, file one issue that
   quotes the open critical finding and links the PR, post a comment starting
   `Arbiter ruling: moved to draft` with the link, and stop.
4. List every open finding and check each against the current head. Fix now: any critical defect
   and any way the head fails to do what the PR says. Ship and file: everything else, in one
   `needs-triage` issue quoting each finding and linking its review comment.
5. For fix-now findings, push the smallest fix touching only those lines. Never weaken a test.
   Mutation-prove any test you add or change. Follow CLAUDE.md's commit rules.
6. Post one PR comment starting `Arbiter ruling` with your ruling on every open finding and the
   issue link.
   - Pushed nothing: end with the closing block for the pinned head, verdict APPROVE.
   - Pushed fixes: no verdict. Name `Arbiter base: <sha>` (the head before your first push) and
     the new head, so a fresh reviewer runs the delta check.
7. Never merge, never hand the PR back to Ian, never add the independent-review-exempt label.
````
