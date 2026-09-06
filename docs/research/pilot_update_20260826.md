# Reviewer pilot update, 26 August 2026

For Siddique and Abigail. Everything below is live on <https://ai.csr.uky.edu>.
This note covers what changed since the 25 August meeting, what will look
different from the last time you signed in, and how to recover a forgotten
password.

## The short version

| Change | What you will notice |
|---|---|
| Confidence bands | Every candidate is labelled likely match, check carefully, or no match. |
| New-directory loop | You can create a directory for a fragment that matches nothing, and it becomes a real candidate for later fragments. |
| Shared notes | You see your colleagues' reasoning on a query instead of only your own. |
| Passwords | You can change your own password. Abigail can reset anyone's. |
| One pipeline | The post-processing picker is gone. Everybody sees the same ranking. |
| Full attribution | The token-highlight view now works on essentially every query, not just four demo cases. |

## Confidence bands

The similarity score shown next to each candidate is now banded, using the two
thresholds fixed in the meeting and applied to the displayed SIF+ABTT
similarity:

- **0.7 and above: likely match.** Confirm it if the text agrees.
- **0.5 to 0.7: check carefully.** The model is unsure. Your reading decides.
- **Below 0.5: no match.** The page shows a red callout saying the fragment
  probably does not belong to any existing directory, with the option to create
  a new one.

The backend now owns these two numbers rather than the browser, so the band you
see and the badge the system stores can no longer disagree.

**Please read this part before you start.** On the default model (mT5-base),
**75% of the 2,238 unlabelled fragments score below 0.5**, 12% land between 0.5
and 0.7, and 13% score 0.7 or above. The median top score is 0.26. The red
no-match callout is therefore the normal case, not the exception. That is a
genuine property of the corpus rather than a bug: most unlabelled fragments have
no sibling among the labelled directories. But it does mean the thresholds sort
the queue into a small confident head and a long uncertain tail, and if you
expected the bands to split the work more evenly we should revisit the numbers
after you have worked through a few dozen fragments. The other five models sit
in the same range, between 70% and 74% below 0.5.

## New-directory loop, and "awaiting a match"

When a fragment matches nothing, you can create a directory for it. Two things
are new since the last build:

1. **The directory becomes a live candidate.** Every later fragment is scored
   against it, so the second half of a pair that arrives next week will surface
   the directory you created today. Reviewer directories appear below the model
   candidates, ranked among themselves, capped at five per query.
2. **The badge tells the truth about its state.** A new directory reads
   **awaiting a match** and keeps reading that until *a person* files a second
   fragment into it. It never flips on similarity alone. An earlier draft did
   flip on similarity and marked 57% to 70% of directories as matched the moment
   they were made, which would have been meaningless.

Two cautions. Creating a directory is **permanent**: nothing in the interface
deletes one, because a deletion would orphan the fragments filed into it. And
the same fragment can only ever seed one directory, so a double click is
refused rather than making a duplicate.

## Shared notes

Reopening a query now prefills the most recent note written by **anyone** on the
team, with a line above the box saying who wrote it and when. Previously you saw
only your own notes and had no way to tell that a colleague had already worked
the same fragment.

The parts that stay private to you are the decision fields: the rank you picked
and the outcome you chose are still your own. Only the prose is shared. Nothing
is ever overwritten. Saving appends a new row, and the full history stays in the
export.

## One pipeline

The post-processing variant picker has been removed for everybody. All
reviewers now see the SIF+ABTT ranking at the deployed layer, which was already
the default.

This matters only if you had switched the picker to something else. Compared
with the uncorrected `raw` ranking, the pipeline everyone now sees puts a
different directory in first place for **87%** of fragments on mT5-base (66% on
KaLM and Qwen, 55% to 60% on LaTa, PhilTa and LaBSE). So if you reviewed a batch
with the picker on `raw`, those rank-1 answers have moved and any notes that say
"the first one" may now point at a different directory. Notes tied to a specific
directory name are unaffected.

The SIF coefficient is frozen at 1e-3, as agreed.

## Full attribution coverage

The token-highlight view, which shades the words driving a similarity score,
previously existed for four hand-picked demo pairs. It now covers **39,647
query-candidate-model pairs**, which is every pair the interface can actually
show you across all six models. You can open the attribution panel on any
fragment and any of its ranked candidates and get a real answer instead of an
empty panel.

Highlights are computed at the exact layer and post-processing each ranking
uses, so what you see explains the score you are looking at rather than an
approximation of it.

## How to reset a forgotten password

There is no password reset email on this deployment, so recovery goes through
Abigail. If a reviewer forgets their password, Abigail signs in, opens the
account list, and uses **Reset password** on that reviewer's row. The system
generates a temporary password and shows it **once**, on that screen only, so it
has to be copied and passed to the reviewer straight away; it is never stored in
readable form or emailed. The reviewer signs in with it and is then required to
choose a new password before they can do anything else, and all of their old
sessions are signed out. One deliberate restriction: an administrator cannot
reset their own password this way, so if Abigail is ever locked out, a second
administrator account has to do it for her. That is why a second administrator
should exist before the pilot starts in earnest.

Reviewers who still know their password can change it themselves at any time
from the account menu, which keeps them signed in on the current browser and
signs out every other session.

## Questions worth your answer

1. Are the 0.5 and 0.7 thresholds still right given that 75% of fragments fall
   below the lower one? We can move them without a redeploy of the data.
2. Should a second administrator account exist now, so that a lockout is
   recoverable?
