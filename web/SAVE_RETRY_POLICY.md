# Save retry policy

What the review client may do when a save does not obviously succeed, and why
it does so little. Companion to [FEEDBACK_WRITE_CONTRACT.md](FEEDBACK_WRITE_CONTRACT.md),
which describes what the server accepts.

## The one fact everything follows from

`POST /api/feedback` appends. There is no update, no delete and no idempotency
key, and `feedback.py` commits the feedback row before it writes reviewer
directory membership. So:

- a fault after the request left the browser does **not** mean nothing was
  written;
- a second attempt does **not** replace the first, it adds a second human
  decision to the log;
- nothing in the API can tell the two apart afterwards.

A retry is therefore a decision about the research record, not a transport
detail, and it belongs to the reviewer.

## What the client does

| Situation | What is known | What the client does |
| --- | --- | --- |
| Refused in the browser (no candidate, note-less skip) | Nothing was sent | Explains, keeps the draft, no request |
| 4xx from `/api/feedback` | Nothing was written: every 4xx on this route is raised before `db.insert` | Says "Nothing was saved", keeps the draft, lets the reviewer save again |
| 5xx, dropped connection, unreadable or mismatched body | Unknown | Says it cannot tell, keeps the draft, asks the reviewer to check the last review for the document first |
| Receipt for this request | The row exists | Clears the submitted draft revision, acknowledges, then looks for the next document |
| Next-document lookup fails | The save already succeeded | Says so, offers to repeat **only** the lookup |

Nothing above retries on a timer, on focus, on reconnect or on unmount. The only
repeat request is one the reviewer asks for by pressing a button again, with the
uncertainty in front of them.

## What counts as proof of a write

Only a readable `FeedbackEntry` body that is the row just asked for
(`src/api/feedback.ts`). Every field a new write fills is checked for shape, and
these are compared against the request:

| Field | Must equal |
| --- | --- |
| `query_id`, `model_slug` | the assessment that was sent (slug compared normalised) |
| `variant`, `outcome` | what was sent, when the request carried them |
| `notes` | the note that was sent, exactly |
| `selected_ranks` | the choices that were sent, in the order they were made |
| `correct_rank` | the canonical choice, and `null` for a skip |
| `correct_dir` | the directory `expected_candidate_dirs` gave for that rank, and `null` for a skip or none-of-these |
| `reviewer_account_id` | the account that was signed in when the request went out |

Anything else -- a missing field, a wrong type, a mismatch -- is an *uncertain*
outcome: the draft stays, and the reviewer is told the save could not be
confirmed. The directory and the choices matter as much as the ids because they
are the assignment being recorded; a row that names another directory is a
different decision, not a receipt for this one.

HTTP 200 on its own is not proof: a proxy error page, a truncated body or an answer
to somebody else's request all arrive with a status code, and treating one as a
receipt deletes a draft nobody can show was saved.

`GET /api/feedback/latest` is not proof either. It is the latest review for the
query, merged across reviewers, and it can be stale, can belong to someone else,
and can be answered out of order relative to a save. It seeds an empty form; it
never confirms one.

The dev mock is held to the same contract. `npm run dev:mock` answers
`POST /api/feedback` with a full row built from the payload -- canonical rank,
resolved directory, ordered choices, the note verbatim, the signed-in reviewer --
because a stub that answers `{ success: true }` is refused by the client, so
mock mode could not save at all.

## Why an explicit retry can still be wrong

Even when the reviewer chooses to save again, the second row may differ from the
first in ways nobody sees at the time:

- if the ranking was reloaded in between, a rank number can now name a different
  directory. `expected_candidate_dirs` catches exactly that and rejects the save
  with `CANDIDATE_IDENTITY_CHANGED`, which is a guard against a *changed*
  meaning, not an exactly-once guarantee;
- a duplicate that passes the guard is a real, indistinguishable second
  assessment in the analysis set.

Hence the copy on an uncertain failure asks the reviewer to look at the last
review for the document before pressing anything.

## Duplicate suppression in the client

The client holds one in-flight operation per assessment (reviewer, query, model,
variant): a second Submit, or a Skip racing a Submit, does nothing and reports
`already_pending`. The lock is taken synchronously before the first `await`, so
two calls in the same tick cannot both pass, and it is keyed per assessment, so
a save on one document never blocks another.

This is duplicate *request* suppression in one browser tab. It is not
server-side deduplication: two tabs, two devices or a deliberate second click
after a failure can still append twice.

## Navigation is a separate operation

Moving to the next document happens after an acknowledged save and can fail on
its own. A failed lookup is reported as a failed move, never as a failed save,
and its retry repeats the lookup alone. Saying "save failed" there would send a
reviewer to press Submit again, which is precisely how an append-only log
acquires duplicates.

The move is also abandoned, rather than retried, whenever the assessment it
belongs to is no longer on screen: another query, model, variant or reviewer, a
second visit to the same document, another view, or an unmounted panel. And it
is held when the reviewer has typed since the save, because navigating would
hide unsent work.

A 200 is not by itself a next document (issue #171). `apiFetch`'s generic is a
type assertion, not a check, so a null body or an object with no usable
`file_id` used to throw where nothing was catching, on a promise the caller
discards: silent on screen and an unhandled rejection in the suite. The
response is read at this boundary instead, anything unusable is reported
through the same visible failed-move notice, and a `file_id` of `null` is kept
as what the route means by it -- there is nothing left to review.

## Whose answer is it

Every save, lookup and prefill carries the assessment *and the visit* it started
on. A visit ends when the reviewer leaves that assessment or leaves the review
screen, and also when the panel itself is unmounted and mounted again -- so
returning to the same document, by any route, is a new visit even though the key
is unchanged. A completion is checked against the current visit before it does
anything at all -- before it cancels a timer, replaces a recovery, clears a
draft or shows a notice -- so a save finishing on the document behind can never
steer the one in front. The acknowledgement in the sidebar additionally checks
the signed-in account, so a colleague who signs in next is never shown someone
else's save or offered their draft.

An operation in flight belongs to the assessment, not to the buttons: the
provider holds it, keyed per assessment, so collapsing the sidebar and opening
it again shows a save still in progress rather than an idle Submit. Its outcome
is held the same way. When a save settles, the receipt or the failure is
recorded against the assessment, so an answer that arrives while the controls
are unmounted is shown when they come back: a failure that landed with the
sidebar collapsed still reports itself, with the draft intact, instead of
leaving an idle Submit that invites an unwarned second append.

Those two rules meet at a remount, and they resolve in opposite directions on
purpose. The request is not cancelled -- it is the same assessment -- but the
visit it began in is over, so a success it brings back does not clear the
returned-to draft, does not move the reviewer on and does not raise the saved
toast. The panel says the row was recorded after they left that screen, and
whether what is in the box now is newer work that is still unsent.

The prefill guard is keyed per assessment too, so a save on one document no
longer suppresses a colleague's note on the next, and a prefill is refused if
the box has been edited since the request went out -- including an edit typed
and undone, which is a decision about the note, not an absence of one.

Notices about unsent work are read from the box at the moment they are shown,
never from a snapshot taken when the receipt arrived. A reviewer who keeps
typing while the next-document lookup is out is told their newer draft is
unsent, not that nothing here needs saving.

## Undo is local

The acknowledgement toast can put the submitted draft back in the box. It sends
nothing, and it removes neither the feedback row nor any reviewer-directory
membership the save created. It is also withheld once newer unsent work exists
under the same key, since restoring would overwrite it.
