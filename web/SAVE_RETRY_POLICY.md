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

Only a readable `FeedbackEntry` body whose `query_id`, `model_slug` and `outcome`
match the request just sent (`src/api/feedback.ts`). HTTP 200 is not proof: a
proxy error page, a truncated body or an answer to somebody else's request all
arrive with a status code, and treating one as a receipt deletes a draft nobody
can show was saved.

`GET /api/feedback/latest` is not proof either. It is the latest review for the
query, merged across reviewers, and it can be stale, can belong to someone else,
and can be answered out of order relative to a save. It seeds an empty form; it
never confirms one.

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
second visit to the same document, or an unmounted panel. And it is held when
the reviewer has typed since the save, because navigating would hide unsent work.

## Undo is local

The acknowledgement toast can put the submitted draft back in the box. It sends
nothing, and it removes neither the feedback row nor any reviewer-directory
membership the save created. It is also withheld once newer unsent work exists
under the same key, since restoring would overwrite it.
