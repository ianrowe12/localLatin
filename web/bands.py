"""Confidence bands on the displayed similarity.

One place, deliberately. Two features read the same numbers:

* the reviewer UI (issue #94) paints a candidate list red below ``NO_MATCH``,
  cautious between the bands, and calm above ``VERIFY``;
* the reviewer-directory loop (issue #95) calls a reviewer-created directory
  *matched* once some other query scores at least ``NO_MATCH`` against it.

The two must agree: a directory the backend calls matched has to be one the UI
paints as a match, or the badge and the card contradict each other on the same
screen.

WHO OWNS THE NUMBER. The backend, and the frontend reads it. The band is not
merely presentational here -- the server uses it to decide a persisted
directory's status -- so a hardcoded copy in TypeScript would be a second
source of truth that can silently drift from the one the API answers with.
``GET /api/models`` therefore carries a ``confidence_bands`` object on every
entry (alongside ``default_variant``, which is deployment-wide in the same
way), and the frontend's ``api/bands.ts`` reads it, falling back to these same
literals only before the first response lands.

CALIBRATION. Learned per-model tau under ``sif_abtt`` is 0.34-0.43 on the train
split, so 0.5 is deliberately conservative: a score below it is well under
every model's learned decision boundary, which is what makes an overt "no
match" treatment defensible.

These are similarity thresholds, not probabilities. Changing them changes what
reviewers are shown *and* which directories count as matched, so treat an edit
as a protocol change, not a tweak.
"""

from __future__ import annotations

#: Below this, the top candidate is flagged as "potentially no match" and the
#: new-directory CTA becomes the default action.
NO_MATCH_BAND: float = 0.5

#: At or above this, a candidate is framed as "likely match - verify".
VERIFY_BAND: float = 0.7

#: A reviewer-created directory leaves 'awaiting_match' once another query
#: scores at least this much against it. Same number as NO_MATCH_BAND by
#: definition: "matched" means "the UI would not paint this as a non-match".
REVIEWER_DIR_MATCH_BAND: float = NO_MATCH_BAND
