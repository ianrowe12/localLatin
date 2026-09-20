/**
 * Reviewer-facing wording for provisional directories (issue #162).
 *
 * One module, because the same three facts have to be told the same way in the
 * creation form, the no-match callout, the saved acknowledgement and the tour:
 * creation writes a permanent record on its own, it is not the assessment, and
 * neither one is a claim about the CCL.
 *
 * Sentences only. The creation request, its pending state and where the
 * acknowledgement is mounted belong to the components (issue #161), so a later
 * behavioural change and a later wording change do not have to touch the same
 * lines.
 */

export const DIRECTORY_CREATION_COPY = {
  /**
   * What is left after issue #196 retired the creation flow in the prediction
   * panel. `SavedDirectoryNotice` reads a record it cannot write to, so the
   * only sentence it still needs is the one for a lookup that failed -- and
   * that one matters more than ever: "could not check" must not be shown as
   * "no directory here", now that nothing in this panel offers to make one.
   */
  unresolvedNote:
    'The app could not check whether this document already has a directory of its own. Nothing here depends on the answer; it is said rather than left blank.',

  savedHeading: 'Directory saved',
  /**
   * Deliberately conditional. A directory is scored from the model's q-q
   * matrix, is never offered to its own members, cannot be scored for an
   * excluded query, and only the closest few reach any one list, so
   * "a candidate for every other document" was a promise the app cannot keep.
   */
  savedAvailability:
    'Saved permanently and seeded with this document. It can be offered on other documents this model can score, not on every one.',
  savedIndependence: 'Submitting or skipping your assessment does not undo it.',
} as const

/**
 * The CCL key field on the blue "None of the top N" action (issue #196).
 *
 * Every sentence an evaluator reads about the key lives here, for the same
 * reason the creation copy did: the wording is Prof. Firey's subject matter and
 * the behaviour is not, so they change independently.
 *
 * Two things are said and a third is deliberately not. Said: the field is
 * optional, and what the server will do with what is typed. Not said: anything
 * that asks a general evaluator to go and search the CCL by hand. The field is
 * for a key they already know, which is the case Abigail described -- she
 * recognises the source and the ten candidates simply do not contain it.
 */
export const CCL_KEY_COPY = {
  fieldLabel: 'CCL key of the source, if known',
  placeholder: 'e.g. CTOU.567.16',
  /**
   * The whole of what is said BEFORE the key is recorded (issue #221).
   *
   * It used to be followed by a second sentence spelling out the three things
   * the server might do with a key: match a labelled directory, join a group a
   * colleague started, or start one. Since issue #221 hides the
   * reviewer-directory block, an evaluator can no longer see a group anywhere
   * in this app, so that sentence described mechanics they have no way to
   * observe. It is not lost: the receipt after Record still names the branch
   * the server actually took, which is the sentence that was ever worth
   * reading.
   */
  optionalNote:
    'Optional. Leave it blank to record only that none of the candidates match. If you know the source, type its CCL key and it is saved with your answer.',
  submit: 'Record this answer',
  submitting: 'Recording…',
  /** Heading over the answer already recorded for this document. */
  recordedHeading: 'Your recorded answer',
  /** Reopens the form. Recording again replaces nothing; it adds a new row. */
  change: 'Change this answer',
  changeNote:
    'Recording again adds a new answer to the log. It does not replace this one, and it cannot remove a group.',
  failed:
    'Nothing was recorded. Your answer and the key are still here, so you can try again.',
  uncertain:
    'The app did not get an answer it can trust, so it cannot tell whether this was recorded. Reload the document and check before recording it again.',
} as const

/** What the server did with a key, as the wire reports it. */
export type CclKeyActionName =
  | 'matched_labelled_dir'
  | 'joined_reviewer_dir'
  | 'already_joined'
  | 'created_reviewer_dir'
  | 'seed_taken'

/**
 * The recorded answer, in one sentence, naming the branch the server took.
 *
 * Six sentences because there are six outcomes and they are not the same fact:
 * a permanent group was created, an existing one was joined, the document was
 * already in it, the key names corpus data that the ranking did or did not
 * offer, or no group could be started. This is the ONLY place the distinction
 * is expressed to the evaluator, which is why the whole set lives here and is
 * rendered from the stored row rather than from what the client hoped for.
 *
 * `rank` is `ccl_key_rank`: where the labelled directory stood in the ranking
 * this answer was recorded against, resolved server-side. It is what lets the
 * shortlist clause be true rather than assumed -- the parenthetical used to say
 * "not in the shortlist" about a directory sitting at rank 1.
 */
export function recordedSentence(
  action: CclKeyActionName | null,
  key: string,
  rank: number | null,
): string {
  if (action === null || key === '') {
    return 'Recorded: none of the ranked candidates match.'
  }
  if (action === 'matched_labelled_dir') {
    return rank === null
      ? `Recorded: matches directory ${key}, which the ranking did not offer. Nothing new was created.`
      : `Recorded: matches directory ${key}, which is in the shortlist at #${rank}. Nothing new was created.`
  }
  if (action === 'joined_reviewer_dir') {
    return `Recorded, and this document joined the group under ${key}.`
  }
  if (action === 'already_joined') {
    return `Recorded. This document was already in the group under ${key}, so nothing was added to it.`
  }
  if (action === 'created_reviewer_dir') {
    return `Recorded, and a group under ${key} now holds this document.`
  }
  return `Recorded with the key ${key}. This document already starts a group of its own, and one document can start only one, so no group was started for this key.`
}

/**
 * Attribution for a grouping this reviewer is being shown rather than one they
 * just made (issue #161): recovered after a refused or lost create, or found on
 * the server after a reload. `created_by` is often somebody else, and the label
 * is always the stored one rather than anything this reviewer proposed.
 */
export function savedByNote(creator: string): string {
  return `Created by ${creator}. The name shown is the one it was saved under.`
}

/** A second, older grouping for the same document, listed rather than hidden. */
export function alsoGroupedNote(count: number): string {
  return count === 1
    ? 'This document is also recorded under another directory:'
    : `This document is also recorded under ${count} other directories:`
}

/**
 * A stored grouping that does not list the document seeding it as a member
 * (issue #161).
 *
 * Before issue #160 made creation atomic, the directory row and its seed
 * membership row were written on a shared connection, so an unrelated commit
 * in between could make the directory permanent and leave the membership
 * behind. Those rows still exist, the server still serves them, and nothing in
 * this application can add the missing row or remove the directory.
 *
 * The acknowledgement therefore reports what is stored and stops. It does not
 * fabricate the membership, does not hide the grouping, and does not let
 * "seeded with this document" stand as a claim that the document is filed
 * there -- membership is what a second witness would be counted in, and what
 * `matched` is derived from.
 */
export const SEED_NOT_FILED =
  'The stored record of this grouping does not list this document among its members. The app shows the record as it is rather than adding the missing entry, and no new directory can be started for this document while this one exists.'

/** The same fact as a clause beside one of several groupings. */
export const SEED_NOT_FILED_SHORT =
  'The stored record does not list this document among its members.'

/**
 * A create that ended in a recovery.
 *
 * Deliberately neutral about whose write this row is. The app reaches this
 * message whenever a POST failed and the seed-filtered lookup then found a
 * directory, and those two facts are also exactly what a LOST RESPONSE looks
 * like: the write landed, its 201 never arrived, and the row that comes back is
 * the reviewer's own. Saying "nothing new was created and the name you proposed
 * was not saved" is true for a refusal and false for a lost response, and this
 * layer cannot tell them apart -- the message text of a 409 is not a contract,
 * and a directory's timestamps are the server's clock, not the browser's. So
 * the app reports what it can actually see: a directory exists for this
 * document now, and the details shown are the stored ones.
 */
export const RECOVERED_AFTER_FAILURE =
  'A directory for this document is on the server. The app could not confirm whether your attempt created it, so the details shown are the stored ones rather than what you proposed.'

/**
 * What a low similarity does and does not establish.
 *
 * The band says the model has no useful opinion. It says nothing about whether
 * the CCL holds the source, and a general evaluator is not being asked to go
 * and find out by hand.
 */
export const NO_MATCH_GUIDANCE =
  'A low score means the model has no useful opinion here. It is not evidence that the source is missing from the Carolingian Canon Law (CCL) collections, and you are not expected to search the CCL by hand.'

/**
 * The None control's label, as `MatchPills` actually draws it (issue #157).
 *
 * "Top N" rather than "N model candidates" since issue #196: nothing else is
 * numbered beside them any more, so the shorter phrase Prof. Firey uses is now
 * also the accurate one.
 *
 * Duplicated deliberately rather than imported: this module is a leaf that the
 * prediction list and the tour both read, and reaching into an assessment-panel
 * component for a string would invert that. The drift guard is a test that
 * renders the real pill and compares, not a shared import.
 *
 * The count is the MODEL candidates only. A reviewer directory sits at the
 * anchored rank 11, and rejecting the model's answers says nothing about it.
 */
export function noneOptionLabel(modelCandidateCount: number): string {
  return modelCandidateCount === 1
    ? 'None of the 1 candidate'
    : `None of the top ${modelCandidateCount}`
}

/**
 * Status wording. `matched` follows membership, never a score: a directory is
 * matched once a human has filed a second, distinct witness into it, which is
 * how web/services/reviewer_dirs.py derives it.
 */
export const DIRECTORY_STATUS_COPY = {
  awaiting: 'Awaiting future match',
  matched: 'New directory matched',
  awaitingTitle: (label: string) =>
    `${label} — no second document has been filed into it yet`,
  leadTitle: (label: string, score: string) =>
    `${label} — a related document scores ${score}, unconfirmed by any reviewer`,
  matchedTitle: (label: string) =>
    `${label} — a reviewer filed a second document into it`,
  /**
   * Marker for a grouping this document seeds but is NOT listed in, shown in
   * the same short `· clause` form as `· lead` with the full sentence in the
   * title (issue #161's `SEED_NOT_FILED_SHORT`).
   *
   * It exists because the badge sits on the document's own header, where both
   * statuses are read as being about this document: `matched` in particular
   * would otherwise say a second witness joined the group this document is
   * filed in, when the stored record does not list this document at all. The
   * app cannot add the missing membership row and will not imply one, so it
   * marks the claim instead of making it.
   */
  notFiled: ' · not filed',
} as const
