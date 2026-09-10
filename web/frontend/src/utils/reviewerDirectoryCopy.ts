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
  /** Below the no-match band, where creating is the default top option. */
  openEmphasised: 'New directory / New file',
  /** Above it, where creating is a quiet escape hatch. */
  openQuiet: 'Start a new directory',
  /** One line under the button, before the form is even opened. */
  buttonCaption:
    'Saves a permanent grouping as soon as you confirm it. It does not record your assessment.',

  formHeading: 'Before you create this directory',
  formPoints: [
    'The directory and the name you give it are saved the moment you confirm. The app has no rename, no removal and no way to withdraw a document from a grouping.',
    'This is not an assessment. Submitting or skipping this document afterwards does not undo it.',
    'Create one only if a new grouping is warranted. To reject the ranked candidates without starting a grouping, close this form, record the None option and say why in your notes.',
  ],

  fieldLabel: 'Name the new directory',
  submit: 'Create directory',
  submitting: 'Creating…',
  cancel: 'Cancel',

  /**
   * Cancel, once the request has gone (issue #161).
   *
   * Before the request there is genuinely something to cancel. Afterwards
   * there is not: closing a form does not reach the server, and aborting the
   * fetch would not either, since the write may already be committed. So the
   * control changes its name rather than keeping a promise it cannot keep.
   */
  closePending: 'Close',
  pendingNote:
    'Saving. Closing this form does not cancel the save, and nothing in the app can remove a directory once the server has it.',

  /** Reconciled with the server: the write certainly did not land. */
  failedNotCreated:
    'Nothing was created. Your name is kept, so you can try again.',
  /**
   * The request failed AND the follow-up check failed. Saying "not saved"
   * here would be a guess, and acting on that guess is how a second permanent
   * directory gets created for one document.
   */
  failedUnknown:
    'The app could not confirm whether the directory was created. Check again before trying a different name.',

  checking: 'Checking whether this document already has a directory…',
  checkAgain: 'Check again',
  /**
   * A failed lookup is not an empty lookup. Offering Create here would invite
   * a duplicate the server will refuse.
   */
  unresolvedNote:
    'The app could not check whether this document already has a directory. Creating one now could be refused as a duplicate.',

  savedHeading: 'Directory saved',
  /**
   * Deliberately conditional. A directory is scored from the model's q-q
   * matrix, is never offered to its own members, cannot be scored for an
   * excluded query, and only the closest few reach any one list, so
   * "a candidate for every other document" was a promise the app cannot keep.
   */
  savedAvailability:
    'Saved permanently and seeded with this document. It can be offered as a candidate on other documents this model can score, not on every one.',
  savedIndependence: 'Submitting or skipping your assessment does not undo it.',
} as const

/**
 * Attribution for a grouping this reviewer is being shown rather than one they
 * just made (issue #161): recovered after a refused or lost create, or found on
 * the server after a reload. `created_by` is often somebody else, and the label
 * is always the stored one rather than anything this reviewer proposed.
 */
export function savedByNote(creator: string): string {
  return `Created by ${creator}. The name shown is the one it was saved under.`
}

/** A create that turned into a recovery. The proposed name was not stored. */
export const RECOVERED_INSTEAD_OF_CREATED =
  'This document already had a directory, so nothing new was created and the name you proposed was not saved.'

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
    ? 'None of the 1 model candidate'
    : `None of the ${modelCandidateCount} model candidates`
}

/** Shown in place of the button when this document already seeds a directory. */
export function alreadySeededNote(modelCandidateCount: number): string {
  return `This document already seeds a provisional directory, and only one directory per document is allowed. If none of the ranked candidates fits, record “${noneOptionLabel(modelCandidateCount)}” and explain in your notes.`
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
} as const
