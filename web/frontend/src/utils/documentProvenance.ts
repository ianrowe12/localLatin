/**
 * What kind of text is in a document panel (issue #162).
 *
 * The right-hand panel used to be captioned "Predicted Source" whatever was in
 * it, which is false for a reviewer-created directory: its members are
 * originally unlabeled witnesses that a colleague grouped, not sources the
 * corpus attests. The three kinds a reviewer can be looking at are:
 *
 *   query             the unlabeled witness under review (left panel)
 *   labeled_reference a witness from the labelled corpus, the model's candidate
 *   reviewer_group    an originally unlabeled witness filed into a provisional
 *                     reviewer directory
 *
 * None of these names a historical source identity. A labelled directory is a
 * corpus grouping of hand copies, most of them never edited to modern
 * standards, so "labeled-reference" means "the corpus already groups this
 * witness", not "this text is edited and attested".
 *
 * A leaf module on purpose: it takes the candidate's shape structurally rather
 * than importing the API layer, so panels, cards and the tour can all read the
 * same vocabulary without an import cycle.
 */

export type DocumentProvenance =
  | 'query'
  | 'labeled_reference'
  | 'reviewer_group'
  | 'unknown'

/**
 * Mirrors `REVIEWER_DIR_PREFIX` in web/services/reviewer_dirs.py. The backend
 * guarantees it is disjoint from every labelled directory name, and routes
 * `correct_dir` on exactly this test, so reading provenance off a bare
 * directory name is the backend's own rule rather than a guess.
 */
export const REVIEWER_DIR_ID_PREFIX = 'reviewer-dir-'

export function isReviewerDirId(dirName: string | null | undefined): boolean {
  return !!dirName && dirName.startsWith(REVIEWER_DIR_ID_PREFIX)
}

/** Short caption naming what the reader is looking at. */
export const PROVENANCE_TERMS: Record<DocumentProvenance, string> = {
  query: 'Query witness (unlabeled)',
  labeled_reference: 'Labeled-reference witness',
  reviewer_group: 'Reviewer group member (originally unlabeled)',
  unknown: 'Candidate witness',
}

/**
 * Provenance of a candidate, from whatever the caller actually knows.
 *
 * `source` is the authoritative answer and comes from the current prediction.
 * The directory-name test is the explicit fallback for candidates opened
 * outside the ranked list (the examples gallery passes a directory with no
 * prediction behind it). With neither, the caption stays the non-committal
 * `unknown`: an unidentified candidate must not be captioned as a
 * labeled reference.
 */
export function provenanceOf(candidate: {
  source?: 'model' | 'reviewer' | null
  dirName?: string | null
}): DocumentProvenance {
  if (candidate.source === 'reviewer') return 'reviewer_group'
  if (candidate.source === 'model') return 'labeled_reference'
  if (isReviewerDirId(candidate.dirName)) return 'reviewer_group'
  if (candidate.dirName) return 'labeled_reference'
  return 'unknown'
}

/** The caption line under a document's filename, rank included when known. */
export function provenanceCaption(
  provenance: DocumentProvenance,
  rank?: number | null,
): string {
  const term = PROVENANCE_TERMS[provenance]
  if (provenance === 'query' || rank == null) return term
  return `${term} · Rank ${rank}`
}
