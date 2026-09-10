import type { ReactNode } from 'react'
import {
  provenanceCaption,
  type DocumentProvenance,
} from '../../utils/documentProvenance'

interface DocumentHeaderProps {
  filename: string
  dirLabel?: string
  score?: number
  rank?: number
  side: 'query' | 'candidate'
  /**
   * What kind of text this is (issue #162). The left panel is always the query
   * witness. On the right, `labeled_reference` is a witness the labelled corpus
   * already groups and `reviewer_group` is an originally unlabeled document
   * filed into a provisional reviewer directory: captioning the second one
   * "Predicted Source" asserted a corpus fact that does not exist.
   *
   * Omitted means the caller genuinely does not know -- an off-list candidate
   * opened without a prediction behind it -- and the caption then says only
   * "Candidate witness". It must not silently become "labeled reference".
   */
  provenance?: DocumentProvenance
  /** Status pills that belong to the document itself, e.g. "Awaiting future match". */
  badge?: ReactNode
}

export default function DocumentHeader({
  filename,
  dirLabel,
  score,
  rank,
  side,
  provenance,
  badge,
}: DocumentHeaderProps) {
  const resolved: DocumentProvenance =
    side === 'query' ? 'query' : (provenance ?? 'unknown')

  return (
    <div className="px-3 py-2 border-b border-stone-200 dark:border-stone-700 bg-stone-50 dark:bg-surface-800 flex-shrink-0">
      <div className="flex items-center gap-2">
        <span className="font-medium text-sm truncate">{filename}</span>

        {dirLabel && (
          <span
            className="text-xs text-stone-500 truncate max-w-[200px]"
            title={dirLabel}
          >
            {dirLabel}
          </span>
        )}

        <span className="flex-1" />

        {badge}

        {score != null && (
          <span className="bg-accent/10 text-accent text-xs px-2 py-0.5 rounded-full font-mono whitespace-nowrap">
            Similarity: {score.toFixed(3)}
          </span>
        )}
      </div>

      <div
        data-testid={`document-provenance-${side}`}
        className="font-ui text-xs text-stone-400 dark:text-stone-500 mt-0.5"
      >
        {provenanceCaption(resolved, side === 'candidate' ? rank : null)}
      </div>
    </div>
  )
}
