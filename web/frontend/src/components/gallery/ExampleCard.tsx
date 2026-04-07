import { memo } from 'react'
import { useApp } from '../../contexts/AppContext'
import type { TokenMapExampleCard } from '../../api/tokenMap'

interface ExampleCardProps {
  card: TokenMapExampleCard
}

const BUCKET_STYLES: Record<string, { label: string; className: string }> = {
  correct_similar: {
    label: 'correct similar',
    className:
      'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300',
  },
  correct_not_similar: {
    label: 'correct not similar',
    className:
      'bg-sky-100 text-sky-700 dark:bg-sky-900/30 dark:text-sky-300',
  },
  wrong_similar: {
    label: 'wrong similar',
    className:
      'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300',
  },
  wrong_not_similar: {
    label: 'wrong not similar',
    className:
      'bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-300',
  },
}

function BooleanGlyph({ value, label }: { value: number; label: string }) {
  const ok = value === 1
  return (
    <span
      title={`${label}: ${ok ? 'yes' : 'no'}`}
      className={`inline-flex items-center gap-0.5 text-[10px] font-mono ${
        ok ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'
      }`}
    >
      <span className="text-stone-500 dark:text-stone-400 font-ui">{label}</span>
      <span aria-hidden="true">{ok ? '\u2713' : '\u2717'}</span>
    </span>
  )
}

function ExampleCardInner({ card }: ExampleCardProps) {
  const { navigateToExample } = useApp()
  const bucket =
    BUCKET_STYLES[card.bucket] ?? {
      label: card.bucket,
      className:
        'bg-stone-100 text-stone-700 dark:bg-stone-800 dark:text-stone-300',
    }

  const disabled = card.methods_available.length === 0
  const candidateLabelExtra =
    card.candidate_label && card.candidate_label !== card.candidate_folder_id
      ? ` (${card.candidate_label})`
      : ''

  const baseClasses =
    'text-left p-3 rounded-lg border bg-white dark:bg-surface-800 transition flex flex-col gap-2'
  const interactionClasses = disabled
    ? 'border-stone-200 dark:border-stone-700 opacity-50 cursor-not-allowed'
    : 'border-stone-200 dark:border-stone-700 hover:border-accent hover:shadow-sm cursor-pointer'

  return (
    <button
      type="button"
      disabled={disabled}
      onClick={() => {
        if (disabled) return
        navigateToExample(
          card.model_slug,
          card.query_file_id,
          card.candidate_folder_id,
        )
      }}
      className={`${baseClasses} ${interactionClasses}`}
      aria-label={`Example pair ${card.example_id}: ${card.query_folder_id} vs ${card.candidate_folder_id}`}
    >
      {/* Header: bucket badge */}
      <div className="flex items-center justify-between">
        <span
          className={`px-2 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wide ${bucket.className}`}
        >
          {bucket.label}
        </span>
        <span className="text-[10px] text-stone-400 dark:text-stone-500 font-mono">
          #{card.example_id}
        </span>
      </div>

      {/* Query line */}
      <div
        className="text-xs font-mono text-stone-700 dark:text-stone-200 truncate"
        title={`${card.query_folder_id} / ${card.query_filename}`}
      >
        {card.query_folder_id} / {card.query_filename}
      </div>

      {/* Arrow */}
      <div className="text-stone-400 dark:text-stone-500 text-xs">
        <span aria-hidden="true">{'\u2192'}</span>
      </div>

      {/* Candidate line */}
      <div
        className="text-xs font-mono text-stone-700 dark:text-stone-200 truncate"
        title={`${card.candidate_folder_id}${candidateLabelExtra}`}
      >
        {card.candidate_folder_id}
        {candidateLabelExtra}
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between pt-1 border-t border-stone-100 dark:border-stone-700/60">
        <span className="text-[10px] text-stone-500 dark:text-stone-400 font-ui">
          {card.methods_available.length} methods
        </span>
        <div className="flex items-center gap-2">
          <BooleanGlyph value={card.gold_similar} label="Gold" />
          <BooleanGlyph value={card.baseline_pred} label="Base" />
          <BooleanGlyph value={card.abtt_pred} label="ABTT" />
        </div>
      </div>
    </button>
  )
}

const ExampleCard = memo(ExampleCardInner)
ExampleCard.displayName = 'ExampleCard'

export default ExampleCard
