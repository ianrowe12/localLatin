import type { CountedAnswer } from '../../api/adminReview'

export default function AdminReviewAnswers({ answers }: { answers: CountedAnswer[] }) {
  if (answers.length === 0) {
    return <p className="text-sm text-stone-500 dark:text-stone-400">No counted reviewer answers yet.</p>
  }
  return (
    <ul className="space-y-3">
      {answers.map((answer, index) => (
        <li key={index} className="text-sm break-words min-w-0">
          <p className="font-medium text-stone-800 dark:text-stone-100">
            {answer.reviewer}: {answer.keys.join('; ')}
            {answer.none_of_these ? 'None of these' : ''}
            {answer.unresolved ? `${answer.keys.length ? '; ' : ''}Unresolved pick` : ''}
          </p>
          <p className="text-xs text-stone-500 dark:text-stone-400">
            {answer.model_slug} / {answer.variant ?? 'Legacy variant'}, <time dateTime={answer.timestamp}>{answer.timestamp}</time>
          </p>
          {answer.notes && <p className="mt-1 whitespace-pre-wrap text-stone-600 dark:text-stone-300">{answer.notes}</p>}
        </li>
      ))}
    </ul>
  )
}
