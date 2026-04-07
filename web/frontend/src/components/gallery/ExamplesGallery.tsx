import { useMemo } from 'react'
import { useTokenMapExamplesGrouped } from '../../api/tokenMap'
import { useModels } from '../../api/models'
import ExampleCard from './ExampleCard'

export default function ExamplesGallery() {
  const { data, loading, error } = useTokenMapExamplesGrouped()
  const modelsState = useModels()

  const displayNameBySlug = useMemo(() => {
    const map = new Map<string, string>()
    for (const m of modelsState.data ?? []) {
      map.set(m.slug, m.display_name)
    }
    return map
  }, [modelsState.data])

  if (loading && !data) {
    return (
      <div className="flex-1 overflow-y-auto bg-surface-50 dark:bg-surface-900">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <p className="text-sm text-stone-500 dark:text-stone-400 text-center mt-12">
            Loading examples...
          </p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex-1 overflow-y-auto bg-surface-50 dark:bg-surface-900">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <p className="text-sm text-rose-600 dark:text-rose-400 text-center mt-12">
            Failed to load examples: {error}
          </p>
        </div>
      </div>
    )
  }

  const modelSlugs = data ? Object.keys(data.by_model) : []
  const totalPairs = data
    ? modelSlugs.reduce((sum, slug) => sum + (data.by_model[slug]?.length ?? 0), 0)
    : 0

  if (!data || modelSlugs.length === 0) {
    return (
      <div className="flex-1 overflow-y-auto bg-surface-50 dark:bg-surface-900">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <h1 className="text-2xl font-semibold text-stone-800 dark:text-stone-100 font-latin">
            Browse example pairs
          </h1>
          <p className="text-sm text-stone-500 dark:text-stone-400 mt-8 text-center">
            No example pairs available.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto bg-surface-50 dark:bg-surface-900">
      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-2xl font-semibold text-stone-800 dark:text-stone-100 font-latin">
            Browse example pairs
          </h1>
          <p className="text-sm text-stone-500 dark:text-stone-400 mt-1 font-ui">
            {totalPairs} pairs across {modelSlugs.length} models
          </p>
        </div>

        {/* Per-model sections */}
        <div className="space-y-10">
          {modelSlugs.map((slug) => {
            const cards = data.by_model[slug] ?? []
            const displayName = displayNameBySlug.get(slug) ?? slug
            return (
              <section key={slug}>
                <h2 className="text-sm font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400 mb-3 font-ui">
                  {displayName}
                  <span className="ml-2 text-[11px] font-normal text-stone-400 dark:text-stone-500 normal-case">
                    {cards.length} pairs
                  </span>
                </h2>
                {cards.length === 0 ? (
                  <p className="text-xs text-stone-400 dark:text-stone-500 italic">
                    No examples for this model.
                  </p>
                ) : (
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
                    {cards.map((card) => (
                      <ExampleCard key={card.example_id} card={card} />
                    ))}
                  </div>
                )}
              </section>
            )
          })}
        </div>
      </div>
    </div>
  )
}
