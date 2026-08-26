import { useEffect } from 'react'
import { useApp } from '../../contexts/AppContext'
import { useModels, DEFAULT_MODEL_SLUG } from '../../api/models'

export default function ModelSelector() {
  const { activeModel, setActiveModel } = useApp()
  const { data: models } = useModels()

  // Pick the deployment's preferred model once the list loads (issue #94:
  // mT5-base). /api/models is sorted by display name, so "the first entry" is
  // an accident of the alphabet and must not decide what reviewers see.
  // Falls back to the first served model where mT5-base is not deployed.
  useEffect(() => {
    if (activeModel || !models || models.length === 0) return
    const preferred = models.find((m) => m.slug === DEFAULT_MODEL_SLUG)
    setActiveModel((preferred ?? models[0]).slug)
  }, [activeModel, models, setActiveModel])

  return (
    <select
      data-tour="model-selector"
      value={activeModel}
      onChange={(e) => setActiveModel(e.target.value)}
      className="w-full h-9 px-2 rounded-lg border border-stone-200 dark:border-stone-700
                 bg-white dark:bg-surface-800 text-sm font-ui
                 text-stone-800 dark:text-stone-200
                 focus:outline-none focus:ring-2 focus:ring-accent/30"
      aria-label="Select model"
    >
      {!activeModel && (
        <option value="" disabled>
          Select a model...
        </option>
      )}
      {models?.map((m) => (
        <option key={m.slug} value={m.slug}>
          {m.display_name}
        </option>
      ))}
    </select>
  )
}
