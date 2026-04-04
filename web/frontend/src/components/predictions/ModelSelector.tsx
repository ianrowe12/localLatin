import { useEffect } from 'react'
import { useApp } from '../../contexts/AppContext'
import { useModels } from '../../api/models'

export default function ModelSelector() {
  const { activeModel, setActiveModel } = useApp()
  const { data: models } = useModels()

  // Auto-select first model when models load and none is selected
  useEffect(() => {
    if (!activeModel && models && models.length > 0) {
      setActiveModel(models[0].slug)
    }
  }, [activeModel, models, setActiveModel])

  return (
    <select
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
