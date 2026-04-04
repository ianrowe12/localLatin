interface SkeletonLoaderProps {
  lines?: number
}

export default function SkeletonLoader({ lines = 8 }: SkeletonLoaderProps) {
  return (
    <div className="space-y-3 p-4">
      {Array.from({ length: lines }).map((_, i) => (
        <div
          key={i}
          className="h-4 bg-stone-200 dark:bg-stone-700 rounded animate-pulse"
          style={{ width: `${60 + (i * 17) % 40}%` }}
        />
      ))}
    </div>
  )
}
