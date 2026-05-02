interface ImageGenerationDisplayProps {
  isGenerating: boolean;
  image: string | null;
  progress: { completed: number; total: number } | null;
  error: string | null;
}

export function ImageGenerationDisplay({
  isGenerating,
  image,
  progress,
  error,
}: ImageGenerationDisplayProps) {
  if (!isGenerating && !image && !error) return null;

  return (
    <div className="px-6 py-4">
      {isGenerating && (
        <div className="flex flex-col gap-2">
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-neutral-200 dark:bg-neutral-700">
            <div
              className="h-full rounded-full bg-neutral-800 dark:bg-neutral-200 transition-all"
              style={{
                width: progress
                  ? `${(progress.completed / progress.total) * 100}%`
                  : "100%",
                animation: progress
                  ? "none"
                  : "pulse 1.5s ease-in-out infinite",
              }}
            />
          </div>
          <p className="text-xs text-neutral-500">
            {progress
              ? `${progress.completed}/${progress.total} steps`
              : "Generating..."}
          </p>
        </div>
      )}
      {error && (
        <div className="rounded-lg bg-red-50 dark:bg-red-900/20 p-3 text-sm text-red-600 dark:text-red-400">
          {error}
        </div>
      )}
      {image && (
        <div className="flex flex-col items-start gap-2">
          <img
            src={`data:image/png;base64,${image}`}
            alt="Generated image"
            className="max-w-full rounded-lg"
          />
        </div>
      )}
    </div>
  );
}
