import { useMemo, useState } from "react";
import {
  MagnifyingGlassIcon,
  ArrowDownTrayIcon,
  CheckCircleIcon,
  XMarkIcon,
} from "@heroicons/react/24/outline";
import { useExploreModels } from "@/hooks/useExploreModels";
import { useSavedModels } from "@/hooks/useSavedModels";
import { usePullModel } from "@/hooks/usePullModel";
import { useModels } from "@/hooks/useModels";
import BookmarkButton from "@/components/BookmarkButton";
import { formatBytes } from "@/utils/formatBytes";

export default function Explore() {
  const { data: catalog = [], isLoading, error } = useExploreModels();
  const { savedSet } = useSavedModels();
  const { data: localModels } = useModels();
  const { pull, stateFor } = usePullModel();
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<string | null>(null);

  const installed = useMemo(
    () => new Set(localModels.map((m) => m.model)),
    [localModels],
  );

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return catalog;
    return catalog.filter(
      (m) =>
        m.model.toLowerCase().includes(q) ||
        (m.capabilities ?? []).some((c) => c.toLowerCase().includes(q)),
    );
  }, [catalog, query]);

  const selectedModel = useMemo(
    () => catalog.find((m) => m.model === selected) ?? null,
    [catalog, selected],
  );

  return (
    <main className="flex h-screen w-full flex-col relative">
      <section className="flex-1 overflow-y-auto overscroll-contain relative min-h-0">
        <div className="max-w-[730px] mx-auto w-full px-4 pt-4 pb-20 sm:px-6 sm:pt-6 sm:pb-24 lg:px-8 lg:pt-8 lg:pb-28">
          <h1 className="text-xl font-semibold text-neutral-900 dark:text-neutral-100">
            Explore
          </h1>
          <p className="mt-1 text-sm text-neutral-500 dark:text-neutral-400">
            Browse public models from the Ollama library.
          </p>

          <div className="relative mt-6">
            <MagnifyingGlassIcon className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-neutral-400" />
            <input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search models..."
              aria-label="Search models"
              className="w-full rounded-full border border-neutral-200 bg-white py-1.5 pl-9 pr-3 text-sm text-neutral-800 outline-none focus:border-neutral-400 dark:border-neutral-700 dark:bg-neutral-800 dark:text-neutral-100 dark:focus:border-neutral-500"
            />
          </div>

          {isLoading ? (
            <p className="mt-6 text-sm text-neutral-500 dark:text-neutral-400">
              Loading models...
            </p>
          ) : error ? (
            <p className="mt-6 text-sm text-red-600 dark:text-red-400">
              Could not load the model catalog. Check your connection and try
              again.
            </p>
          ) : filtered.length === 0 ? (
            <p className="mt-6 text-sm text-neutral-500 dark:text-neutral-400">
              No models match “{query}”.
            </p>
          ) : (
            <ul className="mt-4 divide-y divide-neutral-100 dark:divide-neutral-800">
              {filtered.map((model) => {
                const pullState = stateFor(model.model);
                const isInstalled = installed.has(model.model);
                return (
                  <li key={model.model} className="py-3">
                    <div className="flex items-center gap-3">
                      <button
                        type="button"
                        onClick={() => setSelected(model.model)}
                        className="min-w-0 flex-1 text-left cursor-pointer"
                      >
                        <span className="block truncate text-sm font-medium text-neutral-900 dark:text-neutral-100">
                          {model.model}
                        </span>
                        <span className="mt-0.5 flex flex-wrap items-center gap-x-2 text-xs text-neutral-500 dark:text-neutral-400">
                          {model.size ? (
                            <span>{formatBytes(model.size)}</span>
                          ) : null}
                          {(model.capabilities ?? []).map((c) => (
                            <span key={c}>{c}</span>
                          ))}
                          {isInstalled && (
                            <span className="text-green-600 dark:text-green-400">
                              Installed
                            </span>
                          )}
                        </span>
                      </button>

                      <BookmarkButton model={model.model} size="sm" />

                      {pullState.status === "downloading" ? (
                        <span className="w-16 text-right text-xs text-neutral-500 dark:text-neutral-400">
                          {pullState.total > 0
                            ? `${Math.round((pullState.completed / pullState.total) * 100)}%`
                            : "..."}
                        </span>
                      ) : pullState.status === "done" || isInstalled ? (
                        <CheckCircleIcon
                          className="h-5 w-5 text-green-600 dark:text-green-400"
                          title="Installed"
                        />
                      ) : (
                        <button
                          type="button"
                          onClick={() => pull(model.model)}
                          aria-label={`Download ${model.model}`}
                          title={`Download ${model.model}`}
                          className="flex h-8 w-8 items-center justify-center rounded-full text-neutral-400 hover:bg-neutral-100 hover:text-neutral-600 dark:text-neutral-500 dark:hover:bg-neutral-700/60 dark:hover:text-neutral-300 cursor-pointer"
                        >
                          <ArrowDownTrayIcon className="h-5 w-5" />
                        </button>
                      )}
                    </div>
                    {pullState.status === "error" && (
                      <p className="mt-1 text-xs text-red-600 dark:text-red-400">
                        {pullState.message}
                      </p>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </section>

      {selectedModel && (
        <div
          className="absolute inset-0 z-50 flex items-center justify-center bg-black/20 p-4"
          onClick={() => setSelected(null)}
        >
          <div
            role="dialog"
            aria-modal="true"
            aria-label={`${selectedModel.model} details`}
            onClick={(e) => e.stopPropagation()}
            className="w-full max-w-md rounded-2xl bg-white p-5 shadow-xl dark:bg-neutral-800"
          >
            <div className="flex items-start justify-between gap-3">
              <h2 className="text-base font-semibold text-neutral-900 dark:text-neutral-100">
                {selectedModel.model}
              </h2>
              <button
                type="button"
                onClick={() => setSelected(null)}
                aria-label="Close details"
                className="rounded-full p-1 text-neutral-400 hover:bg-neutral-100 hover:text-neutral-600 dark:hover:bg-neutral-700 dark:hover:text-neutral-300 cursor-pointer"
              >
                <XMarkIcon className="h-5 w-5" />
              </button>
            </div>

            <dl className="mt-4 space-y-2 text-sm">
              <div className="flex justify-between gap-4">
                <dt className="text-neutral-500 dark:text-neutral-400">Size</dt>
                <dd className="text-neutral-900 dark:text-neutral-100">
                  {selectedModel.size
                    ? formatBytes(selectedModel.size)
                    : "Unknown"}
                </dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-neutral-500 dark:text-neutral-400">
                  Capabilities
                </dt>
                <dd className="text-right text-neutral-900 dark:text-neutral-100">
                  {(selectedModel.capabilities ?? []).join(", ") || "Unknown"}
                </dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-neutral-500 dark:text-neutral-400">
                  Status
                </dt>
                <dd className="text-neutral-900 dark:text-neutral-100">
                  {installed.has(selectedModel.model)
                    ? "Installed"
                    : "Not installed"}
                </dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-neutral-500 dark:text-neutral-400">
                  Saved
                </dt>
                <dd className="text-neutral-900 dark:text-neutral-100">
                  {savedSet.has(selectedModel.model) ? "Yes" : "No"}
                </dd>
              </div>
            </dl>

            <div className="mt-5 flex items-center justify-end gap-2">
              <BookmarkButton model={selectedModel.model} />
              {!installed.has(selectedModel.model) && (
                <button
                  type="button"
                  onClick={() => pull(selectedModel.model)}
                  className="rounded-full bg-neutral-900 px-4 py-1.5 text-sm text-white hover:bg-neutral-700 dark:bg-white dark:text-neutral-900 dark:hover:bg-neutral-200 cursor-pointer"
                >
                  Download
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
