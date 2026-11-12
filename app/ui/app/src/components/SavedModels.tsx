import { useMemo, useState } from "react";
import { Link } from "@tanstack/react-router";
import { MagnifyingGlassIcon, BookmarkIcon } from "@heroicons/react/24/outline";
import { useSavedModels } from "@/hooks/useSavedModels";
import { useModels } from "@/hooks/useModels";
import BookmarkButton from "@/components/BookmarkButton";

export default function SavedModels() {
  const { savedModels, isLoading, error } = useSavedModels();
  const { data: localModels } = useModels();
  const [query, setQuery] = useState("");

  const installed = useMemo(
    () => new Set(localModels.map((m) => m.model)),
    [localModels],
  );

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return savedModels;
    return savedModels.filter((m) => m.toLowerCase().includes(q));
  }, [savedModels, query]);

  return (
    <main className="flex h-screen w-full flex-col relative">
      <section className="flex-1 overflow-y-auto overscroll-contain relative min-h-0">
        <div className="max-w-[730px] mx-auto w-full px-4 pt-4 pb-20 sm:px-6 sm:pt-6 sm:pb-24 lg:px-8 lg:pt-8 lg:pb-28">
          <h1 className="text-xl font-semibold text-neutral-900 dark:text-neutral-100">
            Saved models
          </h1>
          <p className="mt-1 text-sm text-neutral-500 dark:text-neutral-400">
            Models you bookmarked to revisit later.
          </p>

          {isLoading ? (
            <p className="mt-6 text-sm text-neutral-500 dark:text-neutral-400">
              Loading...
            </p>
          ) : error ? (
            <p className="mt-6 text-sm text-red-600 dark:text-red-400">
              Could not load saved models.
            </p>
          ) : savedModels.length === 0 ? (
            <div className="mt-10 flex flex-col items-center text-center">
              <BookmarkIcon className="h-8 w-8 text-neutral-300 dark:text-neutral-600" />
              <p className="mt-3 text-sm font-medium text-neutral-900 dark:text-neutral-100">
                No saved models yet
              </p>
              <p className="mt-1 max-w-sm text-sm text-neutral-500 dark:text-neutral-400">
                Bookmark models from the model picker to keep them here.
              </p>
              <Link
                to="/c/$chatId"
                params={{ chatId: "new" }}
                className="mt-4 rounded-full bg-neutral-900 px-4 py-1.5 text-sm text-white hover:bg-neutral-700 dark:bg-white dark:text-neutral-900 dark:hover:bg-neutral-200"
              >
                Browse models
              </Link>
            </div>
          ) : (
            <>
              <div className="relative mt-6">
                <MagnifyingGlassIcon className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-neutral-400" />
                <input
                  type="search"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Search saved models..."
                  aria-label="Search saved models"
                  className="w-full rounded-full border border-neutral-200 bg-white py-1.5 pl-9 pr-3 text-sm text-neutral-800 outline-none focus:border-neutral-400 dark:border-neutral-700 dark:bg-neutral-800 dark:text-neutral-100 dark:focus:border-neutral-500"
                />
              </div>

              {filtered.length === 0 ? (
                <p className="mt-6 text-sm text-neutral-500 dark:text-neutral-400">
                  No saved models match “{query}”.
                </p>
              ) : (
                <ul className="mt-4 divide-y divide-neutral-100 dark:divide-neutral-800">
                  {filtered.map((model) => (
                    <li key={model} className="flex items-center gap-3 py-3">
                      <div className="min-w-0 flex-1">
                        <Link
                          to="/c/$chatId"
                          params={{ chatId: "new" }}
                          className="block truncate text-sm font-medium text-neutral-900 hover:underline dark:text-neutral-100"
                        >
                          {model}
                        </Link>
                        <p className="mt-0.5 text-xs text-neutral-500 dark:text-neutral-400">
                          {installed.has(model) ? "Installed" : "Not installed"}
                        </p>
                      </div>
                      <BookmarkButton model={model} size="sm" />
                    </li>
                  ))}
                </ul>
              )}
            </>
          )}
        </div>
      </section>
    </main>
  );
}
