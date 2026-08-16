import { useEffect, useRef, useState } from "react";
import {
  FolderIcon,
  FolderOpenIcon,
  XMarkIcon,
} from "@heroicons/react/24/outline";
import { useProject } from "@/hooks/useProject";

// Header pill showing the active project, with a menu to open a folder,
// reopen a recent project, or close the current one
export function ProjectButton() {
  const {
    project,
    recentProjects,
    openProjectDialog,
    openProjectPath,
    closeProject,
    isOpening,
  } = useProject();
  const [isOpen, setIsOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isOpen) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (!containerRef.current?.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [isOpen]);

  const run = async (action: () => Promise<unknown>) => {
    setIsOpen(false);
    setError(null);
    try {
      await action();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to open project");
    }
  };

  const recents = recentProjects.filter((r) => r !== project?.root);

  return (
    <div
      ref={containerRef}
      className="relative"
      onMouseDown={(e) => e.stopPropagation()}
      onDoubleClick={(e) => e.stopPropagation()}
    >
      <button
        type="button"
        onClick={() => setIsOpen((open) => !open)}
        disabled={isOpening}
        title={project ? project.root : "Open a project folder"}
        className={`flex max-w-56 items-center gap-1.5 rounded-full px-3 py-1.5 text-sm cursor-pointer transition-colors ${
          project
            ? "bg-neutral-100 text-neutral-700 hover:bg-neutral-200 dark:bg-neutral-800 dark:text-neutral-200 dark:hover:bg-neutral-700"
            : "text-neutral-500 hover:bg-neutral-100 dark:text-neutral-400 dark:hover:bg-neutral-800"
        }`}
      >
        {project ? (
          <FolderOpenIcon className="h-4 w-4 flex-shrink-0" />
        ) : (
          <FolderIcon className="h-4 w-4 flex-shrink-0" />
        )}
        <span className="truncate">
          {isOpening ? "Opening…" : project ? project.name : "Open project"}
        </span>
      </button>

      {isOpen && (
        <div className="absolute right-0 top-full z-50 mt-1 w-72 rounded-xl border border-neutral-200 bg-white p-1 shadow-lg dark:border-neutral-700 dark:bg-neutral-800">
          <button
            type="button"
            onClick={() => run(openProjectDialog)}
            className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-sm text-neutral-700 hover:bg-neutral-100 dark:text-neutral-200 dark:hover:bg-neutral-700 cursor-pointer"
          >
            <FolderIcon className="h-4 w-4 flex-shrink-0 text-neutral-400" />
            Open folder…
          </button>

          {recents.length > 0 && (
            <>
              <div className="my-1 border-t border-neutral-200 dark:border-neutral-700" />
              <p className="px-3 py-1 text-xs font-semibold uppercase tracking-wide text-neutral-400">
                Recent
              </p>
              {recents.map((path) => (
                <button
                  key={path}
                  type="button"
                  onClick={() => run(() => openProjectPath(path))}
                  title={path}
                  className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-sm text-neutral-700 hover:bg-neutral-100 dark:text-neutral-200 dark:hover:bg-neutral-700 cursor-pointer"
                >
                  <FolderIcon className="h-4 w-4 flex-shrink-0 text-neutral-400" />
                  <span className="min-w-0 flex-1 truncate" dir="rtl">
                    {path}
                  </span>
                </button>
              ))}
            </>
          )}

          {project && (
            <>
              <div className="my-1 border-t border-neutral-200 dark:border-neutral-700" />
              <button
                type="button"
                onClick={() => run(closeProject)}
                className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-sm text-neutral-700 hover:bg-neutral-100 dark:text-neutral-200 dark:hover:bg-neutral-700 cursor-pointer"
              >
                <XMarkIcon className="h-4 w-4 flex-shrink-0 text-neutral-400" />
                Close project
              </button>
            </>
          )}
        </div>
      )}

      {error && (
        <div className="absolute right-0 top-full z-50 mt-1 w-72 rounded-xl border border-red-200 bg-red-50 p-3 text-sm text-red-700 shadow-lg dark:border-red-800 dark:bg-red-900/40 dark:text-red-300">
          {error}
          <button
            type="button"
            onClick={() => setError(null)}
            className="ml-2 underline cursor-pointer"
          >
            Dismiss
          </button>
        </div>
      )}
    </div>
  );
}
