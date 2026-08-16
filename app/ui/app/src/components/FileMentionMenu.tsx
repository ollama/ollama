import { useEffect, useRef } from "react";
import { DocumentIcon } from "@heroicons/react/24/outline";

// Dropdown listing fuzzy-matched project files while typing "@" in the chat
// input. Rendered above the textarea; keyboard handling lives in ChatForm.
export function FileMentionMenu({
  results,
  selectedIndex,
  onSelect,
  onHover,
}: {
  results: string[];
  selectedIndex: number;
  onSelect: (path: string) => void;
  onHover: (index: number) => void;
}) {
  const listRef = useRef<HTMLDivElement>(null);

  // Keep the selected row visible while navigating with the keyboard
  useEffect(() => {
    const list = listRef.current;
    const item = list?.children[selectedIndex] as HTMLElement | undefined;
    if (item && list) {
      const itemTop = item.offsetTop;
      const itemBottom = itemTop + item.offsetHeight;
      if (itemTop < list.scrollTop) {
        list.scrollTop = itemTop;
      } else if (itemBottom > list.scrollTop + list.clientHeight) {
        list.scrollTop = itemBottom - list.clientHeight;
      }
    }
  }, [selectedIndex]);

  if (results.length === 0) return null;

  return (
    <div className="absolute bottom-full left-0 right-0 z-40 mb-2 overflow-hidden rounded-xl border border-neutral-200 bg-white shadow-lg dark:border-neutral-700 dark:bg-neutral-800">
      <div ref={listRef} className="max-h-64 overflow-y-auto p-1">
        {results.map((path, index) => {
          const slash = path.lastIndexOf("/");
          const dir = slash === -1 ? "" : path.slice(0, slash);
          const name = slash === -1 ? path : path.slice(slash + 1);
          return (
            <button
              key={path}
              type="button"
              // preventDefault keeps focus in the textarea
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => onSelect(path)}
              onMouseEnter={() => onHover(index)}
              className={`flex w-full items-center gap-2 rounded-lg px-3 py-1.5 text-left text-sm cursor-pointer ${
                index === selectedIndex
                  ? "bg-neutral-100 dark:bg-neutral-700"
                  : ""
              }`}
            >
              <DocumentIcon className="h-4 w-4 flex-shrink-0 text-neutral-400" />
              <span className="truncate text-neutral-700 dark:text-neutral-200">
                {name}
              </span>
              {dir && (
                <span className="min-w-0 flex-1 truncate text-xs text-neutral-400">
                  {dir}
                </span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
