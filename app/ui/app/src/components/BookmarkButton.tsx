import { BookmarkIcon } from "@heroicons/react/24/outline";
import { BookmarkIcon as BookmarkIconSolid } from "@heroicons/react/24/solid";
import clsx from "clsx";
import { useSavedModels } from "@/hooks/useSavedModels";

/**
 * Toggles whether a model is in the user's saved collection. Saving a model is
 * independent of downloading it, so this never touches the local model store.
 */
export default function BookmarkButton({
  model,
  className,
  size = "md",
}: {
  model: string;
  className?: string;
  size?: "sm" | "md";
}) {
  const { isSaved, toggleSaved } = useSavedModels();
  const saved = isSaved(model);
  const label = saved ? `Remove ${model} from saved models` : `Save ${model}`;

  return (
    <button
      type="button"
      onClick={(e) => {
        e.preventDefault();
        e.stopPropagation();
        toggleSaved(model);
      }}
      aria-label={label}
      aria-pressed={saved}
      title={label}
      className={clsx(
        "flex items-center justify-center rounded-full transition-colors cursor-pointer",
        "focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500",
        size === "sm" ? "h-7 w-7" : "h-8 w-8",
        saved
          ? "text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-500/10"
          : "text-neutral-400 dark:text-neutral-500 hover:bg-neutral-100 dark:hover:bg-neutral-700/60 hover:text-neutral-600 dark:hover:text-neutral-300",
        className,
      )}
    >
      {saved ? (
        <BookmarkIconSolid className={size === "sm" ? "h-4 w-4" : "h-5 w-5"} />
      ) : (
        <BookmarkIcon className={size === "sm" ? "h-4 w-4" : "h-5 w-5"} />
      )}
    </button>
  );
}
