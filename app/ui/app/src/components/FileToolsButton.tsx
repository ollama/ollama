import { type JSX } from "react";

interface FileToolsButtonProps {
  enabled: boolean;
  active: boolean;
  onToggle: (active: boolean) => void;
}

export default function FileToolsButton({
  enabled,
  active,
  onToggle,
}: FileToolsButtonProps): JSX.Element | null {
  if (!enabled) return null;

  return (
    <button
      type="button"
      onClick={() => onToggle(!active)}
      title="Toggle File Tools"
      className={`flex h-9 w-9 items-center justify-center rounded-full bg-white dark:bg-neutral-700 focus:outline-none transition-all cursor-pointer border border-transparent ${
        active
          ? "text-[rgba(0,115,255,1)]"
          : "text-neutral-800 dark:text-neutral-100"
      }`}
    >
      <svg
        className="h-4 w-4"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"
        />
      </svg>
    </button>
  );
}
