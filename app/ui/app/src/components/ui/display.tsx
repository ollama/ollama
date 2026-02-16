import { clsx } from "clsx";
import { XMarkIcon } from "@heroicons/react/20/solid";

const colors = {
  red: "bg-red-50/80 dark:bg-red-950/50",
  neutral: "bg-neutral-50/80 dark:bg-neutral-800/50",
  amber: "bg-amber-50/80 dark:bg-amber-950/50",
  blue: "bg-blue-50/80 dark:bg-blue-950/50",
  green: "bg-green-50/80 dark:bg-green-950/50",
  zinc: "bg-zinc-50/80 dark:bg-zinc-800/50",
};

const textColors = {
  red: "text-red-600 dark:text-red-400",
  neutral: "text-neutral-600 dark:text-neutral-400",
  amber: "text-amber-600 dark:text-amber-400",
  blue: "text-blue-600 dark:text-blue-400",
  green: "text-green-600 dark:text-green-400",
  zinc: "text-zinc-600 dark:text-zinc-400",
};

const dismissButtonColors = {
  red: "text-red-400 hover:bg-red-100/50 hover:text-red-600 dark:text-red-500 dark:hover:bg-red-900/30 dark:hover:text-red-300",
  neutral:
    "text-neutral-400 hover:bg-neutral-200/50 hover:text-neutral-600 dark:text-neutral-500 dark:hover:bg-neutral-700/50 dark:hover:text-neutral-300",
  amber:
    "text-amber-400 hover:bg-amber-100/50 hover:text-amber-600 dark:text-amber-500 dark:hover:bg-amber-900/30 dark:hover:text-amber-300",
  blue: "text-blue-400 hover:bg-blue-100/50 hover:text-blue-600 dark:text-blue-500 dark:hover:bg-blue-900/30 dark:hover:text-blue-300",
  green:
    "text-green-400 hover:bg-green-100/50 hover:text-green-600 dark:text-green-500 dark:hover:bg-green-900/30 dark:hover:text-green-300",
  zinc: "text-zinc-400 hover:bg-zinc-200/50 hover:text-zinc-600 dark:text-zinc-500 dark:hover:bg-zinc-700/50 dark:hover:text-zinc-300",
};

export interface DisplayAction {
  label: string;
  onClick?: () => void;
  href?: string;
  disabled?: boolean;
  loading?: boolean;
  gradientColors?: string;
}

interface DisplayProps {
  message: string;
  variant?: keyof typeof colors;
  onDismiss?: () => void;
  action?: DisplayAction;
  className?: string;
}

export const Display = ({
  message,
  variant = "neutral",
  onDismiss,
  action,
  className,
}: DisplayProps) => {
  const ActionButton = ({ action }: { action: DisplayAction }) => {
    const buttonClass =
      "px-3 py-1.5 text-xs font-medium text-white bg-zinc-900 border border-zinc-950/90 rounded-full shadow-sm disabled:opacity-50 disabled:cursor-not-allowed dark:text-zinc-950 dark:bg-white dark:border-zinc-950/10 cursor-pointer hover:bg-zinc-800 dark:hover:bg-neutral-100";

    const content = (
      <span>{action.loading ? `${action.label}...` : action.label}</span>
    );

    if (action.href) {
      return (
        <a
          href={action.href}
          target="_blank"
          rel="noopener noreferrer"
          className={buttonClass}
        >
          {content}
        </a>
      );
    }

    return (
      <button
        onClick={action.onClick}
        disabled={action.disabled}
        className={buttonClass}
      >
        {content}
      </button>
    );
  };

  return (
    <div
      className={clsx(
        "mx-auto flex w-full max-w-[730px] items-center justify-between rounded-2xl px-4 py-3 text-sm transition-all duration-200 backdrop-blur-sm",
        colors[variant],
        className,
      )}
    >
      <div className="flex items-center space-x-3 select-text">
        <span className={clsx("leading-relaxed", textColors[variant])}>
          {message}
        </span>
      </div>

      <div className="flex items-center space-x-3">
        {action && <ActionButton action={action} />}
        {onDismiss && (
          <button
            onClick={onDismiss}
            className={clsx(
              "rounded-full p-1.5 cursor-pointer",
              dismissButtonColors[variant],
            )}
          >
            <XMarkIcon className="h-3.5 w-3.5" />
          </button>
        )}
      </div>
    </div>
  );
};
