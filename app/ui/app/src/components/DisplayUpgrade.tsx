import type { ErrorEvent } from "@/gotypes";
import { Display, type DisplayAction } from "@/components/ui/display";

interface DisplayUpgradeProps {
  error: ErrorEvent | null;
  onDismiss: () => void;
  className?: string;
  message?: string;
  label?: string;
  href?: string;
}

export const DisplayUpgrade = ({
  error,
  onDismiss,
  className,
  message,
  label = "Upgrade",
  href = "https://ollama.com/upgrade",
}: DisplayUpgradeProps) => {
  if (!error || error.code !== "usage_limit_upgrade") return null;

  const isUsageLimit = error.code === "usage_limit_upgrade";

  const action: DisplayAction | undefined = isUsageLimit
    ? {
        label,
        href,
        gradientColors: "from-cyan-500/20 via-purple-500/20 to-green-500/20",
      }
    : undefined;

  const variant = isUsageLimit ? "neutral" : "red";

  return (
    <Display
      message={message || error.error || "An error occurred"}
      variant={variant}
      onDismiss={onDismiss}
      action={action}
      className={className}
    />
  );
};
