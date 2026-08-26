import { SparklesIcon } from "@heroicons/react/24/outline";
import { useEffect, useState } from "react";

interface ClaudeDesktopIconProps {
  installed?: boolean;
  className?: string;
}

export function ClaudeDesktopIcon({
  installed = false,
  className = "h-6 w-6",
}: ClaudeDesktopIconProps) {
  const [src, setSrc] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    if (!installed || !window.getClaudeDesktopIcon) {
      return;
    }

    void window
      .getClaudeDesktopIcon()
      .then((icon) => {
        if (active && icon) setSrc(icon);
      })
      .catch(() => undefined);
    return () => {
      active = false;
    };
  }, [installed]);

  if (installed && src) {
    return <img src={src} alt="" className={`${className} object-contain`} />;
  }

  return (
    <SparklesIcon
      aria-hidden="true"
      className={`${className} text-neutral-500 dark:text-neutral-400`}
    />
  );
}
