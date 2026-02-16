import { CheckIcon } from "@heroicons/react/20/solid";
import { Square2StackIcon } from "@heroicons/react/24/outline";
import React, { useState } from "react";

interface CopyButtonProps {
  content: string;
  copyRef?: React.RefObject<HTMLElement | null>;
  removeClasses?: string[];
  size?: "sm" | "md";
  showLabels?: boolean;
  className?: string;
  title?: string;
}

const CopyButton: React.FC<CopyButtonProps> = ({
  content,
  copyRef,
  removeClasses = [],
  size = "sm",
  showLabels = false,
  className = "",
  title = "",
}) => {
  const [isCopied, setIsCopied] = useState(false);

  const handleCopy = async () => {
    try {
      if (copyRef?.current) {
        // For copy response message
        const cloned = copyRef.current.cloneNode(true) as HTMLElement;

        removeClasses.forEach((className) => {
          cloned
            .querySelectorAll(`.${className}`)
            .forEach((element) => element.remove());
        });

        await navigator.clipboard.write([
          new ClipboardItem({
            "text/html": new Blob([cloned.innerHTML], {
              type: "text/html",
            }),
            "text/plain": new Blob([content], { type: "text/plain" }),
          }),
        ]);
      } else {
        await navigator.clipboard.writeText(content);
      }

      setIsCopied(true);
      setTimeout(() => setIsCopied(false), 2000);
    } catch (error) {
      console.error("Clipboard API failed, falling back to plain text", error);
      try {
        await navigator.clipboard.writeText(content);
        setIsCopied(true);
        setTimeout(() => setIsCopied(false), 2000);
      } catch (fallbackError) {
        console.error("Fallback copy also failed:", fallbackError);
      }
    }
  };

  const iconSize = size === "sm" ? "h-3 w-3" : "h-7 w-7";
  const baseClasses =
    size === "sm"
      ? `text-xs px-4 py-2 z-10 rounded-lg hover:cursor-pointer ${className}`
      : `${iconSize} px-1 py-0.5 text-xs cursor-pointer rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800 flex items-center justify-center ${className}`;

  const icon = isCopied ? (
    <CheckIcon className={iconSize} />
  ) : (
    <Square2StackIcon className={iconSize} />
  );

  return (
    <button
      type="button"
      className={baseClasses}
      onClick={handleCopy}
      title={title}
    >
      {showLabels ? (
        <span className="flex items-center gap-1">
          {icon}
          {isCopied ? "Copied" : "Copy"}
        </span>
      ) : (
        icon
      )}
    </button>
  );
};

export default CopyButton;
