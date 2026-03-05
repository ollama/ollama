import { MinusIcon, PlusIcon } from "@heroicons/react/20/solid";
import clsx from "clsx";
import React, { useCallback } from "react";
import { Input } from "./input";

export function NumberInput({
  value,
  onChange,
  min,
  max,
  step = 1,
  disabled = false,
  suffix,
  className,
}: {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
  suffix?: string;
  className?: string;
}) {
  const clampValue = useCallback(
    (v: number) => {
      let clamped = v;
      if (min !== undefined) clamped = Math.max(min, clamped);
      if (max !== undefined) clamped = Math.min(max, clamped);
      return clamped;
    },
    [min, max],
  );

  const handleDecrement = () => {
    if (disabled) return;
    onChange(clampValue(value - step));
  };

  const handleIncrement = () => {
    if (disabled) return;
    onChange(clampValue(value + step));
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.value;
    if (raw === "" || raw === "-") return;
    const parsed = Number(raw);
    if (!Number.isNaN(parsed)) {
      onChange(clampValue(parsed));
    }
  };

  const isAtMin = min !== undefined && value <= min;
  const isAtMax = max !== undefined && value >= max;

  return (
    <div
      data-slot="control"
      className={clsx(className, "flex items-center gap-2")}
    >
      <button
        type="button"
        onClick={handleDecrement}
        disabled={disabled || isAtMin}
        className={clsx([
          // Basic layout
          "inline-flex items-center justify-center rounded-lg size-9 sm:size-8 flex-shrink-0",
          // Border
          "border border-zinc-950/10 dark:border-white/10",
          // Background
          "bg-white dark:bg-white/5",
          // Typography
          "text-zinc-950 dark:text-white",
          // Hover
          "hover:bg-zinc-950/2.5 hover:border-zinc-950/20 dark:hover:bg-white/10 dark:hover:border-white/20",
          // Focus
          "focus:outline-hidden focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-500",
          // Disabled
          "disabled:opacity-50 disabled:pointer-events-none",
        ])}
      >
        <MinusIcon className="size-4" aria-hidden="true" />
      </button>

      <div className="relative flex-1 min-w-0">
        <Input
          type="number"
          value={value}
          onChange={handleChange}
          disabled={disabled}
          className="[&_input]:text-center [&_input]:pr-1 [&_input]:[-moz-appearance:textfield] [&_input::-webkit-inner-spin-button]:appearance-none [&_input::-webkit-outer-spin-button]:appearance-none"
        />
        {suffix && (
          <span className="pointer-events-none absolute inset-y-0 right-3 flex items-center text-sm text-zinc-500 dark:text-zinc-400 sm:right-2.5 sm:text-xs">
            {suffix}
          </span>
        )}
      </div>

      <button
        type="button"
        onClick={handleIncrement}
        disabled={disabled || isAtMax}
        className={clsx([
          // Basic layout
          "inline-flex items-center justify-center rounded-lg size-9 sm:size-8 flex-shrink-0",
          // Border
          "border border-zinc-950/10 dark:border-white/10",
          // Background
          "bg-white dark:bg-white/5",
          // Typography
          "text-zinc-950 dark:text-white",
          // Hover
          "hover:bg-zinc-950/2.5 hover:border-zinc-950/20 dark:hover:bg-white/10 dark:hover:border-white/20",
          // Focus
          "focus:outline-hidden focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-500",
          // Disabled
          "disabled:opacity-50 disabled:pointer-events-none",
        ])}
      >
        <PlusIcon className="size-4" aria-hidden="true" />
      </button>
    </div>
  );
}
