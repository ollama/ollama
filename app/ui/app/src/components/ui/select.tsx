import * as Headless from "@headlessui/react";
import { ChevronUpDownIcon } from "@heroicons/react/20/solid";
import clsx from "clsx";

export function Select({
  value,
  onChange,
  options,
  disabled = false,
  placeholder = "Select an option",
  className,
}: {
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: string }[];
  disabled?: boolean;
  placeholder?: string;
  className?: string;
}) {
  const selectedOption = options.find((o) => o.value === value);

  return (
    <Headless.Listbox value={value} onChange={onChange} disabled={disabled}>
      <span
        data-slot="control"
        className={clsx([
          className,
          // Basic layout
          "relative block w-full",
          // Background color + shadow applied to inset pseudo element, so shadow blends with border in light mode
          "before:absolute before:inset-px before:rounded-[calc(var(--radius-lg)-1px)] before:bg-white before:shadow-sm",
          // Background color is moved to control and shadow is removed in dark mode so hide `before` pseudo
          "dark:before:hidden",
          // Focus ring
          "after:pointer-events-none after:absolute after:inset-0 after:rounded-lg after:ring-transparent after:ring-inset sm:focus-within:after:ring-2 sm:focus-within:after:ring-blue-500",
          // Disabled state
          "has-data-disabled:opacity-50 has-data-disabled:before:bg-zinc-950/5 has-data-disabled:before:shadow-none",
        ])}
      >
        <Headless.ListboxButton
          className={clsx([
            // Basic layout
            "relative block w-full appearance-none rounded-lg px-[calc(--spacing(3.5)-1px)] py-[calc(--spacing(2.5)-1px)] sm:px-[calc(--spacing(3)-1px)] sm:py-[calc(--spacing(1.5)-1px)]",
            // Typography
            "text-left text-base/6 text-zinc-950 sm:text-sm/6 dark:text-white",
            // Border
            "border border-zinc-950/10 data-hover:border-zinc-950/20 dark:border-white/10 dark:data-hover:border-white/20",
            // Background color
            "bg-transparent dark:bg-white/5",
            // Hide default focus styles
            "focus:outline-hidden",
            // Disabled state
            "data-disabled:border-zinc-950/20 dark:data-disabled:border-white/15 dark:data-disabled:bg-white/2.5 dark:data-hover:data-disabled:border-white/15",
            // Right padding for chevron
            "pr-10 sm:pr-9",
          ])}
        >
          <span
            className={clsx(
              "block truncate",
              !selectedOption && "text-zinc-500",
            )}
          >
            {selectedOption ? selectedOption.label : placeholder}
          </span>
          <span className="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-3 sm:pr-2.5">
            <ChevronUpDownIcon
              className="size-5 text-zinc-500 sm:size-4 dark:text-zinc-400"
              aria-hidden="true"
            />
          </span>
        </Headless.ListboxButton>
      </span>

      <Headless.ListboxOptions
        anchor="bottom"
        className={clsx([
          // Basic layout
          "z-50 w-[var(--button-width)] rounded-lg p-1",
          // Background
          "bg-white shadow-lg ring-1 ring-zinc-950/10 dark:bg-neutral-800 dark:ring-white/10",
          // Transitions
          "transition duration-100 ease-in data-leave:data-closed:opacity-0",
          // Margin from button
          "[--anchor-gap:4px]",
        ])}
      >
        {options.map((option) => (
          <Headless.ListboxOption
            key={option.value}
            value={option.value}
            className={clsx([
              // Basic layout
              "relative cursor-default select-none rounded-md px-3 py-2 sm:px-2.5 sm:py-1.5",
              // Typography
              "text-base/6 text-zinc-950 sm:text-sm/6 dark:text-white",
              // Hover & focus
              "data-focus:bg-zinc-950/5 dark:data-focus:bg-white/10",
              // Selected
              "data-selected:font-medium",
            ])}
          >
            <span className="block truncate">{option.label}</span>
          </Headless.ListboxOption>
        ))}
      </Headless.ListboxOptions>
    </Headless.Listbox>
  );
}
