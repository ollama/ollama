import * as Headless from "@headlessui/react";
import clsx from "clsx";

export function RadioGroup({
  className,
  ...props
}: { className?: string } & Omit<
  Headless.RadioGroupProps,
  "as" | "className"
>) {
  return (
    <Headless.RadioGroup
      data-slot="control"
      {...props}
      className={clsx(
        className,
        // Basic groups
        "space-y-3 **:data-[slot=label]:font-normal",
        // With descriptions
        "has-data-[slot=description]:space-y-6 has-data-[slot=description]:**:data-[slot=label]:font-medium",
      )}
    />
  );
}

export function RadioField({
  className,
  ...props
}: { className?: string } & Omit<Headless.FieldProps, "as" | "className">) {
  return (
    <Headless.Field
      data-slot="field"
      {...props}
      className={clsx(
        className,
        // Base layout
        "grid grid-cols-[1.125rem_1fr] gap-x-4 gap-y-1 sm:grid-cols-[1rem_1fr]",
        // Control layout
        "*:data-[slot=control]:col-start-1 *:data-[slot=control]:row-start-1 *:data-[slot=control]:mt-0.75 sm:*:data-[slot=control]:mt-1",
        // Label layout
        "*:data-[slot=label]:col-start-2 *:data-[slot=label]:row-start-1",
        // Description layout
        "*:data-[slot=description]:col-start-2 *:data-[slot=description]:row-start-2",
        // With description
        "has-data-[slot=description]:**:data-[slot=label]:font-medium",
      )}
    />
  );
}

const base = [
  // Basic layout
  "relative isolate flex size-4.75 shrink-0 rounded-full sm:size-4.25",
  // Background color + shadow applied to inset pseudo element, so shadow blends with border in light mode
  "before:absolute before:inset-0 before:-z-10 before:rounded-full before:bg-white before:shadow-sm",
  // Background color when checked
  "group-data-checked:before:bg-(--radio-checked-bg)",
  // Background color is moved to control and shadow is removed in dark mode so hide `before` pseudo
  "dark:before:hidden",
  // Background color applied to control in dark mode
  "dark:bg-white/5 dark:group-data-checked:bg-(--radio-checked-bg)",
  // Border
  "border border-zinc-950/15 group-data-checked:border-transparent group-data-hover:group-data-checked:border-transparent group-data-hover:border-zinc-950/30 group-data-checked:bg-(--radio-checked-border)",
  "dark:border-white/15 dark:group-data-checked:border-white/5 dark:group-data-hover:group-data-checked:border-white/5 dark:group-data-hover:border-white/30",
  // Inner highlight shadow
  "after:absolute after:inset-0 after:rounded-full after:shadow-[inset_0_1px_--theme(--color-white/15%)]",
  "dark:after:-inset-px dark:after:hidden dark:after:rounded-full dark:group-data-checked:after:block",
  // Indicator color (light mode)
  "[--radio-indicator:transparent] group-data-checked:[--radio-indicator:var(--radio-checked-indicator)] group-data-hover:group-data-checked:[--radio-indicator:var(--radio-checked-indicator)] group-data-hover:[--radio-indicator:var(--color-zinc-900)]/10",
  // Indicator color (dark mode)
  "dark:group-data-hover:group-data-checked:[--radio-indicator:var(--radio-checked-indicator)] dark:group-data-hover:[--radio-indicator:var(--color-zinc-700)]",
  // Focus ring
  "group-data-focus:outline group-data-focus:outline-2 group-data-focus:outline-offset-2 group-data-focus:outline-blue-500",
  // Disabled state
  "group-data-disabled:opacity-50",
  "group-data-disabled:border-zinc-950/25 group-data-disabled:bg-zinc-950/5 group-data-disabled:[--radio-checked-indicator:var(--color-zinc-950)]/50 group-data-disabled:before:bg-transparent",
  "dark:group-data-disabled:border-white/20 dark:group-data-disabled:bg-white/2.5 dark:group-data-disabled:[--radio-checked-indicator:var(--color-white)]/50 dark:group-data-checked:group-data-disabled:after:hidden",
];

const colors = {
  "dark/zinc": [
    "[--radio-checked-bg:var(--color-zinc-900)] [--radio-checked-border:var(--color-zinc-950)]/90 [--radio-checked-indicator:var(--color-white)]",
    "dark:[--radio-checked-bg:var(--color-zinc-600)]",
  ],
  "dark/white": [
    "[--radio-checked-bg:var(--color-zinc-900)] [--radio-checked-border:var(--color-zinc-950)]/90 [--radio-checked-indicator:var(--color-white)]",
    "dark:[--radio-checked-bg:var(--color-white)] dark:[--radio-checked-border:var(--color-zinc-950)]/15 dark:[--radio-checked-indicator:var(--color-zinc-900)]",
  ],
  white:
    "[--radio-checked-bg:var(--color-white)] [--radio-checked-border:var(--color-zinc-950)]/15 [--radio-checked-indicator:var(--color-zinc-900)]",
  dark: "[--radio-checked-bg:var(--color-zinc-900)] [--radio-checked-border:var(--color-zinc-950)]/90 [--radio-checked-indicator:var(--color-white)]",
  zinc: "[--radio-checked-indicator:var(--color-white)] [--radio-checked-bg:var(--color-zinc-600)] [--radio-checked-border:var(--color-zinc-700)]/90",
  red: "[--radio-checked-indicator:var(--color-white)] [--radio-checked-bg:var(--color-red-600)] [--radio-checked-border:var(--color-red-700)]/90",
  orange:
    "[--radio-checked-indicator:var(--color-white)] [--radio-checked-bg:var(--color-orange-500)] [--radio-checked-border:var(--color-orange-600)]/90",
  amber:
    "[--radio-checked-bg:var(--color-amber-400)] [--radio-checked-border:var(--color-amber-500)]/80 [--radio-checked-indicator:var(--color-amber-950)]",
  yellow:
    "[--radio-checked-bg:var(--color-yellow-300)] [--radio-checked-border:var(--color-yellow-400)]/80 [--radio-checked-indicator:var(--color-yellow-950)]",
  lime: "[--radio-checked-bg:var(--color-lime-300)] [--radio-checked-border:var(--color-lime-400)]/80 [--radio-checked-indicator:var(--color-lime-950)]",
  green:
    "[--radio-checked-indicator:var(--color-white)] [--radio-checked-bg:var(--color-green-600)] [--radio-checked-border:var(--color-green-700)]/90",
  emerald:
    "[--radio-checked-indicator:var(--color-white)] [--radio-checked-bg:var(--color-emerald-600)] [--radio-checked-border:var(--color-emerald-700)]/90",
  teal: "[--radio-checked-indicator:var(--color-white)] [--radio-checked-bg:var(--color-teal-600)] [--radio-checked-border:var(--color-teal-700)]/90",
  cyan: "[--radio-checked-bg:var(--color-cyan-300)] [--radio-checked-border:var(--color-cyan-400)]/80 [--radio-checked-indicator:var(--color-cyan-950)]",
  sky: "[--radio-checked-indicator:var(--color-white)] [--radio-checked-bg:var(--color-sky-500)] [--radio-checked-border:var(--color-sky-600)]/80",
  blue: "[--radio-checked-indicator:var(--color-white)] [--radio-checked-bg:var(--color-blue-600)] [--radio-checked-border:var(--color-blue-700)]/90",
  indigo:
    "[--radio-checked-indicator:var(--color-white)] [--radio-checked-bg:var(--color-indigo-500)] [--radio-checked-border:var(--color-indigo-600)]/90",
  violet:
    "[--radio-checked-indicator:var(--color-white)] [--radio-checked-bg:var(--color-violet-500)] [--radio-checked-border:var(--color-violet-600)]/90",
  purple:
    "[--radio-checked-indicator:var(--color-white)] [--radio-checked-bg:var(--color-purple-500)] [--radio-checked-border:var(--color-purple-600)]/90",
  fuchsia:
    "[--radio-checked-indicator:var(--color-white)] [--radio-checked-bg:var(--color-fuchsia-500)] [--radio-checked-border:var(--color-fuchsia-600)]/90",
  pink: "[--radio-checked-indicator:var(--color-white)] [--radio-checked-bg:var(--color-pink-500)] [--radio-checked-border:var(--color-pink-600)]/90",
  rose: "[--radio-checked-indicator:var(--color-white)] [--radio-checked-bg:var(--color-rose-500)] [--radio-checked-border:var(--color-rose-600)]/90",
};

type Color = keyof typeof colors;

export function Radio({
  color = "dark/zinc",
  className,
  ...props
}: { color?: Color; className?: string } & Omit<
  Headless.RadioProps,
  "as" | "className" | "children"
>) {
  return (
    <Headless.Radio
      data-slot="control"
      {...props}
      className={clsx(className, "group inline-flex focus:outline-hidden")}
    >
      <span className={clsx([base, colors[color]])}>
        <span
          className={clsx(
            "size-full rounded-full border-[4.5px] border-transparent bg-(--radio-indicator) bg-clip-padding",
            // Forced colors mode
            "forced-colors:border-[Canvas] forced-colors:group-data-checked:border-[Highlight]",
          )}
        />
      </span>
    </Headless.Radio>
  );
}
