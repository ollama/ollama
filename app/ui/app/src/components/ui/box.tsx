import clsx from "clsx";
import React, { forwardRef } from "react";

export interface BoxProps extends React.ComponentPropsWithoutRef<"div"> {
  as?: React.ElementType;
  variant?: "default" | "outlined" | "filled";
  padding?: "none" | "sm" | "md" | "lg";
  radius?: "none" | "sm" | "md" | "lg" | "full";
}

const variants = {
  default: "bg-transparent",
  outlined: "border border-zinc-200 dark:border-zinc-700",
  filled: "bg-zinc-50 dark:bg-zinc-800/50",
};

const paddings = {
  none: "p-0",
  sm: "p-2",
  md: "p-4",
  lg: "p-6",
};

const radiuses = {
  none: "rounded-none",
  sm: "rounded-sm",
  md: "rounded-md",
  lg: "rounded-lg",
  full: "rounded-full",
};

export const Box = forwardRef<HTMLElement, BoxProps>(
  (
    {
      as: Component = "div",
      variant = "default",
      padding = "none",
      radius = "none",
      className,
      children,
      ...props
    },
    ref,
  ) => {
    return (
      <Component
        ref={ref}
        {...props}
        className={clsx(
          variants[variant],
          paddings[padding],
          radiuses[radius],
          className,
        )}
      >
        {children}
      </Component>
    );
  },
);

Box.displayName = "Box";
