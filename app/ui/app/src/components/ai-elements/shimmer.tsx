"use client";

import type { ComponentProps } from "react";
import { memo, useEffect, useState } from "react";

export type ShimmerProps = ComponentProps<"span"> & {
  duration?: number;
};

export const Shimmer = memo(
  ({ className, duration, children, ...props }: ShimmerProps) => {
    const [isShimmering, setIsShimmering] = useState(true);

    useEffect(() => {
      if (!duration) return;

      const timer = setTimeout(() => {
        setIsShimmering(false);
      }, duration * 1000);

      return () => clearTimeout(timer);
    }, [duration]);

    if (!isShimmering && duration) return <span>{children}</span>;

    return (
      <span
        className={`inline-block animate-pulse ${className || ""}`}
        {...props}
      >
        {children}
      </span>
    );
  },
);

Shimmer.displayName = "Shimmer";
