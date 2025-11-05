"use client";

import type { ComponentProps } from "react";
import { memo } from "react";

export type ResponseProps = ComponentProps<"div"> & {
  children: React.ReactNode;
};

export const Response = memo(
  ({ className, children, ...props }: ResponseProps) => (
    <div className={className} {...props}>
      {children}
    </div>
  ),
);

Response.displayName = "Response";
