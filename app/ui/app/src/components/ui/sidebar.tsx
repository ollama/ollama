"use client";

import * as Headless from "@headlessui/react";
import clsx from "clsx";
import { LayoutGroup, motion } from "framer-motion";
import React, { forwardRef, useId } from "react";
import { TouchTarget } from "./button";
import { Link } from "./link";

export function Sidebar({
  className,
  ...props
}: React.ComponentPropsWithoutRef<"nav">) {
  return (
    <nav
      {...props}
      data-role="sidebar"
      className={clsx(className, "flex min-h-0 flex-col")}
    />
  );
}

export function SidebarHeader({
  className,
  ...props
}: React.ComponentPropsWithoutRef<"div">) {
  return (
    <div
      {...props}
      className={clsx(
        className,
        "flex flex-col p-4 dark:border-white/5 [&>[data-slot=section]+[data-slot=section]]:mt-2.5",
      )}
    />
  );
}

export function SidebarBody({
  className,
  ...props
}: React.ComponentPropsWithoutRef<"div">) {
  return (
    <div
      {...props}
      className={clsx(
        className,
        "flex flex-1 flex-col overflow-y-auto overscroll-contain p-4 [&>[data-slot=section]+[data-slot=section]]:mt-8",
      )}
    />
  );
}

export function SidebarFooter({
  className,
  ...props
}: React.ComponentPropsWithoutRef<"div">) {
  return (
    <div
      {...props}
      className={clsx(
        className,
        "flex flex-col border-t border-zinc-950/5 p-4 dark:border-white/5 [&>[data-slot=section]+[data-slot=section]]:mt-2.5",
      )}
    />
  );
}

export function SidebarSection({
  className,
  ...props
}: React.ComponentPropsWithoutRef<"div">) {
  const id = useId();

  return (
    <LayoutGroup id={id}>
      <div
        {...props}
        data-slot="section"
        className={clsx(className, "flex flex-col gap-0.5")}
      />
    </LayoutGroup>
  );
}

export function SidebarDivider({
  className,
  ...props
}: React.ComponentPropsWithoutRef<"hr">) {
  return (
    <hr
      {...props}
      className={clsx(
        className,
        "my-4 border-t border-zinc-950/5 -mx-4 dark:border-white/5",
      )}
    />
  );
}

export function SidebarSpacer({
  className,
  ...props
}: React.ComponentPropsWithoutRef<"div">) {
  return (
    <div
      aria-hidden="true"
      {...props}
      className={clsx(className, "mt-8 flex-1")}
    />
  );
}

export function SidebarHeading({
  className,
  ...props
}: React.ComponentPropsWithoutRef<"h3">) {
  return (
    <h3
      {...props}
      className={clsx(
        className,
        "mb-1 px-2 text-xs/6 font-medium text-zinc-500 dark:text-zinc-400",
      )}
    />
  );
}

type SidebarItemLinkProps = {
  current?: boolean;
  className?: string;
  children: React.ReactNode;
  href: string;
  mask?: { to: string };
} & Omit<Headless.ButtonProps<typeof Link>, "as" | "className">;

type SidebarItemButtonProps = {
  current?: boolean;
  className?: string;
  children: React.ReactNode;
} & Omit<Headless.ButtonProps, "as" | "className"> & {
    href?: never;
  };

type SidebarItemProps = SidebarItemLinkProps | SidebarItemButtonProps;

export const SidebarItem = forwardRef<
  HTMLAnchorElement | HTMLButtonElement,
  SidebarItemProps
>(function SidebarItem(props, ref) {
  const { current, className, children, ...restProps } = props;

  const classes = clsx(
    // Base
    "flex w-full items-center gap-3 rounded-lg px-2 py-2 text-left text-sm/5 font-medium text-zinc-950",
    // Leading icon/icon-only
    "*:data-[slot=icon]:size-5 *:data-[slot=icon]:shrink-0 *:data-[slot=icon]:fill-zinc-500",
    // Trailing icon (down chevron or similar)
    "*:last:data-[slot=icon]:ml-auto *:last:data-[slot=icon]:size-4",
    // Avatar
    "*:data-[slot=avatar]:-m-0.5 *:data-[slot=avatar]:size-6",
    // Hover
    "data-hover:bg-zinc-950/5 data-hover:*:data-[slot=icon]:fill-zinc-950",
    // Active
    "data-active:bg-zinc-950/5 data-active:*:data-[slot=icon]:fill-zinc-950",
    // Current
    "data-current:*:data-[slot=icon]:fill-zinc-950",
    // Dark mode
    "dark:text-white dark:*:data-[slot=icon]:fill-zinc-400",
    "dark:data-hover:bg-white/5 dark:data-hover:*:data-[slot=icon]:fill-white",
    "dark:data-active:bg-white/5 dark:data-active:*:data-[slot=icon]:fill-white",
    "dark:data-current:*:data-[slot=icon]:fill-white",
  );

  return (
    <span className={clsx(className, "relative")}>
      {current && (
        <motion.span
          layoutId="current-indicator"
          className="absolute inset-y-2 -left-4 w-0.5 rounded-full bg-zinc-950 dark:bg-white"
        />
      )}
      {"href" in restProps ? (
        <Headless.CloseButton
          as={Link}
          {...(restProps as Omit<
            SidebarItemLinkProps,
            "current" | "className" | "children"
          >)}
          className={classes}
          data-current={current ? "true" : undefined}
          ref={ref as React.ForwardedRef<HTMLAnchorElement>}
        >
          <TouchTarget>{children}</TouchTarget>
        </Headless.CloseButton>
      ) : (
        <Headless.Button
          {...(restProps as Omit<
            SidebarItemButtonProps,
            "current" | "className" | "children"
          >)}
          className={clsx("cursor-default", classes)}
          data-current={current ? "true" : undefined}
          ref={ref as React.ForwardedRef<HTMLButtonElement>}
        >
          <TouchTarget>{children}</TouchTarget>
        </Headless.Button>
      )}
    </span>
  );
});

export function SidebarLabel({
  className,
  ...props
}: React.ComponentPropsWithoutRef<"span">) {
  return <span {...props} className={clsx(className, "truncate")} />;
}
