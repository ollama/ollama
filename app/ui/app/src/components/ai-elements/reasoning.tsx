"use client";

import { useControllableState } from "@radix-ui/react-use-controllable-state";
import { Collapsible, CollapsibleTrigger } from "@radix-ui/react-collapsible";
import { ChevronDownIcon } from "lucide-react";
import type { ComponentProps } from "react";
import { createContext, memo, useContext, useEffect, useState } from "react";
import { Shimmer } from "./shimmer";

type ReasoningContextValue = {
  isStreaming: boolean;
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
  duration: number;
};

const ReasoningContext = createContext<ReasoningContextValue | null>(null);

const useReasoning = () => {
  const context = useContext(ReasoningContext);
  if (!context) {
    throw new Error("Reasoning components must be used within Reasoning");
  }
  return context;
};

export type ReasoningProps = ComponentProps<typeof Collapsible> & {
  isStreaming?: boolean;
  open?: boolean;
  defaultOpen?: boolean;
  onOpenChange?: (open: boolean) => void;
  duration?: number;
};

const MS_IN_S = 1000;

export const Reasoning = memo(
  ({
    className,
    isStreaming = false,
    open,
    defaultOpen = false,
    onOpenChange,
    duration: durationProp,
    children,
    ...props
  }: ReasoningProps) => {
    const [isOpen, setIsOpen] = useControllableState({
      prop: open,
      defaultProp: defaultOpen,
      onChange: onOpenChange,
    });
    const [duration, setDuration] = useControllableState({
      prop: durationProp,
      defaultProp: 0,
    });

    const [startTime, setStartTime] = useState<number | null>(null);

    // Track duration when streaming starts and ends
    useEffect(() => {
      if (isStreaming) {
        if (startTime === null) {
          setStartTime(Date.now());
        }
      } else if (startTime !== null) {
        setDuration(Math.ceil((Date.now() - startTime) / MS_IN_S));
        setStartTime(null);
      }
    }, [isStreaming, startTime, setDuration]);

    const handleOpenChange = (newOpen: boolean) => {
      setIsOpen(newOpen);
    };

    return (
      <ReasoningContext.Provider
        value={{ isStreaming, isOpen, setIsOpen, duration }}
      >
        <Collapsible
          className={`not-prose mb-4 ${className || ""}`}
          onOpenChange={handleOpenChange}
          open={isOpen}
          {...props}
        >
          {children}
        </Collapsible>
      </ReasoningContext.Provider>
    );
  },
);

export type ReasoningTriggerProps = ComponentProps<typeof CollapsibleTrigger>;

export const getThinkingMessage = (isStreaming: boolean, duration?: number) => {
  if (isStreaming || duration === 0) {
    return <Shimmer duration={1}>Thinking...</Shimmer>;
  }
  if (duration === undefined) {
    return <span>Thought for a few seconds</span>;
  }
  if (duration < 2) {
    return <span>Thought for a moment</span>;
  }
  return <span>Thought for {duration} seconds</span>;
};

export const ReasoningTrigger = memo(
  ({ className, children, ...props }: ReasoningTriggerProps) => {
    const { isStreaming, isOpen, duration } = useReasoning();

    return (
      <CollapsibleTrigger
        className={`flex w-full items-center gap-2 text-muted-foreground text-sm transition-colors hover:text-foreground cursor-pointer ${className || ""}`}
        {...props}
      >
        {children ?? (
          <>
            {/* Light bulb icon */}
            <svg className="w-3 fill-current" viewBox="0 0 14 24" fill="none">
              <path d="M0 6.01562C0 9.76562 2.24609 10.6934 2.87109 17.207C2.91016 17.5586 3.10547 17.7832 3.47656 17.7832H9.58984C9.9707 17.7832 10.166 17.5586 10.2051 17.207C10.8301 10.6934 13.0664 9.76562 13.0664 6.01562C13.0664 2.64648 10.1855 0 6.5332 0C2.88086 0 0 2.64648 0 6.01562ZM1.47461 6.01562C1.47461 3.37891 3.78906 1.47461 6.5332 1.47461C9.27734 1.47461 11.5918 3.37891 11.5918 6.01562C11.5918 8.81836 9.73633 9.48242 8.85742 16.3086H4.21875C3.33008 9.48242 1.47461 8.81836 1.47461 6.01562ZM3.44727 19.8926H9.62891C9.95117 19.8926 10.1953 19.6387 10.1953 19.3164C10.1953 19.0039 9.95117 18.75 9.62891 18.75H3.44727C3.125 18.75 2.87109 19.0039 2.87109 19.3164C2.87109 19.6387 3.125 19.8926 3.44727 19.8926ZM6.5332 22.7246C8.04688 22.7246 9.30664 21.9824 9.4043 20.8594H3.67188C3.74023 21.9824 5.00977 22.7246 6.5332 22.7246Z" />
            </svg>
            {getThinkingMessage(isStreaming, duration)}
            <ChevronDownIcon
              className={`size-4 transition-transform duration-300 ${
                isOpen ? "rotate-180" : "rotate-0"
              }`}
            />
          </>
        )}
      </CollapsibleTrigger>
    );
  },
);

Reasoning.displayName = "Reasoning";
ReasoningTrigger.displayName = "ReasoningTrigger";
