"use client";

import clsx from "clsx";
import type { ComponentProps } from "react";
import {
  useCallback,
  useEffect,
  useRef,
  createContext,
  useContext,
} from "react";
import { StickToBottom, useStickToBottomContext } from "use-stick-to-bottom";

// Create a context to share the "allow scroll" state
const ConversationControlContext = createContext<{
  allowScroll: () => void;
} | null>(null);

export type ConversationProps = ComponentProps<typeof StickToBottom> & {
  isStreaming?: boolean;
};

export const Conversation = ({
  className,
  isStreaming = false,
  children,
  ...props
}: ConversationProps) => {
  const shouldStopScrollRef = useRef(true);

  const allowScroll = useCallback(() => {
    shouldStopScrollRef.current = false;
    setTimeout(() => {
      shouldStopScrollRef.current = true;
    }, 100);
  }, []);

  return (
    <ConversationControlContext.Provider value={{ allowScroll }}>
      <StickToBottom
        className={clsx("relative h-full w-full overflow-y-auto", className)}
        initial="instant"
        resize="instant"
        role="log"
        {...props}
      >
        <ConversationContentInternal
          isStreaming={isStreaming}
          shouldStopScrollRef={shouldStopScrollRef}
        >
          <>{children}</>
        </ConversationContentInternal>
      </StickToBottom>
    </ConversationControlContext.Provider>
  );
};

const ConversationContentInternal = ({
  isStreaming,
  shouldStopScrollRef,
  children,
}: {
  isStreaming: boolean;
  shouldStopScrollRef: React.MutableRefObject<boolean>;
  children: React.ReactNode;
}) => {
  const { stopScroll } = useStickToBottomContext();
  const stopScrollRef = useRef(stopScroll);

  useEffect(() => {
    stopScrollRef.current = stopScroll;
  }, [stopScroll]);

  useEffect(() => {
    if (!isStreaming) return;

    const interval = setInterval(() => {
      if (shouldStopScrollRef.current) {
        stopScrollRef.current();
      }
    }, 16);

    if (shouldStopScrollRef.current) {
      stopScrollRef.current();
    }

    return () => clearInterval(interval);
  }, [isStreaming, shouldStopScrollRef]);

  return <>{children}</>;
};

export type ConversationContentProps = ComponentProps<
  typeof StickToBottom.Content
> & {
  isStreaming?: boolean;
};

export const ConversationContent = ({
  className,
  ...props
}: ConversationContentProps) => {
  return (
    <StickToBottom.Content
      className={clsx("flex flex-col", className)}
      {...props}
    />
  );
};

export type ConversationScrollButtonProps = ComponentProps<"button">;

export const ConversationScrollButton = ({
  className,
  ...props
}: ConversationScrollButtonProps) => {
  const { isAtBottom, scrollToBottom } = useStickToBottomContext();
  const context = useContext(ConversationControlContext);

  const handleScrollToBottom = useCallback(() => {
    console.log("scrollToBottom");
    context?.allowScroll();
    scrollToBottom();
  }, [scrollToBottom, context]);

  return (
    !isAtBottom && (
      <button
        className={clsx(
          "absolute bottom-4 left-[50%] translate-x-[-50%] rounded-full z-50",
          "bg-white dark:bg-neutral-800 border border-neutral-200 dark:border-neutral-700",
          "p-1 shadow-lg hover:shadow-xl transition-all",
          "text-neutral-700 dark:text-neutral-200 hover:scale-105",
          "hover:cursor-pointer",
          className,
        )}
        onClick={handleScrollToBottom}
        type="button"
        aria-label="Scroll to bottom"
        {...props}
      >
        <svg
          className="w-5 h-5"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>
    )
  );
};
