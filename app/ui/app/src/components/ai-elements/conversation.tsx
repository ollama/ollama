"use client";

import clsx from "clsx";
import type { ComponentProps } from "react";
import {
  useCallback,
  useEffect,
  useRef,
  createContext,
  useContext,
  useState,
} from "react";
import { StickToBottom, useStickToBottomContext } from "use-stick-to-bottom";

// Create a context to share the "allow scroll" state and spacer state
const ConversationControlContext = createContext<{
  allowScroll: () => void;
  spacerHeight: number;
  setSpacerHeight: (height: number) => void;
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
  const [spacerHeight, setSpacerHeight] = useState(0);

  const allowScroll = useCallback(() => {
    shouldStopScrollRef.current = false;
    setTimeout(() => {
      shouldStopScrollRef.current = true;
    }, 100);
  }, []);

  return (
    <ConversationControlContext.Provider
      value={{ allowScroll, spacerHeight, setSpacerHeight }}
    >
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

// New wrapper component that includes spacer management
export const ConversationWithSpacer = ({
  isStreaming,
  messageCount,
  children,
  ...props
}: ConversationProps & {
  messageCount: number;
}) => {
  return (
    <Conversation isStreaming={isStreaming} {...props}>
      <SpacerController
        isStreaming={isStreaming ?? false}
        messageCount={messageCount}
      />
      <>{children}</>
    </Conversation>
  );
};

// This component manages the spacer state but doesn't render the spacer itself
const SpacerController = ({
  isStreaming,
  messageCount,
}: {
  isStreaming: boolean;
  messageCount: number;
}) => {
  const context = useContext(ConversationControlContext);
  const { scrollToBottom } = useStickToBottomContext();
  const previousMessageCountRef = useRef(messageCount);
  const scrollContainerRef = useRef<HTMLElement | null>(null);
  const [isActiveInteraction, setIsActiveInteraction] = useState(false);

  // Get reference to scroll container
  useEffect(() => {
    const container = document.querySelector('[role="log"]') as HTMLElement;
    scrollContainerRef.current = container;
  }, []);

  // Calculate spacer height based on actual DOM elements
  const calculateSpacerHeight = useCallback(() => {
    const container = scrollContainerRef.current;
    if (!container) {
      console.log("❌ No container");
      return 0;
    }

    const containerHeight = container.clientHeight;

    // Find all message elements
    const messageElements = container.querySelectorAll(
      "[data-message-index]",
    ) as NodeListOf<HTMLElement>;
    console.log("📝 Found", messageElements.length, "message elements");

    if (messageElements.length === 0) return 0;

    // Log all messages and their roles
    messageElements.forEach((el, i) => {
      const role = el.getAttribute("data-message-role");
      console.log(`  Message ${i}: role="${role}"`);
    });

    // Find the last user message
    let lastUserMessageElement: HTMLElement | null = null;
    let lastUserMessageIndex = -1;

    for (let i = messageElements.length - 1; i >= 0; i--) {
      const el = messageElements[i];
      const role = el.getAttribute("data-message-role");
      if (role === "user") {
        lastUserMessageElement = el;
        lastUserMessageIndex = i;
        console.log("✅ Found user message at index:", i);
        break;
      }
    }

    if (!lastUserMessageElement) {
      console.log("❌ No user message found!");
      return 0;
    }

    // Calculate height of content after the last user message
    let contentHeightAfter = 0;
    for (let i = lastUserMessageIndex + 1; i < messageElements.length; i++) {
      contentHeightAfter += messageElements[i].offsetHeight;
    }

    const userMessageHeight = lastUserMessageElement.offsetHeight;

    // Goal: Position user message at the top with some padding
    // We want the user message to start at around 10% from the top of viewport
    const targetTopPosition = containerHeight * 0.05; // 10% from top

    // Calculate spacer: we need enough space so that when scrolled to bottom:
    // spacerHeight = containerHeight - targetTopPosition - userMessageHeight - contentAfter
    const calculatedHeight =
      containerHeight -
      targetTopPosition -
      userMessageHeight -
      contentHeightAfter;

    const baseHeight = Math.max(0, calculatedHeight);

    console.log(
      "📊 Container:",
      containerHeight,
      "User msg:",
      userMessageHeight,
      "Content after:",
      contentHeightAfter,
      "Target top pos:",
      targetTopPosition,
      "→ Final spacer:",
      baseHeight,
    );

    return baseHeight;
  }, []);

  // When a new message is submitted, set initial spacer height and scroll
  useEffect(() => {
    if (messageCount > previousMessageCountRef.current) {
      console.log("🎯 NEW MESSAGE - Setting spacer");
      // Allow scrolling by temporarily disabling stopScroll
      context?.allowScroll();
      setIsActiveInteraction(true);

      const container = scrollContainerRef.current;

      if (container) {
        // Wait for new message to render in DOM
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            const spacerHeight = calculateSpacerHeight();
            console.log("📏 Calculated spacer:", spacerHeight);
            context?.setSpacerHeight(spacerHeight);

            // Wait for spacer to be added to DOM, then scroll
            requestAnimationFrame(() => {
              requestAnimationFrame(() => {
                // Use the library's scrollToBottom method
                console.log("📜 Scrolling to bottom");
                scrollToBottom("instant");
                console.log("📜 Final scrollTop:", container.scrollTop);
              });
            });
          });
        });
      }
    }
    previousMessageCountRef.current = messageCount;
  }, [messageCount, context, calculateSpacerHeight, scrollToBottom]);

  // Update active interaction state
  useEffect(() => {
    if (isStreaming) {
      setIsActiveInteraction(true);
    }
    // Don't automatically set to false when streaming stops
    // Let the ResizeObserver handle clearing the spacer naturally
  }, [isStreaming]);

  // Use ResizeObserver to recalculate spacer as content changes
  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container || !isActiveInteraction) return;

    const resizeObserver = new ResizeObserver(() => {
      const newHeight = calculateSpacerHeight();
      context?.setSpacerHeight(newHeight);

      // Clear active interaction when spacer reaches 0
      if (newHeight === 0) {
        setIsActiveInteraction(false);
      }
    });

    // Observe all message elements
    const messageElements = container.querySelectorAll("[data-message-index]");
    messageElements.forEach((element) => {
      resizeObserver.observe(element);
    });

    return () => {
      resizeObserver.disconnect();
    };
  }, [isActiveInteraction, calculateSpacerHeight, context]);

  // Remove the effect that clears spacer when not streaming
  // This was causing the spacer to disappear prematurely

  return null;
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
  children,
  ...props
}: ConversationContentProps) => {
  return (
    <StickToBottom.Content
      className={clsx("flex flex-col", className)}
      {...props}
    >
      {children}
    </StickToBottom.Content>
  );
};

// Spacer component that can be placed anywhere in your content
export const ConversationSpacer = () => {
  const context = useContext(ConversationControlContext);
  const spacerHeight = context?.spacerHeight ?? 0;

  console.log("🎨 Spacer render - height:", spacerHeight);

  if (spacerHeight === 0) return null;

  return (
    <div
      style={{
        height: `${spacerHeight}px`,
        flexShrink: 0,
        backgroundColor: "rgba(255,0,0,0.1)", // Temporary for debugging
      }}
      aria-hidden="true"
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
    context?.allowScroll();
    scrollToBottom();
  }, [scrollToBottom, context]);

  // Show button if not at bottom AND spacer is not active (height is 0)
  const shouldShowButton = !isAtBottom && (context?.spacerHeight ?? 0) === 0;

  return (
    shouldShowButton && (
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
