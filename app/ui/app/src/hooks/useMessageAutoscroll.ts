import {
  useRef,
  useCallback,
  useEffect,
  useLayoutEffect,
  useState,
  useMemo,
} from "react";
import type { Message } from "@/gotypes";

// warning: this file is all claude code, needs to be looked into more closely

interface UseMessageAutoscrollOptions {
  messages: Message[];
  isStreaming: boolean;
  chatId: string;
}

interface MessageAutoscrollBehavior {
  handleNewUserMessage: () => void;
  containerRef: React.RefObject<HTMLElement | null>;
  spacerHeight: number;
}

export const useMessageAutoscroll = ({
  messages,
  isStreaming,
  chatId,
}: UseMessageAutoscrollOptions): MessageAutoscrollBehavior => {
  const containerRef = useRef<HTMLElement | null>(null);
  const pendingScrollToUserMessage = useRef(false);
  const [spacerHeight, setSpacerHeight] = useState(0);
  const lastScrollHeightRef = useRef(0);
  const lastScrollTopRef = useRef(0);
  const [isActiveInteraction, setIsActiveInteraction] = useState(false);
  const [hasSubmittedMessage, setHasSubmittedMessage] = useState(false);
  const prevChatIdRef = useRef<string>(chatId);

  // Find the last user message index from React state
  const getLastUserMessageIndex = useCallback(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "user") {
        return i;
      }
    }
    return -1;
  }, [messages]);

  const scrollToMessage = useCallback((messageIndex: number) => {
    if (!containerRef.current || messageIndex < 0) {
      return;
    }

    const container = containerRef.current;
    // select the exact element by its data-message-index to avoid index mismatches
    const targetElement = container.querySelector(
      `[data-message-index="${messageIndex}"]`,
    ) as HTMLElement | null;

    if (!targetElement) return;

    const containerHeight = container.clientHeight;
    const containerStyle = window.getComputedStyle(container);
    const paddingTop = parseFloat(containerStyle.paddingTop) || 0;
    const scrollHeight = container.scrollHeight;
    const messageHeight = targetElement.offsetHeight;

    // Check if the message is large, which is 70% of the container height
    const isLarge = messageHeight > containerHeight * 0.7;

    let targetPosition: number = targetElement.offsetTop - paddingTop; // default to scrolling the message to the top of the window

    if (isLarge) {
      // when the message is large scroll to the bottom of it
      targetPosition = scrollHeight - containerHeight;
    }

    // Ensure we don't scroll past content boundaries
    const maxScroll = scrollHeight - containerHeight;
    const finalPosition = Math.min(Math.max(0, targetPosition), maxScroll);

    container.scrollTo({
      top: finalPosition,
      behavior: "smooth",
    });
  }, []);

  // Calculate and set the spacer height based on container dimensions
  const updateSpacerHeight = useCallback(() => {
    if (!containerRef.current) {
      return;
    }

    const containerHeight = containerRef.current.clientHeight;

    // Find the last user message to calculate spacer for
    const lastUserIndex = getLastUserMessageIndex();

    if (lastUserIndex < 0) {
      setSpacerHeight(0);
      return;
    }

    const messageElements = containerRef.current.querySelectorAll(
      "[data-message-index]",
    ) as NodeListOf<HTMLElement>;

    if (!messageElements || messageElements.length === 0) {
      setSpacerHeight(0);
      return;
    }

    const targetElement = containerRef.current.querySelector(
      `[data-message-index="${lastUserIndex}"]`,
    ) as HTMLElement | null;

    if (!targetElement) {
      setSpacerHeight(0);
      return;
    }

    const elementsAfter = Array.from(messageElements).filter((el) => {
      const idx = Number(el.dataset.messageIndex);
      return Number.isFinite(idx) && idx > lastUserIndex;
    });

    const contentHeightAfterTarget = elementsAfter.reduce(
      (sum, el) => sum + el.offsetHeight,
      0,
    );

    // Calculate the spacer height needed to position the user message at the top
    // Add extra space for assistant response area
    const targetMessageHeight = targetElement.offsetHeight;

    // Calculate spacer to position the last user message at the top
    // For new messages, we want them to appear at the top regardless of content after
    // For large messages, we want to preserve the scroll-to-bottom behavior
    // which shows part of the message and space for streaming response
    let baseHeight: number;

    if (contentHeightAfterTarget === 0) {
      // No content after the user message (new message case)
      // Position it at the top with some padding
      baseHeight = Math.max(0, containerHeight - targetMessageHeight);
    } else {
      // Content exists after the user message
      // Calculate spacer to position user message at top
      baseHeight = Math.max(
        0,
        containerHeight - contentHeightAfterTarget - targetMessageHeight,
      );
    }

    // Only apply spacer height when actively interacting (streaming or pending new message)
    // When just viewing a chat, don't add extra space
    if (!isActiveInteraction) {
      setSpacerHeight(0);
      return;
    }

    // Add extra space for assistant response only when streaming
    const extraSpaceForAssistant = isStreaming ? containerHeight * 0.4 : 0;
    const calculatedHeight = baseHeight + extraSpaceForAssistant;

    setSpacerHeight(calculatedHeight);
  }, [getLastUserMessageIndex, isStreaming, isActiveInteraction]);

  // Handle new user message submission
  const handleNewUserMessage = useCallback(() => {
    // Mark that we're expecting a new message and should scroll to it
    pendingScrollToUserMessage.current = true;
    setIsActiveInteraction(true);
    setHasSubmittedMessage(true);
  }, []);

  // Use layoutEffect to scroll immediately after DOM updates
  useLayoutEffect(() => {
    if (pendingScrollToUserMessage.current) {
      // Find the last user message from current state
      const targetUserIndex = getLastUserMessageIndex();

      if (targetUserIndex >= 0) {
        requestAnimationFrame(() => {
          updateSpacerHeight();
          requestAnimationFrame(() => {
            scrollToMessage(targetUserIndex);
            pendingScrollToUserMessage.current = false;
          });
        });
      } else {
        pendingScrollToUserMessage.current = false;
        // Reset active interaction if no target found
        setIsActiveInteraction(isStreaming);
      }
    }
  }, [
    messages,
    getLastUserMessageIndex,
    scrollToMessage,
    updateSpacerHeight,
    isStreaming,
  ]);

  // Update active interaction state based on streaming and message submission
  useEffect(() => {
    if (
      isStreaming ||
      pendingScrollToUserMessage.current ||
      hasSubmittedMessage
    ) {
      setIsActiveInteraction(true);
    } else {
      setIsActiveInteraction(false);
    }
  }, [isStreaming, hasSubmittedMessage]);

  useEffect(() => {
    if (prevChatIdRef.current !== chatId) {
      setIsActiveInteraction(false);
      setHasSubmittedMessage(false);
      prevChatIdRef.current = chatId;
    }
  }, [chatId]);

  // Recalculate spacer height when messages change
  useEffect(() => {
    updateSpacerHeight();
  }, [messages, updateSpacerHeight]);

  // Use ResizeObserver to handle dynamic content changes
  useEffect(() => {
    if (!containerRef.current) return;

    let resizeTimeout: ReturnType<typeof setTimeout>;
    let immediateUpdate = false;

    const resizeObserver = new ResizeObserver((entries) => {
      // Check if this is a significant height change (like collapsing content)
      let hasSignificantChange = false;
      for (const entry of entries) {
        const element = entry.target as HTMLElement;
        if (
          element.dataset.messageIndex &&
          entry.contentRect.height !== element.offsetHeight
        ) {
          const heightDiff = Math.abs(
            entry.contentRect.height - element.offsetHeight,
          );
          if (heightDiff > 50) {
            hasSignificantChange = true;
            break;
          }
        }
      }

      // For significant changes, update immediately
      if (hasSignificantChange || immediateUpdate) {
        updateSpacerHeight();
        immediateUpdate = false;
      } else {
        // For small changes (like streaming text), debounce
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
          updateSpacerHeight();
        }, 100);
      }
    });

    // Also use MutationObserver for immediate attribute changes
    const mutationObserver = new MutationObserver((mutations) => {
      // Check if any mutations are related to expanding/collapsing
      const hasToggle = mutations.some(
        (mutation) =>
          mutation.type === "attributes" &&
          (mutation.attributeName === "class" ||
            mutation.attributeName === "style" ||
            mutation.attributeName === "open" ||
            mutation.attributeName === "data-expanded"),
      );

      if (hasToggle) {
        immediateUpdate = true;
        updateSpacerHeight();
      }
    });

    // Observe the container and all messages
    resizeObserver.observe(containerRef.current);
    mutationObserver.observe(containerRef.current, {
      attributes: true,
      subtree: true,
      attributeFilter: ["class", "style", "open", "data-expanded"],
    });

    // Observe all message elements for size changes
    const messageElements = containerRef.current.querySelectorAll(
      "[data-message-index]",
    );
    messageElements.forEach((element) => {
      resizeObserver.observe(element);
    });

    return () => {
      clearTimeout(resizeTimeout);
      resizeObserver.disconnect();
      mutationObserver.disconnect();
    };
  }, [messages, updateSpacerHeight]);

  // Track scroll position
  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const handleScroll = () => {
      lastScrollTopRef.current = container.scrollTop;
      lastScrollHeightRef.current = container.scrollHeight;
    };

    container.addEventListener("scroll", handleScroll);

    // Initialize scroll tracking
    lastScrollTopRef.current = container.scrollTop;
    lastScrollHeightRef.current = container.scrollHeight;

    return () => {
      container.removeEventListener("scroll", handleScroll);
    };
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      pendingScrollToUserMessage.current = false;
    };
  }, []);

  return useMemo(
    () => ({
      handleNewUserMessage,
      containerRef,
      spacerHeight,
    }),
    [handleNewUserMessage, containerRef, spacerHeight],
  );
};
