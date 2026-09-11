import {
  useRef,
  useCallback,
  useEffect,
  useLayoutEffect,
  useState,
  useMemo,
} from "react";
import type { Message } from "@/gotypes";

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

const BOTTOM_THRESHOLD = 50;

export const useMessageAutoscroll = ({
  messages,
  isStreaming,
  chatId,
}: UseMessageAutoscrollOptions): MessageAutoscrollBehavior => {
  const containerRef = useRef<HTMLElement | null>(null);
  const [spacerHeight, setSpacerHeight] = useState(0);
  const spacerInDOM = useRef(0);
  const submitted = useRef(false);
  const needsScroll = useRef(false);
  const nearBottom = useRef(true);
  const cachedPaddingTop = useRef(0);

  const getLastUserIndex = useCallback((): number => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "user") return i;
    }
    return -1;
    // eslint-disable-next-line react-hooks/exhaustive-deps -- index only changes on add/remove
  }, [messages.length]);

  const getAnchorElement = useCallback(
    (container: HTMLElement): HTMLElement | null => {
      const idx = getLastUserIndex();
      if (idx < 0) return null;
      return container.querySelector(
        `[data-message-index="${idx}"]`,
      ) as HTMLElement | null;
    },
    [getLastUserIndex],
  );

  const getPaddingTop = useCallback((container: HTMLElement): number => {
    if (!cachedPaddingTop.current) {
      cachedPaddingTop.current =
        parseFloat(getComputedStyle(container).paddingTop) || 0;
    }
    return cachedPaddingTop.current;
  }, []);

  // Spacer so maxScroll = anchor at top of viewport
  const computeSpacer = useCallback((): number => {
    const container = containerRef.current;
    if (!container || !submitted.current) return 0;

    const anchor = getAnchorElement(container);
    if (!anchor) return 0;

    const paddingTop = getPaddingTop(container);
    const contentHeight = container.scrollHeight - spacerInDOM.current;

    return Math.max(
      0,
      anchor.offsetTop - paddingTop + container.clientHeight - contentHeight,
    );
  }, [getAnchorElement, getPaddingTop]);

  const applySpacer = useCallback(() => {
    const h = computeSpacer();
    if (h !== spacerInDOM.current) {
      spacerInDOM.current = h;
      setSpacerHeight(h);
    }
  }, [computeSpacer]);

  const handleNewUserMessage = useCallback(() => {
    submitted.current = true;
    needsScroll.current = true;
    nearBottom.current = true;
  }, []);

  // Recalculate spacer + scroll to anchor if pending
  useLayoutEffect(() => {
    applySpacer();

    if (needsScroll.current) {
      needsScroll.current = false;
      const container = containerRef.current;
      const anchor = container ? getAnchorElement(container) : null;
      if (container && anchor) {
        const paddingTop = getPaddingTop(container);
        const isLarge = anchor.offsetHeight > container.clientHeight * 0.7;

        requestAnimationFrame(() => {
          container.scrollTo({
            top: isLarge
              ? container.scrollHeight - container.clientHeight
              : anchor.offsetTop - paddingTop,
            behavior: "instant",
          });
        });
      }
    }
  }, [messages, applySpacer, getAnchorElement, getPaddingTop]);

  // Auto-scroll during streaming when user hasn't scrolled away
  useEffect(() => {
    const container = containerRef.current;
    if (!isStreaming || !nearBottom.current || !container) return;
    container.scrollTop = container.scrollHeight;
  }, [messages, isStreaming]);

  // Track nearBottom via wheel/touch only (scroll events fire on programmatic scrolls too)
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const isNearBottom = () => {
      const { scrollTop, scrollHeight, clientHeight } = container;
      return scrollTop + clientHeight >= scrollHeight - BOTTOM_THRESHOLD;
    };

    const onWheel = (e: WheelEvent) => {
      if (e.deltaY < 0) {
        nearBottom.current = false;
      } else if (e.deltaY > 0) {
        nearBottom.current = isNearBottom();
      }
    };

    let lastTouchY = 0;
    const onTouchStart = (e: TouchEvent) => {
      lastTouchY = e.touches[0].clientY;
    };
    const onTouchMove = (e: TouchEvent) => {
      const y = e.touches[0].clientY;
      if (y > lastTouchY) {
        // finger moves down = content scrolls up
        nearBottom.current = false;
      } else if (y < lastTouchY) {
        nearBottom.current = isNearBottom();
      }
      lastTouchY = y;
    };

    container.addEventListener("wheel", onWheel, { passive: true });
    container.addEventListener("touchstart", onTouchStart, { passive: true });
    container.addEventListener("touchmove", onTouchMove, { passive: true });
    return () => {
      container.removeEventListener("wheel", onWheel);
      container.removeEventListener("touchstart", onTouchStart);
      container.removeEventListener("touchmove", onTouchMove);
    };
  }, [chatId]);

  // Observe layout changes (images loading, expand/collapse)
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let timeout: ReturnType<typeof setTimeout>;
    const onResize = () => {
      if (submitted.current) {
        applySpacer();
      } else {
        clearTimeout(timeout);
        timeout = setTimeout(applySpacer, 80);
      }
    };

    const ro = new ResizeObserver(onResize);
    ro.observe(container);
    container
      .querySelectorAll("[data-message-index]")
      .forEach((el) => ro.observe(el));

    // Auto-observe new message elements
    const mo = new MutationObserver((mutations) => {
      for (const m of mutations) {
        for (const node of m.addedNodes) {
          if (node instanceof HTMLElement) {
            const el = node.dataset?.messageIndex
              ? node
              : node.querySelector?.("[data-message-index]");
            if (el) ro.observe(el);
          }
        }
      }
    });
    mo.observe(container, { childList: true, subtree: true });

    return () => {
      clearTimeout(timeout);
      ro.disconnect();
      mo.disconnect();
    };
  }, [applySpacer, chatId]);

  // Reset on chat change
  useEffect(() => {
    submitted.current = false;
    nearBottom.current = true;
    cachedPaddingTop.current = 0;
    spacerInDOM.current = 0;
    setSpacerHeight(0);
  }, [chatId]);

  return useMemo(
    () => ({ handleNewUserMessage, containerRef, spacerHeight }),
    [handleNewUserMessage, spacerHeight],
  );
};
