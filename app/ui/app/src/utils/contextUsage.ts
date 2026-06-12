import type { Message } from "@/gotypes";

type ContextMessage = Pick<Message, "role" | "promptEvalCount" | "evalCount">;

type ContextUsageTone = {
  strokeClass: string;
  badgeClass: string;
  textClass: string;
  label: string;
};

export type ContextUsageState = "empty" | "available" | "unavailable";

export type ContextUsageData = {
  state: ContextUsageState;
  latestAssistantMessage: ContextMessage | null;
  displayedAssistantMessage: ContextMessage | null;
  isPendingRefresh: boolean;
  latestPromptTokens: number;
  latestReplyTokens: number;
  estimatedConversationTokens: number;
  rawContextUsageRatio: number | null;
  contextUsageRatio: number | null;
  contextUsagePercent: number | null;
  isContextOverLimit: boolean;
  visibleContextUsageRatio: number;
  overflowTokens: number;
  shouldShowContextUsage: boolean;
  tone: ContextUsageTone;
  hint: string | null;
  summary: string;
  breakdown: string;
  tooltip: string;
};

function formatExactTokenCount(count: number): string {
  return count.toLocaleString();
}

export function getLatestAssistantMessage(
  messages: ContextMessage[],
): ContextMessage | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "assistant") {
      return messages[i];
    }
  }

  return null;
}

function getLatestAssistantMessageWithMetrics(
  messages: ContextMessage[],
): ContextMessage | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i];
    if (message.role === "assistant" && hasContextMetrics(message)) {
      return message;
    }
  }

  return null;
}

export function hasContextMetrics(message: ContextMessage | null): boolean {
  return (
    message !== null &&
    (message.promptEvalCount !== undefined || message.evalCount !== undefined)
  );
}

function getContextUsageTone(
  ratio: number,
  state: ContextUsageState,
  isOverLimit = false,
): ContextUsageTone {
  if (state === "unavailable") {
    return {
      strokeClass: "stroke-neutral-400 dark:stroke-neutral-500",
      badgeClass:
        "bg-neutral-500/12 text-neutral-700 dark:bg-neutral-500/15 dark:text-neutral-300",
      textClass: "text-neutral-600 dark:text-neutral-300",
      label: "Unavailable",
    };
  }

  if (isOverLimit) {
    return {
      strokeClass: "stroke-red-500 dark:stroke-red-400",
      badgeClass:
        "bg-red-500/15 text-red-700 dark:bg-red-500/20 dark:text-red-300",
      textClass: "text-red-600 dark:text-red-300",
      label: "Over limit",
    };
  }

  if (ratio >= 0.9) {
    return {
      strokeClass: "stroke-red-500 dark:stroke-red-400",
      badgeClass:
        "bg-red-500/12 text-red-700 dark:bg-red-500/15 dark:text-red-300",
      textClass: "text-red-600 dark:text-red-300",
      label: "Near limit",
    };
  }

  if (ratio >= 0.75) {
    return {
      strokeClass: "stroke-amber-500 dark:stroke-amber-400",
      badgeClass:
        "bg-amber-500/15 text-amber-700 dark:bg-amber-500/15 dark:text-amber-300",
      textClass: "text-amber-600 dark:text-amber-300",
      label: "Filling up",
    };
  }

  return {
    strokeClass: "stroke-emerald-500 dark:stroke-emerald-400",
    badgeClass:
      "bg-emerald-500/12 text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-300",
    textClass: "text-emerald-600 dark:text-emerald-300",
    label: "Comfortable",
  };
}

function getContextUsageHint(
  ratio: number | null,
  state: ContextUsageState,
  isOverLimit: boolean,
  isStreaming: boolean,
): string | null {
  if (state === "unavailable") {
    if (isStreaming) {
      return "Waiting for token counts from the latest response.";
    }

    return "The latest response did not report token counts.";
  }

  if (isOverLimit) {
    return "Older messages may already be dropped from the prompt.";
  }

  if (ratio !== null && ratio >= 0.9) {
    return "Older messages may start dropping soon.";
  }

  return null;
}

export function buildContextUsageData({
  messages,
  contextLength,
  isStreaming = false,
}: {
  messages: ContextMessage[];
  contextLength?: number;
  isStreaming?: boolean;
}): ContextUsageData {
  const latestAssistantMessage = getLatestAssistantMessage(messages);
  const latestAssistantMessageWithMetrics =
    getLatestAssistantMessageWithMetrics(messages);
  const latestAssistantMissingMetrics =
    latestAssistantMessage !== null && !hasContextMetrics(latestAssistantMessage);
  const isPendingRefresh =
    isStreaming &&
    latestAssistantMissingMetrics &&
    latestAssistantMessageWithMetrics !== null;
  const displayedAssistantMessage = isPendingRefresh
    ? latestAssistantMessageWithMetrics
    : latestAssistantMessage;
  const state: ContextUsageState =
    displayedAssistantMessage === null
      ? "empty"
      : hasContextMetrics(displayedAssistantMessage)
        ? "available"
        : "unavailable";

  const latestPromptTokens =
    state === "available"
      ? (displayedAssistantMessage?.promptEvalCount ?? 0)
      : 0;
  const latestReplyTokens =
    state === "available" ? (displayedAssistantMessage?.evalCount ?? 0) : 0;
  const estimatedConversationTokens = latestPromptTokens + latestReplyTokens;
  const rawContextUsageRatio =
    contextLength && state === "available"
      ? estimatedConversationTokens / contextLength
      : state === "empty" && contextLength
        ? 0
        : null;
  const contextUsageRatio =
    rawContextUsageRatio !== null ? Math.min(rawContextUsageRatio, 1) : null;
  const contextUsagePercent =
    rawContextUsageRatio !== null ? Math.round(rawContextUsageRatio * 100) : null;
  const isContextOverLimit =
    rawContextUsageRatio !== null && rawContextUsageRatio > 1;
  const tone = getContextUsageTone(
    contextUsageRatio ?? 0,
    state,
    isContextOverLimit,
  );
  const hint = getContextUsageHint(
    contextUsageRatio,
    state,
    isContextOverLimit,
    isStreaming,
  );
  const visibleContextUsageRatio =
    contextUsageRatio !== null && estimatedConversationTokens > 0
      ? Math.max(contextUsageRatio, 0.015)
      : 0;
  const overflowTokens =
    contextLength && estimatedConversationTokens > contextLength
      ? estimatedConversationTokens - contextLength
      : 0;

  let summary: string;
  if (state === "unavailable") {
    summary = contextLength
      ? `Context limit ${formatExactTokenCount(contextLength)} tokens`
      : "Token counts unavailable";
  } else if (contextLength) {
    summary = `~${formatExactTokenCount(estimatedConversationTokens)} / ${formatExactTokenCount(contextLength)} tokens${
      overflowTokens > 0
        ? ` (+${formatExactTokenCount(overflowTokens)} over)`
        : ""
    }`;
  } else {
    summary = `~${formatExactTokenCount(estimatedConversationTokens)} tokens`;
  }

  let breakdown = "No completed response yet";
  if (state === "unavailable") {
    breakdown = isStreaming
      ? "Latest response is still streaming"
      : "Latest response did not report token counts";
  } else if (state === "available") {
    breakdown = `Latest prompt ${formatExactTokenCount(latestPromptTokens)}${
      latestReplyTokens > 0
        ? ` + reply ${formatExactTokenCount(latestReplyTokens)}`
        : ""
    }`;

    if (isPendingRefresh) {
      breakdown += " - updating after the current response finishes";
    }
  }

  const tooltip = `Conversation context: ${summary}. ${breakdown}. ${tone.label}.${hint ? ` ${hint}` : ""}`;

  return {
    state,
    latestAssistantMessage,
    displayedAssistantMessage,
    isPendingRefresh,
    latestPromptTokens,
    latestReplyTokens,
    estimatedConversationTokens,
    rawContextUsageRatio,
    contextUsageRatio,
    contextUsagePercent,
    isContextOverLimit,
    visibleContextUsageRatio,
    overflowTokens,
    shouldShowContextUsage:
      contextLength !== undefined || latestAssistantMessage !== null,
    tone,
    hint,
    summary,
    breakdown,
    tooltip,
  };
}
