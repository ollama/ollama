import type { ClaudeDesktopStatus } from "@/types/webview";

export const CLAUDE_INSTALL_TIMEOUT_MS = 120_000;

export function isClaudeConnectionComplete(
  enabled: boolean,
  status: ClaudeDesktopStatus,
) {
  return enabled ? status.connected && !status.startFailed : !status.configured;
}

export function scheduleClaudeInstallTimeout(onTimeout: () => void) {
  return window.setTimeout(onTimeout, CLAUDE_INSTALL_TIMEOUT_MS);
}

// Claude Desktop has a bounded model list. The app supplies the limit when it
// knows it; retain the existing five-model behavior for older app versions.
export const defaultClaudeDesktopMaxModels = 5;

export function claudeDesktopMaxModels(
  status?: ClaudeDesktopStatus | null,
): number {
  return status?.maxModels && status.maxModels > 0
    ? status.maxModels
    : defaultClaudeDesktopMaxModels;
}

export function claudeDesktopMaxModelsMessage(maxModels: number): string {
  return `Claude supports up to ${maxModels} models. Deselect one to add another.`;
}

// addClaudeModelSelection returns the selection with name appended, or an
// unchanged selection plus an error when the Claude model limit is reached.
export function addClaudeModelSelection(
  selection: string[],
  name: string,
  maxModels: number,
): { selection: string[]; error?: string } {
  if (selection.includes(name)) return { selection };
  if (selection.length >= maxModels) {
    return { selection, error: claudeDesktopMaxModelsMessage(maxModels) };
  }
  return { selection: [...selection, name] };
}
