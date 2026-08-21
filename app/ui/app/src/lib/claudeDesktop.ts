import type {
  ClaudeDesktopModelStatus,
  ClaudeDesktopStatus,
} from "@/types/webview";

export const CLAUDE_INSTALL_TIMEOUT_MS = 120_000;

export function isClaudeConnectionComplete(
  enabled: boolean,
  status: ClaudeDesktopStatus,
) {
  const configured = isClaudeConfigured(status);
  return enabled
    ? configured && status.connected && !status.startFailed
    : !configured;
}

export function isClaudeConfigured(status: ClaudeDesktopStatus): boolean {
  return status.configured ?? status.connected;
}

export function claudeDesktopRecoveryMessage(
  statusError?: string,
  actionError?: string | null,
): string | null {
  return statusError || actionError || null;
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

// Unavailable models remain visible for account guidance, but cannot remain
// selected. Choose one available model when filtering would empty the list.
export function claudeDesktopUsableSelection(
  models: ClaudeDesktopModelStatus[],
  selectAllAvailable = false,
  maxModels = defaultClaudeDesktopMaxModels,
): string[] {
  const available = models.filter(
    (model) =>
      model.availability === undefined || model.availability === "available",
  );
  if (selectAllAvailable) {
    return available.slice(0, maxModels).map((model) => model.name);
  }
  const selected = available
    .filter((model) => model.selected)
    .map((model) => model.name);
  if (selected.length > 0) return selected;
  return available.length > 0 ? [available[0].name] : [];
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
