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
