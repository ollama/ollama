import type { ClaudeDesktopStatus } from "@/types/webview";

export function isClaudeConnectionComplete(
  enabled: boolean,
  status: ClaudeDesktopStatus,
) {
  return status.connected === enabled && !status.startFailed;
}

export function shouldShowClaudeConnectedIntro(
  status: ClaudeDesktopStatus,
  introSeen: boolean,
) {
  return status.connected && !status.startFailed && !introSeen;
}
