import type { ClaudeDesktopStatus } from "@/types/webview";

export function isClaudeConnectionComplete(
  enabled: boolean,
  status: ClaudeDesktopStatus,
) {
  return status.connected === enabled && !status.startFailed;
}
