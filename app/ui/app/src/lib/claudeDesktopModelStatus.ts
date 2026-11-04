import type { ClaudeDesktopModelStatus } from "@/types/webview";

export function claudeDesktopModelStatusLabel(
  model: ClaudeDesktopModelStatus,
): string | null {
  switch (model.reason) {
    case "sign_in_required":
      return "Sign in required";
    case "upgrade_required":
      return model.requiredPlan
        ? `${model.requiredPlan[0]?.toUpperCase()}${model.requiredPlan.slice(1)} plan required`
        : "Upgrade required";
    case "verification_unavailable":
      return "Access unavailable";
    case "model_not_installed":
      return "Not installed";
  }

  return null;
}
