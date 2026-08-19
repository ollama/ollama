export const CURRENT_ONBOARDING_VERSION = 1;

export function onboardingConnectUrl(connectUrl: string): string {
  const url = new URL(connectUrl);
  url.searchParams.delete("launch");
  return url.toString();
}

export type OnboardingStep = "intro" | "welcome" | "run";
export type OnboardingAction = "continue" | "skip" | "local";
export type AuthenticationTimeoutAction = "ignore" | "defer" | "fail";

export function nextOnboardingStep(
  step: OnboardingStep,
  action: OnboardingAction,
  isAuthenticated: boolean,
): OnboardingStep {
  if (action === "skip" || action === "local") return "run";
  if (step === "intro") return isAuthenticated ? "run" : "welcome";
  return step;
}

export function authenticationTimeoutAction(
  settled: boolean,
  checking: boolean,
): AuthenticationTimeoutAction {
  if (settled) return "ignore";
  if (checking) return "defer";
  return "fail";
}

export function homeChatId(): "new" {
  return "new";
}
