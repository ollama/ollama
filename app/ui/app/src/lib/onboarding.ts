// Keep in sync with store.CurrentOnboardingVersion in app/store/store.go.
export const CURRENT_ONBOARDING_VERSION = 1;

export type OnboardingAuthMode = "signin" | "signup";

export function onboardingAuthUrl(
  connectUrl: string,
  mode: OnboardingAuthMode,
): string {
  const url = new URL(connectUrl);
  url.searchParams.delete("launch");

  const authUrl = new URL(`/${mode}`, url.origin);
  authUrl.searchParams.set("next", `${url.pathname}${url.search}`);
  return authUrl.toString();
}

export type OnboardingStep = "intro" | "welcome" | "run";
export type OnboardingAction = "continue" | "skip" | "local";
export type AuthenticationTimeoutAction = "ignore" | "defer" | "fail";

export function nextOnboardingStep(
  step: OnboardingStep,
  action: OnboardingAction,
  _isAuthenticated: boolean,
): OnboardingStep {
  if (action === "skip" || action === "local") return "run";
  if (step === "intro") return "welcome";
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
