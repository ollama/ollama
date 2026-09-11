// Keep in sync with store.CurrentOnboardingVersion in app/store/store.go.
export const CURRENT_ONBOARDING_VERSION = 1;

export type OnboardingAuthMode = "signin" | "signup";

export function onboardingConnectUrl(
  connectUrl: string,
  mode: OnboardingAuthMode,
): string {
  const url = new URL(connectUrl);
  url.searchParams.delete("launch");
  if (mode === "signup") {
    url.searchParams.set("signup", "true");
  } else {
    url.searchParams.delete("signup");
  }
  return url.toString();
}

export const AUTHENTICATION_TIMEOUT_MS = 5 * 60 * 1000;

export type OnboardingStep = "intro" | "welcome" | "apps" | "run";

export type OnboardingAction = "continue" | "authenticated" | "local";
export type AuthenticationTimeoutAction = "ignore" | "defer" | "fail";

export function nextOnboardingStep(
  step: OnboardingStep,
  action: OnboardingAction,
  isAuthenticated: boolean,
): OnboardingStep {
  if (action === "local") return "run";
  if (step === "intro" && action === "continue") {
    return isAuthenticated ? "apps" : "welcome";
  }
  if (step === "welcome" && action === "authenticated") return "apps";
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
