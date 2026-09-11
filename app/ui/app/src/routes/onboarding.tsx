import Onboarding from "@/components/Onboarding";
import { getSettings } from "@/api";
import { useSettings } from "@/hooks/useSettings";
import { useUser } from "@/hooks/useUser";
import {
  AUTHENTICATION_TIMEOUT_MS,
  authenticationTimeoutAction,
  CURRENT_ONBOARDING_VERSION,
  homeChatId,
  onboardingConnectUrl,
  type OnboardingAuthMode,
} from "@/lib/onboarding";
import { createFileRoute, redirect, useNavigate } from "@tanstack/react-router";
import { useCallback, useEffect, useRef, useState } from "react";

export const Route = createFileRoute("/onboarding")({
  beforeLoad: async ({ context }) => {
    // Let developers review onboarding without resetting their local app data.
    if (
      import.meta.env.DEV &&
      new URLSearchParams(window.location.search).get("preview") === "1"
    ) {
      return;
    }

    const settingsData = await context.queryClient.ensureQueryData({
      queryKey: ["settings"],
      queryFn: getSettings,
    });

    if (settingsData.settings.OnboardingVersion >= CURRENT_ONBOARDING_VERSION) {
      const chatId = homeChatId();
      throw redirect({
        to: "/c/$chatId",
        params: { chatId },
        mask: { to: "/" },
      });
    }
  },
  component: OnboardingRoute,
});

function OnboardingRoute() {
  const navigate = useNavigate();
  const { settingsData, setSettings } = useSettings();
  const { fetchConnectUrl, refetchUser, isAuthenticated } = useUser();
  const [isAwaitingAuth, setIsAwaitingAuth] = useState(false);
  const [signInError, setSignInError] = useState<string | null>(null);
  const [completionError, setCompletionError] = useState<string | null>(null);
  const authAttemptRef = useRef(0);

  const completeOnboarding = useCallback(async (): Promise<boolean> => {
    setCompletionError(null);

    try {
      if (!settingsData) {
        throw new Error("Settings are not loaded");
      }

      await setSettings({
        OnboardingVersion: CURRENT_ONBOARDING_VERSION,
      });
      return true;
    } catch (error) {
      console.error("Failed to save onboarding state:", error);
      setCompletionError("Unable to save setup. Please try again.");
      return false;
    }
  }, [setSettings, settingsData]);

  const finishSetup = useCallback(() => {
    void completeOnboarding();
  }, [completeOnboarding]);

  const openApps = useCallback(async (): Promise<boolean> => {
    if (!(await completeOnboarding())) return false;
    await navigate({ to: "/connect" });
    return true;
  }, [completeOnboarding, navigate]);

  const retryCompletion = useCallback(() => {
    void completeOnboarding();
  }, [completeOnboarding]);

  const authenticate = useCallback(
    async (mode: OnboardingAuthMode) => {
      setSignInError(null);

      if (isAuthenticated) {
        return;
      }

      const authAttempt = ++authAttemptRef.current;
      setIsAwaitingAuth(true);

      try {
        const result = await fetchConnectUrl();
        if (authAttempt !== authAttemptRef.current) return;
        if (!result.data) {
          throw new Error("No sign-in URL was returned");
        }

        window.open(onboardingConnectUrl(result.data, mode), "_blank");
      } catch (error) {
        if (authAttempt !== authAttemptRef.current) return;
        console.error("Failed to start sign in:", error);
        setIsAwaitingAuth(false);
        setSignInError("Unable to start sign in. Please try again.");
      }
    },
    [fetchConnectUrl, isAuthenticated],
  );

  const signIn = useCallback(() => authenticate("signin"), [authenticate]);
  const signUp = useCallback(() => authenticate("signup"), [authenticate]);

  const useLocal = useCallback(() => {
    authAttemptRef.current += 1;
    setIsAwaitingAuth(false);
    setSignInError(null);
    finishSetup();
  }, [finishSetup]);

  useEffect(() => {
    if (!isAwaitingAuth) return;

    let checking = false;
    let settled = false;
    let timeoutPending = false;
    const authAttempt = authAttemptRef.current;

    const failConnection = () => {
      if (settled || authAttempt !== authAttemptRef.current) return;
      settled = true;
      setIsAwaitingAuth(false);
      setSignInError(
        "Connection is taking longer than expected. Please try again.",
      );
    };

    const checkConnection = async () => {
      if (checking || settled || authAttempt !== authAttemptRef.current) return;
      checking = true;

      try {
        const result = await refetchUser();
        if (
          !settled &&
          authAttempt === authAttemptRef.current &&
          result.data?.name
        ) {
          settled = true;
          setIsAwaitingAuth(false);
          window.activateOllama?.();
        }
      } catch (error) {
        console.error("Failed to check sign-in status:", error);
      } finally {
        checking = false;
        if (timeoutPending) failConnection();
      }
    };

    void checkConnection();
    const pollingInterval = window.setInterval(checkConnection, 1000);
    const timeout = window.setTimeout(() => {
      const action = authenticationTimeoutAction(settled, checking);
      if (action === "ignore") return;
      if (action === "defer") {
        timeoutPending = true;
        return;
      }
      failConnection();
    }, AUTHENTICATION_TIMEOUT_MS);

    window.addEventListener("focus", checkConnection);
    return () => {
      settled = true;
      window.clearInterval(pollingInterval);
      window.clearTimeout(timeout);
      window.removeEventListener("focus", checkConnection);
    };
  }, [isAwaitingAuth, refetchUser]);

  return (
    <Onboarding
      completionError={completionError}
      isAuthenticated={isAuthenticated}
      isSigningIn={isAwaitingAuth}
      signInError={signInError}
      onOpenApps={openApps}
      onSignIn={signIn}
      onSignUp={signUp}
      onRetryCompletion={retryCompletion}
      onUseLocal={useLocal}
    />
  );
}
