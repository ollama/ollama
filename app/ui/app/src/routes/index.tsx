import { useEffect, useRef } from "react";
import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { getSettings } from "@/api";
import { CURRENT_ONBOARDING_VERSION, homeChatId } from "@/lib/onboarding";
import { userQueryOptions } from "@/hooks/useUser";
import LoadingScreen from "@/components/LoadingScreen";

export const Route = createFileRoute("/")({
  component: RouteComponent,
});

function RouteComponent() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const navigatedRef = useRef(false);

  const { data, isFetching } = useQuery({
    queryKey: ["settings"],
    queryFn: getSettings,
    retry: false,
    staleTime: 0,
    refetchOnMount: "always",
    refetchInterval: (query) => {
      // Once we have settings, stop polling.
      return query.state.data ? false : 1500;
    },
  });

  useEffect(() => {
    if (navigatedRef.current) return;
    // If we have settings, route based on onboarding status, then home view.
    if (data?.settings && !isFetching) {
      navigatedRef.current = true;
      if (data.settings.OnboardingVersion < CURRENT_ONBOARDING_VERSION) {
        navigate({ to: "/onboarding", replace: true });
        return;
      }
      const chatId = homeChatId();
      navigate({ to: "/c/$chatId", params: { chatId }, replace: true });
      return;
    }
  }, [data, isFetching, navigate]);

  useEffect(() => {
    // Keep the deadline independent of polling and query notifications.
    const t = setTimeout(() => {
      if (!navigatedRef.current) {
        navigatedRef.current = true;
        navigate({ to: "/c/$chatId", params: { chatId: "new" }, replace: true });
      }
    }, 5000);
    return () => clearTimeout(t);
  }, [navigate]);

  // Pre-warm the user query so the chat route can render immediately.
  useEffect(() => {
    void queryClient.prefetchQuery(userQueryOptions);
  }, [queryClient]);

  return <LoadingScreen message="Opening Ollama..." />;
}
