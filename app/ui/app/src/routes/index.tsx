import { createFileRoute, redirect } from "@tanstack/react-router";
import { getSettings } from "@/api";
import { CURRENT_ONBOARDING_VERSION, homeChatId } from "@/lib/onboarding";

export const Route = createFileRoute("/")({
  beforeLoad: async ({ context }) => {
    const settingsData = await context.queryClient.ensureQueryData({
      queryKey: ["settings"],
      queryFn: getSettings,
    });
    if (settingsData.settings.OnboardingVersion < CURRENT_ONBOARDING_VERSION) {
      throw redirect({ to: "/onboarding" });
    }

    const chatId = homeChatId();

    throw redirect({
      to: "/c/$chatId",
      params: { chatId },
      mask: {
        to: "/",
      },
    });
  },
});
