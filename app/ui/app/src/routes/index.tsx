import { createFileRoute, redirect } from "@tanstack/react-router";
import { getSettings } from "@/api";
import { resolveHomeChatId } from "@/lib/homeView";

export const Route = createFileRoute("/")({
  beforeLoad: async ({ context }) => {
    const settingsData = await context.queryClient.ensureQueryData({
      queryKey: ["settings"],
      queryFn: getSettings,
    });
    const chatId = resolveHomeChatId(settingsData?.settings?.LastHomeView);

    throw redirect({
      to: "/c/$chatId",
      params: { chatId },
      mask: {
        to: "/",
      },
    });
  },
});
