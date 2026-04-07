import { createFileRoute, redirect } from "@tanstack/react-router";
import { getSettings } from "@/api";

export const Route = createFileRoute("/")({
  beforeLoad: async ({ context }) => {
    const settingsData = await context.queryClient.ensureQueryData({
      queryKey: ["settings"],
      queryFn: getSettings,
    });
    const lastHomeView = settingsData?.settings?.LastHomeView ?? "chat";
    const chatId = lastHomeView === "chat" ? "new" : "launch";

    throw redirect({
      to: "/c/$chatId",
      params: { chatId },
      mask: {
        to: "/",
      },
    });
  },
});
