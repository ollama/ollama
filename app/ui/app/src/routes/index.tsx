import { createFileRoute, redirect } from "@tanstack/react-router";
import { getSettings } from "@/api";

export const Route = createFileRoute("/")({
  beforeLoad: async ({ context }) => {
    const settingsData = await context.queryClient.ensureQueryData({
      queryKey: ["settings"],
      queryFn: getSettings,
    });
    const chatId =
      settingsData?.settings?.LastHomeView === "chat" ? "new" : "launch";

    throw redirect({
      to: "/c/$chatId",
      params: { chatId },
      mask: {
        to: "/",
      },
    });
  },
});
