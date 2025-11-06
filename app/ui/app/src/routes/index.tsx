import { createFileRoute, redirect } from "@tanstack/react-router";

export const Route = createFileRoute("/")({
  beforeLoad: () => {
    throw redirect({
      to: "/c/$chatId",
      params: { chatId: "new" },
      mask: {
        to: "/",
      },
    });
  },
});
