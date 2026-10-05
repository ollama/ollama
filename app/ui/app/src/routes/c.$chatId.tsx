import { createFileRoute, redirect } from "@tanstack/react-router";
import { History } from "@/components/History";

export const Route = createFileRoute("/c/$chatId")({
  component: RouteComponent,
  beforeLoad: ({ params }) => {
    if (params.chatId === "launch") {
      throw redirect({
        to: "/c/$chatId",
        params: { chatId: "new" },
        mask: { to: "/" },
      });
    }
  },
});

function RouteComponent() {
  const { chatId } = Route.useParams();
  const navigate = Route.useNavigate();
  return (
    <History
      chatId={chatId}
      onSelect={(id) => void navigate({ params: { chatId: id } })}
    />
  );
}
