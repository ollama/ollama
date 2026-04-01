import { createFileRoute } from "@tanstack/react-router";
import Settings from "@/components/Settings";

export const Route = createFileRoute("/settings")({
  component: Settings,
});
