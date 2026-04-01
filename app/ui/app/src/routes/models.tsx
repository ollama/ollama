import { createFileRoute } from "@tanstack/react-router";
import ModelManager from "@/components/ModelManager";

export const Route = createFileRoute("/models")({
  component: ModelManager,
});
