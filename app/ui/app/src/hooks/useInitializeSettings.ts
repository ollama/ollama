import { useQuery } from "@tanstack/react-query";
import { getSettings } from "@/api";

export function useInitializeSettings() {
  // This hook ensures settings are fetched on app startup
  useQuery({
    queryKey: ["settings"],
    queryFn: getSettings,
  });
}
