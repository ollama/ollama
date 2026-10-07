import type { QueryClient } from "@tanstack/react-query";
import {
  fetchHealth,
  getChats,
  getModelRecommendations,
  getModels,
} from "@/api";

type PrefetchClient = Pick<QueryClient, "prefetchQuery">;

export function preloadChatData(queryClient: PrefetchClient) {
  return Promise.all([
    queryClient.prefetchQuery({
      queryKey: ["chats"],
      queryFn: getChats,
    }),
    queryClient.prefetchQuery({
      queryKey: ["health"],
      queryFn: fetchHealth,
    }),
    queryClient.prefetchQuery({
      queryKey: ["models", ""],
      queryFn: () => getModels(""),
      gcTime: 10 * 60 * 1000,
    }),
    queryClient.prefetchQuery({
      queryKey: ["modelRecommendations"],
      queryFn: getModelRecommendations,
      staleTime: 5 * 60 * 1000,
      gcTime: 30 * 60 * 1000,
    }),
  ]);
}
