import { useQuery } from "@tanstack/react-query";
import { fetchHealth } from "@/api";

export function useHealth() {
  const healthQuery = useQuery({
    queryKey: ["health"],
    queryFn: fetchHealth,
    refetchInterval: (query) => {
      // If the server is not healthy, poll every 10ms
      // Once healthy, stop polling
      return query.state.data === false ? 10 : false;
    },
    refetchIntervalInBackground: true,
    retry: false, // Don't retry, just return false
    staleTime: 0, // Always consider stale so we keep polling
  });

  return {
    isHealthy: healthQuery.data ?? false,
    isChecking: healthQuery.isLoading,
  };
}
