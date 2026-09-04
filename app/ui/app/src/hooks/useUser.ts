import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { fetchUser, fetchConnectUrl, disconnectUser, UIClientError } from "@/api";

export function useUser() {
  const queryClient = useQueryClient();

  const userQuery = useQuery({
    queryKey: ["user"],
    queryFn: async () => {
      const result = await fetchUser();
      return result;
    },
    staleTime: 5 * 60 * 1000, // Consider data stale after 5 minutes
    gcTime: 10 * 60 * 1000, // Keep in cache for 10 minutes
    retry: (failureCount, error) => {
      // Do not retry definitive client errors (401/403). During an ollama.com
      // auth outage, retrying these recreates the sign-in/verification loop
      // reported in ollama/ollama#17471.
      if (error instanceof UIClientError && error.status >= 400 && error.status < 500) return false;
      return failureCount < 3;
    },
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 10000),
    refetchOnMount: true, // Always fetch when component mounts
  });

  // Mutation to refresh user data
  const refreshUser = useMutation({
    mutationFn: () => fetchUser(),
    onSuccess: (data) => {
      queryClient.setQueryData(["user"], data);
    },
  });

  // Query for connect URL (only fetched when needed)
  const connectUrlQuery = useQuery({
    queryKey: ["connectUrl"],
    queryFn: fetchConnectUrl,
    enabled: false, // Don't fetch automatically
    staleTime: Infinity, // Connect URL doesn't change
  });

  const disconnectMutation = useMutation({
    mutationFn: disconnectUser,
    onSuccess: () => {
      queryClient.setQueryData(["user"], null);
    },
  });

  const isLoading = userQuery.isLoading || userQuery.isFetching;
  const isAuthenticated = Boolean(userQuery.data?.name);

  return {
    user: userQuery.data,
    isLoading,
    isError: userQuery.isError,
    error: userQuery.error,
    isAuthenticated,
    refreshUser: refreshUser.mutate,
    isRefreshing: refreshUser.isPending,
    refetchUser: userQuery.refetch,
    fetchConnectUrl: connectUrlQuery.refetch,
    connectUrl: connectUrlQuery.data,
    disconnectUser: disconnectMutation.mutate,
  };
}
