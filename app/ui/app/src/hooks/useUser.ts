import {
  useQuery,
  useMutation,
  useQueryClient,
  queryOptions,
} from "@tanstack/react-query";
import { fetchUser, fetchConnectUrl, disconnectUser } from "@/api";

export const userQueryOptions = queryOptions({
  queryKey: ["user"],
  queryFn: fetchUser,
  staleTime: 5 * 60 * 1000,
  gcTime: 10 * 60 * 1000,
  retry: 10,
  retryDelay: (attemptIndex: number) => Math.min(500 * attemptIndex, 2000),
});

export function useUser() {
  const queryClient = useQueryClient();

  const userQuery = useQuery(userQueryOptions);

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
    onMutate: async () => {
      await queryClient.cancelQueries({ queryKey: ["user"] });
      const previousUser = queryClient.getQueryData(["user"]);
      queryClient.setQueryData(["user"], null);

      return { previousUser };
    },
    onError: (_error, _variables, context) => {
      queryClient.setQueryData(["user"], context?.previousUser);
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
    disconnectUser: disconnectMutation.mutateAsync,
  };
}
