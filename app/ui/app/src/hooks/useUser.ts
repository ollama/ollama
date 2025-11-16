import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { fetchUser, fetchConnectUrl, disconnectUser } from "@/api";

export function useUser() {
  const queryClient = useQueryClient();
  const [initialDataLoaded, setInitialDataLoaded] = useState(false);

  // Wait for initial data to be loaded
  useEffect(() => {
    const initialPromise = window.__initialUserDataPromise;
    if (initialPromise) {
      initialPromise.finally(() => {
        setInitialDataLoaded(true);
      });
    } else {
      setInitialDataLoaded(true);
    }
  }, []);

  const userQuery = useQuery({
    queryKey: ["user"],
    queryFn: () => fetchUser(),
    staleTime: 5 * 60 * 1000, // Consider data stale after 5 minutes
    gcTime: 10 * 60 * 1000, // Keep in cache for 10 minutes
    initialData: null, // Start with null to prevent flashing
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

  return {
    user: userQuery.data,
    isLoading:
      !initialDataLoaded ||
      (userQuery.isLoading && userQuery.data === undefined), // Show loading until initial data is loaded
    isError: userQuery.isError,
    error: userQuery.error,
    isAuthenticated: Boolean(userQuery.data?.name),
    refreshUser: refreshUser.mutate,
    isRefreshing: refreshUser.isPending,
    refetchUser: userQuery.refetch,
    fetchConnectUrl: connectUrlQuery.refetch,
    connectUrl: connectUrlQuery.data,
    disconnectUser: disconnectMutation.mutate,
  };
}
