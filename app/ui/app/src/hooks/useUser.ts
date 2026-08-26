import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { fetchUser, fetchConnectUrl, disconnectUser } from "@/api";
import {
  createContext,
  createElement,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";

export const ACCOUNT_CONNECTION_POLL_INTERVAL_MS = 1000;
export const ACCOUNT_CONNECTION_TIMEOUT_MS = 5 * 60 * 1000;
const accountConnectionTimeoutMessage =
  "Connection is taking longer than expected. Please try again.";

function useUserValue() {
  const queryClient = useQueryClient();
  const [isAwaitingConnection, setIsAwaitingConnection] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const connectionAttemptRef = useRef(0);

  const userQuery = useQuery({
    queryKey: ["user"],
    queryFn: async () => {
      const result = await fetchUser();
      return result;
    },
    staleTime: 5 * 60 * 1000, // Consider data stale after 5 minutes
    gcTime: 10 * 60 * 1000, // Keep in cache for 10 minutes
    retry: 10,
    retryDelay: (attemptIndex) => Math.min(500 * attemptIndex, 2000),
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
    onMutate: () =>
      queryClient.cancelQueries({ queryKey: ["user"], exact: true }),
    onSuccess: () => {
      queryClient.setQueryData(["user"], null);
    },
  });

  const isLoading = userQuery.isLoading;
  const isAuthenticated = Boolean(userQuery.data?.name);
  const isConnecting = connectUrlQuery.isFetching || isAwaitingConnection;
  const refetchConnectUrl = connectUrlQuery.refetch;
  const refetchUser = userQuery.refetch;

  const connectUser = useCallback(async () => {
    if (isAuthenticated || isConnecting) return;

    const connectionAttempt = ++connectionAttemptRef.current;
    setConnectionError(null);

    try {
      const { data: connectUrl } = await refetchConnectUrl();
      if (connectionAttempt !== connectionAttemptRef.current) return;
      if (!connectUrl) throw new Error("No sign-in URL was returned");

      window.open(connectUrl, "_blank");
      setIsAwaitingConnection(true);
    } catch (error) {
      if (connectionAttempt !== connectionAttemptRef.current) return;
      console.error("Failed to start sign in:", error);
      setConnectionError("Unable to start sign in. Please try again.");
    }
  }, [isAuthenticated, isConnecting, refetchConnectUrl]);

  useEffect(() => {
    if (!isAwaitingConnection) return;

    const connectionAttempt = connectionAttemptRef.current;
    let checking = false;
    let settled = false;

    const finishConnection = (error: string | null) => {
      if (settled || connectionAttempt !== connectionAttemptRef.current) {
        return false;
      }
      settled = true;
      setIsAwaitingConnection(false);
      setConnectionError(error);
      return true;
    };

    const checkConnection = async () => {
      if (
        checking ||
        settled ||
        connectionAttempt !== connectionAttemptRef.current
      ) {
        return;
      }
      checking = true;

      try {
        const result = await refetchUser();
        if (result.data?.name) finishConnection(null);
      } catch (error) {
        console.error("Failed to check sign-in status:", error);
      } finally {
        checking = false;
      }
    };

    void checkConnection();
    const pollingInterval = window.setInterval(
      checkConnection,
      ACCOUNT_CONNECTION_POLL_INTERVAL_MS,
    );
    const timeout = window.setTimeout(() => {
      if (finishConnection(accountConnectionTimeoutMessage)) {
        void queryClient.cancelQueries({ queryKey: ["user"], exact: true });
      }
    }, ACCOUNT_CONNECTION_TIMEOUT_MS);

    window.addEventListener("focus", checkConnection);
    return () => {
      settled = true;
      window.clearInterval(pollingInterval);
      window.clearTimeout(timeout);
      window.removeEventListener("focus", checkConnection);
    };
  }, [isAwaitingConnection, queryClient, refetchUser]);

  useEffect(() => {
    if (!isAuthenticated) return;
    setIsAwaitingConnection(false);
    setConnectionError(null);
  }, [isAuthenticated]);

  useEffect(
    () => () => {
      connectionAttemptRef.current += 1;
    },
    [],
  );

  return {
    user: userQuery.data,
    isLoading,
    isError: userQuery.isError,
    error: userQuery.error,
    isAuthenticated,
    connectUser,
    isConnecting,
    connectionError,
    refreshUser: refreshUser.mutate,
    isRefreshing: refreshUser.isPending,
    refetchUser,
    fetchConnectUrl: connectUrlQuery.refetch,
    connectUrl: connectUrlQuery.data,
    disconnectUser: disconnectMutation.mutate,
  };
}

type UserContextValue = ReturnType<typeof useUserValue>;

const UserContext = createContext<UserContextValue | null>(null);

export function UserProvider({ children }: { children: ReactNode }) {
  const value = useUserValue();
  return createElement(UserContext.Provider, { value }, children);
}

export function useUser(): UserContextValue {
  const value = useContext(UserContext);
  if (!value) {
    throw new Error("useUser must be used within a UserProvider");
  }
  return value;
}
