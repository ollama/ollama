import { useQuery } from "@tanstack/react-query";
import { getCloudStatus, type CloudStatusResponse } from "@/api";

export function useCloudStatus() {
  const cloudQuery = useQuery<CloudStatusResponse | null>({
    queryKey: ["cloudStatus"],
    queryFn: getCloudStatus,
    retry: false,
    staleTime: 60 * 1000,
  });

  return {
    cloudStatus: cloudQuery.data,
    cloudDisabled: cloudQuery.data?.disabled ?? false,
    isKnown: cloudQuery.data !== null && cloudQuery.data !== undefined,
    isLoading: cloudQuery.isLoading,
    isError: cloudQuery.isError,
    error: cloudQuery.error,
  };
}
