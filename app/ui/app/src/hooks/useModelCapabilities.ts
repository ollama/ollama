import { useQuery } from "@tanstack/react-query";
import { getModelCapabilities } from "@/api";
import { ModelCapabilitiesResponse } from "@/gotypes";

export function useModelCapabilities(modelName: string | undefined) {
  return useQuery<ModelCapabilitiesResponse, Error>({
    queryKey: ["modelCapabilities", modelName],
    queryFn: () => {
      return getModelCapabilities(modelName!);
    },
    enabled: !!modelName, // Only run query if modelName is provided
    gcTime: 60 * 60 * 1000, // Keep in cache for 1 hour
    staleTime: 60 * 60 * 1000, // Consider data stale after 1 hour
    retry: 3,
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 5000),
  });
}

export function useHasVisionCapability(modelName: string | undefined) {
  const { data: capabilitiesResponse } = useModelCapabilities(modelName);
  return capabilitiesResponse?.capabilities?.includes("vision") ?? false;
}

export function useHasToolsCapability(modelName: string | undefined) {
  const { data: capabilitiesResponse } = useModelCapabilities(modelName);
  return capabilitiesResponse?.capabilities?.includes("tools") ?? false;
}

export interface ModelThinking {
  values: (string | boolean)[];
  default: string | boolean;
  stringLevels: string[];
  hasLevels: boolean;
  hasOff: boolean;
  isToggleOnly: boolean;
}

export function useModelThinking(
  modelName: string | undefined,
): ModelThinking | undefined {
  const { data: capabilitiesResponse } = useModelCapabilities(modelName);
  const thinking = capabilitiesResponse?.thinking;
  const hasCapability =
    capabilitiesResponse?.capabilities?.includes("thinking") ?? false;

  if (!hasCapability || !thinking || !thinking.values?.length) {
    return undefined;
  }

  const stringLevels = thinking.values.filter(
    (v: any) => typeof v === "string",
  ) as string[];
  const hasLevels = stringLevels.length > 0;
  const hasOff = thinking.values.includes(false);

  return {
    values: thinking.values as (string | boolean)[],
    default: thinking.default as string | boolean,
    stringLevels,
    hasLevels,
    hasOff,
    isToggleOnly: !hasLevels,
  };
}
