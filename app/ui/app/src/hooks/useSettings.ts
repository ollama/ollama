import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Settings } from "@/gotypes";
import { getSettings, updateSettings } from "@/api";
import { useMemo, useCallback } from "react";

// TODO(hoyyeva): remove turboEnabled when we remove Migration logic in useSelectedModel.ts
interface SettingsState {
  turboEnabled: boolean;
  webSearchEnabled: boolean;
  selectedModel: string;
  sidebarOpen: boolean;
  lastHomeView: string;
  thinkEnabled: boolean;
  thinkLevel: string;
}

// Type for partial settings updates
type SettingsUpdate = Partial<{
  TurboEnabled: boolean;
  WebSearchEnabled: boolean;
  ThinkEnabled: boolean;
  ThinkLevel: string;
  SelectedModel: string;
  SidebarOpen: boolean;
  LastHomeView: string;
}>;

export function useSettings() {
  const queryClient = useQueryClient();

  // Fetch settings with useQuery
  const { data: settingsData, error } = useQuery({
    queryKey: ["settings"],
    queryFn: getSettings,
  });

  // Update settings with useMutation
  const updateSettingsMutation = useMutation({
    mutationFn: updateSettings,
    onSuccess: () => {
      // Invalidate the query to ensure fresh data
      queryClient.invalidateQueries({ queryKey: ["settings"] });
    },
  });

  // Extract settings with defaults
  const settings: SettingsState = useMemo(
    () => ({
      turboEnabled: settingsData?.settings?.TurboEnabled ?? false,
      webSearchEnabled: settingsData?.settings?.WebSearchEnabled ?? false,
      thinkEnabled: settingsData?.settings?.ThinkEnabled ?? false,
      thinkLevel: settingsData?.settings?.ThinkLevel ?? "none",
      selectedModel: settingsData?.settings?.SelectedModel ?? "",
      sidebarOpen: settingsData?.settings?.SidebarOpen ?? false,
      lastHomeView: settingsData?.settings?.LastHomeView ?? "launch",
    }),
    [settingsData?.settings],
  );

  const { mutateAsync } = updateSettingsMutation;

  // Single function to update most settings.
  //
  // Only the fields that actually change are sent, and the comparison is made
  // against the cache rather than a render-time copy of the settings. Sending
  // the whole object from a copy that has gone stale reverts fields another
  // view just saved; that view then re-applies them, reverting these in turn,
  // and the two keep writing to /api/v1/settings for the life of the process.
  // Dropping writes that change nothing keeps effects that re-assert a setting
  // from issuing a request every time the settings are refetched.
  const setSettings = useCallback(
    async (updates: SettingsUpdate) => {
      const current = queryClient.getQueryData<{ settings: Settings }>([
        "settings",
      ])?.settings;
      if (!current) return;

      const changed = Object.fromEntries(
        Object.entries(updates).filter(
          ([key, value]) => value !== current[key as keyof Settings],
        ),
      ) as SettingsUpdate;

      if (Object.keys(changed).length === 0) return;

      await mutateAsync(changed);
    },
    [queryClient, mutateAsync],
  );

  return useMemo(
    () => ({
      settings,
      settingsData: settingsData?.settings,
      error,
      setSettings,
    }),
    [settings, settingsData?.settings, error, setSettings],
  );
}
