import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Settings } from "@/gotypes";
import { getSettings, updateSettings } from "@/api";
import { useMemo, useCallback } from "react";

interface SettingsState {
  sidebarOpen: boolean;
  lastHomeView: string;
  onboardingVersion: number;
}

// Type for partial settings updates
type SettingsUpdate = Partial<{
  SidebarOpen: boolean;
  LastHomeView: string;
  OnboardingVersion: number;
}>;

export function useSettings({
  refetchInterval,
}: { refetchInterval?: number } = {}) {
  const queryClient = useQueryClient();

  // Fetch settings with useQuery
  const { data: settingsData, error } = useQuery({
    queryKey: ["settings"],
    queryFn: getSettings,
    refetchInterval,
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
      sidebarOpen: settingsData?.settings?.SidebarOpen ?? false,
      lastHomeView: settingsData?.settings?.LastHomeView ?? "chat",
      onboardingVersion: settingsData?.settings?.OnboardingVersion ?? 0,
    }),
    [settingsData?.settings],
  );

  // Single function to update most settings
  const setSettings = useCallback(
    async (updates: SettingsUpdate) => {
      if (!settingsData?.settings) return;

      const updatedSettings = new Settings({
        ...settingsData.settings,
        ...updates,
      });

      await updateSettingsMutation.mutateAsync(updatedSettings);
    },
    [settingsData?.settings, updateSettingsMutation],
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
