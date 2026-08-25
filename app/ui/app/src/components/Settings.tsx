import { useEffect, useState, useCallback, useRef } from "react";
import { Switch } from "@/components/ui/switch";
import { Text } from "@/components/ui/text";
import { Input } from "@/components/ui/input";
import { Field, Label, Description } from "@/components/ui/fieldset";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import {
  ClaudeDesktopModelsSettings,
  type ClaudeDesktopModelsSettingsHandle,
} from "@/components/ClaudeDesktopModelsSettings";
import {
  WifiIcon,
  FolderIcon,
  BoltIcon,
  WrenchIcon,
  CloudIcon,
  CogIcon,
  ArrowDownTrayIcon,
  ArrowPathIcon,
  Squares2X2Icon,
} from "@heroicons/react/20/solid";
import { Settings as SettingsType } from "@/gotypes";
import { isWindowsPlatform } from "@/lib/platform";
import { settingsMutationScope } from "@/lib/settingsMutationScope";
import { useUser } from "@/hooks/useUser";
import { useCloudStatus } from "@/hooks/useCloudStatus";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useBlocker } from "@tanstack/react-router";
import {
  getSettings,
  type CloudStatusSource,
  type CloudStatusResponse,
  updateCloudSetting,
  updateSettings,
  getInferenceCompute,
} from "@/api";

function AnimatedDots() {
  return (
    <span className="inline-flex">
      <span className="animate-pulse">.</span>
      <span className="animate-pulse" style={{ animationDelay: "0.2s" }}>
        .
      </span>
      <span className="animate-pulse" style={{ animationDelay: "0.4s" }}>
        .
      </span>
    </span>
  );
}

interface SettingsDefaultsActions {
  updateSettings: (settings: SettingsType) => Promise<unknown>;
  updateCloud: (enabled: boolean) => Promise<unknown>;
  updateShowAppsInMenu: (visible: boolean) => Promise<unknown>;
  resetClaudeMappings: () => Promise<boolean>;
  currentSettings: SettingsType;
  currentShowAppsInMenu: boolean;
  cloudSource: CloudStatusSource;
  onSaved: () => void;
}

interface CloudUpdateRequest {
  enabled: boolean;
  requestId: number;
}

let latestCloudRequestId = 0;
const savedConfirmationDuration = 3000;

export async function applySettingsDefaults({
  updateSettings,
  updateCloud,
  updateShowAppsInMenu,
  resetClaudeMappings,
  currentSettings,
  currentShowAppsInMenu,
  cloudSource,
  onSaved,
}: SettingsDefaultsActions): Promise<void> {
  const cloudNeedsReset = cloudSource === "config" || cloudSource === "both";
  const rollbacks: Array<() => Promise<unknown>> = [];

  try {
    if (cloudNeedsReset) {
      await updateCloud(true);
      rollbacks.push(() => updateCloud(false));
    }

    await updateSettings(
      new SettingsType({
        Expose: false,
        Browser: false,
        Models: "",
        Agent: false,
        Tools: false,
        ContextLength: currentSettings.ContextLength,
        AutoUpdateEnabled: true,
      }),
    );
    rollbacks.push(() => updateSettings(currentSettings));

    await updateShowAppsInMenu(true);
    rollbacks.push(() => updateShowAppsInMenu(currentShowAppsInMenu));

    // Apply Claude last so no later settings failure can leave its mappings
    // reset while the rest of the page rolls back.
    if (!(await resetClaudeMappings())) {
      throw new Error("Claude model mappings could not be reset");
    }
  } catch (error) {
    const rollbackErrors: unknown[] = [];
    for (const rollback of rollbacks.reverse()) {
      try {
        await rollback();
      } catch (rollbackError) {
        rollbackErrors.push(rollbackError);
      }
    }
    if (rollbackErrors.length > 0) {
      console.error("Failed to roll back settings reset:", rollbackErrors);
    }
    throw error;
  }

  onSaved();
}

export default function Settings() {
  const queryClient = useQueryClient();
  const [showSaved, setShowSaved] = useState(false);
  const [restartMessage, setRestartMessage] = useState(false);
  const [showAppsInMenu, setShowAppsInMenuState] = useState(true);
  const [showAppsInMenuPending, setShowAppsInMenuPending] = useState(false);
  const [resettingToDefaults, setResettingToDefaults] = useState(false);
  const [resetError, setResetError] = useState<string | null>(null);
  const [hasClaudeDraftChanges, setHasClaudeDraftChanges] = useState(false);
  const claudeModelsSettingsRef =
    useRef<ClaudeDesktopModelsSettingsHandle>(null);
  const savedConfirmationTimeoutRef = useRef<number | null>(null);
  useBlocker({
    shouldBlockFn: () =>
      !window.confirm("Discard unapplied Claude routing changes?"),
    enableBeforeUnload: hasClaudeDraftChanges,
    disabled: !hasClaudeDraftChanges,
  });
  const {
    user,
    isAuthenticated,
    refreshUser,
    isRefreshing,
    refetchUser,
    fetchConnectUrl,
    isLoading,
    disconnectUser,
  } = useUser();
  const [isAwaitingConnection, setIsAwaitingConnection] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [pollingInterval, setPollingInterval] = useState<number | null>(null);
  const {
    cloudDisabled,
    cloudStatus,
    isKnown: cloudStatusKnown,
  } = useCloudStatus();

  const showSavedConfirmation = useCallback(() => {
    if (savedConfirmationTimeoutRef.current !== null) {
      window.clearTimeout(savedConfirmationTimeoutRef.current);
    }
    setShowSaved(true);
    savedConfirmationTimeoutRef.current = window.setTimeout(() => {
      setShowSaved(false);
      savedConfirmationTimeoutRef.current = null;
    }, savedConfirmationDuration);
  }, []);

  useEffect(
    () => () => {
      if (savedConfirmationTimeoutRef.current !== null) {
        window.clearTimeout(savedConfirmationTimeoutRef.current);
      }
    },
    [],
  );

  const {
    data: settingsData,
    isLoading: loading,
    error,
  } = useQuery({
    queryKey: ["settings"],
    queryFn: getSettings,
  });

  const settings = settingsData?.settings || null;

  const { data: inferenceComputeResponse } = useQuery({
    queryKey: ["inferenceCompute"],
    queryFn: getInferenceCompute,
  });

  const defaultContextLength = inferenceComputeResponse?.defaultContextLength;

  const updateSettingsMutation = useMutation({
    scope: settingsMutationScope,
    mutationFn: updateSettings,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["settings"] });
    },
  });

  const updateCloudMutation = useMutation({
    scope: settingsMutationScope,
    mutationFn: ({ enabled }: CloudUpdateRequest) =>
      updateCloudSetting(enabled),
    onMutate: async ({ enabled, requestId }: CloudUpdateRequest) => {
      await queryClient.cancelQueries({ queryKey: ["cloudStatus"] });

      const previous = queryClient.getQueryData<CloudStatusResponse | null>([
        "cloudStatus",
      ]);
      if (requestId !== latestCloudRequestId) return { previous };
      const envForcesDisabled =
        previous?.source === "env" || previous?.source === "both";

      queryClient.setQueryData<CloudStatusResponse | null>(
        ["cloudStatus"],
        previous
          ? {
              ...previous,
              disabled: !enabled || envForcesDisabled,
            }
          : {
              disabled: !enabled,
              source: "config",
            },
      );

      return { previous };
    },
    onError: (_error, request, context) => {
      if (request.requestId !== latestCloudRequestId) return;
      if (context?.previous !== undefined) {
        queryClient.setQueryData(["cloudStatus"], context.previous);
      }
    },
    onSuccess: (status, request) => {
      if (request.requestId !== latestCloudRequestId) return;
      queryClient.setQueryData<CloudStatusResponse | null>(
        ["cloudStatus"],
        status,
      );
    },
    onSettled: (_status, _error, request) => {
      if (request.requestId !== latestCloudRequestId) return;
      queryClient.invalidateQueries({ queryKey: ["models"] });
      queryClient.invalidateQueries({ queryKey: ["cloudStatus"] });
    },
  });

  const requestCloudUpdate = (enabled: boolean) => {
    const requestId = ++latestCloudRequestId;
    return updateCloudMutation.mutateAsync({ enabled, requestId });
  };

  useEffect(() => {
    refetchUser();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    window
      .getShowAppsInMenu?.()
      .then(setShowAppsInMenuState)
      .catch((error) =>
        console.error("Failed to load menu app visibility:", error),
      );
  }, []);

  useEffect(() => {
    const handleFocus = () => {
      if (isAwaitingConnection && pollingInterval) {
        // Stop polling when window gets focus
        clearInterval(pollingInterval);
        setPollingInterval(null);
        // Reset awaiting connection state
        setIsAwaitingConnection(false);
        // Make one last refresh request
        refreshUser();
      }
    };

    window.addEventListener("focus", handleFocus);

    return () => {
      window.removeEventListener("focus", handleFocus);
    };
  }, [isAwaitingConnection, refreshUser, pollingInterval]);

  // Check if user is authenticated after refresh
  useEffect(() => {
    if (isAwaitingConnection && isAuthenticated) {
      setIsAwaitingConnection(false);
      setConnectionError(null);
      if (pollingInterval) {
        clearInterval(pollingInterval);
        setPollingInterval(null);
      }
    }
  }, [isAuthenticated, isAwaitingConnection, pollingInterval]);

  // Cleanup interval on unmount
  useEffect(() => {
    return () => {
      if (pollingInterval) {
        clearInterval(pollingInterval);
      }
    };
  }, [pollingInterval]);

  const handleChange = useCallback(
    (field: keyof SettingsType, value: boolean | string | number) => {
      if (settings) {
        const updatedSettings = new SettingsType({
          ...settings,
          [field]: value,
        });

        // If context length is being changed, show restart message
        if (field === "ContextLength" && value !== settings.ContextLength) {
          setRestartMessage(true);
          // Hide restart message after 3 seconds
          window.setTimeout(
            () => setRestartMessage(false),
            savedConfirmationDuration,
          );
        }

        updateSettingsMutation.mutate(updatedSettings, {
          onSuccess: showSavedConfirmation,
        });
      }
    },
    [settings, showSavedConfirmation, updateSettingsMutation],
  );

  const updateShowAppsInMenuVisibility = async (checked: boolean) => {
    const previous = showAppsInMenu;
    setShowAppsInMenuState(checked);
    setShowAppsInMenuPending(true);
    try {
      await window.setShowAppsInMenu?.(checked);
    } catch (error) {
      setShowAppsInMenuState(previous);
      throw error;
    } finally {
      setShowAppsInMenuPending(false);
    }
  };

  const handleShowAppsInMenu = (checked: boolean) => {
    void updateShowAppsInMenuVisibility(checked)
      .then(showSavedConfirmation)
      .catch((error) =>
        console.error("Failed to update menu app visibility:", error),
      );
  };

  const handleCloudUpdate = (enabled: boolean) => {
    void requestCloudUpdate(enabled)
      .then(showSavedConfirmation)
      .catch((error) =>
        console.error("Failed to update cloud setting:", error),
      );
  };

  const cloudOverriddenByEnv =
    cloudStatus?.source === "env" || cloudStatus?.source === "both";
  const cloudToggleDisabled = cloudOverriddenByEnv;

  const handleResetToDefaults = async () => {
    const cloudSource = cloudStatus?.source;
    if (!settings || resettingToDefaults || !cloudSource) return;

    setResettingToDefaults(true);
    if (savedConfirmationTimeoutRef.current !== null) {
      window.clearTimeout(savedConfirmationTimeoutRef.current);
      savedConfirmationTimeoutRef.current = null;
    }
    setShowSaved(false);
    setRestartMessage(false);
    setResetError(null);
    try {
      await applySettingsDefaults({
        updateSettings: (defaultSettings) =>
          updateSettingsMutation.mutateAsync(defaultSettings),
        updateCloud: requestCloudUpdate,
        updateShowAppsInMenu: updateShowAppsInMenuVisibility,
        resetClaudeMappings: async () =>
          (await claudeModelsSettingsRef.current?.resetToDefaults()) ?? true,
        currentSettings: settings,
        currentShowAppsInMenu: showAppsInMenu,
        cloudSource,
        onSaved: showSavedConfirmation,
      });
    } catch (error) {
      console.error("Failed to reset settings:", error);
      setResetError(
        "Ollama could not reset every setting. Check the settings above and try again.",
      );
    } finally {
      setResettingToDefaults(false);
    }
  };

  const handleConnectOllamaAccount = async () => {
    setConnectionError(null);

    // If user is already authenticated, no need to connect
    if (isAuthenticated) {
      return;
    }

    try {
      // If we don't have a user or user has no name, get connect URL
      if (!user || !user?.name) {
        const { data: connectUrl } = await fetchConnectUrl();
        if (connectUrl) {
          window.open(connectUrl, "_blank");
          setIsAwaitingConnection(true);
          // Start polling every 5 seconds
          const interval = setInterval(() => {
            refreshUser();
          }, 5000);
          setPollingInterval(interval);
        } else {
          setConnectionError("Failed to get connect URL");
        }
      }
    } catch (error) {
      console.error("Error connecting to Ollama account:", error);
      setConnectionError(
        error instanceof Error
          ? error.message
          : "Failed to connect to Ollama account",
      );
      setIsAwaitingConnection(false);
    }
  };

  if (loading) {
    return null;
  }

  if (error || !settings) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <div className="text-red-500">Failed to load settings</div>
      </div>
    );
  }

  const isWindows = isWindowsPlatform();

  return (
    <main className="flex min-h-0 w-full flex-1 flex-col select-none dark:bg-neutral-900">
      <div className="w-full p-6 overflow-y-auto flex-1 overscroll-contain">
        <fieldset
          disabled={resettingToDefaults}
          aria-busy={resettingToDefaults}
          className="mx-auto max-w-4xl space-y-4 border-0 p-0"
        >
          {/* Connect Ollama Account */}
          <div className="overflow-hidden rounded-xl bg-white dark:bg-neutral-800">
            <div className="p-4">
              <Field>
                {isLoading ? (
                  // Loading skeleton, this will only happen if the app started recently
                  <div className="flex items-center justify-between">
                    <div className="space-y-2">
                      <div className="h-4 bg-neutral-200 dark:bg-neutral-700 rounded animate-pulse w-24"></div>
                      <div className="h-3 bg-neutral-200 dark:bg-neutral-700 rounded animate-pulse w-32"></div>
                    </div>
                    <div className="h-10 w-10 bg-neutral-200 dark:bg-neutral-700 rounded-full animate-pulse"></div>
                  </div>
                ) : user && user.name ? (
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="flex items-center space-x-2">
                        <Label className="text-sm font-medium text-neutral-900 dark:text-white">
                          {user?.name}
                        </Label>
                      </div>
                      <Description className="text-sm text-neutral-500 dark:text-neutral-400">
                        {user?.email}
                      </Description>
                      <div className="flex items-center space-x-2 mt-2">
                        {user?.plan === "free" && (
                          <Button
                            type="button"
                            color="dark"
                            className="px-3 py-2 text-sm font-medium bg-black/90 backdrop-blur-sm text-white rounded-lg border border-white/10 shadow-2xl transition-all duration-300 ease-out relative overflow-hidden group"
                            onClick={() =>
                              window.open(
                                "https://ollama.com/upgrade",
                                "_blank",
                              )
                            }
                          >
                            <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/20 via-purple-500/20 to-green-500/20 opacity-60 group-hover:opacity-80 transition-opacity duration-300"></div>
                            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000 ease-out"></div>
                            <span className="relative z-10 flex items-center space-x-2">
                              <span>Upgrade</span>
                            </span>
                          </Button>
                        )}
                        <Button
                          type="button"
                          color="white"
                          className="px-3 py-2 text-sm"
                          onClick={() =>
                            window.open("https://ollama.com/settings", "_blank")
                          }
                        >
                          Manage
                        </Button>
                        <Button
                          type="button"
                          color="zinc"
                          className="px-3 py-2 text-sm"
                          onClick={() => disconnectUser()}
                        >
                          Sign out
                        </Button>
                      </div>
                    </div>
                    {user?.avatarurl && (
                      <img
                        src={user.avatarurl}
                        alt={user?.name}
                        className="h-10 w-10 rounded-full bg-neutral-200 dark:bg-neutral-700 flex-shrink-0"
                        onError={(e) => {
                          const target = e.target as HTMLImageElement;
                          target.className = "hidden";
                        }}
                      />
                    )}
                  </div>
                ) : (
                  <div className="flex items-center justify-between">
                    <div>
                      <Label>Ollama account</Label>
                      <Description>Not connected</Description>
                    </div>
                    <Button
                      type="button"
                      color="white"
                      onClick={handleConnectOllamaAccount}
                      disabled={isRefreshing || isAwaitingConnection}
                    >
                      {isRefreshing || isAwaitingConnection ? (
                        <AnimatedDots />
                      ) : (
                        "Sign In"
                      )}
                    </Button>
                  </div>
                )}
              </Field>
              {connectionError && (
                <div className="mt-3 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                  <Text className="text-sm text-red-600 dark:text-red-400">
                    {connectionError}
                  </Text>
                </div>
              )}
            </div>
          </div>
          {/* Local Configuration */}
          <div className="relative overflow-hidden rounded-xl bg-white dark:bg-neutral-800">
            <div className="space-y-4 p-4">
              <Field>
                <div className="flex items-start justify-between gap-4">
                  <div className="flex items-start space-x-3 flex-1">
                    <CloudIcon className="mt-1 h-5 w-5 flex-shrink-0 text-black dark:text-neutral-100" />
                    <div>
                      <Label>Cloud</Label>
                      <Description>
                        {cloudOverriddenByEnv
                          ? "The OLLAMA_NO_CLOUD environment variable is currently forcing cloud off."
                          : "Enable cloud models and web search."}
                      </Description>
                    </div>
                  </div>
                  <div className="flex-shrink-0">
                    <Switch
                      checked={!cloudDisabled}
                      disabled={cloudToggleDisabled}
                      onChange={(checked) => {
                        if (cloudOverriddenByEnv) {
                          return;
                        }
                        handleCloudUpdate(checked);
                      }}
                    />
                  </div>
                </div>
              </Field>

              {!isWindows && (
                <Field>
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex flex-1 items-start space-x-3">
                      <Squares2X2Icon className="mt-1 h-5 w-5 flex-shrink-0 text-black dark:text-neutral-100" />
                      <div>
                        <Label>Show apps in menu</Label>
                        <Description>
                          Show connected apps at the top of the Ollama menu.
                        </Description>
                      </div>
                    </div>
                    <div className="flex-shrink-0">
                      <Switch
                        checked={showAppsInMenu}
                        disabled={showAppsInMenuPending}
                        onChange={handleShowAppsInMenu}
                      />
                    </div>
                  </div>
                </Field>
              )}

              {/* Auto Update */}
              <Field>
                <div className="flex items-start justify-between gap-4">
                  <div className="flex items-start space-x-3 flex-1">
                    <ArrowDownTrayIcon className="mt-1 h-5 w-5 flex-shrink-0 text-black dark:text-neutral-100" />
                    <div>
                      <Label>Auto-download updates</Label>
                      <Description>
                        {settings.AutoUpdateEnabled
                          ? "Automatically download updates when available."
                          : "Updates will not be downloaded automatically."}
                      </Description>
                    </div>
                  </div>
                  <div className="flex-shrink-0">
                    <Switch
                      checked={settings.AutoUpdateEnabled}
                      onChange={(checked) =>
                        handleChange("AutoUpdateEnabled", checked)
                      }
                    />
                  </div>
                </div>
              </Field>

              {/* Expose Ollama */}
              <Field>
                <div className="flex items-start justify-between gap-4">
                  <div className="flex items-start space-x-3 flex-1">
                    <WifiIcon className="mt-1 h-5 w-5 flex-shrink-0 text-black dark:text-neutral-100" />
                    <div>
                      <Label>Expose Ollama to the network</Label>
                      <Description>
                        Allow other devices or services to access Ollama.
                      </Description>
                    </div>
                  </div>
                  <div className="flex-shrink-0">
                    <Switch
                      checked={settings.Expose}
                      onChange={(checked) => handleChange("Expose", checked)}
                    />
                  </div>
                </div>
              </Field>

              {/* Model Directory */}
              <Field>
                <div className="flex items-start space-x-3">
                  <FolderIcon className="mt-1 h-5 w-5 flex-shrink-0 text-black dark:text-neutral-100" />
                  <div className="w-full">
                    <Label>Model location</Label>
                    <Description>Location where models are stored.</Description>
                    <div className="mt-2 flex items-center space-x-2">
                      <Input
                        value={settings.Models || ""}
                        onChange={(e) => handleChange("Models", e.target.value)}
                        readOnly
                      />
                      <Button
                        type="button"
                        color="white"
                        className="px-2"
                        onClick={async () => {
                          if (window.webview?.selectModelsDirectory) {
                            try {
                              const directory =
                                await window.webview.selectModelsDirectory();
                              if (directory) {
                                handleChange("Models", directory);
                              }
                            } catch (error) {
                              console.error(
                                "Error selecting models directory:",
                                error,
                              );
                            }
                          }
                        }}
                      >
                        <FolderIcon className="w-4 h-4 mr-1" />
                        Browse
                      </Button>
                    </div>
                  </div>
                </div>
              </Field>

              {/* Context Length */}
              <Field>
                <div className="flex items-start space-x-3">
                  <CogIcon className="mt-1 h-5 w-5 flex-shrink-0 text-black dark:text-neutral-100" />
                  <div className="w-full">
                    <Label>Context length</Label>
                    <Description>
                      Context length determines how much of your conversation
                      local LLMs can remember and use to generate responses.
                    </Description>
                    <div className="mt-3">
                      <Slider
                        value={
                          settings.ContextLength || defaultContextLength || 0
                        }
                        onChange={(value) => {
                          handleChange("ContextLength", value);
                        }}
                        disabled={!defaultContextLength}
                        options={[
                          { value: 4096, label: "4k" },
                          { value: 8192, label: "8k" },
                          { value: 16384, label: "16k" },
                          { value: 32768, label: "32k" },
                          { value: 65536, label: "64k" },
                          { value: 131072, label: "128k" },
                          { value: 262144, label: "256k" },
                        ]}
                      />
                    </div>
                  </div>
                </div>
              </Field>
            </div>
          </div>

          {!isWindows && (
            <ClaudeDesktopModelsSettings
              ref={claudeModelsSettingsRef}
              includeCloudModels={
                isAuthenticated && cloudStatusKnown && !cloudDisabled
              }
              onDraftChange={setHasClaudeDraftChanges}
            />
          )}

          {/* Agent Mode */}
          {window.OLLAMA_TOOLS && (
            <div className="overflow-hidden rounded-xl bg-white dark:bg-neutral-800">
              <div className="space-y-4 p-4">
                <Field>
                  <div className="flex items-center justify-between">
                    <div className="flex items-start space-x-3">
                      <BoltIcon className="mt-1 h-5 w-5 flex-shrink-0 text-black dark:text-neutral-100" />
                      <div>
                        <Label>Enable Agent Mode</Label>
                        <Description>
                          Use multi-turn tools to fulfill user requests
                        </Description>
                      </div>
                    </div>
                    <Switch
                      checked={settings.Agent}
                      onChange={(checked) => handleChange("Agent", checked)}
                    />
                  </div>
                </Field>

                {/* Tools Mode */}
                <Field>
                  <div className="flex items-center justify-between">
                    <div className="flex items-start space-x-3">
                      <WrenchIcon className="mt-1 h-5 w-5 flex-shrink-0 text-black dark:text-neutral-100" />
                      <div>
                        <Label>Enable Tools Mode</Label>
                        <Description>
                          Use single-turn tools to fulfill user requests
                        </Description>
                      </div>
                    </div>
                    <Switch
                      checked={settings.Tools}
                      onChange={(checked) => handleChange("Tools", checked)}
                    />
                  </div>
                </Field>
              </div>
            </div>
          )}

          {/* Reset button */}
          <div className="flex items-center justify-between gap-4 px-4">
            {resetError ? (
              <p
                role="alert"
                className="text-xs text-red-600 dark:text-red-400"
              >
                {resetError}
              </p>
            ) : (
              <span />
            )}
            <Button
              type="button"
              color="white"
              className="px-3"
              disabled={resettingToDefaults || !cloudStatusKnown}
              onClick={() => void handleResetToDefaults()}
            >
              {resettingToDefaults && (
                <ArrowPathIcon data-slot="icon" className="animate-spin" />
              )}
              {resettingToDefaults ? "Resetting…" : "Reset to defaults"}
            </Button>
          </div>
        </fieldset>

        {/* Saved indicator */}
        {(showSaved || restartMessage) && (
          <div className="fixed bottom-4 left-1/2 transform -translate-x-1/2 transition-opacity duration-300 z-50">
            <Badge
              color="green"
              className="!bg-green-500 !text-white dark:!bg-green-600"
            >
              Saved
            </Badge>
          </div>
        )}
      </div>
    </main>
  );
}
