import { useEffect, useState, useCallback } from "react";
import { Tab, TabGroup, TabList, TabPanel, TabPanels } from "@headlessui/react";
import { Text } from "@/components/ui/text";
import { Field, Label, Description } from "@/components/ui/fieldset";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  BoltIcon,
  WrenchIcon,
  XMarkIcon,
  ArrowLeftIcon,
} from "@heroicons/react/20/solid";
import { Switch } from "@/components/ui/switch";
import { Settings as SettingsType } from "@/gotypes";
import { useNavigate } from "@tanstack/react-router";
import { useUser } from "@/hooks/useUser";
import { useCloudStatus } from "@/hooks/useCloudStatus";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  getSettings,
  type CloudStatusResponse,
  updateCloudSetting,
  updateSettings,
  getInferenceCompute,
} from "@/api";
import GeneralTab from "@/components/settings/GeneralTab";
import GpuPerformanceTab from "@/components/settings/GpuPerformanceTab";
import GenerationDefaultsTab from "@/components/settings/GenerationDefaultsTab";
import NetworkSecurityTab from "@/components/settings/NetworkSecurityTab";
import clsx from "clsx";

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

const tabs = [
  { name: "General" },
  { name: "GPU & Performance" },
  { name: "Generation" },
  { name: "Network" },
];

export default function Settings() {
  const queryClient = useQueryClient();
  const [showSaved, setShowSaved] = useState(false);
  const [restartMessage, setRestartMessage] = useState(false);
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
  const navigate = useNavigate();
  const {
    cloudDisabled,
    cloudStatus,
    isLoading: cloudStatusLoading,
  } = useCloudStatus();

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
    mutationFn: updateSettings,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["settings"] });
      setShowSaved(true);
      setTimeout(() => setShowSaved(false), 1500);
    },
  });

  const updateCloudMutation = useMutation({
    mutationFn: (enabled: boolean) => updateCloudSetting(enabled),
    onMutate: async (enabled: boolean) => {
      await queryClient.cancelQueries({ queryKey: ["cloudStatus"] });

      const previous = queryClient.getQueryData<CloudStatusResponse | null>([
        "cloudStatus",
      ]);
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
    onError: (_error, _enabled, context) => {
      if (context?.previous !== undefined) {
        queryClient.setQueryData(["cloudStatus"], context.previous);
      }
    },
    onSuccess: (status) => {
      queryClient.setQueryData<CloudStatusResponse | null>(
        ["cloudStatus"],
        status,
      );
      queryClient.invalidateQueries({ queryKey: ["models"] });
      queryClient.invalidateQueries({ queryKey: ["cloudStatus"] });

      setShowSaved(true);
      setTimeout(() => setShowSaved(false), 1500);
    },
  });

  useEffect(() => {
    refetchUser();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

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
    (
      field: keyof SettingsType,
      value: boolean | string | number | null,
    ) => {
      if (settings) {
        const updatedSettings = new SettingsType({
          ...settings,
          [field]: value,
        });

        // If context length is being changed, show restart message
        if (field === "ContextLength" && value !== settings.ContextLength) {
          setRestartMessage(true);
          // Hide restart message after 3 seconds
          setTimeout(() => setRestartMessage(false), 3000);
        }

        updateSettingsMutation.mutate(updatedSettings);
      }
    },
    [settings, updateSettingsMutation],
  );

  const handleResetToDefaults = () => {
    if (settings) {
      const defaultSettings = new SettingsType({
        Expose: false,
        Browser: false,
        Models: "",
        Agent: false,
        Tools: false,
        ContextLength: 0,
        // General
        DebugLogging: false,
        KeepAliveDuration: "",
        // GPU & Performance
        FlashAttention: false,
        KvCacheType: "",
        NumParallel: 0,
        GpuOverhead: 0,
        SchedSpread: false,
        EnableVulkan: false,
        // Generation Defaults
        DefaultTemperature: null,
        DefaultTopK: null,
        DefaultTopP: null,
        DefaultMinP: null,
        DefaultRepeatPenalty: null,
        DefaultRepeatLastN: null,
        DefaultSeed: null,
        DefaultNumPredict: null,
        // Network & Security
        OllamaHost: "",
        HttpProxy: "",
        HttpsProxy: "",
        NoProxy: "",
        CorsOrigins: "",
        AllowedRemotes: "",
      });
      updateSettingsMutation.mutate(defaultSettings);
    }
  };

  const cloudOverriddenByEnv =
    cloudStatus?.source === "env" || cloudStatus?.source === "both";
  const cloudToggleDisabled =
    cloudStatusLoading || updateCloudMutation.isPending || cloudOverriddenByEnv;

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
      <div className="flex min-h-screen items-center justify-center">
        <div className="text-red-500">Failed to load settings</div>
      </div>
    );
  }

  const isWindows = navigator.platform.toLowerCase().includes("win");

  return (
    <main className="flex h-screen w-full flex-col select-none dark:bg-neutral-900">
      <header
        className="w-full flex flex-none justify-between h-[52px] py-2.5 items-center border-b border-neutral-200 dark:border-neutral-800 select-none"
        onMouseDown={() => window.drag && window.drag()}
        onDoubleClick={() => window.doubleClick && window.doubleClick()}
      >
        <h1
          className={`${isWindows ? "pl-4" : "pl-24"} flex items-center font-rounded text-md font-medium dark:text-white`}
        >
          {isWindows && (
            <button
              onClick={() => navigate({ to: "/" })}
              className="hover:bg-neutral-100 mr-3 dark:hover:bg-neutral-800 rounded-full p-1.5"
            >
              <ArrowLeftIcon className="w-5 h-5 dark:text-white" />
            </button>
          )}
          Settings
        </h1>
        {!isWindows && (
          <button
            onClick={() => navigate({ to: "/" })}
            className="p-1 hover:bg-neutral-100 mr-3 dark:hover:bg-neutral-800 rounded-full"
          >
            <XMarkIcon className="w-6 h-6 dark:text-white" />
          </button>
        )}
      </header>
      <div className="w-full p-6 overflow-y-auto flex-1 overscroll-contain">
        <div className="space-y-4 max-w-2xl mx-auto">
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

          {/* Tabbed Settings */}
          <TabGroup>
            <TabList className="flex space-x-1 rounded-xl bg-neutral-100 dark:bg-neutral-800 p-1">
              {tabs.map((tab) => (
                <Tab
                  key={tab.name}
                  className={({ selected }) =>
                    clsx(
                      "w-full rounded-lg py-2 text-sm font-medium leading-5 transition-colors",
                      "focus:outline-hidden",
                      selected
                        ? "bg-white dark:bg-neutral-700 text-neutral-900 dark:text-white shadow-sm"
                        : "text-neutral-500 dark:text-neutral-400 hover:text-neutral-700 dark:hover:text-neutral-300 hover:bg-white/50 dark:hover:bg-neutral-700/50",
                    )
                  }
                >
                  {tab.name}
                </Tab>
              ))}
            </TabList>
            <TabPanels className="mt-3">
              <TabPanel>
                <GeneralTab
                  settings={settings}
                  onChange={handleChange}
                  cloudDisabled={cloudDisabled}
                  cloudStatus={cloudStatus}
                  cloudOverriddenByEnv={cloudOverriddenByEnv}
                  cloudToggleDisabled={cloudToggleDisabled}
                  updateCloudMutation={updateCloudMutation}
                  defaultContextLength={defaultContextLength}
                  inferenceComputeResponse={inferenceComputeResponse}
                />
              </TabPanel>
              <TabPanel>
                <GpuPerformanceTab
                  settings={settings}
                  onChange={handleChange}
                />
              </TabPanel>
              <TabPanel>
                <GenerationDefaultsTab
                  settings={settings}
                  onChange={handleChange}
                />
              </TabPanel>
              <TabPanel>
                <NetworkSecurityTab
                  settings={settings}
                  onChange={handleChange}
                />
              </TabPanel>
            </TabPanels>
          </TabGroup>

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
          <div className="mt-6 flex justify-end px-4">
            <Button
              type="button"
              color="white"
              className="px-3"
              onClick={handleResetToDefaults}
            >
              Reset to defaults
            </Button>
          </div>
        </div>

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
