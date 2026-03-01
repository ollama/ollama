import type { UseMutationResult } from "@tanstack/react-query";
import { Switch } from "@/components/ui/switch";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Field, Label, Description } from "@/components/ui/fieldset";
import {
  WifiIcon,
  FolderIcon,
  CogIcon,
  CloudIcon,
  ArrowDownTrayIcon,
  BugAntIcon,
  ClockIcon,
} from "@heroicons/react/20/solid";
import type { Settings } from "@/gotypes";
import type { InferenceComputeResponse } from "@/gotypes";
import type { CloudStatusResponse } from "@/api";

interface GeneralTabProps {
  settings: Settings;
  onChange: (field: keyof Settings, value: boolean | string | number) => void;
  cloudDisabled: boolean;
  cloudStatus: CloudStatusResponse | null | undefined;
  cloudOverriddenByEnv: boolean;
  cloudToggleDisabled: boolean;
  updateCloudMutation: UseMutationResult<CloudStatusResponse, Error, boolean>;
  defaultContextLength: number | undefined;
  inferenceComputeResponse: InferenceComputeResponse | undefined;
}

export default function GeneralTab({
  settings,
  onChange,
  cloudDisabled,
  cloudOverriddenByEnv,
  cloudToggleDisabled,
  updateCloudMutation,
  defaultContextLength,
}: GeneralTabProps) {
  return (
    <div className="overflow-hidden rounded-xl bg-white dark:bg-neutral-800">
      <div className="space-y-4 p-4">
        {/* Cloud */}
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
                  updateCloudMutation.mutate(checked);
                }}
              />
            </div>
          </div>
        </Field>

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
                onChange={(checked) => onChange("AutoUpdateEnabled", checked)}
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
                onChange={(checked) => onChange("Expose", checked)}
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
                  onChange={(e) => onChange("Models", e.target.value)}
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
                          onChange("Models", directory);
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
                  value={settings.ContextLength || defaultContextLength || 0}
                  onChange={(value) => {
                    onChange("ContextLength", value);
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

        {/* Debug Logging */}
        <Field>
          <div className="flex items-start justify-between gap-4">
            <div className="flex items-start space-x-3 flex-1">
              <BugAntIcon className="mt-1 h-5 w-5 flex-shrink-0 text-black dark:text-neutral-100" />
              <div>
                <Label>Debug logging</Label>
                <Description>
                  Enable verbose debug logging for troubleshooting.
                </Description>
              </div>
            </div>
            <div className="flex-shrink-0">
              <Switch
                checked={settings.DebugLogging}
                onChange={(checked) => onChange("DebugLogging", checked)}
              />
            </div>
          </div>
        </Field>

        {/* Keep-Alive Duration */}
        <Field>
          <div className="flex items-start space-x-3">
            <ClockIcon className="mt-1 h-5 w-5 flex-shrink-0 text-black dark:text-neutral-100" />
            <div className="w-full">
              <Label>Keep-alive duration</Label>
              <Description>
                How long models stay loaded in memory after last use. Use &quot;-1&quot; to keep loaded indefinitely.
              </Description>
              <div className="mt-2">
                <Input
                  value={settings.KeepAliveDuration || ""}
                  onChange={(e) => onChange("KeepAliveDuration", e.target.value)}
                  placeholder="5m"
                />
              </div>
            </div>
          </div>
        </Field>
      </div>
    </div>
  );
}
