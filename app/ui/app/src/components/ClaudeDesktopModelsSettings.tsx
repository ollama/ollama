import { getClaudeDesktopAvailableModels } from "@/api";
import { Button } from "@/components/ui/button";
import { Description, Field, Label } from "@/components/ui/fieldset";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import {
  addClaudeModelSelection,
  claudeDesktopRecoveryMessage,
  claudeDesktopMaxModels,
  claudeDesktopMaxModelsMessage,
  claudeDesktopUsableSelection,
} from "@/lib/claudeDesktop";
import type {
  ClaudeDesktopModelStatus,
  ClaudeDesktopStatus,
} from "@/types/webview";
import { ArrowPathIcon } from "@heroicons/react/20/solid";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

interface ClaudeDesktopModelsSettingsProps {
  initialStatus?: ClaudeDesktopStatus;
  initialLocalModels?: string[];
}

function isInvalidModelName(name: string): boolean {
  const normalized = name.trim().toLowerCase().replace(/[-:]+/g, " ");
  return normalized === "ollama cloud";
}

function visibleModels(
  status: ClaudeDesktopStatus,
): ClaudeDesktopModelStatus[] {
  return (status.models ?? []).filter(
    (model) => !isInvalidModelName(model.name) && model.reason !== "cloud_off",
  );
}

function selectedModelNames(status: ClaudeDesktopStatus): string[] {
  return claudeDesktopUsableSelection(
    visibleModels(status),
    status.modelSource !== "user",
    claudeDesktopMaxModels(status),
  );
}

function modelAccessLabel(model: ClaudeDesktopModelStatus): string | null {
  switch (model.reason) {
    case "sign_in_required":
      return "Sign in required";
    case "upgrade_required":
      return model.requiredPlan
        ? `${model.requiredPlan} plan required`
        : "Upgrade required";
    case "verification_unavailable":
      return "Access unavailable";
    case "model_not_installed":
      return "Not installed";
    default:
      return null;
  }
}

function explicitCloudName(name: string): string {
  return name.endsWith(":cloud") ? name : `${name}:cloud`;
}

function formatModelList(names: string[]): string {
  if (names.length < 2) return names[0] ?? "";
  if (names.length === 2) return `${names[0]} or ${names[1]}`;
  return `${names.slice(0, -1).join(", ")}, or ${names[names.length - 1]}`;
}

export function ClaudeDesktopModelsSettings({
  initialStatus,
  initialLocalModels,
}: ClaudeDesktopModelsSettingsProps) {
  const [status, setStatus] = useState<ClaudeDesktopStatus | null>(
    initialStatus ?? null,
  );
  const [models, setModels] = useState<ClaudeDesktopModelStatus[]>(() =>
    initialStatus ? visibleModels(initialStatus) : [],
  );
  const [selection, setSelection] = useState<string[]>(() =>
    initialStatus ? selectedModelNames(initialStatus) : [],
  );
  const [localModels, setLocalModels] = useState<string[]>(
    initialLocalModels ?? [],
  );
  const [searchQuery, setSearchQuery] = useState("");
  const [pickerOpen, setPickerOpen] = useState(false);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [restarting, setRestarting] = useState(false);
  const [autoModeOverride, setAutoModeOverride] = useState<boolean | null>(
    null,
  );
  const pickerRef = useRef<HTMLDivElement>(null);

  const applyStatus = useCallback((next: ClaudeDesktopStatus) => {
    setStatus(next);
    setModels(visibleModels(next));
    setSelection(selectedModelNames(next));
    setError(null);
  }, []);

  const refreshStatus = useCallback(async () => {
    if (!window.getClaudeDesktopStatus) return;
    try {
      applyStatus(await window.getClaudeDesktopStatus());
    } catch {
      setError("Ollama could not read the Claude connection status.");
    }
  }, [applyStatus]);

  useEffect(() => {
    if (!initialStatus) void refreshStatus();
    const handleFocus = () => void refreshStatus();
    window.addEventListener("focus", handleFocus);
    return () => window.removeEventListener("focus", handleFocus);
  }, [initialStatus, refreshStatus]);

  useEffect(() => {
    if (!status) return;
    setModels(visibleModels(status));
    setSelection(selectedModelNames(status));
  }, [status]);

  useEffect(() => {
    if (initialLocalModels || !status?.used) return;
    let cancelled = false;
    setModelsLoading(true);
    void getClaudeDesktopAvailableModels()
      .then((installed) => {
        if (!cancelled) {
          setLocalModels(installed.map((model) => model.model));
        }
      })
      .catch(() => {
        if (!cancelled) setError("Ollama could not load your models.");
      })
      .finally(() => {
        if (!cancelled) setModelsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [initialLocalModels, status?.used]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        pickerRef.current &&
        !pickerRef.current.contains(event.target as Node)
      ) {
        setPickerOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const matchingLocalModels = useMemo(() => {
    const current = new Set(
      models.map((model) =>
        model.cloud ? explicitCloudName(model.name) : model.name,
      ),
    );
    const query = searchQuery.trim().toLowerCase();
    return localModels
      .filter(
        (name) =>
          !current.has(name) &&
          !isInvalidModelName(name) &&
          (!query || name.toLowerCase().includes(query)),
      )
      .sort((left, right) => left.localeCompare(right));
  }, [localModels, models, searchQuery]);

  const toggleModel = (name: string) => {
    setError(null);
    setSelection((current) => {
      if (!current.includes(name)) {
        const result = addClaudeModelSelection(
          current,
          name,
          claudeDesktopMaxModels(status),
        );
        if (result.error) {
          setError(result.error);
          return current;
        }
        return result.selection;
      }
      if (current.length === 1) {
        setError("Select at least one model for Claude.");
        return current;
      }
      return current.filter((model) => model !== name);
    });
  };

  const addLocalModel = (name: string) => {
    const maxModels = claudeDesktopMaxModels(status);
    const result = addClaudeModelSelection(selection, name, maxModels);
    if (result.error) {
      setError(result.error);
      return;
    }
    const cloud = name.endsWith(":cloud");
    setModels((current) => [
      ...current,
      {
        name,
        displayName: name,
        cloud,
        selected: true,
        availability: "available",
      },
    ]);
    setSelection(result.selection);
    setSearchQuery("");
    setPickerOpen(false);
    setError(null);
  };

  const hasAvailableSelection = selection.some((name) => {
    const model = models.find((candidate) => candidate.name === name);
    return !model?.availability || model.availability === "available";
  });

  const confirmRestartIfRunning = (message: string) =>
    !status?.running ||
    window.confirm(`${message} Any running task will stop.`);

  const restartClaude = async () => {
    if (!window.restartClaudeDesktop) {
      setError("Claude restart is available in the Ollama macOS app.");
      return;
    }
    if (selection.length === 0) {
      setError("Select at least one model for Claude.");
      return;
    }
    if (!hasAvailableSelection) {
      setError("Select a model available to your account.");
      return;
    }
    if (
      !confirmRestartIfRunning("Restart Claude Desktop to update its models?")
    )
      return;

    setError(null);
    setRestarting(true);
    try {
      const result = await window.restartClaudeDesktop(selection);
      applyStatus(result.status);
      if (result.error) setError(result.error);
    } catch {
      setError("Ollama could not restart Claude.");
    } finally {
      setRestarting(false);
    }
  };

  const toggleAutoMode = async (checked: boolean) => {
    if (!window.setClaudeDesktopAutoMode) {
      setError("Auto mode is available in the Ollama macOS app.");
      return;
    }
    if (!confirmRestartIfRunning("Restart Claude to change auto mode?")) return;

    setError(null);
    setAutoModeOverride(checked);
    setRestarting(true);
    try {
      const result = await window.setClaudeDesktopAutoMode(checked);
      applyStatus(result.status);
      if (result.error) setError(result.error);
    } catch {
      setError("Ollama could not update Claude auto mode.");
    } finally {
      setAutoModeOverride(null);
      setRestarting(false);
    }
  };

  if (!status?.supported || !status.used) {
    return null;
  }

  const maxModels = claudeDesktopMaxModels(status);
  const selectionFull = selection.length >= maxModels;
  const autoModeModels = models.filter((model) => model.autoMode);
  const autoModeModelNames = autoModeModels.map((model) => model.name);
  const autoModeAvailable =
    selection.length > 0 &&
    selection.some((name) =>
      autoModeModels.some((model) => model.name === name),
    );
  const autoMode =
    autoModeAvailable && (autoModeOverride ?? status.autoMode ?? false);
  const autoModeDescription = autoModeAvailable
    ? "Let Claude decide when to ask before making changes."
    : autoModeModelNames.length > 0
      ? `Select one of ${formatModelList(autoModeModelNames)} to use auto mode.`
      : "Auto mode needs a recommended model from Ollama.com.";
  const guidance =
    claudeDesktopRecoveryMessage(status.error, error) ??
    (!hasAvailableSelection && models.length > 0
      ? "Select a model available to your account."
      : selectionFull
        ? claudeDesktopMaxModelsMessage(maxModels)
        : status.connected
          ? "Restart Claude to refresh its model list."
          : "These models will be available when Claude starts.");

  return (
    <section aria-labelledby="apps-settings-heading" className="space-y-2">
      <h2
        id="apps-settings-heading"
        className="px-1 text-xs font-medium uppercase tracking-wider text-neutral-400 dark:text-neutral-500"
      >
        Apps
      </h2>
      <div
        aria-labelledby="claude-models-settings-heading"
        className="overflow-visible rounded-xl bg-white p-4 dark:bg-neutral-800"
      >
        <div className="flex items-start space-x-3">
          <img
            src="/launch-icons/claude.svg"
            alt=""
            className="mt-0.5 h-5 w-5 flex-shrink-0"
          />
          <div className="min-w-0 flex-1">
            <div className="flex items-center justify-between gap-3">
              <h2
                id="claude-models-settings-heading"
                className="text-sm font-medium text-neutral-900 dark:text-white"
              >
                Claude
              </h2>
              {status.modelSource === "fallback" && models.length > 0 && (
                <span className="text-xs text-neutral-400">
                  Built-in defaults
                </span>
              )}
            </div>

            <div className="mt-3 grid grid-cols-2 gap-x-5 gap-y-2 max-[850px]:grid-cols-1">
              {models.map((model) => {
                const selected = selection.includes(model.name);
                const accessLabel = modelAccessLabel(model);
                const unavailable =
                  model.availability !== undefined &&
                  model.availability !== "available";
                return (
                  <label
                    key={model.name}
                    className="flex min-w-0 cursor-pointer items-center gap-2 rounded-md py-1 text-sm text-neutral-600 dark:text-neutral-300"
                    title={accessLabel ?? model.description}
                  >
                    <input
                      type="checkbox"
                      checked={selected}
                      disabled={
                        restarting ||
                        (!selected && (selectionFull || unavailable))
                      }
                      onChange={() => toggleModel(model.name)}
                      className="h-4 w-4 rounded border-neutral-300 accent-neutral-900 dark:border-neutral-600 dark:accent-white"
                    />
                    <span className="truncate">{model.displayName}</span>
                    {accessLabel && (
                      <span className="flex-shrink-0 text-xs text-neutral-400">
                        {accessLabel}
                      </span>
                    )}
                  </label>
                );
              })}
            </div>

            <div ref={pickerRef} className="relative mt-3">
              <Input
                type="search"
                value={searchQuery}
                onFocus={() => setPickerOpen(true)}
                onChange={(event) => {
                  setSearchQuery(event.target.value);
                  setPickerOpen(true);
                }}
                placeholder="Search Ollama models"
                aria-label="Search Ollama models"
                aria-expanded={pickerOpen}
                aria-controls="claude-local-models"
                disabled={restarting}
                autoComplete="off"
              />
              {pickerOpen && (
                <div
                  id="claude-local-models"
                  className="absolute z-20 mt-1 max-h-52 w-full overflow-y-auto rounded-lg border border-neutral-200 bg-white p-1 shadow-lg dark:border-neutral-600 dark:bg-neutral-800"
                >
                  {modelsLoading ? (
                    <p className="px-3 py-2 text-sm text-neutral-400">
                      Loading models…
                    </p>
                  ) : selectionFull ? (
                    <p className="px-3 py-2 text-sm text-neutral-400">
                      {claudeDesktopMaxModelsMessage(maxModels)}
                    </p>
                  ) : matchingLocalModels.length > 0 ? (
                    matchingLocalModels.map((name) => (
                      <button
                        key={name}
                        type="button"
                        onClick={() => addLocalModel(name)}
                        className="block w-full rounded-md px-3 py-2 text-left text-sm text-neutral-700 hover:bg-neutral-100 dark:text-neutral-200 dark:hover:bg-neutral-700"
                      >
                        {name}
                      </button>
                    ))
                  ) : (
                    <p className="px-3 py-2 text-sm text-neutral-400">
                      No models found.
                    </p>
                  )}
                </div>
              )}
            </div>

            <Field className="mt-3 border-t border-neutral-200 pt-3 dark:border-neutral-700">
              <div className="flex items-center justify-between gap-4">
                <div className="min-w-0">
                  <Label>Enable auto mode</Label>
                  <Description>{autoModeDescription}</Description>
                </div>
                <Switch
                  checked={autoMode}
                  disabled={restarting || !autoModeAvailable}
                  onChange={(checked) => void toggleAutoMode(checked)}
                  className="flex-shrink-0"
                />
              </div>
            </Field>

            <div className="mt-3 flex items-center justify-between gap-4 max-sm:flex-col max-sm:items-stretch">
              <p
                role={error || status.error ? "alert" : undefined}
                className="text-xs leading-5 text-neutral-500 dark:text-neutral-400"
              >
                {guidance}
              </p>
              <Button
                type="button"
                color="white"
                onClick={restartClaude}
                disabled={
                  restarting || selection.length === 0 || !hasAvailableSelection
                }
                className="flex-shrink-0 max-sm:w-full"
              >
                {restarting && (
                  <ArrowPathIcon data-slot="icon" className="animate-spin" />
                )}
                {restarting
                  ? status.connected
                    ? "Restarting…"
                    : "Starting…"
                  : status.connected
                    ? "Restart Claude"
                    : "Start Claude"}
              </Button>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
