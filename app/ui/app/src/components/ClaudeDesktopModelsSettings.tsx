import { getClaudeDesktopAvailableModels } from "@/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  addClaudeModelSelection,
  claudeDesktopMaxModels,
  claudeDesktopMaxModelsMessage,
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
  includeCloudModels?: boolean;
}

function isInvalidModelName(name: string): boolean {
  const normalized = name.trim().toLowerCase().replace(/[-:]+/g, " ");
  return normalized === "ollama cloud";
}

function visibleModels(
  status: ClaudeDesktopStatus,
  includeCloudModels: boolean,
): ClaudeDesktopModelStatus[] {
  return (status.models ?? []).filter(
    (model) =>
      !isInvalidModelName(model.name) && (includeCloudModels || !model.cloud),
  );
}

function selectedModelNames(
  status: ClaudeDesktopStatus,
  includeCloudModels: boolean,
): string[] {
  return visibleModels(status, includeCloudModels)
    .filter((model) => model.selected)
    .map((model) => model.name);
}

function explicitCloudName(name: string): string {
  return name.endsWith(":cloud") ? name : `${name}:cloud`;
}

export function ClaudeDesktopModelsSettings({
  initialStatus,
  initialLocalModels,
  includeCloudModels = false,
}: ClaudeDesktopModelsSettingsProps) {
  const [status, setStatus] = useState<ClaudeDesktopStatus | null>(
    initialStatus ?? null,
  );
  const [models, setModels] = useState<ClaudeDesktopModelStatus[]>(() =>
    initialStatus ? visibleModels(initialStatus, includeCloudModels) : [],
  );
  const [selection, setSelection] = useState<string[]>(() =>
    initialStatus ? selectedModelNames(initialStatus, includeCloudModels) : [],
  );
  const [localModels, setLocalModels] = useState<string[]>(
    initialLocalModels ?? [],
  );
  const [searchQuery, setSearchQuery] = useState("");
  const [pickerOpen, setPickerOpen] = useState(false);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [restarting, setRestarting] = useState(false);
  const pickerRef = useRef<HTMLDivElement>(null);

  const applyStatus = useCallback(
    (next: ClaudeDesktopStatus) => {
      setStatus(next);
      setModels(visibleModels(next, includeCloudModels));
      setSelection(selectedModelNames(next, includeCloudModels));
    },
    [includeCloudModels],
  );

  const refreshStatus = useCallback(async () => {
    if (!window.getClaudeDesktopStatus) return;
    try {
      applyStatus(await window.getClaudeDesktopStatus());
      setError(null);
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
    setModels(visibleModels(status, includeCloudModels));
    setSelection(selectedModelNames(status, includeCloudModels));
  }, [includeCloudModels, status]);

  useEffect(() => {
    if (initialLocalModels || !status?.used) return;
    let cancelled = false;
    setModelsLoading(true);
    void getClaudeDesktopAvailableModels(includeCloudModels)
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
  }, [includeCloudModels, initialLocalModels, status?.used]);

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
      { name, displayName: name, cloud, selected: true },
    ]);
    setSelection(result.selection);
    setSearchQuery("");
    setPickerOpen(false);
    setError(null);
  };

  const restartClaude = async () => {
    if (!window.restartClaudeDesktop) {
      setError("Claude restart is available in the Ollama macOS app.");
      return;
    }
    if (selection.length === 0) {
      setError("Select at least one model for Claude.");
      return;
    }
    if (
      status?.running &&
      !window.confirm(
        "Restart Claude Desktop to update its models? Any running task will stop.",
      )
    ) {
      return;
    }

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

  if (!status?.supported || !status.used) {
    return null;
  }

  const maxModels = claudeDesktopMaxModels(status);
  const selectionFull = selection.length >= maxModels;

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
              {models.map((model) => (
                <label
                  key={model.name}
                  className="flex min-w-0 cursor-pointer items-center gap-2 rounded-md py-1 text-sm text-neutral-600 dark:text-neutral-300"
                  title={model.description}
                >
                  <input
                    type="checkbox"
                    checked={selection.includes(model.name)}
                    disabled={
                      restarting ||
                      (!selection.includes(model.name) && selectionFull)
                    }
                    onChange={() => toggleModel(model.name)}
                    className="h-4 w-4 rounded border-neutral-300 accent-neutral-900 dark:border-neutral-600 dark:accent-white"
                  />
                  <span className="truncate">{model.displayName}</span>
                </label>
              ))}
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

            <div className="mt-3 flex items-center justify-between gap-4 max-sm:flex-col max-sm:items-stretch">
              <p
                role={error ? "alert" : undefined}
                className={`text-xs leading-5 ${error ? "text-red-600 dark:text-red-400" : "text-neutral-400"}`}
              >
                {error ??
                  (selectionFull
                    ? claudeDesktopMaxModelsMessage(maxModels)
                    : status.connected
                      ? "Restart Claude to refresh its model list."
                      : "These models will be available when Claude starts.")}
              </p>
              <Button
                type="button"
                color="white"
                onClick={restartClaude}
                disabled={restarting || selection.length === 0}
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
