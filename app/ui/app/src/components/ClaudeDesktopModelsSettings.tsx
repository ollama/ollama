import { getClaudeDesktopAvailableModels } from "@/api";
import { Button } from "@/components/ui/button";
import { Description, Field, Label } from "@/components/ui/fieldset";
import { Switch } from "@/components/ui/switch";
import { claudeDesktopRecoveryMessage } from "@/lib/claudeDesktop";
import { claudeDesktopModelStatusLabel } from "@/lib/claudeDesktopModelStatus";
import type {
  ClaudeDesktopActionResult,
  ClaudeDesktopMappingStatus,
  ClaudeDesktopModelStatus,
  ClaudeDesktopStatus,
} from "@/types/webview";
import {
  ArrowPathIcon,
  ArrowRightIcon,
  CheckIcon,
  ChevronUpDownIcon,
  MagnifyingGlassIcon,
} from "@heroicons/react/20/solid";
import { Popover, PopoverButton, PopoverPanel } from "@headlessui/react";
import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from "react";

export interface ClaudeDesktopModelsSettingsHandle {
  resetToDefaults: () => Promise<boolean>;
}

interface ClaudeDesktopModelsSettingsProps {
  initialStatus?: ClaudeDesktopStatus;
  initialLocalModels?: string[];
  initialCloudModels?: string[];
  includeCloudModels?: boolean;
  onDraftChange?: (hasChanges: boolean) => void;
}

const fallbackRoutes: ClaudeDesktopMappingStatus[] = [
  { routeId: "claude-fable-5", routeName: "Fable 5" },
  { routeId: "claude-opus-5", routeName: "Opus 5" },
  { routeId: "claude-sonnet-5", routeName: "Sonnet 5" },
  {
    routeId: "claude-haiku-4-5-20251001",
    routeName: "Haiku 4.5",
  },
  { routeId: "claude-sonnet-4-6", routeName: "Sonnet 4.6" },
];

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

function modelIsAvailable(model: ClaudeDesktopModelStatus): boolean {
  return !model.availability || model.availability === "available";
}

function initialMappings(
  status: ClaudeDesktopStatus,
): ClaudeDesktopMappingStatus[] {
  const models = visibleModels(status);
  const known = new Set(models.map((model) => model.name));
  const available = new Set(
    models.filter(modelIsAvailable).map((model) => model.name),
  );
  const routes = (
    status.mappings?.length ? status.mappings : fallbackRoutes
  ).map((route) => ({ ...route }));

  if (!status.mappings?.length) {
    const selected = models.filter(
      (model) => model.selected && available.has(model.name),
    );
    selected.slice(0, routes.length).forEach((model, index) => {
      routes[index].model = model.name;
    });
  }

  for (const route of routes) {
    if (route.model && !known.has(route.model)) route.model = undefined;
  }
  if (!routes.some((route) => route.model)) {
    const first = models.find(modelIsAvailable);
    if (first && routes.length > 0) routes[0].model = first.name;
  }
  return routes;
}

function mappingsEqual(
  left: ClaudeDesktopMappingStatus[],
  right: ClaudeDesktopMappingStatus[],
): boolean {
  return (
    left.length === right.length &&
    left.every(
      (route, index) =>
        route.routeId === right[index]?.routeId &&
        (route.model ?? "") === (right[index]?.model ?? ""),
    )
  );
}

function mappingRecord(
  mappings: ClaudeDesktopMappingStatus[],
): Record<string, string> {
  return Object.fromEntries(
    mappings
      .filter((route) => route.model)
      .map((route) => [route.routeId, route.model ?? ""]),
  );
}

function formatModelList(names: string[]): string {
  if (names.length < 2) return names[0] ?? "";
  if (names.length === 2) return `${names[0]} or ${names[1]}`;
  return `${names.slice(0, -1).join(", ")}, or ${names[names.length - 1]}`;
}

interface ClaudeModelPickerProps {
  id: string;
  routeName: string;
  value?: string;
  models: ClaudeDesktopModelStatus[];
  disabled: boolean;
  onChange: (model: string) => void;
}

function ClaudeModelPicker({
  id,
  routeName,
  value,
  models,
  disabled,
  onChange,
}: ClaudeModelPickerProps) {
  return (
    <Popover className="relative min-w-0">
      <PopoverButton
        id={id}
        aria-label={`Ollama model for ${routeName}`}
        aria-haspopup="listbox"
        disabled={disabled}
        className="flex min-h-9 w-full items-center gap-2 rounded-lg bg-neutral-50 px-3 py-1.5 text-left text-sm text-neutral-800 outline-none ring-1 ring-inset ring-neutral-200 hover:bg-neutral-100 focus:ring-2 focus:ring-blue-500 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-neutral-700 dark:text-neutral-100 dark:ring-neutral-600 dark:hover:bg-neutral-600"
      >
        <span
          className={`min-w-0 flex-1 truncate ${value ? "" : "text-neutral-400"}`}
        >
          {value || "Select a model"}
        </span>
        <ChevronUpDownIcon className="h-4 w-4 flex-shrink-0 text-neutral-400" />
      </PopoverButton>

      <PopoverPanel
        anchor={{ to: "bottom end", gap: 8, padding: 8 }}
        className="z-50 flex w-[var(--button-width)] min-w-64 flex-col overflow-hidden rounded-2xl border border-neutral-100 bg-white text-[15px] text-neutral-800 shadow-xl shadow-black/5 [--anchor-max-height:19rem] dark:border-neutral-600/40 dark:bg-neutral-800 dark:text-white"
      >
        {({ close }) => (
          <ClaudeModelPickerOptions
            routeName={routeName}
            value={value}
            models={models}
            onChange={(model) => {
              onChange(model);
              close();
            }}
          />
        )}
      </PopoverPanel>
    </Popover>
  );
}

function ClaudeModelPickerOptions({
  routeName,
  value,
  models,
  onChange,
}: Pick<
  ClaudeModelPickerProps,
  "routeName" | "value" | "models" | "onChange"
>) {
  const [query, setQuery] = useState("");
  const searchRef = useRef<HTMLInputElement>(null);
  const normalizedQuery = query.trim().toLowerCase();
  const filteredModels = models.filter((model) =>
    model.displayName.toLowerCase().includes(normalizedQuery),
  );

  useEffect(() => {
    searchRef.current?.focus({ preventScroll: true });
  }, []);

  return (
    <>
      <div className="flex flex-none items-center gap-2 border-b border-neutral-100 px-3 py-2 dark:border-neutral-700">
        <MagnifyingGlassIcon className="h-4 w-4 flex-shrink-0 text-neutral-400" />
        <input
          ref={searchRef}
          type="text"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Find model..."
          aria-label={`Find model for ${routeName}`}
          autoCorrect="off"
          autoComplete="off"
          className="min-w-0 flex-1 border-none bg-transparent py-0.5 outline-none"
        />
      </div>
      <div role="listbox" className="min-h-0 overflow-y-auto py-1">
        {filteredModels.map((model) => {
          const available = modelIsAvailable(model);
          const statusLabel = claudeDesktopModelStatusLabel(model);
          const selected = value === model.name;
          return (
            <button
              key={model.name}
              type="button"
              role="option"
              aria-selected={selected}
              disabled={!available}
              onClick={() => onChange(model.name)}
              className="flex w-full cursor-pointer items-start gap-2 px-3 py-2 text-left hover:bg-neutral-100 focus:bg-neutral-100 focus:outline-none disabled:cursor-not-allowed disabled:opacity-45 dark:hover:bg-neutral-700/60 dark:focus:bg-neutral-700/60"
            >
              <span className="mt-0.5 h-4 w-4 flex-shrink-0">
                {selected && <CheckIcon className="h-4 w-4" />}
              </span>
              <span className="min-w-0 flex-1">
                <span className="block truncate">{model.displayName}</span>
                {statusLabel && (
                  <span className="mt-0.5 block truncate text-xs text-neutral-400">
                    {statusLabel}
                  </span>
                )}
              </span>
            </button>
          );
        })}
        {filteredModels.length === 0 && (
          <p className="px-3 py-2 text-neutral-400">No models found</p>
        )}
      </div>
    </>
  );
}

export const ClaudeDesktopModelsSettings = forwardRef<
  ClaudeDesktopModelsSettingsHandle,
  ClaudeDesktopModelsSettingsProps
>(function ClaudeDesktopModelsSettings(
  {
    initialStatus,
    initialLocalModels,
    initialCloudModels,
    includeCloudModels = false,
    onDraftChange,
  },
  ref,
) {
  const [status, setStatus] = useState<ClaudeDesktopStatus | null>(
    initialStatus ?? null,
  );
  const [models, setModels] = useState<ClaudeDesktopModelStatus[]>(() =>
    initialStatus ? visibleModels(initialStatus) : [],
  );
  const [mappings, setMappings] = useState<ClaudeDesktopMappingStatus[]>(() =>
    initialStatus ? initialMappings(initialStatus) : [],
  );
  const [savedMappings, setSavedMappings] = useState<
    ClaudeDesktopMappingStatus[]
  >(() => (initialStatus ? initialMappings(initialStatus) : []));
  const [localModels, setLocalModels] = useState<string[]>(
    initialLocalModels ?? [],
  );
  const [accountCloudModels, setAccountCloudModels] = useState<string[]>(
    initialCloudModels ?? [],
  );
  const [modelsLoading, setModelsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [applying, setApplying] = useState(false);
  const [resettingMappings, setResettingMappings] = useState(false);
  const [autoModeApplying, setAutoModeApplying] = useState(false);
  const [autoModeOverride, setAutoModeOverride] = useState<boolean | null>(
    null,
  );
  const draftRef = useRef({ mappings, savedMappings });
  const statusRequestRef = useRef(0);
  const operationInFlightRef = useRef(false);
  draftRef.current = { mappings, savedMappings };

  const applyStatus = useCallback(
    (next: ClaudeDesktopStatus, preserveDraft = false) => {
      const nextMappings = initialMappings(next);
      const draft = draftRef.current;
      const keepDraft =
        preserveDraft && !mappingsEqual(draft.mappings, draft.savedMappings);
      setStatus(next);
      setModels(visibleModels(next));
      if (!keepDraft) {
        setMappings(nextMappings);
        setSavedMappings(nextMappings);
      }
      setError(null);
    },
    [],
  );

  const refreshStatus = useCallback(async () => {
    if (!window.getClaudeDesktopStatus) return;
    const request = ++statusRequestRef.current;
    try {
      const next = await window.getClaudeDesktopStatus();
      if (
        request === statusRequestRef.current &&
        !operationInFlightRef.current
      ) {
        applyStatus(next, true);
      }
    } catch {
      if (
        request === statusRequestRef.current &&
        !operationInFlightRef.current
      ) {
        setError("Ollama could not read the Claude connection status.");
      }
    }
  }, [applyStatus]);

  useEffect(() => {
    if (!initialStatus) void refreshStatus();
    const handleFocus = () => void refreshStatus();
    window.addEventListener("focus", handleFocus);
    return () => window.removeEventListener("focus", handleFocus);
  }, [initialStatus, refreshStatus]);

  useEffect(() => {
    if (initialLocalModels || !status?.used) return;
    let cancelled = false;
    setModelsLoading(true);
    void getClaudeDesktopAvailableModels(includeCloudModels)
      .then((installed) => {
        if (!cancelled) {
          setLocalModels(installed.map((model) => model.model));
          setAccountCloudModels(
            installed
              .filter((model) => model.isCloud())
              .map((model) => model.model),
          );
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

  const catalogModels = useMemo(() => {
    const current = new Set(models.map((model) => model.name));
    const installed: ClaudeDesktopModelStatus[] = localModels
      .filter((name) => !current.has(name) && !isInvalidModelName(name))
      .sort((left, right) => left.localeCompare(right))
      .map((name) => ({
        name,
        displayName: name,
        selected: false,
        availability: "available",
      }));
    return [...models, ...installed];
  }, [localModels, models]);

  const hasDraftChanges = !mappingsEqual(mappings, savedMappings);
  const assignedModels = mappings
    .map((route) => route.model)
    .filter((model): model is string => Boolean(model));
  const hasInvalidMapping = assignedModels.some((name) => {
    const model = catalogModels.find((candidate) => candidate.name === name);
    return !model || !modelIsAvailable(model);
  });
  const busy = applying || resettingMappings || autoModeApplying;

  useEffect(() => {
    onDraftChange?.(hasDraftChanges);
  }, [hasDraftChanges, onDraftChange]);

  const updateMapping = (routeId: string, model: string) => {
    setError(null);
    setMappings((current) =>
      current.map((route) =>
        route.routeId === routeId
          ? { ...route, model: model || undefined }
          : route,
      ),
    );
  };

  const runMappingAction = useCallback(
    async (
      action: (restartConfirmed: boolean) => Promise<ClaudeDesktopActionResult>,
      failureMessage: string,
    ): Promise<boolean> => {
      try {
        let result = await action(false);
        if (result.restartConfirmationRequired) {
          applyStatus(result.status, true);
          if (
            !window.confirm(
              "Restart Claude Desktop? Any running task will stop.",
            )
          ) {
            return false;
          }
          result = await action(true);
        }
        ++statusRequestRef.current;
        if (result.error) {
          applyStatus(result.status, !result.mappingsApplied);
          setError(result.error);
          return Boolean(result.mappingsApplied);
        }
        applyStatus(result.status);
        return true;
      } catch {
        setError(failureMessage);
        return false;
      }
    },
    [applyStatus],
  );

  const applyChanges = async () => {
    const applyMappings = window.applyClaudeDesktopMappings;
    if (!applyMappings) {
      setError(
        "Claude routing settings are available in the Ollama macOS app.",
      );
      return;
    }
    if (assignedModels.length === 0) {
      setError("Choose at least one Ollama model for Claude.");
      return;
    }
    if (hasInvalidMapping) {
      setError("Choose models available to your account and device.");
      return;
    }
    if (operationInFlightRef.current) return;

    const mappingsToApply = mappingRecord(mappings);
    setApplying(true);
    setError(null);
    operationInFlightRef.current = true;
    ++statusRequestRef.current;
    try {
      await runMappingAction(
        (restartConfirmed) => applyMappings(mappingsToApply, restartConfirmed),
        "Ollama could not apply the Claude model mappings.",
      );
    } finally {
      ++statusRequestRef.current;
      operationInFlightRef.current = false;
      setApplying(false);
    }
  };

  const toggleAutoMode = async (checked: boolean) => {
    if (!window.setClaudeDesktopAutoMode) {
      setError("Auto mode is available in the Ollama macOS app.");
      return;
    }
    setError(null);
    setAutoModeOverride(checked);
    setAutoModeApplying(true);
    operationInFlightRef.current = true;
    ++statusRequestRef.current;
    try {
      let result = await window.setClaudeDesktopAutoMode(checked, false);
      if (result.restartConfirmationRequired) {
        applyStatus(result.status, true);
        if (
          !window.confirm(
            "Restart Claude to change auto mode? Any running task will stop.",
          )
        ) {
          return;
        }
        result = await window.setClaudeDesktopAutoMode(checked, true);
      }
      ++statusRequestRef.current;
      applyStatus(result.status);
      if (result.error) setError(result.error);
    } catch {
      setError("Ollama could not update Claude auto mode.");
    } finally {
      ++statusRequestRef.current;
      operationInFlightRef.current = false;
      setAutoModeOverride(null);
      setAutoModeApplying(false);
    }
  };

  const resetToDefaults = useCallback(async (): Promise<boolean> => {
    if (operationInFlightRef.current) return false;

    const resetMappings = window.resetClaudeDesktopMappings;
    if (!resetMappings) {
      setError("Ollama could not reset the Claude model mappings.");
      return false;
    }

    setResettingMappings(true);
    setError(null);
    operationInFlightRef.current = true;
    ++statusRequestRef.current;
    try {
      return await runMappingAction(
        resetMappings,
        "Ollama could not reset the Claude model mappings.",
      );
    } finally {
      ++statusRequestRef.current;
      operationInFlightRef.current = false;
      setResettingMappings(false);
    }
  }, [runMappingAction]);

  useImperativeHandle(ref, () => ({ resetToDefaults }), [resetToDefaults]);

  if (!status?.supported || !status.used) return null;

  const autoModeModelNames = Array.from(
    new Set([
      ...models.filter((model) => model.autoMode).map((model) => model.name),
      ...accountCloudModels,
    ]),
  );
  const autoModeModelSet = new Set(autoModeModelNames);
  const autoModeAvailable =
    !hasDraftChanges &&
    assignedModels.length > 0 &&
    assignedModels.some((name) => autoModeModelSet.has(name));
  const autoMode = autoModeAvailable
    ? (autoModeOverride ?? status.autoMode ?? false)
    : (status.autoMode ?? false);
  const autoModeDescription = hasDraftChanges
    ? "Start or restart Claude to apply model changes before changing auto mode."
    : autoModeAvailable
      ? "Let Claude decide when to ask before making changes."
      : accountCloudModels.length > 0
        ? "Select a cloud model from Ollama.com to use auto mode."
        : autoModeModelNames.length > 0
          ? `Select one of ${formatModelList(autoModeModelNames)} to use auto mode.`
          : "Auto mode needs a cloud model available to your Ollama.com account.";

  const guidance =
    claudeDesktopRecoveryMessage(status.error, error) ??
    (hasDraftChanges && status.running
      ? "Restarting Claude will stop any running task."
      : null);

  return (
    <section aria-labelledby="apps-settings-heading" className="space-y-2">
      <h2
        id="apps-settings-heading"
        className="px-1 text-xs font-medium uppercase tracking-wider text-neutral-400 dark:text-neutral-500"
      >
        Apps
      </h2>
      <div
        aria-labelledby="claude-settings-heading"
        className="overflow-visible rounded-xl bg-white p-4 dark:bg-neutral-800"
      >
        <div className="flex items-start space-x-3">
          <img
            src="/launch-icons/claude.svg"
            alt=""
            className="mt-0.5 h-5 w-5 flex-shrink-0"
          />
          <div className="min-w-0 flex-1">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2
                  id="claude-settings-heading"
                  className="text-sm font-medium text-neutral-900 dark:text-white"
                >
                  Claude
                </h2>
                <p className="mt-1 text-base/6 text-zinc-500 sm:text-sm/6 dark:text-zinc-400">
                  Choose which Ollama model Claude uses for each model option.
                </p>
              </div>
              <Button
                type="button"
                color="white"
                onClick={applyChanges}
                disabled={
                  busy ||
                  assignedModels.length === 0 ||
                  hasInvalidMapping ||
                  (status.running && !hasDraftChanges)
                }
                className="flex-shrink-0"
              >
                {(applying || resettingMappings) && (
                  <ArrowPathIcon data-slot="icon" className="animate-spin" />
                )}
                {resettingMappings
                  ? "Resetting…"
                  : applying
                    ? status.running
                      ? "Restarting…"
                      : "Starting…"
                    : status.running
                      ? "Restart Claude"
                      : "Start Claude"}
              </Button>
            </div>

            <div className="mt-4 w-full max-w-xl space-y-1">
              {mappings.map((mapping) => (
                <div
                  key={mapping.routeId}
                  className="relative grid min-h-12 grid-cols-[5.5rem_3.75rem_minmax(0,1fr)] items-center gap-2 py-1 max-sm:grid-cols-1 max-sm:gap-2"
                >
                  <div className="min-w-0">
                    <span className="block text-sm font-medium text-neutral-800 dark:text-neutral-200">
                      {mapping.routeName}
                    </span>
                  </div>
                  <ArrowRightIcon
                    aria-hidden="true"
                    className="absolute left-[6.6625rem] h-4 w-4 -translate-x-1/2 text-neutral-300 dark:text-neutral-500 max-sm:hidden"
                  />
                  <div className="col-start-3 w-2/3 min-w-0 max-sm:col-start-auto max-sm:w-full">
                    <ClaudeModelPicker
                      id={`claude-route-${mapping.routeId}`}
                      routeName={mapping.routeName}
                      value={mapping.model ?? ""}
                      disabled={busy || modelsLoading}
                      models={catalogModels}
                      onChange={(model) =>
                        updateMapping(mapping.routeId, model)
                      }
                    />
                  </div>
                </div>
              ))}
            </div>

            <Field className="mt-3 w-full max-w-xl border-t border-neutral-200 pt-3 dark:border-neutral-700">
              <div className="flex items-center justify-between gap-4">
                <div className="min-w-0">
                  <Label>Enable auto mode</Label>
                  <Description>{autoModeDescription}</Description>
                </div>
                <Switch
                  checked={autoMode}
                  disabled={busy || !autoModeAvailable}
                  onChange={(checked) => void toggleAutoMode(checked)}
                  className="flex-shrink-0"
                />
              </div>
            </Field>

            {guidance && (
              <p
                role={error || status.error ? "alert" : "status"}
                className="mt-3 w-full max-w-xl text-xs leading-5 text-neutral-500 dark:text-neutral-400"
              >
                {guidance}
              </p>
            )}
          </div>
        </div>
      </div>
    </section>
  );
});
