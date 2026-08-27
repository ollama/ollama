import { Button } from "@/components/ui/button";
import type {
  CodexDesktopModelsSettings as ModelsSettings,
  CodexDesktopModelsSettingsResult,
} from "@/types/webview";
import {
  ArrowPathIcon,
  CheckIcon,
  MagnifyingGlassIcon,
  XMarkIcon,
} from "@heroicons/react/20/solid";
import { Popover, PopoverButton, PopoverPanel } from "@headlessui/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

interface CodexDesktopModelsSettingsProps {
  initialSettings?: ModelsSettings;
  onDraftChange?: (hasChanges: boolean) => void;
}

function selectionsEqual(
  left: string[] | null | undefined,
  right: string[] | null | undefined,
): boolean {
  left ??= [];
  right ??= [];
  return (
    left.length === right.length &&
    left.every((model, index) => model === right[index])
  );
}

function normalizeSettings(
  settings: ModelsSettings & {
    selected?: string[] | null;
    available?: string[] | null;
  },
): ModelsSettings {
  return {
    ...settings,
    selected: settings.selected ?? [],
    available: settings.available ?? [],
    maxModels: settings.maxModels || 5,
  };
}

function ModelOptions({
  available,
  selected,
  maxModels,
  onToggle,
}: {
  available: string[];
  selected: string[];
  maxModels: number;
  onToggle: (model: string) => void;
}) {
  const [query, setQuery] = useState("");
  const searchRef = useRef<HTMLInputElement>(null);
  const selectedSet = useMemo(() => new Set(selected), [selected]);
  const normalizedQuery = query.trim().toLowerCase();
  const filtered = available.filter((model) =>
    model.toLowerCase().includes(normalizedQuery),
  );

  useEffect(() => {
    searchRef.current?.focus({ preventScroll: true });
  }, []);

  return (
    <>
      <div className="flex items-center gap-2 border-b border-neutral-100 px-3 py-2 dark:border-neutral-700">
        <MagnifyingGlassIcon className="h-4 w-4 shrink-0 text-neutral-400" />
        <input
          ref={searchRef}
          type="text"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Find model..."
          aria-label="Find ChatGPT model"
          autoCorrect="off"
          autoComplete="off"
          className="min-w-0 flex-1 border-none bg-transparent py-0.5 outline-none"
        />
      </div>
      <div
        role="listbox"
        aria-multiselectable="true"
        className="min-h-0 overflow-y-auto py-1"
      >
        {filtered.map((model) => {
          const checked = selectedSet.has(model);
          const disabled = !checked && selected.length >= maxModels;
          return (
            <button
              key={model}
              type="button"
              role="option"
              aria-selected={checked}
              disabled={disabled}
              onClick={() => onToggle(model)}
              className="flex w-full cursor-pointer items-center gap-2 px-3 py-2 text-left hover:bg-neutral-100 focus:bg-neutral-100 focus:outline-none disabled:cursor-not-allowed disabled:opacity-40 dark:hover:bg-neutral-700/60 dark:focus:bg-neutral-700/60"
            >
              <span className="h-4 w-4 shrink-0">
                {checked && <CheckIcon className="h-4 w-4" />}
              </span>
              <span className="min-w-0 flex-1 truncate">{model}</span>
            </button>
          );
        })}
        {filtered.length === 0 && (
          <p className="px-3 py-2 text-neutral-400">No models found</p>
        )}
      </div>
    </>
  );
}

export function CodexDesktopModelsSettings({
  initialSettings,
  onDraftChange,
}: CodexDesktopModelsSettingsProps) {
  const normalizedInitialSettings = initialSettings
    ? normalizeSettings(initialSettings)
    : null;
  const [settings, setSettings] = useState<ModelsSettings | null>(
    normalizedInitialSettings,
  );
  const [selected, setSelected] = useState<string[]>(
    normalizedInitialSettings?.selected ?? [],
  );
  const [saved, setSaved] = useState<string[]>(
    normalizedInitialSettings?.selected ?? [],
  );
  const [loading, setLoading] = useState(!initialSettings);
  const [applying, setApplying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const draftRef = useRef({ selected, saved });
  draftRef.current = { selected, saved };

  const applyResult = useCallback(
    (result: CodexDesktopModelsSettingsResult, preserveDraft = false) => {
      const nextSettings = normalizeSettings(result.settings);
      const keepDraft =
        preserveDraft &&
        !selectionsEqual(draftRef.current.selected, draftRef.current.saved);
      setSettings(nextSettings);
      if (!keepDraft) {
        setSelected(nextSettings.selected);
        setSaved(nextSettings.selected);
      }
      setError(result.error ?? null);
    },
    [],
  );

  const refresh = useCallback(async () => {
    if (!window.getCodexDesktopModelsSettings || applying) return;
    try {
      applyResult(await window.getCodexDesktopModelsSettings(), true);
    } catch {
      setError("Ollama could not load the ChatGPT model settings.");
    } finally {
      setLoading(false);
    }
  }, [applyResult, applying]);

  useEffect(() => {
    if (!initialSettings) void refresh();
    const onFocus = () => void refresh();
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  }, [initialSettings, refresh]);

  const hasChanges = !selectionsEqual(selected, saved);
  useEffect(() => {
    onDraftChange?.(hasChanges);
  }, [hasChanges, onDraftChange]);

  const maxModels = settings?.maxModels ?? 5;
  const available = useMemo(
    () => Array.from(new Set([...(settings?.available ?? []), ...selected])),
    [selected, settings?.available],
  );

  const toggleModel = (model: string) => {
    setError(null);
    setSelected((current) => {
      if (current.includes(model)) {
        return current.filter((name) => name !== model);
      }
      if (current.length >= maxModels) return current;
      return [...current, model];
    });
  };

  const applyChanges = async () => {
    if (!window.applyCodexDesktopModels) {
      setError("ChatGPT model settings are available in the Ollama macOS app.");
      return;
    }
    if (selected.length === 0) {
      setError("Choose at least one model for ChatGPT.");
      return;
    }
    if (
      settings?.running &&
      !window.confirm(
        "Restart ChatGPT · Ollama to apply these models? Any running task in that window will stop.",
      )
    ) {
      return;
    }

    setApplying(true);
    setError(null);
    try {
      const result = await window.applyCodexDesktopModels(selected);
      if (result.error) {
        setSettings(normalizeSettings(result.settings));
        setError(result.error);
        return;
      }
      applyResult(result);
    } catch {
      setError("Ollama could not apply the ChatGPT models.");
    } finally {
      setApplying(false);
    }
  };

  if (!settings?.supported && !loading) return null;

  return (
    <div
      aria-labelledby="chatgpt-model-settings-heading"
      className="overflow-visible rounded-xl bg-white p-4 dark:bg-neutral-800"
    >
      <div className="flex items-start space-x-3">
        <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center">
          <img
            src="/launch-icons/codex-color.svg"
            alt=""
            className="h-6 w-6 max-w-none"
          />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h2
                id="chatgpt-model-settings-heading"
                className="text-sm font-medium text-neutral-900 dark:text-white"
              >
                ChatGPT
              </h2>
              <p className="mt-1 text-base/6 text-zinc-500 sm:text-sm/6 dark:text-zinc-400">
                Select up to {maxModels} Ollama models to use in ChatGPT.
              </p>
            </div>
            <Button
              type="button"
              color="white"
              onClick={() => void applyChanges()}
              disabled={
                loading ||
                applying ||
                selected.length === 0 ||
                (settings?.running && !hasChanges)
              }
              className="shrink-0"
            >
              {applying && (
                <ArrowPathIcon data-slot="icon" className="animate-spin" />
              )}
              {applying
                ? settings?.running
                  ? "Restarting…"
                  : "Starting…"
                : settings?.running
                  ? "Save & restart ChatGPT"
                  : "Save & start ChatGPT"}
            </Button>
          </div>

          <div className="mt-4 w-full max-w-xl">
            <Popover className="relative w-full">
              <div
                data-testid="chatgpt-model-picker"
                className="relative flex min-h-10 w-full flex-wrap items-center gap-1.5 rounded-lg bg-neutral-50 px-2 py-1.5 ring-1 ring-inset ring-neutral-200 hover:bg-neutral-100 dark:bg-neutral-700 dark:ring-neutral-600 dark:hover:bg-neutral-600"
              >
                <PopoverButton
                  aria-label="Add ChatGPT model"
                  disabled={loading || applying}
                  className="absolute inset-0 rounded-lg outline-none focus-visible:ring-2 focus-visible:ring-blue-500 disabled:cursor-not-allowed"
                >
                  <span className="sr-only">Choose ChatGPT models</span>
                </PopoverButton>
                <div className="pointer-events-none relative z-10 flex min-w-0 flex-1 flex-wrap items-center gap-1.5">
                  {selected.map((model) => (
                    <button
                      key={model}
                      type="button"
                      aria-label={`Remove ${model}`}
                      disabled={applying}
                      onClick={() => toggleModel(model)}
                      className="pointer-events-auto inline-flex max-w-full items-center gap-1 rounded-md bg-neutral-200/70 px-2 py-1 text-sm text-neutral-700 hover:bg-neutral-200 disabled:opacity-50 dark:bg-neutral-600 dark:text-neutral-100 dark:hover:bg-neutral-500"
                    >
                      <span className="truncate">{model}</span>
                      <XMarkIcon className="h-3.5 w-3.5 shrink-0" />
                    </button>
                  ))}
                  {selected.length === 0 && (
                    <span className="px-1 py-1 text-sm text-neutral-400">
                      {loading ? "Loading models…" : "Select models"}
                    </span>
                  )}
                </div>
              </div>
              <PopoverPanel
                anchor={{ to: "bottom start", gap: 8, padding: 8 }}
                data-testid="chatgpt-model-options"
                className="z-50 flex w-[var(--button-width)] max-w-[calc(100vw-1rem)] flex-col overflow-hidden rounded-2xl border border-neutral-100 bg-white text-[15px] text-neutral-800 shadow-xl shadow-black/5 [--anchor-max-height:19rem] dark:border-neutral-600/40 dark:bg-neutral-800 dark:text-white"
              >
                <ModelOptions
                  available={available}
                  selected={selected}
                  maxModels={maxModels}
                  onToggle={toggleModel}
                />
              </PopoverPanel>
            </Popover>
          </div>

          {error && (
            <p
              role="alert"
              className="mt-3 w-full max-w-xl text-xs leading-5 text-red-600 dark:text-red-400"
            >
              {error}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
