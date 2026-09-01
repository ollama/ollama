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
  const [highlightedIndex, setHighlightedIndex] = useState(-1);
  const searchRef = useRef<HTMLInputElement>(null);
  const optionRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const selectedSet = useMemo(() => new Set(selected), [selected]);
  const normalizedQuery = query.trim().toLowerCase();
  const filtered = available.filter((model) =>
    model.toLowerCase().includes(normalizedQuery),
  );

  useEffect(() => {
    searchRef.current?.focus({ preventScroll: true });
  }, []);

  useEffect(() => {
    setHighlightedIndex(-1);
  }, [normalizedQuery]);

  useEffect(() => {
    if (highlightedIndex < 0) return;
    optionRefs.current[highlightedIndex]?.scrollIntoView({ block: "nearest" });
  }, [highlightedIndex]);

  const optionIsDisabled = (model: string) =>
    !selectedSet.has(model) && selected.length >= maxModels;

  const moveHighlight = (direction: -1 | 1) => {
    setHighlightedIndex((current) => {
      if (filtered.length === 0) return -1;
      const start = current < 0 ? (direction === 1 ? -1 : 0) : current;
      for (let offset = 1; offset <= filtered.length; offset += 1) {
        const candidate =
          (start + direction * offset + filtered.length) % filtered.length;
        if (!optionIsDisabled(filtered[candidate])) return candidate;
      }
      return -1;
    });
  };

  return (
    <>
      <div className="flex items-center gap-2 border-b border-neutral-100 px-3 py-2 dark:border-neutral-700">
        <MagnifyingGlassIcon className="h-4 w-4 shrink-0 text-neutral-400" />
        <input
          ref={searchRef}
          type="text"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "ArrowDown") {
              event.preventDefault();
              moveHighlight(1);
            } else if (event.key === "ArrowUp") {
              event.preventDefault();
              moveHighlight(-1);
            } else if (
              event.key === "Enter" &&
              highlightedIndex >= 0 &&
              highlightedIndex < filtered.length
            ) {
              event.preventDefault();
              onToggle(filtered[highlightedIndex]);
            }
          }}
          placeholder="Find model..."
          aria-label="Find ChatGPT model"
          role="combobox"
          aria-expanded="true"
          aria-controls="chatgpt-model-options-listbox"
          aria-activedescendant={
            highlightedIndex >= 0
              ? `chatgpt-model-option-${highlightedIndex}`
              : undefined
          }
          autoCorrect="off"
          autoComplete="off"
          className="min-w-0 flex-1 border-none bg-transparent py-0.5 outline-none"
        />
      </div>
      <div
        id="chatgpt-model-options-listbox"
        role="listbox"
        aria-multiselectable="true"
        className="min-h-0 overflow-y-auto py-1"
      >
        {filtered.map((model, index) => {
          const checked = selectedSet.has(model);
          const disabled = optionIsDisabled(model);
          return (
            <button
              key={model}
              id={`chatgpt-model-option-${index}`}
              ref={(element) => {
                optionRefs.current[index] = element;
              }}
              type="button"
              role="option"
              aria-selected={checked}
              disabled={disabled}
              onClick={() => onToggle(model)}
              onMouseEnter={() => setHighlightedIndex(index)}
              className={`flex w-full cursor-pointer items-center gap-2 px-3 py-2 text-left hover:bg-neutral-100 focus:bg-neutral-100 focus:outline-none disabled:cursor-not-allowed disabled:opacity-40 dark:hover:bg-neutral-700/60 dark:focus:bg-neutral-700/60 ${
                highlightedIndex === index
                  ? "bg-neutral-100 dark:bg-neutral-700/60"
                  : ""
              }`}
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
  const [notice, setNotice] = useState<string | null>(null);
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
    if (applying) return;
    if (!window.getCodexDesktopModelsSettings) {
      setError("ChatGPT model settings are unavailable in this Ollama build.");
      setLoading(false);
      return;
    }
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
        `${settings.connected ? "Restart ChatGPT to update its Ollama models?" : "Restart ChatGPT to add Ollama models?"} Your ChatGPT profile will remain available. Any running task will stop.`,
      )
    ) {
      return;
    }

    setApplying(true);
    setError(null);
    setNotice(null);
    try {
      const result = await window.applyCodexDesktopModels(selected);
      if (result.error) {
        setSettings(normalizeSettings(result.settings));
        setError(result.error);
        return;
      }
      applyResult(result);
      setNotice("Your selected Ollama models are ready in ChatGPT.");
    } catch {
      setError("Ollama could not apply the ChatGPT models.");
    } finally {
      setApplying(false);
    }
  };

  if (!settings?.supported && !loading && !error) return null;

  const connected = settings?.connected ?? false;
  const busy = applying;

  return (
    <div
      aria-labelledby="chatgpt-model-settings-heading"
      className="overflow-visible rounded-xl bg-white p-4 dark:bg-neutral-800"
    >
      <div className="flex items-start space-x-3">
        <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center">
          <img
            src="/launch-icons/codex.svg"
            alt=""
            className="h-5 w-5 dark:hidden"
          />
          <img
            src="/launch-icons/codex-dark.svg"
            alt=""
            className="hidden h-5 w-5 dark:block"
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
                Choose up to {maxModels} Ollama models to use in ChatGPT.
              </p>
            </div>
            <div className="shrink-0">
              <Button
                type="button"
                color="white"
                onClick={() => void applyChanges()}
                disabled={
                  loading ||
                  busy ||
                  selected.length === 0 ||
                  (connected && settings?.running && !hasChanges)
                }
              >
                {applying && (
                  <ArrowPathIcon data-slot="icon" className="animate-spin" />
                )}
                {applying
                  ? settings?.running
                    ? "Restarting…"
                    : "Starting…"
                  : connected
                    ? settings?.running
                      ? "Save & restart ChatGPT"
                      : hasChanges
                        ? "Save & start ChatGPT"
                        : "Start ChatGPT"
                    : settings?.running
                      ? "Save & restart ChatGPT"
                      : "Save & start ChatGPT"}
              </Button>
            </div>
          </div>

          <div className="mt-4 w-full max-w-xl">
            <Popover className="relative w-full">
              <div
                data-testid="chatgpt-model-picker"
                className="relative flex min-h-10 w-full flex-wrap items-center gap-1.5 rounded-lg bg-neutral-50 px-2 py-1.5 ring-1 ring-inset ring-neutral-200 hover:bg-neutral-100 dark:bg-neutral-700 dark:ring-neutral-600 dark:hover:bg-neutral-600"
              >
                <PopoverButton
                  aria-label="Add ChatGPT model"
                  disabled={loading || busy}
                  className="absolute inset-0 rounded-lg outline-none focus-visible:ring-2 focus-visible:ring-blue-500 disabled:cursor-not-allowed"
                >
                  <span className="sr-only">Choose ChatGPT models</span>
                </PopoverButton>
                <div className="pointer-events-none relative z-10 flex min-w-0 flex-1 flex-wrap items-center gap-1.5">
                  {selected.map((model) => (
                    <span
                      key={model}
                      className="pointer-events-none inline-flex max-w-full items-stretch overflow-hidden rounded-md bg-neutral-200/70 text-sm text-neutral-700 dark:bg-neutral-600 dark:text-neutral-100"
                    >
                      <span className="min-w-0 py-1 pl-2 pr-1">
                        <span className="block truncate">{model}</span>
                      </span>
                      <button
                        type="button"
                        aria-label={`Remove ${model}`}
                        disabled={busy}
                        onClick={(event) => {
                          event.stopPropagation();
                          toggleModel(model);
                        }}
                        className="pointer-events-auto inline-flex shrink-0 items-center px-1.5 hover:bg-neutral-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 disabled:opacity-50 dark:hover:bg-neutral-500"
                      >
                        <XMarkIcon className="h-3.5 w-3.5" />
                      </button>
                    </span>
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
          {!error && notice && (
            <p
              role="status"
              className="mt-3 w-full max-w-xl text-xs leading-5 text-neutral-500 dark:text-neutral-400"
            >
              {notice}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
