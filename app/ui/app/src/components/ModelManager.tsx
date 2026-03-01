import { useState, useCallback, useRef, useEffect, useMemo } from "react";
import { useNavigate } from "@tanstack/react-router";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowLeftIcon,
  XMarkIcon,
  MagnifyingGlassIcon,
  TrashIcon,
  ArrowDownTrayIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  CircleStackIcon,
  DocumentDuplicateIcon,
  ChevronUpDownIcon,
  CheckIcon,
} from "@heroicons/react/20/solid";
import {
  getModelsDetailed,
  deleteModel,
  pullModel,
  showModel,
  copyModel,
  getModelSettings,
  updateModelSettings,
  deleteModelSettings,
  type ShowModelResponse,
  type DetailedModel,
  type ModelSettingsData,
} from "@/api";

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(
    Math.floor(Math.log(bytes) / Math.log(k)),
    sizes.length - 1,
  );
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
}

function formatDate(date: unknown): string {
  if (!date) return "—";
  const d = date instanceof Date ? date : new Date(String(date));
  if (isNaN(d.getTime())) return "—";
  // Guard against epoch (Jan 1 1970) from new Date(null)
  if (d.getFullYear() < 2000) return "—";
  return d.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type SortKey = "name-asc" | "name-desc" | "size-asc" | "size-desc" | "date-asc" | "date-desc";

const SORT_OPTIONS: { key: SortKey; label: string }[] = [
  { key: "name-asc", label: "Name A → Z" },
  { key: "name-desc", label: "Name Z → A" },
  { key: "size-desc", label: "Size (largest)" },
  { key: "size-asc", label: "Size (smallest)" },
  { key: "date-desc", label: "Newest first" },
  { key: "date-asc", label: "Oldest first" },
];

interface PullState {
  modelName: string;
  status: string;
  digest?: string;
  total?: number;
  completed?: number;
  error?: string;
  done: boolean;
}

// ---------------------------------------------------------------------------
// ModelSettingsPanel (inline in expanded ModelCard)
// ---------------------------------------------------------------------------

function ModelSettingsPanel({ modelName }: { modelName: string }) {
  const [settings, setSettings] = useState<ModelSettingsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [dirty, setDirty] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    getModelSettings(modelName)
      .then((data) => {
        if (!cancelled) {
          setSettings(data);
          setLoading(false);
        }
      })
      .catch((err) => {
        console.warn("Failed to load model settings:", err);
        if (!cancelled) {
          setSettings({ model: modelName });
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [modelName]);

  const handleChange = (field: keyof ModelSettingsData, value: string) => {
    if (!settings) return;
    setDirty(true);

    const numVal = value === "" ? undefined : Number(value);
    if (field === "system_prompt") {
      setSettings({ ...settings, [field]: value });
    } else {
      setSettings({ ...settings, [field]: numVal });
    }
  };

  const handleSave = async () => {
    if (!settings) return;
    setSaving(true);
    try {
      await updateModelSettings(modelName, settings);
      setDirty(false);
    } catch (err) {
      console.error("Failed to save model settings:", err);
    } finally {
      setSaving(false);
    }
  };

  const handleReset = async () => {
    setSaving(true);
    try {
      await deleteModelSettings(modelName);
      setSettings({ model: modelName });
      setDirty(false);
    } catch (err) {
      console.error("Failed to reset model settings:", err);
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="animate-pulse h-3 bg-neutral-200 dark:bg-neutral-700 rounded w-1/3 mt-2" />
    );
  }

  const inputClass =
    "w-full rounded-md border border-neutral-300 dark:border-neutral-600 bg-white dark:bg-neutral-900 px-2 py-1 text-xs text-neutral-900 dark:text-white focus:outline-none focus:ring-1 focus:ring-blue-500";

  return (
    <div className="mt-3 pt-3 border-t border-neutral-100 dark:border-neutral-700 space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-neutral-500 dark:text-neutral-400">
          Per-model settings
        </span>
        <div className="flex items-center gap-1.5">
          <button
            onClick={handleReset}
            disabled={saving}
            className="text-[10px] text-neutral-400 hover:text-neutral-600 dark:hover:text-neutral-200 disabled:opacity-50"
          >
            Reset
          </button>
          {dirty && (
            <button
              onClick={handleSave}
              disabled={saving}
              className="text-[10px] px-2 py-0.5 rounded bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50"
            >
              {saving ? "Saving..." : "Save"}
            </button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-x-3 gap-y-2">
        <div>
          <label className="block text-[10px] text-neutral-500 dark:text-neutral-400 mb-0.5">
            Temperature
          </label>
          <input
            type="number"
            step="0.1"
            min="0"
            max="2"
            value={settings?.temperature ?? ""}
            onChange={(e) => handleChange("temperature", e.target.value)}
            placeholder="default"
            className={inputClass}
          />
        </div>
        <div>
          <label className="block text-[10px] text-neutral-500 dark:text-neutral-400 mb-0.5">
            Context length
          </label>
          <input
            type="number"
            step="1024"
            min="0"
            value={settings?.context_length ?? ""}
            onChange={(e) => handleChange("context_length", e.target.value)}
            placeholder="default"
            className={inputClass}
          />
        </div>
        <div>
          <label className="block text-[10px] text-neutral-500 dark:text-neutral-400 mb-0.5">
            Top K
          </label>
          <input
            type="number"
            step="1"
            min="0"
            value={settings?.top_k ?? ""}
            onChange={(e) => handleChange("top_k", e.target.value)}
            placeholder="default"
            className={inputClass}
          />
        </div>
        <div>
          <label className="block text-[10px] text-neutral-500 dark:text-neutral-400 mb-0.5">
            Top P
          </label>
          <input
            type="number"
            step="0.05"
            min="0"
            max="1"
            value={settings?.top_p ?? ""}
            onChange={(e) => handleChange("top_p", e.target.value)}
            placeholder="default"
            className={inputClass}
          />
        </div>
      </div>

      <div>
        <label className="block text-[10px] text-neutral-500 dark:text-neutral-400 mb-0.5">
          System prompt
        </label>
        <textarea
          value={settings?.system_prompt ?? ""}
          onChange={(e) => handleChange("system_prompt", e.target.value)}
          placeholder="Override system prompt for this model..."
          rows={2}
          className={`${inputClass} resize-y`}
        />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// ModelCard
// ---------------------------------------------------------------------------

function ModelCard({
  model,
  onDelete,
  onCopy,
  isPulling,
  selected,
  onToggleSelect,
  batchMode,
}: {
  model: DetailedModel;
  onDelete: (name: string) => void;
  onCopy: (name: string) => void;
  isPulling: boolean;
  selected: boolean;
  onToggleSelect: (name: string) => void;
  batchMode: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const [details, setDetails] = useState<ShowModelResponse | null>(null);
  const [loadingDetails, setLoadingDetails] = useState(false);

  const handleExpand = useCallback(async () => {
    if (expanded) {
      setExpanded(false);
      return;
    }
    setExpanded(true);
    if (!details) {
      setLoadingDetails(true);
      try {
        const data = await showModel(model.model);
        setDetails(data);
      } catch (err) {
        console.warn(`Failed to load details for ${model.model}:`, err);
      } finally {
        setLoadingDetails(false);
      }
    }
  }, [expanded, details, model.model]);

  const handleDelete = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      onDelete(model.model);
    },
    [model.model, onDelete],
  );

  const handleCopy = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      onCopy(model.model);
    },
    [model.model, onCopy],
  );

  const handleCheckbox = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      onToggleSelect(model.model);
    },
    [model.model, onToggleSelect],
  );

  return (
    <div className="rounded-xl bg-white dark:bg-neutral-800 overflow-hidden">
      <div
        className="flex items-center p-4 cursor-pointer hover:bg-neutral-50 dark:hover:bg-neutral-700/50"
        onClick={handleExpand}
      >
        {/* Batch selection checkbox */}
        {batchMode && (
          <button
            onClick={handleCheckbox}
            className={`mr-3 flex-shrink-0 h-5 w-5 rounded border-2 flex items-center justify-center transition-colors ${
              selected
                ? "bg-blue-600 border-blue-600 text-white"
                : "border-neutral-300 dark:border-neutral-600 hover:border-blue-400"
            }`}
          >
            {selected && <CheckIcon className="h-3.5 w-3.5" />}
          </button>
        )}

        <CircleStackIcon className="h-5 w-5 flex-shrink-0 text-neutral-400 dark:text-neutral-500 mr-3" />
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-medium text-neutral-900 dark:text-white truncate">
            {model.model}
          </h3>
          <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-0.5">
            {model.size ? formatBytes(model.size) : ""}
            {model.size && model.modified_at ? " · " : ""}
            {model.modified_at ? `Modified ${formatDate(model.modified_at)}` : ""}
          </p>
        </div>
        <div className="flex items-center gap-1 ml-2">
          <button
            onClick={handleCopy}
            disabled={isPulling}
            className="p-1.5 rounded-lg text-neutral-400 hover:text-blue-500 hover:bg-blue-50 dark:hover:bg-blue-900/20 transition-colors disabled:opacity-50"
            title="Copy / Alias"
          >
            <DocumentDuplicateIcon className="h-4 w-4" />
          </button>
          <button
            onClick={handleDelete}
            disabled={isPulling}
            className="p-1.5 rounded-lg text-neutral-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors disabled:opacity-50"
            title="Delete model"
          >
            <TrashIcon className="h-4 w-4" />
          </button>
          {expanded ? (
            <ChevronUpIcon className="h-4 w-4 text-neutral-400" />
          ) : (
            <ChevronDownIcon className="h-4 w-4 text-neutral-400" />
          )}
        </div>
      </div>

      {expanded && (
        <div className="px-4 pb-4 border-t border-neutral-100 dark:border-neutral-700">
          <div className="pt-3 grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
            {model.digest && (
              <>
                <span className="text-neutral-500 dark:text-neutral-400">Digest</span>
                <span className="text-neutral-900 dark:text-neutral-200 font-mono truncate">
                  {model.digest.substring(0, 12)}
                </span>
              </>
            )}
            {(model.details?.family || details?.details?.family) && (
              <>
                <span className="text-neutral-500 dark:text-neutral-400">Family</span>
                <span className="text-neutral-900 dark:text-neutral-200">
                  {model.details?.family || details?.details?.family}
                </span>
              </>
            )}
            {(model.details?.parameter_size || details?.details?.parameter_size) && (
              <>
                <span className="text-neutral-500 dark:text-neutral-400">Parameters</span>
                <span className="text-neutral-900 dark:text-neutral-200">
                  {model.details?.parameter_size || details?.details?.parameter_size}
                </span>
              </>
            )}
            {(model.details?.quantization_level || details?.details?.quantization_level) && (
              <>
                <span className="text-neutral-500 dark:text-neutral-400">Quantization</span>
                <span className="text-neutral-900 dark:text-neutral-200">
                  {model.details?.quantization_level || details?.details?.quantization_level}
                </span>
              </>
            )}
            {(model.details?.format || details?.details?.format) && (
              <>
                <span className="text-neutral-500 dark:text-neutral-400">Format</span>
                <span className="text-neutral-900 dark:text-neutral-200">
                  {model.details?.format || details?.details?.format}
                </span>
              </>
            )}
            {loadingDetails && (
              <div className="col-span-2 mt-1">
                <div className="animate-pulse h-3 bg-neutral-200 dark:bg-neutral-700 rounded w-1/3" />
              </div>
            )}
            {details?.system && (
              <div className="col-span-2 mt-2">
                <span className="text-neutral-500 dark:text-neutral-400 block mb-1">
                  System prompt
                </span>
                <p className="text-neutral-900 dark:text-neutral-200 bg-neutral-50 dark:bg-neutral-900 rounded-lg p-2 text-xs whitespace-pre-wrap max-h-24 overflow-y-auto">
                  {details.system}
                </p>
              </div>
            )}
          </div>

          {/* Per-model settings */}
          <ModelSettingsPanel modelName={model.model} />
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Popular models list for Pull dialog
// ---------------------------------------------------------------------------

const POPULAR_MODELS = [
  { name: "llama3.2", desc: "Meta Llama 3.2", sizes: ["1b", "3b"] },
  { name: "gemma3", desc: "Google Gemma 3", sizes: ["1b", "4b", "12b", "27b"] },
  { name: "qwen3", desc: "Alibaba Qwen 3", sizes: ["0.6b", "1.7b", "4b", "8b", "14b", "30b"] },
  { name: "phi4", desc: "Microsoft Phi 4", sizes: ["14b"] },
  { name: "mistral", desc: "Mistral 7B", sizes: ["7b"] },
  { name: "deepseek-r1", desc: "DeepSeek R1", sizes: ["1.5b", "7b", "8b", "14b", "32b", "70b"] },
  { name: "llama3.3", desc: "Meta Llama 3.3", sizes: ["70b"] },
  { name: "nomic-embed-text", desc: "Nomic Embed", sizes: ["v1.5"] },
];

// ---------------------------------------------------------------------------
// Pull Dialog
// ---------------------------------------------------------------------------

function PullDialog({
  onClose,
  onPull,
  pullState,
  installedModels,
}: {
  onClose: () => void;
  onPull: (name: string) => void;
  pullState: PullState | null;
  installedModels: string[];
}) {
  const [modelName, setModelName] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const name = modelName.trim();
    if (name) {
      onPull(name);
    }
  };

  const handleSelectModel = (name: string) => {
    setModelName(name);
    inputRef.current?.focus();
  };

  const isPulling = pullState !== null && !pullState.done && !pullState.error;
  const progress =
    pullState?.total && pullState.total > 0
      ? Math.min(
          Math.round(((pullState.completed ?? 0) / pullState.total) * 100),
          100,
        )
      : 0;

  const filteredPopular = modelName.trim()
    ? POPULAR_MODELS.filter(
        (m) =>
          m.name.toLowerCase().includes(modelName.toLowerCase()) ||
          m.desc.toLowerCase().includes(modelName.toLowerCase()),
      )
    : POPULAR_MODELS;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-white dark:bg-neutral-800 rounded-2xl shadow-xl w-full max-w-md mx-4 overflow-hidden">
        <div className="p-4 border-b border-neutral-200 dark:border-neutral-700 flex items-center justify-between">
          <h2 className="text-sm font-medium text-neutral-900 dark:text-white">
            Pull Model
          </h2>
          <button
            onClick={onClose}
            disabled={isPulling}
            className="p-1 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-700 disabled:opacity-50"
          >
            <XMarkIcon className="h-5 w-5 text-neutral-500" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-4 space-y-4">
          <div>
            <label className="block text-xs font-medium text-neutral-500 dark:text-neutral-400 mb-1">
              Model name
            </label>
            <input
              ref={inputRef}
              type="text"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              placeholder="e.g. llama3.2, gemma3:4b, qwen3:8b"
              disabled={isPulling}
              className="w-full rounded-lg border border-neutral-300 dark:border-neutral-600 bg-white dark:bg-neutral-900 px-3 py-2 text-sm text-neutral-900 dark:text-white placeholder-neutral-400 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
            />
          </div>

          {/* Popular models list */}
          {!pullState && filteredPopular.length > 0 && (
            <div>
              <label className="block text-xs font-medium text-neutral-500 dark:text-neutral-400 mb-2">
                Popular models
              </label>
              <div className="max-h-48 overflow-y-auto space-y-1 rounded-lg border border-neutral-200 dark:border-neutral-700 p-1">
                {filteredPopular.map((m) => {
                  const isInstalled = installedModels.some(
                    (installed) =>
                      installed === m.name ||
                      installed.startsWith(m.name + ":"),
                  );
                  return (
                    <button
                      key={m.name}
                      type="button"
                      onClick={() => handleSelectModel(m.name)}
                      className="w-full text-left px-2.5 py-1.5 rounded-md hover:bg-neutral-100 dark:hover:bg-neutral-700 flex items-center justify-between gap-2 group"
                    >
                      <div className="min-w-0">
                        <span className="text-sm text-neutral-900 dark:text-white">
                          {m.name}
                        </span>
                        <span className="text-xs text-neutral-400 dark:text-neutral-500 ml-2">
                          {m.desc}
                        </span>
                      </div>
                      <div className="flex items-center gap-1.5 flex-shrink-0">
                        {isInstalled && (
                          <span className="text-[10px] text-green-600 dark:text-green-400 font-medium">
                            installed
                          </span>
                        )}
                        <span className="text-[10px] text-neutral-400 dark:text-neutral-500">
                          {m.sizes.join(", ")}
                        </span>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* Progress */}
          {pullState && (
            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <span className="text-neutral-500 dark:text-neutral-400 truncate flex-1">
                  {pullState.error
                    ? "Error"
                    : pullState.done
                      ? "Complete"
                      : pullState.status || "Pulling..."}
                </span>
                {pullState.total && pullState.total > 0 && !pullState.done && (
                  <span className="text-neutral-900 dark:text-neutral-200 ml-2">
                    {formatBytes(pullState.completed ?? 0)} /{" "}
                    {formatBytes(pullState.total)} ({progress}%)
                  </span>
                )}
              </div>
              {pullState.total && pullState.total > 0 && (
                <div className="h-2 w-full rounded-full bg-neutral-200 dark:bg-neutral-700">
                  <div
                    className={`h-2 rounded-full transition-all ${
                      pullState.error
                        ? "bg-red-500"
                        : pullState.done
                          ? "bg-green-500"
                          : "bg-blue-500"
                    }`}
                    style={{ width: `${progress}%` }}
                  />
                </div>
              )}
              {pullState.error && (
                <p className="text-xs text-red-500 dark:text-red-400">
                  {pullState.error}
                </p>
              )}
              {pullState.done && !pullState.error && (
                <p className="text-xs text-green-600 dark:text-green-400">
                  Model pulled successfully.
                </p>
              )}
            </div>
          )}

          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={onClose}
              disabled={isPulling}
              className="px-3 py-1.5 rounded-lg text-sm text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-700 disabled:opacity-50"
            >
              {pullState?.done ? "Close" : "Cancel"}
            </button>
            {!pullState?.done && (
              <button
                type="submit"
                disabled={!modelName.trim() || isPulling}
                className="px-3 py-1.5 rounded-lg text-sm bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isPulling ? "Pulling..." : "Pull"}
              </button>
            )}
          </div>
        </form>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Copy Dialog
// ---------------------------------------------------------------------------

function CopyDialog({
  sourceModel,
  onClose,
  onCopied,
}: {
  sourceModel: string;
  onClose: () => void;
  onCopied: () => void;
}) {
  const [newName, setNewName] = useState("");
  const [copying, setCopying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const name = newName.trim();
    if (!name) return;

    setCopying(true);
    setError(null);
    try {
      await copyModel(sourceModel, name);
      onCopied();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to copy model");
    } finally {
      setCopying(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-white dark:bg-neutral-800 rounded-2xl shadow-xl w-full max-w-sm mx-4 overflow-hidden">
        <div className="p-4 border-b border-neutral-200 dark:border-neutral-700 flex items-center justify-between">
          <h2 className="text-sm font-medium text-neutral-900 dark:text-white">
            Copy / Alias Model
          </h2>
          <button
            onClick={onClose}
            disabled={copying}
            className="p-1 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-700 disabled:opacity-50"
          >
            <XMarkIcon className="h-5 w-5 text-neutral-500" />
          </button>
        </div>
        <form onSubmit={handleSubmit} className="p-4 space-y-3">
          <div>
            <label className="block text-xs font-medium text-neutral-500 dark:text-neutral-400 mb-1">
              Source
            </label>
            <p className="text-sm text-neutral-900 dark:text-white font-mono bg-neutral-50 dark:bg-neutral-900 rounded-lg px-3 py-2">
              {sourceModel}
            </p>
          </div>
          <div>
            <label className="block text-xs font-medium text-neutral-500 dark:text-neutral-400 mb-1">
              New name
            </label>
            <input
              ref={inputRef}
              type="text"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="e.g. my-custom-model"
              disabled={copying}
              className="w-full rounded-lg border border-neutral-300 dark:border-neutral-600 bg-white dark:bg-neutral-900 px-3 py-2 text-sm text-neutral-900 dark:text-white placeholder-neutral-400 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
            />
          </div>
          {error && (
            <p className="text-xs text-red-500 dark:text-red-400">{error}</p>
          )}
          <div className="flex justify-end gap-2 pt-1">
            <button
              type="button"
              onClick={onClose}
              disabled={copying}
              className="px-3 py-1.5 rounded-lg text-sm text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-700 disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!newName.trim() || copying}
              className="px-3 py-1.5 rounded-lg text-sm bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {copying ? "Copying..." : "Copy"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Batch Action Bar
// ---------------------------------------------------------------------------

function BatchActionBar({
  count,
  onDelete,
  onCancel,
  deleting,
}: {
  count: number;
  onDelete: () => void;
  onCancel: () => void;
  deleting: boolean;
}) {
  return (
    <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-40 flex items-center gap-3 rounded-2xl bg-neutral-900 dark:bg-neutral-700 shadow-xl px-5 py-3 text-white">
      <span className="text-sm font-medium">
        {count} model{count !== 1 ? "s" : ""} selected
      </span>
      <div className="h-4 w-px bg-neutral-600" />
      <button
        onClick={onDelete}
        disabled={deleting}
        className="flex items-center gap-1.5 text-sm text-red-400 hover:text-red-300 disabled:opacity-50"
      >
        <TrashIcon className="h-4 w-4" />
        {deleting ? "Deleting..." : "Delete"}
      </button>
      <button
        onClick={onCancel}
        disabled={deleting}
        className="text-sm text-neutral-400 hover:text-white disabled:opacity-50"
      >
        Cancel
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sort Dropdown
// ---------------------------------------------------------------------------

function SortDropdown({
  value,
  onChange,
}: {
  value: SortKey;
  onChange: (key: SortKey) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const current = SORT_OPTIONS.find((o) => o.key === value);

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs text-neutral-600 dark:text-neutral-300 border border-neutral-200 dark:border-neutral-700 hover:bg-neutral-50 dark:hover:bg-neutral-800"
      >
        <ChevronUpDownIcon className="h-3.5 w-3.5" />
        {current?.label ?? "Sort"}
      </button>
      {open && (
        <div className="absolute right-0 top-full mt-1 w-40 rounded-xl bg-white dark:bg-neutral-800 shadow-lg border border-neutral-200 dark:border-neutral-700 py-1 z-50">
          {SORT_OPTIONS.map((opt) => (
            <button
              key={opt.key}
              onClick={() => {
                onChange(opt.key);
                setOpen(false);
              }}
              className={`w-full text-left px-3 py-1.5 text-xs hover:bg-neutral-100 dark:hover:bg-neutral-700 ${
                opt.key === value
                  ? "text-blue-600 dark:text-blue-400 font-medium"
                  : "text-neutral-700 dark:text-neutral-300"
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sorting helper
// ---------------------------------------------------------------------------

function sortModels(models: DetailedModel[], key: SortKey): DetailedModel[] {
  const sorted = [...models];
  switch (key) {
    case "name-asc":
      return sorted.sort((a, b) => a.model.localeCompare(b.model));
    case "name-desc":
      return sorted.sort((a, b) => b.model.localeCompare(a.model));
    case "size-desc":
      return sorted.sort((a, b) => (b.size || 0) - (a.size || 0));
    case "size-asc":
      return sorted.sort((a, b) => (a.size || 0) - (b.size || 0));
    case "date-desc":
      return sorted.sort(
        (a, b) =>
          (b.modified_at?.getTime?.() ?? 0) - (a.modified_at?.getTime?.() ?? 0),
      );
    case "date-asc":
      return sorted.sort(
        (a, b) =>
          (a.modified_at?.getTime?.() ?? 0) - (b.modified_at?.getTime?.() ?? 0),
      );
    default:
      return sorted;
  }
}

// ---------------------------------------------------------------------------
// Main ModelManager
// ---------------------------------------------------------------------------

export default function ModelManager() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const isWindows = navigator.platform.toLowerCase().includes("win");

  const [search, setSearch] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("name-asc");
  const [showPullDialog, setShowPullDialog] = useState(false);
  const [pullState, setPullState] = useState<PullState | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Batch selection
  const [selectedModels, setSelectedModels] = useState<Set<string>>(new Set());
  const [batchDeleting, setBatchDeleting] = useState(false);
  const batchMode = selectedModels.size > 0;

  // Copy dialog
  const [copySource, setCopySource] = useState<string | null>(null);

  const {
    data: models,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["localModels"],
    queryFn: () => getModelsDetailed(),
  });

  const filteredAndSorted = useMemo(() => {
    if (!models) return [];
    const filtered = search
      ? models.filter((m) => m.model.toLowerCase().includes(search.toLowerCase()))
      : models;
    return sortModels(filtered, sortKey);
  }, [models, search, sortKey]);

  // Clear selection when models change
  useEffect(() => {
    setSelectedModels((prev) => {
      if (!models) return new Set();
      const modelNames = new Set(models.map((m) => m.model));
      const next = new Set<string>();
      for (const name of prev) {
        if (modelNames.has(name)) next.add(name);
      }
      return next;
    });
  }, [models]);

  const toggleSelect = useCallback((name: string) => {
    setSelectedModels((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }, []);

  const handleDelete = useCallback(
    async (modelName: string) => {
      const confirmed = window.confirm(
        `Are you sure you want to delete "${modelName}"? This cannot be undone.`,
      );
      if (!confirmed) return;

      try {
        await deleteModel(modelName);
        queryClient.invalidateQueries({ queryKey: ["localModels"] });
        queryClient.invalidateQueries({ queryKey: ["models"] });
      } catch (err) {
        console.error("Failed to delete model:", err);
        window.alert(
          `Failed to delete model: ${err instanceof Error ? err.message : "Unknown error"}`,
        );
      }
    },
    [queryClient],
  );

  const handleBatchDelete = useCallback(async () => {
    const names = Array.from(selectedModels);
    const confirmed = window.confirm(
      `Are you sure you want to delete ${names.length} model${names.length !== 1 ? "s" : ""}?\n\n${names.join("\n")}\n\nThis cannot be undone.`,
    );
    if (!confirmed) return;

    setBatchDeleting(true);
    const errors: string[] = [];
    for (const name of names) {
      try {
        await deleteModel(name);
      } catch (err) {
        errors.push(`${name}: ${err instanceof Error ? err.message : "Unknown error"}`);
      }
    }
    setBatchDeleting(false);
    setSelectedModels(new Set());
    queryClient.invalidateQueries({ queryKey: ["localModels"] });
    queryClient.invalidateQueries({ queryKey: ["models"] });

    if (errors.length > 0) {
      window.alert(`Some models failed to delete:\n\n${errors.join("\n")}`);
    }
  }, [selectedModels, queryClient]);

  const handleCancelBatch = useCallback(() => {
    setSelectedModels(new Set());
  }, []);

  const handleCopy = useCallback((name: string) => {
    setCopySource(name);
  }, []);

  const handleCopied = useCallback(() => {
    setCopySource(null);
    queryClient.invalidateQueries({ queryKey: ["localModels"] });
    queryClient.invalidateQueries({ queryKey: ["models"] });
  }, [queryClient]);

  const handlePull = useCallback(
    async (modelName: string) => {
      setPullState({
        modelName,
        status: "Starting...",
        done: false,
      });

      const controller = new AbortController();
      abortControllerRef.current = controller;

      try {
        for await (const event of pullModel(modelName, controller.signal)) {
          // Handle server-streamed error events
          if ((event as any).error) {
            throw new Error((event as any).error);
          }
          setPullState((prev) => ({
            ...prev!,
            status: event.status,
            digest: event.digest,
            total: event.total ?? prev?.total,
            completed: event.completed ?? prev?.completed,
          }));
        }
        setPullState((prev) => ({
          ...prev!,
          done: true,
          status: "Complete",
        }));
        queryClient.invalidateQueries({ queryKey: ["localModels"] });
        queryClient.invalidateQueries({ queryKey: ["models"] });
      } catch (err) {
        if (controller.signal.aborted) {
          setPullState((prev) => prev ? ({ ...prev, done: true, status: "Cancelled" }) : null);
          return;
        }
        setPullState((prev) => ({
          ...prev!,
          error: err instanceof Error ? err.message : "Unknown error",
          done: true,
        }));
      } finally {
        abortControllerRef.current = null;
      }
    },
    [queryClient],
  );

  const handleClosePullDialog = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setShowPullDialog(false);
    setPullState(null);
  }, []);

  const isPulling = pullState !== null && !pullState.done && !pullState.error;

  // Select all visible models
  const handleSelectAll = useCallback(() => {
    const visibleNames = filteredAndSorted.map((m) => m.model);
    setSelectedModels((prev) => {
      const allSelected = visibleNames.every((n) => prev.has(n));
      if (allSelected) {
        // Deselect all
        return new Set();
      }
      // Select all visible
      return new Set(visibleNames);
    });
  }, [filteredAndSorted]);

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
          Models
        </h1>
        <div className="flex items-center gap-2 pr-4">
          <button
            onClick={() => setShowPullDialog(true)}
            disabled={isPulling}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50"
          >
            <ArrowDownTrayIcon className="h-4 w-4" />
            Pull
          </button>
          {!isWindows && (
            <button
              onClick={() => navigate({ to: "/" })}
              className="p-1 hover:bg-neutral-100 dark:hover:bg-neutral-800 rounded-full"
            >
              <XMarkIcon className="w-6 h-6 dark:text-white" />
            </button>
          )}
        </div>
      </header>

      <div className="w-full p-6 overflow-y-auto flex-1 overscroll-contain">
        <div className="space-y-4 max-w-2xl mx-auto">
          {/* Search + Sort + Batch select controls */}
          <div className="flex items-center gap-2">
            <div className="relative flex-1">
              <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-neutral-400" />
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search models..."
                className="w-full rounded-xl border border-neutral-200 dark:border-neutral-700 bg-white dark:bg-neutral-800 pl-9 pr-4 py-2 text-sm text-neutral-900 dark:text-white placeholder-neutral-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <SortDropdown value={sortKey} onChange={setSortKey} />
            {filteredAndSorted.length > 1 && (
              <button
                onClick={handleSelectAll}
                className="px-2.5 py-1.5 rounded-lg text-xs text-neutral-600 dark:text-neutral-300 border border-neutral-200 dark:border-neutral-700 hover:bg-neutral-50 dark:hover:bg-neutral-800"
              >
                {filteredAndSorted.length > 0 &&
                filteredAndSorted.every((m) => selectedModels.has(m.model))
                  ? "Deselect all"
                  : "Select all"}
              </button>
            )}
          </div>

          {/* Model list */}
          {isLoading ? (
            <div className="space-y-3">
              {[1, 2, 3].map((i) => (
                <div
                  key={i}
                  className="rounded-xl bg-white dark:bg-neutral-800 p-4"
                >
                  <div className="animate-pulse flex items-center gap-3">
                    <div className="h-5 w-5 bg-neutral-200 dark:bg-neutral-700 rounded" />
                    <div className="flex-1 space-y-2">
                      <div className="h-4 bg-neutral-200 dark:bg-neutral-700 rounded w-1/3" />
                      <div className="h-3 bg-neutral-200 dark:bg-neutral-700 rounded w-1/4" />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : error ? (
            <div className="rounded-xl bg-white dark:bg-neutral-800 p-4 text-sm text-red-500 dark:text-red-400">
              Failed to load models.
            </div>
          ) : filteredAndSorted.length > 0 ? (
            <div className="space-y-2">
              <p className="text-xs text-neutral-500 dark:text-neutral-400 px-1">
                {filteredAndSorted.length} model
                {filteredAndSorted.length !== 1 ? "s" : ""}
                {search ? " found" : " installed"}
              </p>
              {filteredAndSorted.map((model) => (
                <ModelCard
                  key={model.model}
                  model={model}
                  onDelete={handleDelete}
                  onCopy={handleCopy}
                  isPulling={isPulling}
                  selected={selectedModels.has(model.model)}
                  onToggleSelect={toggleSelect}
                  batchMode={batchMode}
                />
              ))}
            </div>
          ) : models && models.length > 0 ? (
            <div className="rounded-xl bg-white dark:bg-neutral-800 p-4 text-sm text-neutral-500 dark:text-neutral-400">
              No models match &quot;{search}&quot;
            </div>
          ) : (
            <div className="rounded-xl bg-white dark:bg-neutral-800 p-6 text-center">
              <CircleStackIcon className="h-8 w-8 text-neutral-300 dark:text-neutral-600 mx-auto mb-3" />
              <p className="text-sm text-neutral-500 dark:text-neutral-400">
                No models installed yet.
              </p>
              <button
                onClick={() => setShowPullDialog(true)}
                className="mt-3 text-sm text-blue-600 dark:text-blue-400 hover:underline"
              >
                Pull your first model
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Batch action bar */}
      {batchMode && (
        <BatchActionBar
          count={selectedModels.size}
          onDelete={handleBatchDelete}
          onCancel={handleCancelBatch}
          deleting={batchDeleting}
        />
      )}

      {/* Pull dialog */}
      {showPullDialog && (
        <PullDialog
          onClose={handleClosePullDialog}
          onPull={handlePull}
          pullState={pullState}
          installedModels={models?.map((m) => m.model) ?? []}
        />
      )}

      {/* Copy dialog */}
      {copySource && (
        <CopyDialog
          sourceModel={copySource}
          onClose={() => setCopySource(null)}
          onCopied={handleCopied}
        />
      )}
    </main>
  );
}
