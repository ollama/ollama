import { useState, useCallback, useRef, useEffect } from "react";
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
} from "@heroicons/react/20/solid";
import {
  getModelsDetailed,
  deleteModel,
  pullModel,
  showModel,
  type ShowModelResponse,
  type DetailedModel,
} from "@/api";

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
  // Model.modified_at may be a Time object (empty class), a Date, or a string
  const d = date instanceof Date ? date : new Date(String(date));
  if (isNaN(d.getTime())) return "—";
  return d.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

interface PullState {
  modelName: string;
  status: string;
  digest?: string;
  total?: number;
  completed?: number;
  error?: string;
  done: boolean;
}

function ModelCard({
  model,
  onDelete,
  isPulling,
}: {
  model: DetailedModel;
  onDelete: (name: string) => void;
  isPulling: boolean;
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
      } catch {
        // silently fail - details section just won't show extra info
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

  return (
    <div className="rounded-xl bg-white dark:bg-neutral-800 overflow-hidden">
      <div
        className="flex items-center p-4 cursor-pointer hover:bg-neutral-50 dark:hover:bg-neutral-700/50"
        onClick={handleExpand}
      >
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
        <div className="flex items-center gap-2 ml-2">
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
                <span className="text-neutral-500 dark:text-neutral-400">
                  Digest
                </span>
                <span className="text-neutral-900 dark:text-neutral-200 font-mono truncate">
                  {model.digest.substring(0, 12)}
                </span>
              </>
            )}
            {(model.details?.family || details?.details?.family) && (
              <>
                <span className="text-neutral-500 dark:text-neutral-400">
                  Family
                </span>
                <span className="text-neutral-900 dark:text-neutral-200">
                  {model.details?.family || details?.details?.family}
                </span>
              </>
            )}
            {(model.details?.parameter_size || details?.details?.parameter_size) && (
              <>
                <span className="text-neutral-500 dark:text-neutral-400">
                  Parameters
                </span>
                <span className="text-neutral-900 dark:text-neutral-200">
                  {model.details?.parameter_size || details?.details?.parameter_size}
                </span>
              </>
            )}
            {(model.details?.quantization_level || details?.details?.quantization_level) && (
              <>
                <span className="text-neutral-500 dark:text-neutral-400">
                  Quantization
                </span>
                <span className="text-neutral-900 dark:text-neutral-200">
                  {model.details?.quantization_level || details?.details?.quantization_level}
                </span>
              </>
            )}
            {(model.details?.format || details?.details?.format) && (
              <>
                <span className="text-neutral-500 dark:text-neutral-400">
                  Format
                </span>
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
        </div>
      )}
    </div>
  );
}

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
          Math.round((pullState.completed ?? 0) / pullState.total * 100),
          100,
        )
      : 0;

  // Filter popular models based on search input
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

export default function ModelManager() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const isWindows = navigator.platform.toLowerCase().includes("win");
  const [search, setSearch] = useState("");
  const [showPullDialog, setShowPullDialog] = useState(false);
  const [pullState, setPullState] = useState<PullState | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const {
    data: models,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["localModels"],
    queryFn: () => getModelsDetailed(),
  });

  const filteredModels = models?.filter((m) =>
    m.model.toLowerCase().includes(search.toLowerCase()),
  );

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
        if (controller.signal.aborted) return;
        setPullState((prev) => ({
          ...prev!,
          error:
            err instanceof Error ? err.message : "Unknown error",
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
          {/* Search */}
          <div className="relative">
            <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-neutral-400" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search models..."
              className="w-full rounded-xl border border-neutral-200 dark:border-neutral-700 bg-white dark:bg-neutral-800 pl-9 pr-4 py-2 text-sm text-neutral-900 dark:text-white placeholder-neutral-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
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
          ) : filteredModels && filteredModels.length > 0 ? (
            <div className="space-y-2">
              <p className="text-xs text-neutral-500 dark:text-neutral-400 px-1">
                {filteredModels.length} model
                {filteredModels.length !== 1 ? "s" : ""} installed
              </p>
              {filteredModels.map((model) => (
                <ModelCard
                  key={model.model}
                  model={model}
                  onDelete={handleDelete}
                  isPulling={isPulling}
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

      {showPullDialog && (
        <PullDialog
          onClose={handleClosePullDialog}
          onPull={handlePull}
          pullState={pullState}
          installedModels={models?.map((m) => m.model) ?? []}
        />
      )}
    </main>
  );
}
