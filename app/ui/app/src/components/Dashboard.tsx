import { useNavigate } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import {
  ArrowLeftIcon,
  XMarkIcon,
  CpuChipIcon,
  ServerStackIcon,
  ClockIcon,
  CircleStackIcon,
} from "@heroicons/react/20/solid";
import {
  getInferenceCompute,
  getRunningModels,
  type ProcessModelResponse,
} from "@/api";

function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
}

function formatTimeRemaining(expiresAt: string): string {
  const expires = new Date(expiresAt);
  const now = new Date();
  const diff = expires.getTime() - now.getTime();
  if (diff <= 0) return "Expiring...";
  const minutes = Math.floor(diff / 60000);
  if (minutes < 60) return `${minutes}m remaining`;
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  return `${hours}h ${remainingMinutes}m remaining`;
}

function GpuCard({
  gpu,
}: {
  gpu: {
    library: string;
    variant: string;
    compute: string;
    driver: string;
    name: string;
    vram: string;
  };
}) {
  return (
    <div className="rounded-xl bg-white p-4 dark:bg-neutral-800">
      <div className="flex items-start space-x-3">
        <CpuChipIcon className="mt-0.5 h-5 w-5 flex-shrink-0 text-blue-500 dark:text-blue-400" />
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-medium text-neutral-900 dark:text-white truncate">
            {gpu.name}
          </h3>
          <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1">
            <div className="text-xs text-neutral-500 dark:text-neutral-400">
              VRAM
            </div>
            <div className="text-xs text-neutral-900 dark:text-neutral-200">
              {gpu.vram}
            </div>
            <div className="text-xs text-neutral-500 dark:text-neutral-400">
              Library
            </div>
            <div className="text-xs text-neutral-900 dark:text-neutral-200">
              {gpu.library}
            </div>
            {gpu.compute && (
              <>
                <div className="text-xs text-neutral-500 dark:text-neutral-400">
                  Compute
                </div>
                <div className="text-xs text-neutral-900 dark:text-neutral-200">
                  {gpu.compute}
                </div>
              </>
            )}
            {gpu.driver && (
              <>
                <div className="text-xs text-neutral-500 dark:text-neutral-400">
                  Driver
                </div>
                <div className="text-xs text-neutral-900 dark:text-neutral-200">
                  {gpu.driver}
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function ModelCard({ model }: { model: ProcessModelResponse }) {
  const vramPercent =
    model.size > 0 ? Math.round((model.size_vram / model.size) * 100) : 0;

  return (
    <div className="rounded-xl bg-white p-4 dark:bg-neutral-800">
      <div className="flex items-start space-x-3">
        <ServerStackIcon className="mt-0.5 h-5 w-5 flex-shrink-0 text-green-500 dark:text-green-400" />
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-medium text-neutral-900 dark:text-white truncate">
            {model.name}
          </h3>
          <div className="mt-2 space-y-2">
            {/* VRAM usage bar */}
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-neutral-500 dark:text-neutral-400">
                  GPU offload
                </span>
                <span className="text-neutral-900 dark:text-neutral-200">
                  {formatBytes(model.size_vram)} / {formatBytes(model.size)} ({vramPercent}%)
                </span>
              </div>
              <div className="h-1.5 w-full rounded-full bg-neutral-200 dark:bg-neutral-700">
                <div
                  className="h-1.5 rounded-full bg-green-500 dark:bg-green-400 transition-all"
                  style={{ width: `${vramPercent}%` }}
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-x-4 gap-y-1">
              <div className="text-xs text-neutral-500 dark:text-neutral-400">
                Context
              </div>
              <div className="text-xs text-neutral-900 dark:text-neutral-200">
                {model.context_length > 0
                  ? model.context_length.toLocaleString()
                  : "—"}
              </div>
              {model.details && (
                <>
                  <div className="text-xs text-neutral-500 dark:text-neutral-400">
                    Parameters
                  </div>
                  <div className="text-xs text-neutral-900 dark:text-neutral-200">
                    {model.details.parameter_size || "—"}
                  </div>
                  <div className="text-xs text-neutral-500 dark:text-neutral-400">
                    Quantization
                  </div>
                  <div className="text-xs text-neutral-900 dark:text-neutral-200">
                    {model.details.quantization_level || "—"}
                  </div>
                </>
              )}
            </div>

            {/* Expiry */}
            <div className="flex items-center space-x-1 text-xs text-neutral-500 dark:text-neutral-400">
              <ClockIcon className="h-3.5 w-3.5" />
              <span>{formatTimeRemaining(model.expires_at)}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function Dashboard() {
  const navigate = useNavigate();
  const isWindows = navigator.platform.toLowerCase().includes("win");

  const { data: inferenceData, isLoading: gpuLoading } = useQuery({
    queryKey: ["inferenceCompute"],
    queryFn: getInferenceCompute,
  });

  const { data: processData, isLoading: modelsLoading } = useQuery({
    queryKey: ["runningModels"],
    queryFn: getRunningModels,
    refetchInterval: 5000,
  });

  const gpus = inferenceData?.inferenceComputes ?? [];
  const models = processData?.models ?? [];

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
          Dashboard
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
        <div className="space-y-6 max-w-2xl mx-auto">
          {/* GPU Section */}
          <section>
            <div className="flex items-center space-x-2 mb-3">
              <CpuChipIcon className="h-4 w-4 text-neutral-500 dark:text-neutral-400" />
              <h2 className="text-sm font-medium text-neutral-500 dark:text-neutral-400 uppercase tracking-wide">
                GPU Devices
              </h2>
            </div>
            {gpuLoading ? (
              <div className="rounded-xl bg-white dark:bg-neutral-800 p-4">
                <div className="animate-pulse space-y-2">
                  <div className="h-4 bg-neutral-200 dark:bg-neutral-700 rounded w-1/3" />
                  <div className="h-3 bg-neutral-200 dark:bg-neutral-700 rounded w-1/2" />
                </div>
              </div>
            ) : gpus.length > 0 ? (
              <div className="space-y-3">
                {gpus.map((gpu, i) => (
                  <GpuCard key={i} gpu={gpu} />
                ))}
              </div>
            ) : (
              <div className="rounded-xl bg-white dark:bg-neutral-800 p-4 text-sm text-neutral-500 dark:text-neutral-400">
                No GPU devices detected — using CPU.
              </div>
            )}
          </section>

          {/* Running Models Section */}
          <section>
            <div className="flex items-center space-x-2 mb-3">
              <CircleStackIcon className="h-4 w-4 text-neutral-500 dark:text-neutral-400" />
              <h2 className="text-sm font-medium text-neutral-500 dark:text-neutral-400 uppercase tracking-wide">
                Loaded Models
                {models.length > 0 && (
                  <span className="ml-2 inline-flex items-center justify-center rounded-full bg-green-100 dark:bg-green-900/30 px-2 py-0.5 text-xs font-medium text-green-700 dark:text-green-400">
                    {models.length}
                  </span>
                )}
              </h2>
            </div>
            {modelsLoading ? (
              <div className="rounded-xl bg-white dark:bg-neutral-800 p-4">
                <div className="animate-pulse space-y-2">
                  <div className="h-4 bg-neutral-200 dark:bg-neutral-700 rounded w-1/3" />
                  <div className="h-3 bg-neutral-200 dark:bg-neutral-700 rounded w-1/2" />
                </div>
              </div>
            ) : models.length > 0 ? (
              <div className="space-y-3">
                {models.map((model, i) => (
                  <ModelCard key={model.digest || i} model={model} />
                ))}
              </div>
            ) : (
              <div className="rounded-xl bg-white dark:bg-neutral-800 p-4 text-sm text-neutral-500 dark:text-neutral-400">
                No models currently loaded. Models are loaded when you start a
                conversation.
              </div>
            )}
          </section>

          {/* System Info Summary */}
          {inferenceData && (
            <section>
              <div className="rounded-xl bg-white dark:bg-neutral-800 p-4">
                <div className="flex items-center space-x-2 mb-2">
                  <ServerStackIcon className="h-4 w-4 text-neutral-500 dark:text-neutral-400" />
                  <h3 className="text-sm font-medium text-neutral-500 dark:text-neutral-400">
                    System Info
                  </h3>
                </div>
                <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                  <span className="text-neutral-500 dark:text-neutral-400">
                    Default context length
                  </span>
                  <span className="text-neutral-900 dark:text-neutral-200">
                    {inferenceData.defaultContextLength.toLocaleString()}
                  </span>
                  <span className="text-neutral-500 dark:text-neutral-400">
                    GPU count
                  </span>
                  <span className="text-neutral-900 dark:text-neutral-200">
                    {gpus.length}
                  </span>
                  <span className="text-neutral-500 dark:text-neutral-400">
                    Loaded models
                  </span>
                  <span className="text-neutral-900 dark:text-neutral-200">
                    {models.length}
                  </span>
                </div>
              </div>
            </section>
          )}
        </div>
      </div>
    </main>
  );
}
