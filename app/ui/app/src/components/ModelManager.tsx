import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  ArrowDownTrayIcon,
  TrashIcon,
  ChartBarIcon,
  XMarkIcon,
  CheckCircleIcon,
} from "@heroicons/react/20/solid";

interface ModelInfo {
  id: string;
  name: string;
  size: number;
  description: string;
  provider: string;
  version: string;
  downloaded: boolean;
  downloaded_at?: string;
  last_used?: string;
  usage_count: number;
  local_path?: string;
  required_ram: number;
  context_window: number;
  capabilities: string[];
}

interface DownloadProgress {
  model_id: string;
  status: string;
  bytes_downloaded: number;
  total_bytes: number;
  percentage: number;
  speed: number;
  eta: number;
  error?: string;
}

const formatBytes = (bytes: number): string => {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + " " + sizes[i];
};

export default function ModelManager() {
  const queryClient = useQueryClient();
  const [selectedTab, setSelectedTab] = useState<"available" | "local">("available");
  const [downloadingModels, setDownloadingModels] = useState<Set<string>>(new Set());

  const { data: availableModels, isLoading: loadingAvailable } = useQuery({
    queryKey: ["models", "available"],
    queryFn: async () => {
      const res = await fetch("/api/models/available");
      if (!res.ok) throw new Error("Failed to fetch available models");
      const data = await res.json();
      return data.models as ModelInfo[];
    },
  });

  const { data: localModels, isLoading: loadingLocal } = useQuery({
    queryKey: ["models", "local"],
    queryFn: async () => {
      const res = await fetch("/api/models/local");
      if (!res.ok) throw new Error("Failed to fetch local models");
      const data = await res.json();
      return data.models as ModelInfo[];
    },
  });

  const { data: storageStats } = useQuery({
    queryKey: ["models", "storage"],
    queryFn: async () => {
      const res = await fetch("/api/models/storage");
      if (!res.ok) throw new Error("Failed to fetch storage stats");
      return res.json();
    },
  });

  const downloadMutation = useMutation({
    mutationFn: async (modelId: string) => {
      const res = await fetch("/api/models/download", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_id: modelId,
          download_url: `https://models.ollama.ai/${modelId}`,
        }),
      });
      if (!res.ok) throw new Error("Failed to start download");
      return res.json();
    },
    onSuccess: (_, modelId) => {
      setDownloadingModels(prev => new Set(prev).add(modelId));
      // Start polling for progress
      const interval = setInterval(async () => {
        const res = await fetch(`/api/models/${modelId}/progress`);
        if (res.ok) {
          const progress: DownloadProgress = await res.json();
          if (progress.status === "completed" || progress.status === "failed") {
            clearInterval(interval);
            setDownloadingModels(prev => {
              const next = new Set(prev);
              next.delete(modelId);
              return next;
            });
            queryClient.invalidateQueries({ queryKey: ["models"] });
          }
        }
      }, 1000);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: async (modelId: string) => {
      const res = await fetch(`/api/models/${modelId}`, {
        method: "DELETE",
      });
      if (!res.ok) throw new Error("Failed to delete model");
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["models"] });
    },
  });

  const models = selectedTab === "available" ? availableModels : localModels;
  const isLoading = selectedTab === "available" ? loadingAvailable : loadingLocal;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-white">Model Management</h3>
          <p className="text-sm text-gray-400">
            Download, manage, and compare AI models
          </p>
        </div>
        {storageStats && (
          <div className="text-right">
            <div className="text-sm text-gray-400">Storage Used</div>
            <div className="text-lg font-semibold text-white">
              {formatBytes(storageStats.total_size)} / {formatBytes(storageStats.disk_total)}
            </div>
            <div className="text-xs text-gray-500">
              {storageStats.disk_usage_pct.toFixed(1)}% used
            </div>
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-gray-700">
        <button
          onClick={() => setSelectedTab("available")}
          className={`px-4 py-2 text-sm font-medium transition-colors ${
            selectedTab === "available"
              ? "text-white border-b-2 border-blue-500"
              : "text-gray-400 hover:text-gray-300"
          }`}
        >
          Available Models
        </button>
        <button
          onClick={() => setSelectedTab("local")}
          className={`px-4 py-2 text-sm font-medium transition-colors ${
            selectedTab === "local"
              ? "text-white border-b-2 border-blue-500"
              : "text-gray-400 hover:text-gray-300"
          }`}
        >
          Downloaded Models ({localModels?.length || 0})
        </button>
      </div>

      {/* Model List */}
      {isLoading ? (
        <div className="text-center py-8 text-gray-400">Loading models...</div>
      ) : models && models.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {models.map((model) => (
            <div
              key={model.id}
              className="rounded-lg border border-gray-700 bg-gray-800 p-4 space-y-3"
            >
              {/* Header */}
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <h4 className="font-medium text-white">{model.name}</h4>
                    <Badge>{model.provider}</Badge>
                  </div>
                  <p className="text-sm text-gray-400">{model.description}</p>
                </div>
              </div>

              {/* Stats */}
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>
                  <span className="text-gray-500">Size:</span>
                  <span className="text-white ml-1">{formatBytes(model.size)}</span>
                </div>
                <div>
                  <span className="text-gray-500">RAM:</span>
                  <span className="text-white ml-1">{formatBytes(model.required_ram)}</span>
                </div>
                <div>
                  <span className="text-gray-500">Context:</span>
                  <span className="text-white ml-1">{model.context_window.toLocaleString()}</span>
                </div>
                <div>
                  <span className="text-gray-500">Version:</span>
                  <span className="text-white ml-1">{model.version}</span>
                </div>
              </div>

              {/* Capabilities */}
              {model.capabilities && model.capabilities.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {model.capabilities.map((cap) => (
                    <span
                      key={cap}
                      className="text-xs bg-gray-700 px-2 py-1 rounded"
                    >
                      {cap}
                    </span>
                  ))}
                </div>
              )}

              {/* Actions */}
              <div className="flex gap-2 pt-2 border-t border-gray-700">
                {model.downloaded ? (
                  <>
                    <div className="flex items-center gap-2 flex-1 text-xs text-green-500">
                      <CheckCircleIcon className="h-4 w-4" />
                      <span>Downloaded</span>
                      {model.usage_count > 0 && (
                        <span className="text-gray-400">
                          • Used {model.usage_count} times
                        </span>
                      )}
                    </div>
                    <Button
                      onClick={() => deleteMutation.mutate(model.id)}
                      disabled={deleteMutation.isPending}
                      variant="ghost"
                      size="sm"
                      className="text-red-500 hover:text-red-400"
                    >
                      <TrashIcon className="h-4 w-4" />
                    </Button>
                  </>
                ) : downloadingModels.has(model.id) ? (
                  <div className="flex items-center gap-2 flex-1 text-xs text-blue-500">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                    <span>Downloading...</span>
                  </div>
                ) : (
                  <Button
                    onClick={() => downloadMutation.mutate(model.id)}
                    disabled={downloadMutation.isPending}
                    size="sm"
                    className="flex items-center gap-2"
                  >
                    <ArrowDownTrayIcon className="h-4 w-4" />
                    Download
                  </Button>
                )}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center py-8 text-gray-400">
          {selectedTab === "available" ? "No models available" : "No models downloaded"}
        </div>
      )}
    </div>
  );
}
