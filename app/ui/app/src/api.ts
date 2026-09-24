import {
  ChatResponse,
  ChatsResponse,
  InferenceComputeResponse,
  Model,
  Settings,
  User,
} from "@/gotypes";
import { ollamaClient as ollama } from "./lib/ollama-client";
import type { ModelResponse } from "ollama/browser";
import { API_BASE, OLLAMA_DOT_COM } from "./lib/config";

// Extend Model class with utility methods
declare module "@/gotypes" {
  interface Model {
    isCloud(): boolean;
  }
}

Model.prototype.isCloud = function (): boolean {
  return this.model.endsWith("cloud");
};

export type CloudStatusSource = "env" | "config" | "both" | "none";
export interface CloudStatusResponse {
  disabled: boolean;
  source: CloudStatusSource;
}

export interface IntegrationStatus {
  id: string;
  name: string;
  description: string;
  installed?: boolean;
  command?: string;
}

export type IntegrationStatuses = IntegrationStatus[];

export async function getIntegrationStatuses(): Promise<IntegrationStatuses> {
  const response = await fetch(`${API_BASE}/api/v1/integrations`);
  if (!response.ok) {
    throw new Error(`Failed to fetch integration statuses: ${response.status}`);
  }
  return response.json();
}

export async function fetchUser(): Promise<User | null> {
  const response = await fetch(`${API_BASE}/api/me`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
  });

  if (response.ok) {
    const userData: User = await response.json();

    if (userData.avatarurl && !userData.avatarurl.startsWith("http")) {
      userData.avatarurl = `${OLLAMA_DOT_COM}${userData.avatarurl}`;
    }

    return userData;
  }

  if (response.status === 401 || response.status === 403) {
    return null;
  }

  throw new Error(`Failed to fetch user: ${response.status}`);
}

export async function fetchConnectUrl(): Promise<string> {
  const response = await fetch(`${API_BASE}/api/me`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
  });

  if (response.status === 401) {
    const data = await response.json();
    if (data.signin_url) {
      const connectUrl = new URL(data.signin_url);
      connectUrl.searchParams.set("launch", "true");
      return connectUrl.toString();
    }
  }

  throw new Error("Failed to fetch connect URL");
}

export async function disconnectUser(): Promise<void> {
  const response = await fetch(`${API_BASE}/api/signout`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    throw new Error("Failed to disconnect user");
  }
}

export async function getClaudeDesktopAvailableModels(
  includeCloudModels = false,
): Promise<Model[]> {
  try {
    const [localResult, cloudResult] = await Promise.all([
      ollama.list(),
      includeCloudModels
        ? fetch(`${API_BASE}/api/v1/models/cloud`)
            .then(async (response) => {
              if (!response.ok) {
                throw new Error(`cloud model list returned ${response.status}`);
              }
              return (await response.json()) as { models?: ModelResponse[] };
            })
            .catch((error) => {
              console.warn("Failed to fetch cloud models:", error);
              return { models: [] };
            })
        : Promise.resolve({ models: [] as ModelResponse[] }),
    ]);

    const localModels = localResult.models.filter((model: ModelResponse) => {
      const response = model as ModelResponse & {
        remote_model?: string;
        remote_host?: string;
      };
      const name = model.name.replace(/:latest$/, "");
      return (
        !response.remote_model &&
        !response.remote_host &&
        !name.endsWith("cloud")
      );
    });
    const cloudModels = (cloudResult.models ?? []).map((model) => {
      const name = model.name.replace(/:latest$/, "");
      const tag = name.slice(name.lastIndexOf(":") + 1).toLowerCase();
      const explicitCloud =
        name.endsWith(":cloud") ||
        (name.includes(":") && tag.endsWith("-cloud"));
      return {
        ...model,
        name: explicitCloud ? name : `${name}:cloud`,
      };
    });

    const seen = new Set<string>();
    return [...localModels, ...cloudModels]
      .filter((model: ModelResponse) => {
        const base = model.name.replace(/:latest$/, "").replace(/:cloud$/, "");
        if (!base || seen.has(base)) return false;

        const families = model.details?.families;
        const supported =
          !families ||
          families.length === 0 ||
          !families.every((family: string) =>
            family.toLowerCase().includes("bert"),
          );
        if (supported) seen.add(base);
        return supported;
      })
      .map(
        (model: ModelResponse) =>
          new Model({
            model: model.name.replace(/:latest$/, ""),
            digest: model.digest,
            modified_at: model.modified_at
              ? new Date(model.modified_at)
              : undefined,
          }),
      );
  } catch (err) {
    throw new Error(`Failed to fetch Ollama models: ${err}`);
  }
}

export async function getSettings(): Promise<{
  settings: Settings;
}> {
  const response = await fetch(`${API_BASE}/api/v1/settings`);
  if (!response.ok) {
    throw new Error("Failed to fetch settings");
  }
  const data = await response.json();
  return {
    settings: new Settings(data.settings),
  };
}

export async function updateSettings(settings: Settings): Promise<{
  settings: Settings;
}> {
  const response = await fetch(`${API_BASE}/api/v1/settings`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(settings),
  });
  if (!response.ok) {
    const error = await response.text();
    throw new Error(error || "Failed to update settings");
  }
  const data = await response.json();
  return {
    settings: new Settings(data.settings),
  };
}

export async function updateCloudSetting(
  enabled: boolean,
): Promise<CloudStatusResponse> {
  const response = await fetch(`${API_BASE}/api/v1/cloud`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ enabled }),
  });
  if (!response.ok) {
    const error = await response.text();
    throw new Error(error || "Failed to update cloud setting");
  }

  const data = await response.json();
  return {
    disabled: Boolean(data.disabled),
    source: (data.source as CloudStatusSource) || "none",
  };
}

export async function getChats() {
  const response = await fetch(`${API_BASE}/api/v1/chats`);
  if (!response.ok) throw new Error("Could not load your chats.");
  return new ChatsResponse(await response.json()).chatInfos;
}

export async function getChat(chatId: string) {
  const response = await fetch(
    `${API_BASE}/api/v1/chat/${encodeURIComponent(chatId)}`,
  );
  if (!response.ok) throw new Error("Could not load this chat.");
  return new ChatResponse(await response.json()).chat;
}

export interface ExportResult {
  path: string;
  warnings?: string[];
}

export interface ExportProgress {
  completed: number;
  total: number;
}

export async function exportChat(chatId: string): Promise<ExportResult | null> {
  const response = await fetch(
    `${API_BASE}/api/v1/chat/${encodeURIComponent(chatId)}/export`,
    { method: "POST" },
  );
  const data = await response.json();
  if (!response.ok)
    throw new Error(data.error ?? "Could not export your chat.");
  return data;
}

export async function exportAllChats(
  signal: AbortSignal,
  onProgress: (progress: ExportProgress) => void,
): Promise<ExportResult | null> {
  const response = await fetch(`${API_BASE}/api/v1/chats/export`, {
    method: "POST",
    signal,
  });
  if (!response.ok) {
    const data = await response.json();
    throw new Error(data.error ?? "Could not export your chats.");
  }
  const reader = response.body?.getReader();
  if (!reader) throw new Error("Could not read export progress.");
  const decoder = new TextDecoder();
  let pending = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      pending += decoder.decode(value, { stream: !done });
      const lines = pending.split("\n");
      pending = lines.pop()!;
      for (const line of lines) {
        if (!line.trim()) continue;
        const update: ExportProgress | ExportResult | { error: string } | null =
          JSON.parse(line);
        if (update === null || "path" in update) return update;
        if ("error" in update) throw new Error(update.error);
        onProgress(update);
      }
      if (done) throw new Error("Export stopped before it finished.");
    }
  } finally {
    reader.releaseLock();
  }
}

export async function deleteChat(chatId: string): Promise<void> {
  const response = await fetch(
    `${API_BASE}/api/v1/chat/${encodeURIComponent(chatId)}`,
    {
      method: "DELETE",
    },
  );
  if (!response.ok) {
    const error = await response.text();
    throw new Error(error || "Failed to delete chat");
  }
}

export async function getInferenceCompute(): Promise<InferenceComputeResponse> {
  const response = await fetch(`${API_BASE}/api/v1/inference-compute`);
  if (!response.ok) {
    throw new Error(
      `Failed to fetch inference compute: ${response.statusText}`,
    );
  }

  const data = await response.json();
  return new InferenceComputeResponse(data);
}

export async function getCloudStatus(): Promise<CloudStatusResponse | null> {
  const response = await fetch(`${API_BASE}/api/v1/cloud`);
  if (!response.ok) {
    throw new Error(`Failed to fetch cloud status: ${response.status}`);
  }

  const data = await response.json();
  return {
    disabled: Boolean(data.disabled),
    source: (data.source as CloudStatusSource) || "none",
  };
}
