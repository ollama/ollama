import {
  ChatResponse,
  ChatsResponse,
  ChatEvent,
  DownloadEvent,
  ErrorEvent,
  InferenceCompute,
  InferenceComputeResponse,
  ModelCapabilitiesResponse,
  Model,
  ChatRequest,
  Settings,
  User,
} from "@/gotypes";
import { parseJsonlFromResponse } from "./util/jsonl-parsing";
import { ollamaClient as ollama } from "./lib/ollama-client";
import type { ModelResponse } from "ollama/browser";

// Extend Model class with utility methods
declare module "@/gotypes" {
  interface Model {
    isCloud(): boolean;
  }
}

Model.prototype.isCloud = function (): boolean {
  return this.model.endsWith("cloud");
};

const API_BASE = import.meta.env.DEV ? "http://127.0.0.1:3001" : "";

// Helper function to convert Uint8Array to base64
function uint8ArrayToBase64(uint8Array: Uint8Array): string {
  const chunkSize = 0x8000; // 32KB chunks to avoid stack overflow
  let binary = "";

  for (let i = 0; i < uint8Array.length; i += chunkSize) {
    const chunk = uint8Array.subarray(i, i + chunkSize);
    binary += String.fromCharCode(...chunk);
  }

  return btoa(binary);
}

export async function fetchUser(): Promise<User | null> {
  try {
    const response = await fetch(`${API_BASE}/api/v1/me`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (response.ok) {
      const userData: User = await response.json();
      return userData;
    }

    return null;
  } catch (error) {
    console.error("Error fetching user:", error);
    return null;
  }
}

export async function fetchConnectUrl(): Promise<string> {
  const response = await fetch(`${API_BASE}/api/v1/connect`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    throw new Error("Failed to fetch connect URL");
  }

  const data = await response.json();
  return data.connect_url;
}

export async function disconnectUser(): Promise<void> {
  const response = await fetch(`${API_BASE}/api/v1/disconnect`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    throw new Error("Failed to disconnect user");
  }
}

export async function getChats(): Promise<ChatsResponse> {
  const response = await fetch(`${API_BASE}/api/v1/chats`);
  const data = await response.json();
  return new ChatsResponse(data);
}

export async function getChat(chatId: string): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE}/api/v1/chat/${chatId}`);
  const data = await response.json();
  return new ChatResponse(data);
}

export async function getModels(query?: string): Promise<Model[]> {
  try {
    const { models: modelsResponse } = await ollama.list();

    let models: Model[] = modelsResponse
      .filter((m: ModelResponse) => {
        const families = m.details?.families;

        if (!families || families.length === 0) {
          return true;
        }

        const isBertOnly = families.every((family: string) =>
          family.toLowerCase().includes("bert"),
        );

        return !isBertOnly;
      })
      .map((m: ModelResponse) => {
        // Remove the latest tag from the returned model
        const modelName = m.name.replace(/:latest$/, "");

        return new Model({
          model: modelName,
          digest: m.digest,
          modified_at: m.modified_at ? new Date(m.modified_at) : undefined,
        });
      });

    // Filter by query if provided
    if (query) {
      const normalizedQuery = query.toLowerCase().trim();

      const filteredModels = models.filter((m: Model) => {
        return m.model.toLowerCase().startsWith(normalizedQuery);
      });

      let exactMatch = false;
      for (const m of filteredModels) {
        if (m.model.toLowerCase() === normalizedQuery) {
          exactMatch = true;
          break;
        }
      }

      // Add query if it's in the registry and not already in the list
      if (!exactMatch) {
        const result = await getModelUpstreamInfo(new Model({ model: query }));
        const existsUpstream = !!result.digest && !result.error;
        if (existsUpstream) {
          filteredModels.push(new Model({ model: query }));
        }
      }

      models = filteredModels;
    }

    return models;
  } catch (err) {
    throw new Error(`Failed to fetch models: ${err}`);
  }
}

export async function getModelCapabilities(
  modelName: string,
): Promise<ModelCapabilitiesResponse> {
  try {
    const showResponse = await ollama.show({ model: modelName });

    return new ModelCapabilitiesResponse({
      capabilities: Array.isArray(showResponse.capabilities)
        ? showResponse.capabilities
        : [],
    });
  } catch (error) {
    // Model might not be downloaded yet, return empty capabilities
    console.error(`Failed to get capabilities for ${modelName}:`, error);
    return new ModelCapabilitiesResponse({ capabilities: [] });
  }
}

export type ChatEventUnion = ChatEvent | DownloadEvent | ErrorEvent;

export async function* sendMessage(
  chatId: string,
  message: string,
  model: Model,
  attachments?: Array<{ filename: string; data: Uint8Array }>,
  signal?: AbortSignal,
  index?: number,
  webSearch?: boolean,
  fileTools?: boolean,
  forceUpdate?: boolean,
  think?: boolean | string,
): AsyncGenerator<ChatEventUnion> {
  // Convert Uint8Array to base64 for JSON serialization
  const serializedAttachments = attachments?.map((att) => ({
    filename: att.filename,
    data: uint8ArrayToBase64(att.data),
  }));

  const response = await fetch(`${API_BASE}/api/v1/chat/${chatId}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(
      new ChatRequest({
        model: model.model,
        prompt: message,
        ...(index !== undefined ? { index } : {}),
        ...(serializedAttachments !== undefined
          ? { attachments: serializedAttachments }
          : {}),
        // Always send web_search as a boolean value (default to false)
        web_search: webSearch ?? false,
        file_tools: fileTools ?? false,
        ...(forceUpdate !== undefined ? { forceUpdate } : {}),
        ...(think !== undefined ? { think } : {}),
      }),
    ),
    signal,
  });

  for await (const event of parseJsonlFromResponse<ChatEventUnion>(response)) {
    switch (event.eventName) {
      case "download":
        yield new DownloadEvent(event);
        break;
      case "error":
        yield new ErrorEvent(event);
        break;
      default:
        yield new ChatEvent(event);
        break;
    }
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

export async function renameChat(chatId: string, title: string): Promise<void> {
  const response = await fetch(`${API_BASE}/api/v1/chat/${chatId}/rename`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ title: title.trim() }),
  });
  if (!response.ok) {
    const error = await response.text();
    throw new Error(error || "Failed to rename chat");
  }
}

export async function deleteChat(chatId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/api/v1/chat/${chatId}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    const error = await response.text();
    throw new Error(error || "Failed to delete chat");
  }
}

// Get upstream information for model staleness checking
export async function getModelUpstreamInfo(
  model: Model,
): Promise<{ digest?: string; pushTime: number; error?: string }> {
  try {
    const response = await fetch(`${API_BASE}/api/v1/model/upstream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: model.model,
      }),
    });

    if (!response.ok) {
      console.warn(
        `Failed to check upstream digest for ${model.model}: ${response.status}`,
      );
      return { pushTime: 0 };
    }

    const data = await response.json();

    if (data.error) {
      console.warn(`Upstream digest check: ${data.error}`);
      return { error: data.error, pushTime: 0 };
    }

    return { digest: data.digest, pushTime: data.pushTime || 0 };
  } catch (error) {
    console.warn(`Error checking model staleness:`, error);
    return { pushTime: 0 };
  }
}

export async function* pullModel(
  modelName: string,
  signal?: AbortSignal,
): AsyncGenerator<{
  status: string;
  digest?: string;
  total?: number;
  completed?: number;
  done?: boolean;
}> {
  const response = await fetch(`${API_BASE}/api/v1/models/pull`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ name: modelName }),
    signal,
  });

  if (!response.ok) {
    throw new Error(`Failed to pull model: ${response.statusText}`);
  }

  for await (const event of parseJsonlFromResponse<{
    status: string;
    digest?: string;
    total?: number;
    completed?: number;
    done?: boolean;
  }>(response)) {
    yield event;
  }
}

export async function getInferenceCompute(): Promise<InferenceCompute[]> {
  const response = await fetch(`${API_BASE}/api/v1/inference-compute`);
  if (!response.ok) {
    throw new Error(
      `Failed to fetch inference compute: ${response.statusText}`,
    );
  }

  const data = await response.json();
  const inferenceComputeResponse = new InferenceComputeResponse(data);
  return inferenceComputeResponse.inferenceComputes || [];
}

export async function fetchHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE}/api/v1/health`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (response.ok) {
      const data = await response.json();
      return data.healthy || false;
    }

    return false;
  } catch (error) {
    console.error("Error checking health:", error);
    return false;
  }
}

// Multi-API Provider Management (Phase 1)

export interface Provider {
  type: string;
  name: string;
  api_key?: string;
  base_url?: string;
}

export interface ProviderModel {
  id: string;
  name: string;
  display_name: string;
  context_window: number;
  capabilities: string[];
}

export async function listProviders(): Promise<Provider[]> {
  const response = await fetch(`${API_BASE}/api/providers`);
  if (!response.ok) {
    throw new Error("Failed to fetch providers");
  }
  const data = await response.json();
  return data.providers;
}

export async function addProvider(
  provider: Provider,
): Promise<{ message: string }> {
  const response = await fetch(`${API_BASE}/api/providers`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(provider),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || "Failed to add provider");
  }

  return response.json();
}

export async function deleteProvider(id: string): Promise<void> {
  const response = await fetch(`${API_BASE}/api/providers/${id}`, {
    method: "DELETE",
  });

  if (!response.ok) {
    throw new Error("Failed to delete provider");
  }
}

export async function getProviderModels(
  providerType: string,
  apiKey: string,
): Promise<ProviderModel[]> {
  const response = await fetch(
    `${API_BASE}/api/providers/${providerType}/models?api_key=${encodeURIComponent(apiKey)}`,
  );

  if (!response.ok) {
    throw new Error("Failed to fetch provider models");
  }

  const data = await response.json();
  return data.models;
}

export async function validateProvider(
  providerType: string,
  apiKey: string,
): Promise<boolean> {
  const response = await fetch(
    `${API_BASE}/api/providers/${providerType}/validate`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ api_key: apiKey }),
    },
  );

  if (!response.ok) {
    return false;
  }

  const data = await response.json();
  return data.valid;
}

// Workspace Management (Phase 2)

export interface WorkspaceRules {
  prohibitions: string[];
  requirements: string[];
  code_style: string[];
  system_prompt?: string;
}

export async function getWorkspaceRules(
  workspacePath: string,
): Promise<WorkspaceRules> {
  const response = await fetch(
    `${API_BASE}/api/workspace/rules?workspace=${encodeURIComponent(workspacePath)}`,
  );

  if (!response.ok) {
    throw new Error("Failed to fetch workspace rules");
  }

  const data = await response.json();
  return data.rules;
}

export async function updateWorkspaceRules(
  workspacePath: string,
  content: string,
): Promise<void> {
  const response = await fetch(
    `${API_BASE}/api/workspace/rules?workspace=${encodeURIComponent(workspacePath)}`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ content }),
    },
  );

  if (!response.ok) {
    throw new Error("Failed to update workspace rules");
  }
}

// RAG (Phase 6)

export interface RAGSearchResult {
  chunk: string;
  score: number;
  metadata: Record<string, string>;
}

export async function ingestDocument(
  workspacePath: string,
  title: string,
  content: string,
  metadata?: Record<string, string>,
): Promise<void> {
  const response = await fetch(`${API_BASE}/api/rag/ingest`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ workspace_path: workspacePath, title, content, metadata }),
  });

  if (!response.ok) {
    throw new Error("Failed to ingest document");
  }
}

export async function searchRAG(
  workspacePath: string,
  query: string,
  topK: number = 5,
): Promise<RAGSearchResult[]> {
  const response = await fetch(`${API_BASE}/api/rag/search`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ workspace_path: workspacePath, query, top_k: topK }),
  });

  if (!response.ok) {
    throw new Error("Failed to search documents");
  }

  const data = await response.json();
  return data.results;
}

// Templates (Phase 5)

export interface Template {
  id: string;
  name: string;
  description: string;
}

export async function listTemplates(): Promise<Template[]> {
  const response = await fetch(`${API_BASE}/api/templates`);

  if (!response.ok) {
    throw new Error("Failed to fetch templates");
  }

  const data = await response.json();
  return data.templates;
}

export async function renderTemplate(
  templateId: string,
  variables: Record<string, string>,
): Promise<string> {
  const response = await fetch(`${API_BASE}/api/templates/render`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ template_id: templateId, variables }),
  });

  if (!response.ok) {
    throw new Error("Failed to render template");
  }

  const data = await response.json();
  return data.rendered;
}

// Voice I/O (Phase 11)

export async function transcribeAudio(
  audioFile: File,
  apiKey: string,
): Promise<string> {
  const formData = new FormData();
  formData.append("file", audioFile);
  formData.append("api_key", apiKey);

  const response = await fetch(`${API_BASE}/api/voice/transcribe`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error("Failed to transcribe audio");
  }

  const data = await response.json();
  return data.text;
}

export async function synthesizeSpeech(
  text: string,
  apiKey: string,
  voice: string = "alloy",
): Promise<Blob> {
  const response = await fetch(`${API_BASE}/api/voice/synthesize`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ text, api_key: apiKey, voice }),
  });

  if (!response.ok) {
    throw new Error("Failed to synthesize speech");
  }

  return response.blob();
}
