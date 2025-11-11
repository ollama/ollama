import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Field, Label } from "@/components/ui/fieldset";
import { Badge } from "@/components/ui/badge";
import {
  PaperAirplaneIcon,
  SparklesIcon,
} from "@heroicons/react/20/solid";
import {
  listProviders,
  getProviderModels,
  type Provider,
  type ProviderModel,
} from "@/api";

interface Message {
  role: "user" | "assistant" | "system";
  content: string;
}

interface MultiProviderChatPanelProps {
  apiKey?: string;
}

export default function MultiProviderChatPanel({
  apiKey,
}: MultiProviderChatPanelProps) {
  const [selectedProvider, setSelectedProvider] = useState<Provider | null>(
    null,
  );
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState("");
  const [providerApiKey, setProviderApiKey] = useState(apiKey || "");

  const { data: providers } = useQuery({
    queryKey: ["providers"],
    queryFn: listProviders,
  });

  const { data: models, isLoading: modelsLoading } = useQuery({
    queryKey: ["provider-models", selectedProvider?.type, providerApiKey],
    queryFn: () => {
      if (!selectedProvider || !providerApiKey) return [];
      return getProviderModels(selectedProvider.type, providerApiKey);
    },
    enabled: !!selectedProvider && !!providerApiKey,
  });

  const sendMessageMutation = useMutation({
    mutationFn: async (message: string) => {
      if (!selectedProvider || !selectedModel || !providerApiKey) {
        throw new Error("Please select provider, model, and enter API key");
      }

      const newMessages = [...messages, { role: "user" as const, content: message }];
      setMessages(newMessages);

      const response = await fetch(
        `/api/providers/${selectedProvider.type}/chat`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            model: selectedModel,
            messages: newMessages,
            api_key: providerApiKey,
          }),
        },
      );

      if (!response.ok) {
        throw new Error("Failed to send message");
      }

      const data = await response.json();
      return data;
    },
    onSuccess: (data) => {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.message.content },
      ]);
      setInputMessage("");
    },
    onError: (error) => {
      console.error("Chat error:", error);
      alert("Failed to send message: " + (error instanceof Error ? error.message : "Unknown error"));
    },
  });

  const handleSendMessage = () => {
    if (!inputMessage.trim()) return;
    sendMessageMutation.mutate(inputMessage);
  };

  const clearChat = () => {
    setMessages([]);
  };

  return (
    <div className="flex flex-col h-full space-y-4">
      {/* Provider and Model Selection */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-4 rounded-lg border border-gray-700 bg-gray-800">
        <Field>
          <Label>Provider</Label>
          <select
            value={selectedProvider?.type || ""}
            onChange={(e) => {
              const provider = providers?.find((p) => p.type === e.target.value);
              setSelectedProvider(provider || null);
              setSelectedModel("");
            }}
            className="w-full rounded-md border border-gray-600 bg-gray-700 px-3 py-2 text-white"
          >
            <option value="">Select provider...</option>
            {providers?.map((provider) => (
              <option key={provider.type} value={provider.type}>
                {provider.name}
              </option>
            ))}
          </select>
        </Field>

        <Field>
          <Label>Model</Label>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            disabled={!selectedProvider || modelsLoading}
            className="w-full rounded-md border border-gray-600 bg-gray-700 px-3 py-2 text-white disabled:opacity-50"
          >
            <option value="">
              {modelsLoading ? "Loading models..." : "Select model..."}
            </option>
            {models?.map((model) => (
              <option key={model.id} value={model.id}>
                {model.display_name}
              </option>
            ))}
          </select>
        </Field>

        <Field>
          <Label>API Key</Label>
          <Input
            type="password"
            placeholder="Enter API key..."
            value={providerApiKey}
            onChange={(e) => setProviderApiKey(e.target.value)}
          />
        </Field>
      </div>

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 rounded-lg border border-gray-700 bg-gray-900 min-h-[400px]">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-400">
            <div className="text-center">
              <SparklesIcon className="h-12 w-12 mx-auto mb-4 text-gray-600" />
              <p>Select a provider and model to start chatting</p>
            </div>
          </div>
        ) : (
          messages.map((message, idx) => (
            <div
              key={idx}
              className={`flex ${
                message.role === "user" ? "justify-end" : "justify-start"
              }`}
            >
              <div
                className={`max-w-[80%] rounded-lg px-4 py-2 ${
                  message.role === "user"
                    ? "bg-blue-600 text-white"
                    : "bg-gray-800 text-white border border-gray-700"
                }`}
              >
                <div className="flex items-center gap-2 mb-1">
                  <Badge className="text-xs">
                    {message.role === "user" ? "You" : selectedProvider?.name || "Assistant"}
                  </Badge>
                </div>
                <p className="whitespace-pre-wrap text-sm">{message.content}</p>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Chat Input */}
      <div className="flex gap-2">
        <Input
          type="text"
          placeholder="Type your message..."
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSendMessage();
            }
          }}
          disabled={sendMessageMutation.isPending}
          className="flex-1"
        />
        <Button
          onClick={handleSendMessage}
          disabled={
            sendMessageMutation.isPending ||
            !selectedProvider ||
            !selectedModel ||
            !providerApiKey ||
            !inputMessage.trim()
          }
          className="flex items-center gap-2"
        >
          <PaperAirplaneIcon className="h-5 w-5" />
          Send
        </Button>
        {messages.length > 0 && (
          <Button onClick={clearChat} variant="outline">
            Clear
          </Button>
        )}
      </div>

      {sendMessageMutation.isPending && (
        <div className="text-sm text-gray-400 text-center">
          Sending message...
        </div>
      )}
    </div>
  );
}
