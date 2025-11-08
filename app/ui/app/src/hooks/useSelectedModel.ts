import { useEffect, useMemo, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { useModels } from "./useModels";
import { useChat } from "./useChats";
import { useSettings } from "./useSettings.ts";
import { Model } from "@/gotypes";
import { FEATURED_MODELS } from "@/utils/mergeModels";
import { getTotalVRAM } from "@/utils/vram.ts";
import { getInferenceCompute } from "@/api";

export function recommendDefaultModel(totalVRAM: number): string {
  const vram = Math.max(0, Number(totalVRAM) || 0);

  if (vram < 6) {
    return "gemma3:1b";
  } else if (vram < 16) {
    return "gemma3:4b";
  }
  return "gpt-oss:20b";
}

export function useSelectedModel(currentChatId?: string, searchQuery?: string) {
  const { settings, setSettings } = useSettings();
  const { data: models = [], isLoading } = useModels(searchQuery || "");
  const { data: chatData, isLoading: isChatLoading } = useChat(
    currentChatId && currentChatId !== "new" ? currentChatId : "",
  );

  const { data: inferenceComputes = [] } = useQuery({
    queryKey: ["inference-compute"],
    queryFn: getInferenceCompute,
    enabled: !settings.selectedModel, // Only fetch if no model is selected
  });

  const totalVRAM = useMemo(
    () => getTotalVRAM(inferenceComputes),
    [inferenceComputes],
  );

  const recommendedModel = useMemo(
    () => recommendDefaultModel(totalVRAM),
    [totalVRAM],
  );

  // Track which chat we've already restored the model for
  const restoredChatRef = useRef<string | null>(null);

  const selectedModel: Model | null = useMemo(() => {
    // if airplane mode is on and selected model ends with cloud,
    // switch to recommended default model
    if (settings.airplaneMode && settings.selectedModel?.endsWith("cloud")) {
      return (
        models.find((m) => m.model === recommendedModel) ||
        models.find((m) => m.isCloud) ||
        models.find((m) => m.digest === undefined || m.digest === "") ||
        models[0] ||
        null
      );
    }

    // Migration logic: if turboEnabled is true and selectedModel is a base model,
    // migrate to the cloud version and disable turboEnabled permanently
    // TODO: remove this logic in a future release
    const baseModelsToMigrate = [
      "gpt-oss:20b",
      "gpt-oss:120b",
      "deepseek-v3.1:671b",
      "qwen3-coder:480b",
    ];
    const shouldMigrate =
      !settings.airplaneMode &&
      settings.turboEnabled &&
      baseModelsToMigrate.includes(settings.selectedModel);

    if (shouldMigrate) {
      const cloudModel = `${settings.selectedModel}cloud`;
      return (
        models.find((m) => m.model === cloudModel) ||
        new Model({
          model: cloudModel,
          cloud: true,
          ollama_host: false,
        })
      );
    }

    return (
      models.find((m) => m.model === settings.selectedModel) ||
      (settings.selectedModel &&
        new Model({
          model: settings.selectedModel,
          cloud: FEATURED_MODELS.some(
            (f) => f.endsWith("cloud") && f === settings.selectedModel,
          ),
          ollama_host: false,
        })) ||
      null
    );
  }, [models, settings.selectedModel, settings.airplaneMode, recommendedModel]);

  useEffect(() => {
    if (!selectedModel) return;

    if (
      settings.airplaneMode &&
      settings.selectedModel?.endsWith("cloud") &&
      selectedModel.model !== settings.selectedModel
    ) {
      setSettings({ SelectedModel: selectedModel.model });
    }

    if (
      !settings.airplaneMode &&
      settings.turboEnabled &&
      selectedModel.model !== settings.selectedModel
    ) {
      setSettings({ SelectedModel: selectedModel.model, TurboEnabled: false });
    }
  }, [selectedModel, settings.airplaneMode, settings.selectedModel]);

  // Set model from chat history when chat data loads
  useEffect(() => {
    // Only run this effect if we have a valid currentChatId
    if (!currentChatId || currentChatId === "new") {
      return;
    }

    if (
      chatData?.chat?.messages &&
      !isChatLoading &&
      restoredChatRef.current !== currentChatId
    ) {
      // Find the most recent model used in this chat
      const messages = [...chatData.chat.messages].reverse();
      for (const message of messages) {
        if (message.model) {
          const chatModelName = message.model;

          if (chatModelName !== settings.selectedModel) {
            setSettings({ SelectedModel: chatModelName });
          }

          // Mark this chat as restored
          restoredChatRef.current = currentChatId;
          return;
        }
      }
      // Mark this chat as processed even if no model was found
      restoredChatRef.current = currentChatId;
    }
  }, [
    currentChatId,
    chatData,
    isChatLoading,
    settings.selectedModel,
    setSettings,
  ]);

  // On initial load, if no model is selected, set default model
  useEffect(() => {
    if (
      isLoading ||
      inferenceComputes.length === 0 ||
      models.length === 0 ||
      settings.selectedModel
    ) {
      return;
    }

    const defaultModel =
      models.find((m) => m.model === recommendedModel) ||
      models.find((m) => m.isCloud()) ||
      models.find((m) => m.digest === undefined || m.digest === "") ||
      models[0];

    if (defaultModel) {
      setSettings({ SelectedModel: defaultModel.model });
    }
  }, [
    isLoading,
    inferenceComputes.length,
    models.length,
    settings.selectedModel,
  ]);

  // Add the selected model to the models list if it's not already there
  const allModels = useMemo(() => {
    if (!selectedModel || models.find((m) => m.model === selectedModel.model)) {
      return models;
    }

    return [...models, selectedModel];
  }, [models, selectedModel]);

  return {
    selectedModel,
    setSettings,
    models: allModels,
    loading: isLoading,
  };
}
