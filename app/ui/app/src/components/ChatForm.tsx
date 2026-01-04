import Logo from "@/components/Logo";
import { ModelPicker } from "@/components/ModelPicker";
import { WebSearchButton } from "@/components/WebSearchButton";
import { ImageThumbnail } from "@/components/ImageThumbnail";
import { isImageFile } from "@/utils/imageUtils";
import {
  useRef,
  useState,
  useEffect,
  useLayoutEffect,
  useCallback,
} from "react";
import {
  useSendMessage,
  useIsStreaming,
  useCancelMessage,
} from "@/hooks/useChats";
import { useNavigate } from "@tanstack/react-router";
import { useSelectedModel } from "@/hooks/useSelectedModel";
import { useHasVisionCapability } from "@/hooks/useModelCapabilities";
import { useUser } from "@/hooks/useUser";
import { DisplayLogin } from "@/components/DisplayLogin";
import { ErrorEvent, Message } from "@/gotypes";
import { useSettings } from "@/hooks/useSettings";
import { ThinkButton } from "./ThinkButton";
import { ErrorMessage } from "./ErrorMessage";
import { processFiles } from "@/utils/fileValidation";
import type { ImageData } from "@/types/webview";
import { PlusIcon } from "@heroicons/react/24/outline";

export type ThinkingLevel = "low" | "medium" | "high";

interface FileAttachment {
  filename: string;
  data: Uint8Array;
  type?: string; // MIME type
}

interface MessageInput {
  content: string;
  attachments: Array<{
    id: string;
    filename: string;
    data?: Uint8Array; // undefined for existing files from editing
  }>;
  fileErrors: Array<{ filename: string; error: string }>;
}

interface ChatFormProps {
  hasMessages: boolean;
  onSubmit?: (
    message: string,
    options: {
      attachments?: FileAttachment[];
      index?: number;
      webSearch?: boolean;
      fileTools?: boolean;
      think?: boolean | string;
    },
  ) => void;
  autoFocus?: boolean;
  chatId?: string;
  isDownloadingModel?: boolean;
  isDisabled?: boolean;
  // Editing props - when provided, ChatForm enters edit mode
  editingMessage?: {
    content: string;
    index: number;
    originalMessage: Message;
  } | null;
  onCancelEdit?: () => void;
  onFilesReceived?: (
    callback: (
      files: Array<{ filename: string; data: Uint8Array; type?: string }>,
      errors: Array<{ filename: string; error: string }>,
    ) => void,
  ) => void;
}

function ChatForm({
  hasMessages,
  onSubmit,
  autoFocus = false,
  chatId = "new",
  isDownloadingModel = false,
  isDisabled = false,
  editingMessage,
  onCancelEdit,
  onFilesReceived,
}: ChatFormProps) {
  const [message, setMessage] = useState<MessageInput>({
    content: "",
    attachments: [],
    fileErrors: [],
  });
  const [isEditing, setIsEditing] = useState<boolean>(false);
  const compositionEndTimeoutRef = useRef<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const thinkButtonRef = useRef<HTMLButtonElement>(null);
  const thinkingLevelButtonRef = useRef<HTMLButtonElement>(null);
  const webSearchButtonRef = useRef<HTMLButtonElement>(null);
  const modelPickerRef = useRef<HTMLButtonElement>(null);
  const submitButtonRef = useRef<HTMLButtonElement>(null);

  const { mutate: sendMessageMutation } = useSendMessage(chatId);
  const navigate = useNavigate();
  const isStreaming = useIsStreaming(chatId);
  const cancelMessage = useCancelMessage();
  const isDownloading = isDownloadingModel;
  const { selectedModel } = useSelectedModel();
  const hasVisionCapability = useHasVisionCapability(selectedModel?.model);
  const { isAuthenticated, isLoading: isLoadingUser } = useUser();
  const [loginPromptFeature, setLoginPromptFeature] = useState<
    "webSearch" | "turbo" | null
  >(null);
  const [fileUploadError, setFileUploadError] = useState<ErrorEvent | null>(
    null,
  );

  const handleThinkingLevelDropdownToggle = (isOpen: boolean) => {
    if (
      isOpen &&
      modelPickerRef.current &&
      (modelPickerRef.current as any).closeDropdown
    ) {
      (modelPickerRef.current as any).closeDropdown();
    }
  };

  const handleModelPickerDropdownToggle = (isOpen: boolean) => {
    if (
      isOpen &&
      thinkingLevelButtonRef.current &&
      (thinkingLevelButtonRef.current as any).closeDropdown
    ) {
      (thinkingLevelButtonRef.current as any).closeDropdown();
    }
  };

  const {
    settings: {
      webSearchEnabled,
      airplaneMode,
      thinkEnabled,
      thinkLevel: settingsThinkLevel,
    },
    setSettings,
  } = useSettings();

  // current supported models for web search
  const modelLower = selectedModel?.model.toLowerCase() || "";
  const supportsWebSearch =
    modelLower.startsWith("gpt-oss") ||
    modelLower.startsWith("qwen3") ||
    modelLower.startsWith("deepseek-v3");
  // Use per-chat thinking level instead of global
  const thinkLevel: ThinkingLevel =
    settingsThinkLevel === "none" || !settingsThinkLevel
      ? "medium"
      : (settingsThinkLevel as ThinkingLevel);
  const setThinkingLevel = (newLevel: ThinkingLevel) => {
    setSettings({ ThinkLevel: newLevel });
  };

  const modelSupportsThinkingLevels =
    selectedModel?.model.toLowerCase().startsWith("gpt-oss") || false;
  const supportsThinkToggling =
    selectedModel?.model.toLowerCase().startsWith("deepseek-v3.1") || false;

  useEffect(() => {
    if (supportsThinkToggling && thinkEnabled && webSearchEnabled) {
      setSettings({ WebSearchEnabled: false });
    }
  }, [
    selectedModel?.model,
    supportsThinkToggling,
    thinkEnabled,
    webSearchEnabled,
    setSettings,
  ]);

  const removeFile = (index: number) => {
    setMessage((prev) => ({
      ...prev,
      attachments: prev.attachments.filter((_, i) => i !== index),
    }));
  };

  const removeFileError = (index: number) => {
    setMessage((prev) => ({
      ...prev,
      fileErrors: prev.fileErrors.filter((_, i) => i !== index),
    }));
  };

  // Create stable callback for file handling
  const handleFilesReceived = useCallback(
    (
      files: Array<{ filename: string; data: Uint8Array; type?: string }>,
      errors: Array<{ filename: string; error: string }> = [],
    ) => {
      if (files.length > 0) {
        setFileUploadError(null);

        const newAttachments = files.map((file) => ({
          id: crypto.randomUUID(),
          filename: file.filename,
          data: file.data,
        }));

        setMessage((prev) => ({
          ...prev,
          attachments: [...prev.attachments, ...newAttachments],
        }));
      }

      // Add validation errors to form state
      if (errors.length > 0) {
        setMessage((prev) => ({
          ...prev,
          fileErrors: [...prev.fileErrors, ...errors],
        }));
      }
    },
    [],
  );

  useEffect(() => {
    if (onFilesReceived) {
      onFilesReceived(handleFilesReceived);
    }
  }, [onFilesReceived, handleFilesReceived]);

  // Determine if login banner should be shown
  const shouldShowLoginBanner =
    !isLoadingUser &&
    !isAuthenticated &&
    ((webSearchEnabled && supportsWebSearch) ||
      (selectedModel?.isCloud() && !airplaneMode));

  // Determine which feature to highlight in the banner
  const getActiveFeatureForBanner = () => {
    if (!isAuthenticated) {
      if (loginPromptFeature) return loginPromptFeature;
      if (webSearchEnabled && selectedModel?.isCloud() && !airplaneMode)
        return "webSearch";
      if (webSearchEnabled) return "webSearch";
      if (selectedModel?.isCloud() && !airplaneMode) return "turbo";
    }
    return null;
  };

  const activeFeatureForBanner = getActiveFeatureForBanner();

  const resetChatForm = () => {
    setMessage({
      content: "",
      attachments: [],
      fileErrors: [],
    });

    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  // Clear loginPromptFeature when user becomes authenticated or no features are enabled
  useEffect(() => {
    if (
      isAuthenticated ||
      (!webSearchEnabled && !!selectedModel?.isCloud() && !airplaneMode)
    ) {
      setLoginPromptFeature(null);
    }
  }, [isAuthenticated, webSearchEnabled, selectedModel, airplaneMode]);

  // When entering edit mode, populate the composition with existing data
  useEffect(() => {
    if (!editingMessage) {
      // Clear composition and reset textarea height when not editing
      resetChatForm();
      return;
    }

    const existingAttachments =
      editingMessage.originalMessage?.attachments || [];
    setMessage({
      content: editingMessage.content,
      attachments: existingAttachments.map((att) => ({
        id: crypto.randomUUID(),
        filename: att.filename,
        // No data for existing files - backend will handle them
      })),
      fileErrors: [],
    });
  }, [editingMessage]);

  // Focus and setup textarea when editing
  useLayoutEffect(() => {
    if (editingMessage && textareaRef.current) {
      textareaRef.current.focus();
      textareaRef.current.style.transition =
        "height 0.2s ease-out, opacity 0.3s ease-in";
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height =
        Math.min(textareaRef.current.scrollHeight, 24 * 8) + "px";
    }
  }, [editingMessage]);

  // Clear composition and reset textarea height when chatId changes
  useEffect(() => {
    resetChatForm();
  }, [chatId]);

  // Auto-focus textarea when autoFocus is true or when streaming completes (but not when editing)
  useEffect(() => {
    if ((autoFocus || !isStreaming) && textareaRef.current && !editingMessage) {
      const timer = setTimeout(
        () => {
          textareaRef.current?.focus();
        },
        autoFocus ? 0 : 100,
      );
      return () => clearTimeout(timer);
    }
  }, [autoFocus, isStreaming, editingMessage]);

  const focusChatFormInput = () => {
    // Focus textarea after model selection or navigation
    if (textareaRef.current) {
      setTimeout(() => {
        textareaRef.current?.focus();
      }, 100);
    }
  };

  // Navigation helper function
  const navigateToNextElement = useCallback(
    (current: HTMLElement, direction: "next" | "prev") => {
      const elements = [
        textareaRef,
        modelSupportsThinkingLevels ? thinkingLevelButtonRef : thinkButtonRef,
        webSearchButtonRef,
        modelPickerRef,
        submitButtonRef,
      ]
        .map((ref) => ref.current)
        .filter(Boolean) as HTMLElement[];
      const index = elements.indexOf(current);
      if (index === -1) return;
      const nextIndex =
        direction === "next"
          ? (index + 1) % elements.length
          : (index - 1 + elements.length) % elements.length;
      elements[nextIndex].focus();
    },
    [],
  );

  // Focus textarea when navigating to a chat (when chatId changes)
  useEffect(() => {
    if (chatId !== "new") {
      focusChatFormInput();
    }
  }, [chatId]);

  // Global keyboard and paste event handlers
  useEffect(() => {
    const focusTextareaIfAppropriate = (target: HTMLElement) => {
      if (
        !textareaRef.current ||
        textareaRef.current === document.activeElement
      ) {
        return;
      }

      const isEditableTarget =
        target.tagName === "INPUT" ||
        target.tagName === "TEXTAREA" ||
        target.contentEditable === "true" ||
        target.closest("input") ||
        target.closest("textarea") ||
        target.closest("[contenteditable='true']");

      if (!isEditableTarget) {
        textareaRef.current.focus();
      }
    };

    const handleKeyDown = (e: KeyboardEvent) => {
      // Handle escape key for canceling
      if (e.key === "Escape") {
        e.preventDefault();
        if (editingMessage && onCancelEdit) {
          handleCancelEdit();
        } else if (isStreaming) {
          handleCancel();
        }
        return;
      }

      // Handle Tab navigation between controls
      if (e.key === "Tab" && e.target !== textareaRef.current) {
        const target = e.target as HTMLElement;
        const focusableElements = [
          modelSupportsThinkingLevels
            ? thinkingLevelButtonRef.current
            : thinkButtonRef.current,
          webSearchButtonRef.current,
          modelPickerRef.current,
          submitButtonRef.current,
        ].filter(Boolean) as HTMLElement[];

        if (focusableElements.includes(target)) {
          e.preventDefault();
          if (e.shiftKey) {
            navigateToNextElement(target, "prev");
          } else {
            navigateToNextElement(target, "next");
          }
          return;
        }
      }

      // Handle paste shortcuts
      const isPasteShortcut = (e.ctrlKey || e.metaKey) && e.key === "v";
      if (isPasteShortcut) {
        focusTextareaIfAppropriate(e.target as HTMLElement);
        return;
      }

      // Handle auto-focus when typing printable characters
      const target = e.target as HTMLElement;
      const isInInputField =
        target.tagName === "INPUT" ||
        target.tagName === "TEXTAREA" ||
        target.contentEditable === "true";

      if (
        !isInInputField &&
        e.key.length === 1 &&
        !e.ctrlKey &&
        !e.metaKey &&
        !e.altKey &&
        textareaRef.current
      ) {
        textareaRef.current.focus();
      }
    };

    const handlePaste = (e: ClipboardEvent) => {
      focusTextareaIfAppropriate(e.target as HTMLElement);
    };

    window.addEventListener("keydown", handleKeyDown);
    document.addEventListener("paste", handlePaste);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      document.removeEventListener("paste", handlePaste);
    };
  }, [isStreaming, editingMessage, onCancelEdit, navigateToNextElement]);

  const handleSubmit = async () => {
    if (!message.content.trim() || isStreaming || isDownloading) return;

    // Check if cloud mode is enabled but user is not authenticated
    if (shouldShowLoginBanner) {
      return;
    }

    // Prepare attachments for submission
    const attachmentsToSend: FileAttachment[] = message.attachments.map(
      (att) => ({
        filename: att.filename,
        data: att.data || new Uint8Array(0), // Empty data for existing files
      }),
    );

    const useWebSearch = supportsWebSearch && webSearchEnabled && !airplaneMode;
    const useThink = modelSupportsThinkingLevels
      ? thinkLevel
      : supportsThinkToggling
        ? thinkEnabled
        : undefined;

    if (onSubmit) {
      onSubmit(message.content, {
        attachments: attachmentsToSend,
        index: undefined,
        webSearch: useWebSearch,
        think: useThink,
      });
    } else {
      sendMessageMutation({
        message: message.content,
        attachments: attachmentsToSend,
        webSearch: useWebSearch,
        think: useThink,
        onChatEvent: (event) => {
          if (event.eventName === "chat_created" && event.chatId) {
            navigate({
              to: "/c/$chatId",
              params: {
                chatId: event.chatId,
              },
            });
          }
        },
      });
    }

    // Clear composition after successful submission
    setMessage({
      content: "",
      attachments: [],
      fileErrors: [],
    });

    // Reset textarea height and refocus after submit
    setTimeout(() => {
      if (textareaRef.current) {
        textareaRef.current.style.height = "auto";
        textareaRef.current.focus();
      }
    }, 100);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Handle Enter to submit
    if (e.key === "Enter" && !e.shiftKey && !isEditing) {
      e.preventDefault();
      if (!isStreaming && !isDownloading) {
        handleSubmit();
      }
      return;
    }

    // Handle Tab navigation
    if (e.key === "Tab") {
      e.preventDefault();
      const focusableElements = [
        modelSupportsThinkingLevels
          ? thinkingLevelButtonRef.current
          : thinkButtonRef.current,
        webSearchButtonRef.current,
        modelPickerRef.current,
        submitButtonRef.current,
      ].filter(Boolean);

      if (e.shiftKey) {
        // Shift+Tab: focus last focusable element
        const lastElement = focusableElements[focusableElements.length - 1];
        lastElement?.focus();
      } else {
        // Tab: focus first focusable element
        const firstElement = focusableElements[0];
        firstElement?.focus();
      }
      return;
    }
  };

  const handleCompositionStart = () => {
    if (compositionEndTimeoutRef.current) {
      window.clearTimeout(compositionEndTimeoutRef.current);
    }
    setIsEditing(true);
  };

  const handleCompositionEnd = () => {
    // Add a small delay to handle the timing issue where Enter keydown
    // fires immediately after composition end
    compositionEndTimeoutRef.current = window.setTimeout(() => {
      setIsEditing(false);
    }, 10);
  };

  const handleCancel = () => {
    cancelMessage(chatId);
  };

  const handleCancelEdit = () => {
    // Clear composition and call parent callback
    setMessage({
      content: "",
      attachments: [],
      fileErrors: [],
    });

    onCancelEdit?.();

    // Focus the textarea after canceling edit mode
    setTimeout(() => {
      textareaRef.current?.focus();
    }, 0);
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    Array.from(files).forEach((file) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
    });

    // Reset file input
    if (e.target) {
      e.target.value = "";
    }
  };

  // Auto-resize textarea function
  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setMessage((prev) => ({ ...prev, content: e.target.value }));

    // Reset height to auto to get the correct scrollHeight, then cap at 8 lines
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 24 * 8) + "px";
  };

  const handleFilesUpload = async () => {
    try {
      setFileUploadError(null);

      const results = await window.webview?.selectMultipleFiles();
      if (results && results.length > 0) {
        // Convert native dialog results to File objects
        const files = results
          .map((result: ImageData) => {
            if (result.dataURL) {
              // Convert dataURL back to File object
              const base64Data = result.dataURL.split(",")[1];
              const mimeType = result.dataURL.split(";")[0].split(":")[1];
              const binaryString = atob(base64Data);
              const bytes = new Uint8Array(binaryString.length);
              for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
              }

              const blob = new Blob([bytes], { type: mimeType });
              const file = new File([blob], result.filename, {
                type: mimeType,
              });
              return file;
            }
            return null;
          })
          .filter(Boolean) as File[];

        if (files.length > 0) {
          const { validFiles, errors } = await processFiles(files, {
            selectedModel,
            hasVisionCapability,
          });

          // Send processed files and errors to the same handler as FileUpload
          if (validFiles.length > 0 || errors.length > 0) {
            handleFilesReceived(validFiles, errors);
          }
        }
      }
    } catch (error) {
      console.error("Error selecting multiple files:", error);

      const errorEvent = new ErrorEvent({
        eventName: "error" as const,
        error:
          error instanceof Error ? error.message : "Failed to select files",
        code: "file_selection_error",
        details:
          "An error occurred while trying to open the file selection dialog. Please try again.",
      });

      setFileUploadError(errorEvent);
    }
  };
  return (
    <div className={`pb-3 px-3 ${hasMessages ? "mt-auto" : "my-auto"}`}>
      {chatId === "new" && <Logo />}

      {shouldShowLoginBanner && (
        <DisplayLogin
          error={
            new ErrorEvent({
              eventName: "error",
              error:
                activeFeatureForBanner === "webSearch"
                  ? "Web search requires authentication"
                  : "Cloud models require authentication",
              code: "cloud_unauthorized",
            })
          }
          message={
            activeFeatureForBanner === "webSearch"
              ? "Web search requires an Ollama account"
              : "Cloud models require an Ollama account"
          }
          className="mb-4"
          onDismiss={() => {
            // Disable the active features when dismissing
            if (webSearchEnabled) setSettings({ WebSearchEnabled: false });
            setLoginPromptFeature(null);
          }}
        />
      )}

      {/* File upload error message */}
      {fileUploadError && <ErrorMessage error={fileUploadError} />}
      <div
        className={`relative mx-auto flex bg-neutral-100 w-full max-w-[768px] flex-col items-center rounded-3xl pb-2 pt-4 dark:bg-neutral-800 dark:border-neutral-700 min-h-[88px] transition-opacity duration-200 ${isDisabled ? "opacity-50" : "opacity-100"}`}
      >
        {isDisabled && (
          // overlay to block interaction
          <div className="absolute inset-0 z-50 rounded-3xl" />
        )}
        {editingMessage && (
          <div className="w-full px-5 pb-2">
            <p className="text-xs text-neutral-500 dark:text-neutral-400">
              Press ESC to cancel editing
            </p>
          </div>
        )}
        {(message.attachments.length > 0 || message.fileErrors.length > 0) && (
          <div className="flex gap-2 overflow-x-auto px-3 pt pb-3 w-full scrollbar-hide">
            {message.attachments.map((attachment, index) => (
              <div
                key={attachment.id}
                className="group flex items-center gap-2 py-2 px-3 rounded-lg bg-neutral-50 dark:bg-neutral-700/50 hover:bg-neutral-100 dark:hover:bg-neutral-700 transition-colors flex-shrink-0"
              >
                {isImageFile(attachment.filename) ? (
                  <ImageThumbnail
                    image={{
                      filename: attachment.filename,
                      data: attachment.data || new Uint8Array(0),
                    }}
                    className="w-8 h-8 object-cover rounded-md flex-shrink-0"
                  />
                ) : (
                  <svg
                    className="w-4 h-4 text-neutral-400 dark:text-neutral-500 flex-shrink-0"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    />
                  </svg>
                )}
                <span className="text-sm text-neutral-700 dark:text-neutral-300 max-w-[150px] truncate">
                  {attachment.filename}
                </span>
                <button
                  type="button"
                  onClick={() => removeFile(index)}
                  className="opacity-0 group-hover:opacity-100 transition-opacity text-neutral-400 hover:text-neutral-600 dark:text-neutral-500 dark:hover:text-neutral-300 -mr-1 cursor-pointer"
                  aria-label={`Remove ${attachment.filename}`}
                >
                  <svg
                    className="w-4 h-4"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M6 18L18 6M6 6l12 12"
                    />
                  </svg>
                </button>
              </div>
            ))}
            {message.fileErrors.map((fileError, index) => (
              <div
                key={`error-${index}`}
                className="group flex items-center gap-2 py-2 px-3 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 flex-shrink-0"
              >
                <svg
                  className="w-4 h-4 text-red-500 flex-shrink-0"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                  />
                </svg>
                <span className="text-sm text-red-700 dark:text-red-300 max-w-[100px] truncate">
                  {fileError.filename}
                </span>
                <span className="text-xs text-red-600 dark:text-red-400 opacity-75">
                  • {fileError.error}
                </span>
                <button
                  type="button"
                  onClick={() => removeFileError(index)}
                  className="opacity-0 group-hover:opacity-100 transition-opacity text-red-400 hover:text-red-600 dark:text-red-500 dark:hover:text-red-300 -mr-1 ml-auto"
                  aria-label={`Remove ${fileError.filename}`}
                >
                  <svg
                    className="w-4 h-4"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M6 18L18 6M6 6l12 12"
                    />
                  </svg>
                </button>
              </div>
            ))}
          </div>
        )}

        <div className="relative w-full px-5">
          <textarea
            ref={textareaRef}
            value={message.content}
            onChange={handleTextareaChange}
            placeholder="Send a message"
            disabled={isDisabled}
            className={`allow-context-menu w-full overflow-y-auto text-neutral-700 outline-none resize-none border-none bg-transparent dark:text-white placeholder:text-neutral-400 dark:placeholder:text-neutral-500 min-h-[24px] leading-6 transition-opacity duration-300 ${
              editingMessage ? "animate-fade-in" : ""
            }`}
            rows={1}
            onKeyDown={handleKeyDown}
            onCompositionStart={handleCompositionStart}
            onCompositionEnd={handleCompositionEnd}
          />
        </div>

        {/* Controls */}
        <div className="flex w-full items-center justify-end gap-2 px-3 pt-2">
          {/* Tool buttons - animate from underneath model picker */}
          {!isDisabled && (
            <div className="flex-1 flex justify-end items-center gap-2">
              <div className={`flex gap-2`}>
                {/* File Upload Buttons */}
                <button
                  type="button"
                  onClick={handleFilesUpload}
                  className="flex h-9 w-9 items-center justify-center rounded-full bg-white dark:bg-neutral-700 focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer border border-transparent"
                  title="Upload multiple files"
                >
                  <PlusIcon className="w-4.5 h-4.5 stroke-2 text-neutral-500 dark:text-neutral-400" />
                </button>
                {/* Thinking Level Button */}
                {modelSupportsThinkingLevels && (
                  <>
                    <ThinkButton
                      mode="thinkingLevel"
                      ref={thinkingLevelButtonRef}
                      isVisible={modelSupportsThinkingLevels}
                      currentLevel={thinkLevel}
                      onLevelChange={setThinkingLevel}
                      onDropdownToggle={handleThinkingLevelDropdownToggle}
                    />
                  </>
                )}
                {/* Think Button turn on and off */}
                {supportsThinkToggling && !modelSupportsThinkingLevels && (
                  <>
                    <ThinkButton
                      mode="think"
                      ref={thinkButtonRef}
                      isVisible={
                        supportsThinkToggling && !modelSupportsThinkingLevels
                      }
                      isActive={thinkEnabled}
                      onToggle={() => {
                        // DeepSeek-v3 specific - thinking and web search are mutually exclusive
                        if (supportsThinkToggling) {
                          const enable = !thinkEnabled;
                          setSettings({
                            ThinkEnabled: enable,
                            ...(enable ? { WebSearchEnabled: false } : {}),
                          });
                          return;
                        }
                        setSettings({ ThinkEnabled: !thinkEnabled });
                      }}
                    />
                  </>
                )}
                <WebSearchButton
                  ref={webSearchButtonRef}
                  isVisible={supportsWebSearch && airplaneMode === false}
                  isActive={webSearchEnabled}
                  onToggle={() => {
                    if (!webSearchEnabled && !isAuthenticated) {
                      setLoginPromptFeature("webSearch");
                    }
                    const enable = !webSearchEnabled;
                    if (supportsThinkToggling && enable) {
                      setSettings({
                        WebSearchEnabled: true,
                        ThinkEnabled: false,
                      });
                      return;
                    }
                    setSettings({ WebSearchEnabled: enable });
                  }}
                />
              </div>
            </div>
          )}

          {/* Model picker and submit button */}
          <div className="flex items-center gap-2 relative z-20">
            <ModelPicker
              ref={modelPickerRef}
              chatId={chatId}
              onModelSelect={focusChatFormInput}
              onEscape={focusChatFormInput}
              isDisabled={isDisabled}
              onDropdownToggle={handleModelPickerDropdownToggle}
            />
            <button
              ref={submitButtonRef}
              onClick={
                isStreaming || isDownloading ? handleCancel : handleSubmit
              }
              disabled={
                !isStreaming &&
                !isDownloading &&
                (!message.content.trim() ||
                  shouldShowLoginBanner ||
                  message.fileErrors.length > 0)
              }
              className={`flex items-center justify-center h-9 w-9 rounded-full disabled:cursor-default cursor-pointer bg-black text-white dark:bg-white dark:text-black disabled:opacity-10 focus:outline-none focus:ring-2 focus:ring-blue-500`}
            >
              {isStreaming || isDownloading ? (
                <svg
                  className="h-3 w-3 fill-current"
                  viewBox="0 0 15 15"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path d="M0 12.3838C0 13.6055 0.738281 14.3262 1.96875 14.3262H12.3486C13.5879 14.3262 14.3174 13.6055 14.3174 12.3838V1.94238C14.3174 0.720703 13.5879 0 12.3486 0H1.96875C0.738281 0 0 0.720703 0 1.94238V12.3838Z" />
                </svg>
              ) : (
                <svg
                  className="h-3.5 w-3.5 fill-current"
                  viewBox="0 0 14 17"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path d="M0.918802 7.73542C1.19144 7.73542 1.43401 7.63188 1.60065 7.45804L3.59348 5.48929L6.7957 1.89614L10.0107 5.48929L12.0067 7.45804C12.179 7.63188 12.416 7.73542 12.6886 7.73542C13.2182 7.73542 13.6074 7.33974 13.6074 6.80466C13.6074 6.54785 13.5149 6.3174 13.3131 6.10998L7.51833 0.306385C7.32603 0.106874 7.06851 0 6.8029 0C6.5373 0 6.2782 0.106874 6.08748 0.306385L0.299881 6.10998C0.0996671 6.3174 0 6.54785 0 6.80466C0 7.33974 0.389177 7.73542 0.918802 7.73542ZM6.8029 16.6848C7.36909 16.6848 7.76073 16.2909 7.76073 15.7136V4.79494L7.65544 1.93059C7.65544 1.40993 7.31091 1.06066 6.8029 1.06066C6.29332 1.06066 5.94879 1.40993 5.94879 1.93059L5.8435 4.79494V15.7136C5.8435 16.2909 6.23672 16.6848 6.8029 16.6848Z" />
                </svg>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Hidden file input for fallback */}
      <input
        ref={fileInputRef}
        type="file"
        accept="*/*"
        multiple
        onChange={handleFileInputChange}
        className="hidden"
      />
    </div>
  );
}

export default ChatForm;
