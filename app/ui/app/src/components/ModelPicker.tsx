import {
  useState,
  useRef,
  useEffect,
  forwardRef,
  type JSX,
  useImperativeHandle,
} from "react";
import { Model } from "@/gotypes";
import { useSelectedModel } from "@/hooks/useSelectedModel";
import { useSettings } from "@/hooks/useSettings";
import { useQueryClient } from "@tanstack/react-query";
import { getModelUpstreamInfo } from "@/api";
import { ArrowDownTrayIcon } from "@heroicons/react/24/outline";

const stalenessCheckCache = new Map<string, number>();

export const ModelPicker = forwardRef<
  HTMLButtonElement,
  {
    chatId?: string;
    onModelSelect?: () => void;
    onEscape?: () => void;
    onDropdownToggle?: (isOpen: boolean) => void;
    isDisabled?: boolean;
  }
>(function ModelPicker(
  { chatId, onModelSelect, onEscape, onDropdownToggle, isDisabled },
  ref,
): JSX.Element {
  const [isOpen, setIsOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const { selectedModel, setSettings, models, loading } = useSelectedModel(
    chatId,
    searchQuery,
  );
  const { settings } = useSettings();
  const dropdownRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const queryClient = useQueryClient();
  const modelListRef = useRef<{
    scrollToSelectedModel: () => void;
    scrollToTop: () => void;
  }>(null);

  const checkModelStaleness = async (model: Model) => {
    if (
      !model ||
      !model.model ||
      model.digest === undefined ||
      model.digest === ""
    )
      return;

    // Check cache - only check staleness every 5 minutes per model
    const now = Date.now();
    const lastChecked = stalenessCheckCache.get(model.model);
    if (lastChecked && now - lastChecked < 5 * 60 * 1000) return;
    stalenessCheckCache.set(model.model, now);

    try {
      const upstreamInfo = await getModelUpstreamInfo(model);

      // Compare local digest with upstream digest
      let isStale =
        model.digest &&
        upstreamInfo.digest &&
        model.digest !== upstreamInfo.digest;

      // If the model has a modified time and upstream has a push time,
      // check if the model was modified after the push time - if so, it's not stale
      if (isStale && model.modified_at && upstreamInfo.pushTime > 0) {
        const modifiedAtTime =
          new Date(model.modified_at as string | number | Date).getTime() /
          1000;
        if (modifiedAtTime > upstreamInfo.pushTime) {
          isStale = false;
        }
      }

      if (isStale) {
        const currentStaleModels =
          queryClient.getQueryData<Map<string, boolean>>(["staleModels"]) ||
          new Map();
        const newMap = new Map(currentStaleModels);
        newMap.set(model.model, true);
        queryClient.setQueryData(["staleModels"], newMap);
      }
    } catch (error) {
      console.error("Failed to check model staleness:", error);
    }
  };

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  useEffect(() => {
    if (ref && typeof ref === "object" && ref.current) {
      (ref.current as any).closeDropdown = () => setIsOpen(false);
    }
  }, [ref, setIsOpen]);

  // Focus search when opened and refresh models
  // Clear search when closed
  useEffect(() => {
    if (isOpen) {
      searchInputRef.current?.focus();
      modelListRef.current?.scrollToSelectedModel();
    } else {
      setSearchQuery("");
    }
  }, [isOpen]);

  // When searching, scroll to top of list
  useEffect(() => {
    if (searchQuery && modelListRef.current) {
      modelListRef.current.scrollToTop();
    }
  }, [searchQuery]);

  useEffect(() => {
    if (selectedModel && !loading) {
      checkModelStaleness(selectedModel);
    }
  }, [selectedModel?.model, loading]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!isOpen) return;

      if (event.key === "Escape") {
        event.preventDefault();
        setIsOpen(false);
        onEscape?.();
        return;
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onEscape]);

  const handleModelSelect = (model: Model) => {
    setSettings({ SelectedModel: model.model });
    setIsOpen(false);
    onModelSelect?.();
  };

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        ref={ref}
        type="button"
        title="Select model"
        onClick={() => {
          const newState = !isOpen;
          setIsOpen(newState);
          onDropdownToggle?.(newState);
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            const newState = !isOpen;
            setIsOpen(newState);
            onDropdownToggle?.(newState);
          }
        }}
        onMouseDown={(e) => e.stopPropagation()}
        onDoubleClick={(e) => e.stopPropagation()}
        className="flex items-center select-none gap-1.5 rounded-full px-3.5 py-1.5 bg-white dark:bg-neutral-700 text-neutral-800 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:text-neutral-100 cursor-pointer"
      >
        <div className="flex items-center gap-2">
          <span>
            {isDisabled
              ? "Loading..."
              : selectedModel?.model || "Select a model"}
          </span>
        </div>
        <svg
          className="h-3 w-3 opacity-70"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M19 9l-7 7-7-7"
          />
        </svg>
      </button>
      {isOpen && (
        <div className="absolute right-0 text-[15px] bottom-full mb-2 z-50 w-64 rounded-2xl overflow-hidden bg-white border border-neutral-100 text-neutral-800 shadow-xl shadow-black/5 backdrop-blur-lg dark:border-neutral-600/40 dark:bg-neutral-800 dark:text-white dark:ring-black/20">
          <div className="px-1 py-2 border-b border-neutral-100 dark:border-neutral-700">
            <input
              ref={searchInputRef}
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Find model..."
              autoCorrect="off"
              className="w-full px-2 py-0.5 bg-transparent border-none border-neutral-200 rounded-md outline-none focus:border-neutral-400 dark:border-neutral-600 dark:focus:border-neutral-400"
            />
          </div>

          <ModelList
            ref={modelListRef}
            models={models}
            selectedModel={selectedModel}
            onModelSelect={handleModelSelect}
            airplaneMode={settings.airplaneMode}
            isOpen={isOpen}
          />
        </div>
      )}
    </div>
  );
});

export const ModelList = forwardRef(function ModelList(
  {
    models,
    selectedModel,
    onModelSelect,
    airplaneMode,
    isOpen,
  }: {
    models: Model[];
    selectedModel: Model | null;
    onModelSelect: (model: Model) => void;
    airplaneMode: boolean;
    isOpen: boolean;
  },
  ref,
): JSX.Element {
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const [highlightedIndex, setHighlightedIndex] = useState(-1);

  useImperativeHandle(ref, () => ({
    scrollToSelectedModel: () => {
      if (!selectedModel || !scrollContainerRef.current) return;
      const selectedIndex = models.findIndex(
        (m) => m.model === selectedModel.model,
      );
      if (selectedIndex !== -1) scrollToItem(selectedIndex);
    },
    scrollToTop: () => {
      if (scrollContainerRef.current) scrollContainerRef.current.scrollTop = 0;
    },
  }));

  // Handle keyboard navigation
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!isOpen || models.length === 0) return;

      switch (event.key) {
        case "ArrowDown":
          event.preventDefault();
          setHighlightedIndex((prev) => {
            const next = prev < models.length - 1 ? prev + 1 : 0;
            scrollToItem(next);
            return next;
          });
          break;
        case "ArrowUp":
          event.preventDefault();
          setHighlightedIndex((prev) => {
            const next = prev > 0 ? prev - 1 : models.length - 1;
            scrollToItem(next);
            return next;
          });
          break;
        case "Enter":
          event.preventDefault();
          if (highlightedIndex >= 0 && highlightedIndex < models.length) {
            onModelSelect(models[highlightedIndex]);
          }
          break;
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, models, highlightedIndex, onModelSelect]);

  // Scroll active item into view
  const scrollToItem = (index: number) => {
    if (scrollContainerRef.current && index >= 0) {
      const container = scrollContainerRef.current;
      const item = container.children[index] as HTMLElement;
      if (item) {
        // Calculate the exact scroll position to center the item
        const containerHeight = container.clientHeight;
        const itemTop = item.offsetTop;
        const itemHeight = item.clientHeight;
        // Position the item in the center of the container
        container.scrollTop = itemTop - containerHeight / 2 + itemHeight / 2;
      }
    }
  };

  return (
    <div
      ref={scrollContainerRef}
      className="h-64 overflow-y-auto overflow-x-hidden"
    >
      {models.length === 0 ? (
        <div className="px-3 py-2 text-neutral-500 dark:text-neutral-400">
          No models found
        </div>
      ) : (
        models.map((model, index) => {
          return (
            <div key={`${model.model}-${model.digest || "no-digest"}-${index}`}>
              <button
                onClick={() => onModelSelect(model)}
                onMouseEnter={() => setHighlightedIndex(index)}
                className={`flex w-full items-center gap-2 px-3 py-2 hover:bg-neutral-100 dark:hover:bg-neutral-700/60 focus:outline-none cursor-pointer ${
                  highlightedIndex === index ||
                  selectedModel?.model === model.model
                    ? "bg-neutral-100 dark:bg-neutral-700/60"
                    : ""
                }`}
              >
                <span className="flex-1 text-left truncate min-w-0">
                  {model.model}
                </span>
                {model.isCloud() && (
                  <svg
                    className="h-3 fill-current text-neutral-500 dark:text-neutral-400"
                    viewBox="0 0 20 15"
                    strokeWidth={1}
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path d="M4.01511 14.5861H14.2304C16.9183 14.5861 19.0002 12.5509 19.0002 9.9403C19.0002 7.30491 16.8911 5.3046 14.0203 5.3046C12.9691 3.23016 11.0602 2 8.69505 2C5.62816 2 3.04822 4.32758 2.72935 7.47455C1.12954 7.95356 0.0766602 9.29431 0.0766602 10.9757C0.0766602 12.9913 1.55776 14.5861 4.01511 14.5861ZM4.02056 13.1261C2.46452 13.1261 1.53673 12.2938 1.53673 11.0161C1.53673 9.91553 2.24207 9.12934 3.51367 8.79302C3.95684 8.68258 4.11901 8.48427 4.16138 8.00729C4.39317 5.3613 6.29581 3.46007 8.69505 3.46007C10.5231 3.46007 11.955 4.48273 12.8385 6.26013C13.0338 6.65439 13.2626 6.7882 13.7488 6.7882C16.1671 6.7882 17.5337 8.19719 17.5337 9.97707C17.5337 11.7526 16.1242 13.1261 14.2852 13.1261H4.02056Z" />
                  </svg>
                )}
                {model.digest === undefined &&
                  (airplaneMode || !model.isCloud()) && (
                    <ArrowDownTrayIcon
                      className="h-4 w-4 text-neutral-500 dark:text-neutral-400"
                      strokeWidth={1.75}
                    />
                  )}
              </button>
            </div>
          );
        })
      )}
    </div>
  );
});
