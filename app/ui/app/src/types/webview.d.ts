// Type declarations for webview API functions

interface ImageData {
  filename: string;
  path: string;
  dataURL: string; // base64 encoded file data
}

interface MenuItem {
  label: string;
  enabled?: boolean;
  separator?: boolean;
}

interface ClaudeDesktopStatus {
  supported: boolean;
  used: boolean;
  installed: boolean;
  configured?: boolean;
  connected: boolean;
  running: boolean;
  startFailed: boolean;
  portConflict: boolean;
  gatewayPort?: number;
  routedRequests?: number;
  error?: string;
  autoMode?: boolean;
  modelSource?: "user" | "endpoint" | "fallback";
  maxModels?: number;
  models?: ClaudeDesktopModelStatus[];
  mappings?: ClaudeDesktopMappingStatus[];
}

interface ClaudeDesktopMappingStatus {
  routeId: string;
  routeName: string;
  model?: string;
}

interface ClaudeDesktopModelStatus {
  name: string;
  displayName: string;
  description?: string;
  cloud?: boolean;
  selected: boolean;
  autoMode?: boolean;
  availability?: "unknown" | "available" | "unavailable";
  reason?:
    | "cloud_off"
    | "sign_in_required"
    | "upgrade_required"
    | "verification_unavailable"
    | "model_not_installed";
  requiredPlan?: string;
}

interface ClaudeDesktopActionResult {
  status: ClaudeDesktopStatus;
  error?: string;
  mappingsApplied?: boolean;
  restartConfirmationRequired?: boolean;
}

type ClaudeDesktopInstallResult = "opened" | "cancelled" | "failed";

interface WebviewAPI {
  selectFile: () => Promise<ImageData | null>;
  selectMultipleFiles: () => Promise<ImageData[] | null>;
  selectModelsDirectory: () => Promise<string | null>;
  selectWorkingDirectory: () => Promise<string | null>;
}

declare global {
  interface Window {
    webview?: WebviewAPI;
    drag?: () => void;
    doubleClick?: () => void;
    activateOllama?: () => void;
    getClaudeDesktopStatus?: () => Promise<ClaudeDesktopStatus>;
    getClaudeDesktopConnectionSummary?: () => Promise<ClaudeDesktopStatus>;
    getClaudeDesktopRequestCount?: () => Promise<number>;
    setClaudeDesktopConnected?: (
      enabled: boolean,
      restartConfirmed: boolean,
    ) => Promise<ClaudeDesktopActionResult>;
    prepareClaudeDesktopConnection?: () => Promise<ClaudeDesktopActionResult>;
    openClaudeDesktop?: () => Promise<string>;
    installClaudeDesktop?: () => Promise<ClaudeDesktopInstallResult>;
    getShowAppsInMenu?: () => Promise<boolean>;
    setShowAppsInMenu?: (visible: boolean) => Promise<void>;
    applyClaudeDesktopMappings?: (
      mappings: Record<string, string>,
      restartConfirmed: boolean,
    ) => Promise<ClaudeDesktopActionResult>;
    resetClaudeDesktopMappings?: (
      restartConfirmed: boolean,
    ) => Promise<ClaudeDesktopActionResult>;
    setClaudeDesktopAutoMode?: (
      enabled: boolean,
      restartConfirmed: boolean,
    ) => Promise<ClaudeDesktopActionResult>;
    setOnboardingWindow?: (enabled: boolean) => void;
    menu: (items: MenuItem[]) => Promise<string | null>;
    OLLAMA_TOOLS?: boolean;
    OLLAMA_WEBSEARCH?: boolean;
    OLLAMA_PLATFORM?: "darwin" | "windows";
  }

  namespace JSX {
    interface IntrinsicElements {
      input: React.DetailedHTMLProps<
        React.InputHTMLAttributes<HTMLInputElement> & {
          webkitdirectory?: string;
          directory?: string;
        },
        HTMLInputElement
      >;
    }
  }

  interface File {
    readonly webkitRelativePath: string;
  }
}

export type {
  ClaudeDesktopActionResult,
  ClaudeDesktopInstallResult,
  ClaudeDesktopMappingStatus,
  ClaudeDesktopModelStatus,
  ClaudeDesktopStatus,
  ContextMenuItem,
  ContextMenuResult,
  ImageData,
  WebviewAPI,
};
