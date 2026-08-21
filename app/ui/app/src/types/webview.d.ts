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
  connected: boolean;
  running: boolean;
  startFailed: boolean;
  portConflict: boolean;
  gatewayPort?: number;
  error?: string;
  modelSource?: "user" | "endpoint" | "fallback";
  maxModels?: number;
  models?: ClaudeDesktopModelStatus[];
}

interface ClaudeDesktopModelStatus {
  name: string;
  displayName: string;
  description?: string;
  cloud?: boolean;
  selected: boolean;
}

interface ClaudeDesktopActionResult {
  status: ClaudeDesktopStatus;
  error?: string;
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
    setClaudeDesktopConnected?: (
      enabled: boolean,
    ) => Promise<ClaudeDesktopActionResult>;
    openClaudeDesktop?: () => Promise<string>;
    installClaudeDesktop?: () => Promise<ClaudeDesktopInstallResult>;
    getShowAppsInMenu?: () => Promise<boolean>;
    setShowAppsInMenu?: (visible: boolean) => Promise<void>;
    restartClaudeDesktop?: (
      models: string[],
    ) => Promise<ClaudeDesktopActionResult>;
    setOnboardingWindow?: (enabled: boolean) => void;
    menu: (items: MenuItem[]) => Promise<string | null>;
    OLLAMA_TOOLS?: boolean;
    OLLAMA_WEBSEARCH?: boolean;
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
  ClaudeDesktopModelStatus,
  ClaudeDesktopStatus,
  ContextMenuItem,
  ContextMenuResult,
  ImageData,
  WebviewAPI,
};
