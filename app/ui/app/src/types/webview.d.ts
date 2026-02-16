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

export type { ImageData, WebviewAPI, ContextMenuItem, ContextMenuResult };
