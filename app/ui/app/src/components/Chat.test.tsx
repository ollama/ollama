import type { ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, expect, it, vi } from "vitest";
import { ErrorEvent } from "@/gotypes";
import Chat from "./Chat";

vi.mock("@tanstack/react-router", () => ({ useNavigate: () => vi.fn() }));
vi.mock("@/hooks/useSelectedModel", () => ({
  useSelectedModel: () => ({ selectedModel: null }),
}));
vi.mock("@/hooks/useUser", () => ({ useUser: () => ({ user: null }) }));
vi.mock("@/hooks/useHealth", () => ({
  useHealth: () => ({ isHealthy: true }),
}));
vi.mock("@/hooks/useModelCapabilities", () => ({
  useHasVisionCapability: () => false,
}));
vi.mock("@/hooks/useMessageAutoscroll", () => ({
  useMessageAutoscroll: () => ({
    containerRef: { current: null },
    spacerHeight: 0,
    handleNewUserMessage: vi.fn(),
  }),
}));
vi.mock("./FileUpload", () => ({
  FileUpload: ({ children }: { children: ReactNode }) => <>{children}</>,
}));
vi.mock("./ChatForm", () => ({ default: () => <form data-chat-form /> }));
vi.mock("./MessageList", () => ({ default: () => null }));

import { StreamingProvider } from "@/contexts/StreamingContext";

afterEach(() => vi.unstubAllGlobals());

it.each(["Load failed", null])(
  "renders new-chat request errors beside the form: %s",
  (message) => {
    vi.stubGlobal("navigator", { platform: "MacIntel" });
    const client = new QueryClient();
    client.setQueryData(
      ["chatError", ""],
      message ? new ErrorEvent({ eventName: "error", error: message }) : null,
    );
    try {
      const html = renderToStaticMarkup(
        <QueryClientProvider client={client}>
          <StreamingProvider>
            <Chat chatId="new" />
          </StreamingProvider>
        </QueryClientProvider>,
      );
      expect(html).toContain("data-chat-form");
      if (message) expect(html).toContain(message);
      else expect(html).not.toContain("<h3>Error</h3>");
    } finally {
      client.clear();
    }
  },
);
