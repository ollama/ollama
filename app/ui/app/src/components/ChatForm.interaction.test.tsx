import { act, create } from "react-test-renderer";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ChatForm from "./ChatForm";

const mocks = vi.hoisted(() => {
  const draftStore = new Map<string, unknown>();
  return {
    draftStore,
    queryClient: {
      getQueryData: (key: unknown[]) => draftStore.get(JSON.stringify(key)),
      setQueryData: (key: unknown[], value: unknown) =>
        draftStore.set(JSON.stringify(key), value),
      cancelQueries: vi.fn().mockResolvedValue(undefined),
      invalidateQueries: vi.fn(),
    },
    onSubmit: vi.fn(),
    sendMessageMutate: vi.fn(),
  };
});

vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => mocks.queryClient,
  useMutation: () => ({ mutate: mocks.sendMessageMutate }),
}));

vi.mock("@tanstack/react-router", () => ({
  useNavigate: () => vi.fn(),
}));

vi.mock("@/hooks/useChats", () => ({
  useSendMessage: () => ({ mutate: mocks.sendMessageMutate }),
  useIsStreaming: () => false,
  useCancelMessage: () => vi.fn(),
}));

vi.mock("@/hooks/useSelectedModel", () => ({
  useSelectedModel: () => ({
    selectedModel: { model: "llama3", isCloud: () => false },
  }),
}));

vi.mock("@/hooks/useModelCapabilities", () => ({
  useHasVisionCapability: () => false,
  useHasToolsCapability: () => false,
}));

vi.mock("@/hooks/useUser", () => ({
  useUser: () => ({ isAuthenticated: true, isLoading: false }),
}));

vi.mock("@/hooks/useSettings", () => ({
  useSettings: () => ({
    settings: {
      webSearchEnabled: false,
      thinkEnabled: false,
      thinkLevel: "medium",
    },
    setSettings: vi.fn(),
  }),
}));

vi.mock("@/hooks/useCloudStatus", () => ({
  useCloudStatus: () => ({ cloudDisabled: false }),
}));

vi.mock("@/utils/fileValidation", () => ({
  processFiles: vi.fn(async (files: unknown[]) => ({
    validFiles: files,
    errors: [],
  })),
}));

vi.mock("@/utils/imageUtils", () => ({
  isImageFile: () => false,
}));

vi.mock("@/gotypes", () => ({
  ErrorEvent: class {
    opts: Record<string, unknown>;
    constructor(opts: Record<string, unknown>) {
      this.opts = opts;
    }
  },
  Message: class {},
}));

vi.mock("@/components/Logo", () => ({ default: () => null }));
vi.mock("@/components/ModelPicker", () => ({
  ModelPicker: () => null,
}));
vi.mock("@/components/WebSearchButton", () => ({
  WebSearchButton: () => null,
}));
vi.mock("@/components/ImageThumbnail", () => ({
  ImageThumbnail: () => null,
}));
vi.mock("./ThinkButton", () => ({ ThinkButton: () => null }));
vi.mock("./ErrorMessage", () => ({ ErrorMessage: () => null }));
vi.mock("@/components/DisplayLogin", () => ({
  DisplayLogin: () => null,
}));

const stubGlobals = () => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("window", {
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    setTimeout: globalThis.setTimeout.bind(globalThis),
    clearTimeout: globalThis.clearTimeout.bind(globalThis),
  });
  vi.stubGlobal("document", {
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  });
};

const typeInto = async (
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  renderer: any,
  value: string,
) => {
  const textarea = renderer.root.findByType("textarea");
  await act(async () => {
    textarea.props.onChange({
      target: { value, style: {} as CSSStyleDeclaration },
    });
  });
};

describe("ChatForm draft persistence", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.draftStore.clear();
    stubGlobals();
  });

  it("restores a typed draft after unmounting and remounting for the same chat", async () => {
    let renderer: ReturnType<typeof create> | undefined;
    try {
      await act(async () => {
        renderer = create(
          <ChatForm
            hasMessages={true}
            chatId="chat-a"
            onSubmit={mocks.onSubmit}
          />,
        );
      });

      await typeInto(renderer, "hello from chat A");
      await act(async () => {
        renderer!.unmount();
      });

      let restored: ReturnType<typeof create> | undefined;
      await act(async () => {
        restored = create(
          <ChatForm
            hasMessages={true}
            chatId="chat-a"
            onSubmit={mocks.onSubmit}
          />,
        );
      });

      const textarea = restored!.root.findByType("textarea");
      expect(textarea.props.value).toBe("hello from chat A");
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("keeps separate drafts per chat when switching between chats", async () => {
    let renderer: ReturnType<typeof create> | undefined;
    try {
      await act(async () => {
        renderer = create(
          <ChatForm
            hasMessages={true}
            chatId="chat-a"
            onSubmit={mocks.onSubmit}
          />,
        );
      });

      await typeInto(renderer, "draft for A");

      await act(async () => {
        renderer!.update(
          <ChatForm
            hasMessages={true}
            chatId="chat-b"
            onSubmit={mocks.onSubmit}
          />,
        );
      });

      // Chat B has no draft yet
      expect(renderer!.root.findByType("textarea").props.value).toBe("");

      // Switch back to chat A: the draft should still be there
      await act(async () => {
        renderer!.update(
          <ChatForm
            hasMessages={true}
            chatId="chat-a"
            onSubmit={mocks.onSubmit}
          />,
        );
      });

      expect(renderer!.root.findByType("textarea").props.value).toBe(
        "draft for A",
      );
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("clears the saved draft after the message is submitted", async () => {
    let renderer: ReturnType<typeof create> | undefined;
    try {
      await act(async () => {
        renderer = create(
          <ChatForm
            hasMessages={true}
            chatId="chat-a"
            onSubmit={mocks.onSubmit}
          />,
        );
      });

      await typeInto(renderer, "to be sent");

      const textarea = renderer!.root.findByType("textarea");
      await act(async () => {
        textarea.props.onKeyDown({
          key: "Enter",
          shiftKey: false,
          preventDefault: vi.fn(),
        });
      });

      expect(mocks.onSubmit).toHaveBeenCalledWith(
        "to be sent",
        expect.any(Object),
      );
      expect(renderer!.root.findByType("textarea").props.value).toBe("");

      // Remount: the draft must not come back
      await act(async () => {
        renderer!.unmount();
      });
      let remounted: ReturnType<typeof create> | undefined;
      await act(async () => {
        remounted = create(
          <ChatForm
            hasMessages={true}
            chatId="chat-a"
            onSubmit={mocks.onSubmit}
          />,
        );
      });
      expect(remounted!.root.findByType("textarea").props.value).toBe("");
    } finally {
      vi.unstubAllGlobals();
    }
  });
});
