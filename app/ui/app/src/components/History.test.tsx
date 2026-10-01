import type { PropsWithChildren } from "react";
import { act, create, type ReactTestRenderer } from "react-test-renderer";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { Navigate } from "@tanstack/react-router";
import {
  QueryClient,
  QueryClientProvider,
  defaultScheduler,
  notifyManager,
} from "@tanstack/react-query";
import { Chat, ChatInfo } from "@/gotypes";
import {
  deleteChat,
  exportChat,
  getChat,
  getChats,
  type ExportResult,
} from "@/api";
import { History } from "./History";

vi.mock("@/api", () => ({
  deleteChat: vi.fn(),
  exportChat: vi.fn(),
  getChat: vi.fn(),
  getChats: vi.fn(),
}));
// Markdown's CSS requires a browser; retain the real message and sidebar UI.
vi.mock("./StreamingMarkdownContent", () => ({
  default: ({ content }: { content: string }) => <div>{content}</div>,
}));
vi.mock("@/components/ui/link", () => ({
  Link: ({ to, children }: PropsWithChildren<{ to: string }>) => (
    <a href={to}>{children}</a>
  ),
}));
vi.mock("@tanstack/react-router", () => ({
  Navigate: () => null,
}));

const date = "2026-09-01T10:00:00Z";
const first = new Chat({
  id: "first",
  title: "Garden",
  created_at: date,
  messages: [
    { role: "user", content: "Keep this history", created_at: date },
    {
      role: "assistant",
      content: "Saved reply",
      model: "llama3.2",
      created_at: date,
    },
    {
      role: "assistant",
      content: "Latest reply",
      model: "gpt-oss:120b-cloud",
      created_at: date,
    },
  ],
});
const second = new Chat({
  id: "second",
  title: "Another conversation",
  created_at: date,
  messages: [{ role: "user", content: "Second chat", created_at: date }],
});
let renderer: ReactTestRenderer;
let client: QueryClient;

beforeEach(() => {
  client = new QueryClient();
  notifyManager.setScheduler(queueMicrotask);
  vi.useFakeTimers();
  vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) =>
    setTimeout(() => callback(0), 16),
  );
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("window", {
    OLLAMA_PLATFORM: "darwin",
    menu: vi.fn().mockResolvedValue("Delete"),
    confirm: vi.fn().mockReturnValue(true),
    setInterval,
    clearInterval,
    setTimeout,
    clearTimeout,
  });
  vi.mocked(getChats)
    .mockReset()
    .mockResolvedValue(
      [first, second].map(
        (chat) =>
          new ChatInfo({
            id: chat.id,
            title: chat.title,
            createdAt: date,
            updatedAt: date,
          }),
      ),
    );
  vi.mocked(getChat)
    .mockReset()
    .mockImplementation(async (id) => (id === first.id ? first : second));
  vi.mocked(exportChat)
    .mockReset()
    .mockResolvedValue({ path: "/exports/Garden" });
  vi.mocked(deleteChat).mockReset().mockResolvedValue(undefined);
});

afterEach(async () => {
  if (renderer) await act(async () => renderer.unmount());
  client.clear();
  notifyManager.setScheduler(defaultScheduler);
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

async function renderHistory() {
  await act(async () => {
    renderer = create(
      <QueryClientProvider client={client}>
        <History chatId="first" />
      </QueryClientProvider>,
    );
  });
  if (renderer.root.findAllByProps({ "aria-label": "Show sidebar" }).length) {
    await click("Show sidebar");
  }
}
function page() {
  return JSON.stringify(renderer.toJSON());
}
async function click(label: string) {
  const button = renderer.root
    .findAllByType("button")
    .find(
      (node) => node.children.includes(label) || node.props.title === label,
    );
  expect(button).toBeDefined();
  await act(async () => {
    await button!.props.onClick();
  });
}

it("shows saved messages and the last model without live chat controls", async () => {
  await renderHistory();
  expect(page()).toContain("Keep this history");
  expect(page()).toContain("Latest reply");
  expect(page()).toContain("Last used:");
  expect(page()).toContain("gpt-oss:120b-cloud");
  expect(renderer.root.findAllByType("textarea")).toHaveLength(0);
  const buttons = renderer.root
    .findAllByType("button")
    .map((button) => button.props["aria-label"] || button.children.join(" "))
    .join(" ");
  expect(buttons).not.toMatch(/Send|Edit|Retry|New Chat|Continue|ChatGPT/);
  expect(renderer.root.findAllByType("a").map((a) => a.props.href)).toEqual(
    expect.arrayContaining([
      "/connect",
      "/settings",
      "https://docs.ollama.com/integrations",
    ]),
  );
  await click("Another conversation");
  expect(page()).toContain("Second chat");
  expect(page()).not.toContain("Last used:");
});

it("exports the selected chat and keeps progress and errors scoped to it", async () => {
  let resolve!: (value: ExportResult | null) => void;
  vi.mocked(exportChat).mockImplementationOnce(
    () =>
      new Promise((accept) => {
        resolve = accept;
      }),
  );
  await renderHistory();
  notifyManager.setScheduler(defaultScheduler);
  await click("Export");
  expect(page()).toContain("Exporting…");
  expect(renderer.root.findByProps({ "aria-busy": true }).props.disabled).toBe(
    true,
  );
  expect(exportChat).not.toHaveBeenCalled();
  expect(
    renderer.root.findByProps({ title: "Another conversation" }).props.disabled,
  ).toBe(true);
  await click("Exporting…");
  await act(async () => vi.advanceTimersByTimeAsync(20));
  expect(exportChat).not.toHaveBeenCalled();
  await act(async () => vi.advanceTimersByTimeAsync(20));
  expect(exportChat).toHaveBeenCalledExactlyOnceWith("first");
  notifyManager.setScheduler(queueMicrotask);
  await act(async () => resolve({ path: "/exports/Garden" }));
  expect(page()).not.toContain("/exports/Garden");
  expect(page()).not.toContain("Saved to");
  vi.mocked(exportChat).mockRejectedValueOnce(new Error("Disk is full"));
  await click("Export");
  await act(async () => vi.advanceTimersByTimeAsync(40));
  expect(page()).toContain("Disk is full");
  expect(renderer.root.findAllByProps({ "aria-busy": true })).toHaveLength(0);
  await click("Another conversation");
  expect(page()).not.toContain("Disk is full");
  expect(page()).not.toContain("/exports/Garden");
  vi.mocked(exportChat).mockResolvedValueOnce(null);
  await click("Export");
  await act(async () => vi.advanceTimersByTimeAsync(40));
  expect(exportChat).toHaveBeenLastCalledWith("second");
  expect(renderer.root.findAllByProps({ "aria-busy": true })).toHaveLength(0);
  expect(page()).not.toContain("Saved to");
  expect(renderer.root.findAllByProps({ role: "alert" })).toHaveLength(0);
});

it("keeps deletion, selecting the next saved chat or opening Apps after the last one", async () => {
  await renderHistory();
  const remove = async (title = "Garden") => {
    await act(async () => {
      await renderer.root
        .findByProps({ title })
        .parent!.props.onContextMenu({ preventDefault: vi.fn() });
    });
  };
  vi.mocked(window.confirm).mockReturnValueOnce(false);
  await remove();
  expect(deleteChat).not.toHaveBeenCalled();

  // A background refresh can return the old list after deletion completes.
  const staleChats = client.getQueryData<ChatInfo[]>(["history-chats"])!;
  let finishRefresh!: (chats: ChatInfo[]) => void;
  vi.mocked(getChats).mockImplementationOnce(
    () =>
      new Promise((resolve) => {
        finishRefresh = resolve;
      }),
  );
  await act(async () => {
    void client.refetchQueries({ queryKey: ["history-chats"] });
  });
  await remove();
  expect(window.menu).toHaveBeenCalledWith([
    { label: "Delete", enabled: true },
  ]);
  expect(deleteChat).toHaveBeenCalledExactlyOnceWith("first");
  expect(page()).not.toContain("Garden");
  expect(page()).toContain("Second chat");
  await act(async () => finishRefresh(staleChats));
  expect(page()).not.toContain("Garden");
  await remove("Another conversation");
  expect(deleteChat).toHaveBeenLastCalledWith("second");
  expect(renderer.root.findByType(Navigate).props).toMatchObject({
    to: "/connect",
    replace: true,
  });
});
