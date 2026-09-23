import type { PropsWithChildren } from "react";
import { act, create, type ReactTestRenderer } from "react-test-renderer";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
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
  expect(page()).toContain("This chat is read-only.");
  expect(renderer.root.findAllByType("textarea")).toHaveLength(0);
  const buttons = renderer.root
    .findAllByType("button")
    .map((button) => button.props["aria-label"] || button.children.join(" "))
    .join(" ");
  expect(buttons).not.toMatch(/Send|Edit|Retry|New Chat|Continue|ChatGPT/);
  expect(renderer.root.findAllByType("a").map((a) => a.props.href)).toEqual(
    expect.arrayContaining(["/connect", "/settings"]),
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
  await click("Export");
  expect(exportChat).toHaveBeenCalledExactlyOnceWith("first");
  expect(page()).toContain("Exporting…");
  expect(
    renderer.root.findByProps({ title: "Another conversation" }).props.disabled,
  ).toBe(true);
  await act(async () => resolve({ path: "/exports/Garden" }));
  expect(page()).toContain("/exports/Garden");
  vi.mocked(exportChat).mockRejectedValueOnce(new Error("Disk is full"));
  await click("Export");
  expect(page()).toContain("Disk is full");
  await click("Another conversation");
  expect(page()).not.toContain("Disk is full");
  expect(page()).not.toContain("/exports/Garden");
  vi.mocked(exportChat).mockResolvedValueOnce(null);
  await click("Export");
  expect(exportChat).toHaveBeenLastCalledWith("second");
  expect(page()).not.toContain("Saved to");
  expect(renderer.root.findAllByProps({ role: "alert" })).toHaveLength(0);
});

it("keeps deletion, including cancelling and selecting the next saved chat", async () => {
  await renderHistory();
  const remove = async () => {
    await act(async () => {
      await renderer.root
        .findByProps({ title: "Garden" })
        .parent!.props.onContextMenu({ preventDefault: vi.fn() });
    });
  };
  vi.mocked(window.confirm).mockReturnValueOnce(false);
  await remove();
  expect(deleteChat).not.toHaveBeenCalled();
  await remove();
  expect(window.menu).toHaveBeenCalledWith([
    { label: "Delete", enabled: true },
  ]);
  expect(deleteChat).toHaveBeenCalledExactlyOnceWith("first");
  expect(page()).not.toContain("Garden");
  expect(page()).toContain("Second chat");
});
