import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { page, userEvent } from "@vitest/browser/context";
import { Model } from "@/gotypes";
import { type PullModelFn, useModelDownload } from "@/hooks/useModelDownload";
import { type DownloadState } from "@/lib/modelDownload";
import { ModelList } from "./ModelPicker";

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  vi.clearAllMocks();
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  vi.unstubAllGlobals();
});

const notInstalled = new Model({ model: "qwen3:9b" });
const installed = new Model({ model: "llama3", digest: "abc123" });

async function renderList(options: {
  downloads?: Record<string, DownloadState>;
  onModelSelect?: (model: Model) => void;
  onDownload?: (modelName: string) => void;
  onCancelDownload?: (modelName: string) => void;
}) {
  const onModelSelect = options.onModelSelect ?? vi.fn();
  const onDownload = options.onDownload ?? vi.fn();
  const onCancelDownload = options.onCancelDownload ?? vi.fn();

  await act(async () =>
    root.render(
      <ModelList
        models={[notInstalled, installed]}
        selectedModel={null}
        onModelSelect={onModelSelect}
        cloudDisabled={false}
        isOpen
        downloads={options.downloads ?? {}}
        onDownload={onDownload}
        onCancelDownload={onCancelDownload}
      />,
    ),
  );

  return { onModelSelect, onDownload, onCancelDownload };
}

it("offers an explicit download action only for models that are not installed", async () => {
  const { onModelSelect, onDownload } = await renderList({});

  const download = page.getByRole("button", { name: "Download qwen3:9b" });
  await expect.element(download).toBeEnabled();
  expect(container.querySelectorAll('[aria-label^="Download "]')).toHaveLength(
    1,
  );

  await act(async () => {
    await userEvent.click(download);
  });

  expect(onDownload).toHaveBeenCalledWith("qwen3:9b");
  expect(onModelSelect).not.toHaveBeenCalled();
});

it("keeps row selection working for installed models", async () => {
  const { onModelSelect } = await renderList({});

  const row = page.getByRole("button", { name: "llama3" });
  await act(async () => {
    await userEvent.click(row);
  });

  expect(onModelSelect).toHaveBeenCalledWith(installed);
});

it("shows layer progress with a cancel action while downloading", async () => {
  const { onModelSelect, onCancelDownload } = await renderList({
    downloads: {
      "qwen3:9b": {
        status: "downloading",
        statusText: "downloading",
        completed: 50,
        total: 100,
      },
    },
  });

  expect(container.textContent).toContain("50 B / 100 B (50%)");

  const cancel = page.getByRole("button", {
    name: "Cancel download of qwen3:9b",
  });
  await act(async () => {
    await userEvent.click(cancel);
  });

  expect(onCancelDownload).toHaveBeenCalledWith("qwen3:9b");
  expect(onModelSelect).not.toHaveBeenCalled();
});

it("offers retry when a download fails", async () => {
  const { onDownload } = await renderList({
    downloads: {
      "qwen3:9b": { status: "error", error: "Failed to pull model" },
    },
  });

  expect(container.textContent).toContain("Failed");

  const retry = page.getByRole("button", {
    name: "Retry download of qwen3:9b",
  });
  await act(async () => {
    await userEvent.click(retry);
  });

  expect(onDownload).toHaveBeenCalledWith("qwen3:9b");
});

it("shows an installed marker while the model list refreshes", async () => {
  await renderList({
    downloads: { "qwen3:9b": { status: "completed" } },
  });

  expect(container.textContent).toContain("Installed");
  expect(container.querySelectorAll('[aria-label^="Download "]')).toHaveLength(
    0,
  );
});

function Probe({
  onCompleted,
  pull,
}: {
  onCompleted: (modelName: string) => void;
  pull: PullModelFn;
}) {
  const { states, start, cancel } = useModelDownload(onCompleted, pull);
  const state = states["qwen3:9b"];

  return (
    <div>
      <button onClick={() => start("qwen3:9b")}>Start download</button>
      <button onClick={() => cancel("qwen3:9b")}>Cancel download</button>
      <span>{state?.status ?? "idle"}</span>
      {state?.error && <span>{state.error}</span>}
    </div>
  );
}

const successPull: PullModelFn = () =>
  (async function* () {
    yield { status: "pulling manifest" };
    yield { status: "downloading", completed: 50, total: 100 };
    yield { status: "success" };
  })();

const blockingPull: PullModelFn = (_modelName, signal) =>
  (async function* () {
    yield { status: "pulling manifest" };
    await new Promise<void>((_resolve, reject) => {
      signal?.addEventListener("abort", () =>
        reject(new DOMException("Aborted", "AbortError")),
      );
    });
  })();

it("runs the pull stream and reports completion", async () => {
  const pull = vi.fn(successPull);
  const onCompleted = vi.fn();

  await act(async () =>
    root.render(<Probe onCompleted={onCompleted} pull={pull} />),
  );
  await act(async () => {
    await userEvent.click(page.getByRole("button", { name: "Start download" }));
  });

  expect(pull).toHaveBeenCalledTimes(1);
  expect(pull.mock.calls[0][0]).toBe("qwen3:9b");
  expect(container.textContent).toContain("completed");
  expect(onCompleted).toHaveBeenCalledWith("qwen3:9b");
});

it("does not start a duplicate pull for the same model", async () => {
  const pull = vi.fn(blockingPull);
  const onCompleted = vi.fn();

  await act(async () =>
    root.render(<Probe onCompleted={onCompleted} pull={pull} />),
  );
  await act(async () => {
    await userEvent.click(page.getByRole("button", { name: "Start download" }));
  });
  await act(async () => {
    await userEvent.click(page.getByRole("button", { name: "Start download" }));
  });

  expect(pull).toHaveBeenCalledTimes(1);
  expect(onCompleted).not.toHaveBeenCalled();
});

it("returns to the available state when a download is cancelled", async () => {
  const pull = vi.fn(blockingPull);
  const onCompleted = vi.fn();

  await act(async () =>
    root.render(<Probe onCompleted={onCompleted} pull={pull} />),
  );
  await act(async () => {
    await userEvent.click(page.getByRole("button", { name: "Start download" }));
  });
  expect(container.textContent).toContain("downloading");

  await act(async () => {
    await userEvent.click(
      page.getByRole("button", { name: "Cancel download" }),
    );
  });

  expect(container.textContent).toContain("idle");
  expect(onCompleted).not.toHaveBeenCalled();
});

it("surfaces a failed pull so the row can offer retry", async () => {
  const pull: PullModelFn = () => {
    throw new Error("Failed to pull model: 500");
  };
  const onCompleted = vi.fn();

  await act(async () =>
    root.render(<Probe onCompleted={onCompleted} pull={pull} />),
  );
  await act(async () => {
    await userEvent.click(page.getByRole("button", { name: "Start download" }));
  });

  expect(container.textContent).toContain("error");
  expect(container.textContent).toContain("Failed to pull model: 500");
  expect(onCompleted).not.toHaveBeenCalled();
});
