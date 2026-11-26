import { Model } from "@/gotypes";
import { page, userEvent } from "@vitest/browser/context";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { ModelList } from "./ModelPicker";

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
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

it("labels a local model that has not been downloaded without adding a download button", async () => {
  const local = new Model({ model: "available-local" });
  const installed = new Model({
    model: "installed-local",
    digest: "installed",
  });
  const cloud = new Model({ model: "available-cloud" });
  const onModelSelect = vi.fn();

  await act(async () =>
    root.render(
      <ModelList
        models={[local, installed, cloud]}
        selectedModel={null}
        onModelSelect={onModelSelect}
        cloudDisabled={false}
        isOpen={true}
      />,
    ),
  );

  const localRow = page.getByRole("button", { name: /available-local/ });
  const status = localRow.getByText("Not downloaded", { exact: true });
  await expect.element(status).toBeVisible();
  await expect
    .element(status)
    .toHaveAttribute("title", "Downloads on first use");
  expect(container.querySelectorAll("button")).toHaveLength(3);
  await expect
    .element(page.getByRole("button", { name: "installed-local", exact: true }))
    .toBeVisible();
  await expect
    .element(page.getByRole("button", { name: "available-cloud", exact: true }))
    .toBeVisible();

  await act(async () => status.click());
  expect(onModelSelect).toHaveBeenCalledExactlyOnceWith(local);
  await act(async () => userEvent.keyboard("{ArrowDown}{Enter}"));
  expect(onModelSelect).toHaveBeenLastCalledWith(installed);

  await act(async () =>
    root.render(
      <ModelList
        models={[cloud]}
        selectedModel={null}
        onModelSelect={onModelSelect}
        cloudDisabled={true}
        isOpen={true}
      />,
    ),
  );
  const cloudStatus = page.getByText("Not downloaded", { exact: true });
  await expect.element(cloudStatus).toBeVisible();
  expect(cloudStatus.element().hasAttribute("title")).toBe(false);
});
