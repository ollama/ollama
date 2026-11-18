import "@/index.css";
import { commands, page } from "@vitest/browser/context";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import StreamingMarkdownContent from "./StreamingMarkdownContent";
import Thinking from "./Thinking";

declare module "@vitest/browser/context" {
  interface BrowserCommands {
    setColorScheme(scheme: "light" | "dark"): Promise<void>;
  }
}

const numberedList = [
  ...Array.from(
    { length: 30 },
    (_, index) => `${index + 1}. color ${index + 1}`,
  ),
  "",
  "A break so the next ordered list can start at three digits.",
  "",
  "100. color 100",
  "101. color 101",
].join("\n");

function ordinal(li: HTMLLIElement) {
  const ol = li.parentElement;
  if (!(ol instanceof HTMLOListElement)) {
    return li.value;
  }
  return ol.start + [...ol.children].indexOf(li);
}

function markerWidth(li: HTMLLIElement) {
  const markerStyle = getComputedStyle(li, "::marker");
  const probe = document.createElement("span");
  probe.textContent = `${ordinal(li)}.`;
  probe.style.font = markerStyle.font;
  probe.style.fontVariantNumeric =
    markerStyle.fontVariantNumeric || "tabular-nums";
  probe.style.position = "absolute";
  probe.style.visibility = "hidden";
  probe.style.whiteSpace = "pre";
  document.body.append(probe);
  const width = probe.getBoundingClientRect().width;
  probe.remove();
  return width;
}

// Outside markers are right-aligned to the item's border box, so a marker
// wider than the list padding paints to the left of the item.
function measureMarker(li: HTMLLIElement, wrapper: HTMLElement) {
  const width = markerWidth(li);
  const liRect = li.getBoundingClientRect();
  const wrapperRect = wrapper.getBoundingClientRect();
  const ol = li.closest("ol");
  if (!ol) {
    throw new Error("list item is not inside an ordered list");
  }
  return {
    text: `${ordinal(li)}.`,
    markerLeft: liRect.left - width,
    markerWidth: width,
    wrapperLeft: wrapperRect.left,
    paddingLeft: Number.parseFloat(getComputedStyle(ol).paddingLeft),
  };
}

function listItem(root: ParentNode, value: number) {
  const item = [...root.querySelectorAll("li")].find(
    (li) => ordinal(li as HTMLLIElement) === value,
  );
  if (!item) {
    throw new Error(`missing list item ${value}`);
  }
  return item as HTMLLIElement;
}

function thinkingWrapper(root: ParentNode) {
  const box = [...root.querySelectorAll("div")].find((element) =>
    element.className.includes("overflow-"),
  );
  if (!box) {
    throw new Error("missing thinking overflow wrapper");
  }
  return box as HTMLElement;
}

function expectMarkersInside(root: ParentNode, wrapper: HTMLElement) {
  for (const value of [1, 10, 30, 100]) {
    const item = measureMarker(listItem(root, value), wrapper);
    const clippedBy = item.wrapperLeft - item.markerLeft;
    expect(
      item.markerLeft,
      `${item.text} clipped by ${clippedBy.toFixed(2)}px (marker ${item.markerWidth.toFixed(2)}px, padding ${item.paddingLeft}px)`,
    ).toBeGreaterThanOrEqual(item.wrapperLeft - 0.5);
    expect(item.paddingLeft).toBeGreaterThanOrEqual(item.markerWidth);
  }
}

async function settle() {
  await act(async () => {
    await new Promise<void>((resolve) => {
      requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
    });
  });
}

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

async function renderThinking(expanded: boolean) {
  await act(async () => {
    root.render(
      <div
        data-testid="thinking-list"
        className="w-[720px] bg-white p-6 dark:bg-neutral-900"
      >
        <Thinking
          thinking={numberedList}
          startTime={new Date(Date.now() - 4000)}
        />
      </div>,
    );
  });
  await settle();
  if (!expanded) {
    return;
  }
  const heading = container.querySelector("h3");
  if (!heading) {
    throw new Error("missing thinking heading");
  }
  await act(async () => {
    heading.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  await settle();
}

describe("thinking numbered lists", () => {
  it.each([
    ["collapsed", "light"],
    ["expanded", "light"],
    ["collapsed", "dark"],
    ["expanded", "dark"],
  ] as const)(
    "keeps multi-digit markers inside the %s thinking box in %s mode",
    async (state, scheme) => {
      await page.viewport(900, 1600);
      await commands.setColorScheme(scheme);
      await renderThinking(state === "expanded");
      const harness = container.querySelector(
        "[data-testid='thinking-list']",
      ) as HTMLElement;
      expectMarkersInside(harness, thinkingWrapper(harness));
    },
  );
});

it("does not change assistant message list padding", async () => {
  await page.viewport(900, 1600);
  await commands.setColorScheme("light");
  await act(async () => {
    root.render(
      <section
        data-testid="assistant-list"
        className="w-[720px] bg-white"
        style={{ overflowY: "auto" }}
      >
        <div className="mx-auto flex max-w-[768px] flex-col px-6">
          <div className="max-w-full prose">
            <StreamingMarkdownContent content={numberedList} />
          </div>
        </div>
      </section>,
    );
  });
  await settle();
  const harness = container.querySelector(
    "[data-testid='assistant-list']",
  ) as HTMLElement;
  const item = measureMarker(listItem(harness, 10), harness);
  expect(item.paddingLeft).toBeCloseTo(26, 0);
  expect(item.markerLeft).toBeGreaterThanOrEqual(
    harness.getBoundingClientRect().left - 0.5,
  );
});
