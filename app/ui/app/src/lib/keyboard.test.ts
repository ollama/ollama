import { describe, expect, it, vi } from "vitest";
import { preventPageSelectAll } from "./keyboard";

function keyboardEvent(overrides: Partial<KeyboardEvent> = {}): KeyboardEvent {
  return {
    key: "a",
    metaKey: true,
    ctrlKey: false,
    target: { tagName: "BODY" } as EventTarget,
    preventDefault: vi.fn(),
    ...overrides,
  } as unknown as KeyboardEvent;
}

describe("preventPageSelectAll", () => {
  it("prevents Command+A from selecting the page", () => {
    const event = keyboardEvent();

    preventPageSelectAll(event);

    expect(event.preventDefault).toHaveBeenCalledOnce();
  });

  it("prevents Ctrl+A from selecting the page", () => {
    const event = keyboardEvent({ metaKey: false, ctrlKey: true });

    preventPageSelectAll(event);

    expect(event.preventDefault).toHaveBeenCalledOnce();
  });

  it.each([
    { tagName: "INPUT" },
    { tagName: "TEXTAREA" },
    { tagName: "DIV", isContentEditable: true },
  ])("keeps Select All working in editable targets", (target) => {
    const event = keyboardEvent({ target: target as EventTarget });

    preventPageSelectAll(event);

    expect(event.preventDefault).not.toHaveBeenCalled();
  });

  it("leaves unmodified A keypresses alone", () => {
    const event = keyboardEvent({ metaKey: false });

    preventPageSelectAll(event);

    expect(event.preventDefault).not.toHaveBeenCalled();
  });
});
