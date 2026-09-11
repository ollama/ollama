import type { ReactNode } from "react";
import { act, create } from "react-test-renderer";
import { afterEach, expect, it, vi } from "vitest";
import { CodexConnectedIntro } from "./CodexConnectedIntro";

vi.mock("@headlessui/react", () => {
  const Container = ({ children }: { children: ReactNode }) => (
    <div>{children}</div>
  );
  return {
    Dialog: Container,
    DialogPanel: Container,
    DialogTitle: Container,
    Description: Container,
  };
});
afterEach(() => vi.unstubAllGlobals());

it("hands Continue to the connection flow, like Claude's intro", async () => {
  const done = vi.fn();
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  let renderer;
  try {
    await act(async () => {
      renderer = create(<CodexConnectedIntro onDone={done} />);
    });
    expect(done).not.toHaveBeenCalled();
    const button = renderer!.root.findByType("button");
    expect(button.children).toEqual(["Continue"]);
    await act(async () => button.props.onClick());
    expect(done).toHaveBeenCalledOnce();
  } finally {
    await act(async () => renderer?.unmount());
  }
});
