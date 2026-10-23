import { act, create, type ReactTestInstance } from "react-test-renderer";
import { describe, expect, it, vi } from "vitest";
import { Switch } from "./ui/switch";
import { ModelMTPSetting } from "./ModelMTPSetting";

const model = "glm5.3-flash:q6-vision";

function textContent(node: ReactTestInstance): string {
  return node.children
    .map((child) => (typeof child === "string" ? child : textContent(child)))
    .join("");
}

function render(
  props: Partial<React.ComponentProps<typeof ModelMTPSetting>> = {},
) {
  let renderer: ReturnType<typeof create> | undefined;
  act(() => {
    renderer = create(
      <ModelMTPSetting
        modelName={model}
        supported
        onChange={vi.fn()}
        {...props}
      />,
    );
  });
  return renderer!;
}

describe("ModelMTPSetting", () => {
  it("is hidden for models without MTP layers", () => {
    const renderer = render({ supported: false, draftNumPredict: 3 });
    expect(
      renderer.root.findAllByProps({ "data-testid": "mtp-toggle" }),
    ).toHaveLength(0);
    act(() => renderer.unmount());
  });

  it("shows an existing draft_num_predict as on", () => {
    const renderer = render({ draftNumPredict: 3 });
    const toggle = renderer.root.findByType(Switch);
    expect(toggle.props["aria-checked"] ?? toggle.props.checked).toBe(true);
    expect(textContent(renderer.root)).toContain(model);
    const depth = renderer.root.findByProps({
      "data-testid": "mtp-draft-depth",
    });
    expect(depth.props.value).toBe(3);
    act(() => renderer.unmount());
  });

  it("shows models without the parameter as off and does not offer a depth", () => {
    const renderer = render({});
    const toggle = renderer.root.findByType(Switch);
    expect(toggle.props.checked).toBe(false);
    expect(
      renderer.root.findAllByProps({ "data-testid": "mtp-draft-depth" }),
    ).toHaveLength(0);
    act(() => renderer.unmount());
  });

  it("turning on asks for the default depth and turning off asks to remove it", () => {
    const onChange = vi.fn();
    const off = render({ onChange });
    off.root.findByType(Switch).props.onChange(true);
    expect(onChange).toHaveBeenCalledWith(3);
    act(() => off.unmount());

    const on = render({ draftNumPredict: 3, onChange });
    on.root.findByType(Switch).props.onChange(false);
    expect(onChange).toHaveBeenCalledWith(0);
    act(() => on.unmount());
  });

  it("changing draft tokens asks for that depth only", () => {
    const onChange = vi.fn();
    const renderer = render({ draftNumPredict: 3, onChange });
    const depth = renderer.root.findByProps({
      "data-testid": "mtp-draft-depth",
    });
    act(() => {
      depth.props.onChange({ target: { value: "2" } });
    });
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onChange).toHaveBeenCalledWith(2);
    act(() => renderer.unmount());
  });
});
