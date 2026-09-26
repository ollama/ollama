import { act, create, type ReactTestRenderer } from "react-test-renderer";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import ReadAloudButton, {
  resetReadAloudPlaybackForTests,
  stopReadAloudPlayback,
} from "./ReadAloudButton";

const OWNER = "chat-1:0";

const { useSettingsMock } = vi.hoisted(() => ({
  useSettingsMock: vi.fn(),
}));

vi.mock("@/hooks/useSettings", () => ({
  useSettings: useSettingsMock,
}));

class FakeUtterance {
  onstart: (() => void) | null = null;
  onend: (() => void) | null = null;
  onerror: (() => void) | null = null;
  voice?: SpeechSynthesisVoice;
  rate?: number;
  volume?: number;

  constructor(readonly text: string) {}
}

function installSpeechSynthesis() {
  const synthesis = {
    speak: vi.fn((utterance: FakeUtterance) => {
      synthesis.speaking = true;
      synthesis.pending = false;
      utterance.onstart?.();
    }),
    pause: vi.fn(() => {
      synthesis.paused = true;
    }),
    resume: vi.fn(() => {
      synthesis.paused = false;
    }),
    cancel: vi.fn(() => {
      synthesis.speaking = false;
      synthesis.pending = false;
      synthesis.paused = false;
    }),
    getVoices: vi.fn(() => []),
    speaking: false,
    pending: false,
    paused: false,
  };

  vi.stubGlobal("window", { speechSynthesis: synthesis });
  vi.stubGlobal("SpeechSynthesisUtterance", FakeUtterance);

  return synthesis;
}

function button(renderer: ReactTestRenderer, label: string) {
  return renderer.root.findByProps({ "aria-label": label });
}

function buttons(renderer: ReactTestRenderer, label: string) {
  return renderer.root.findAllByProps({ "aria-label": label });
}

beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  useSettingsMock.mockReturnValue({
    settings: {
      speechVoice: "",
      speechRate: 1,
      speechVolume: 1,
    },
  });
});

afterEach(() => {
  resetReadAloudPlaybackForTests();
  vi.unstubAllGlobals();
  useSettingsMock.mockReset();
});

describe("ReadAloudButton", () => {
  it("applies saved voice, rate, and volume to each utterance", () => {
    const synthesis = installSpeechSynthesis();
    const voice = { voiceURI: "com.example.voice" } as SpeechSynthesisVoice;
    synthesis.getVoices.mockReturnValue([voice]);
    useSettingsMock.mockReturnValue({
      settings: {
        speechVoice: voice.voiceURI,
        speechRate: 1.5,
        speechVolume: 0.4,
      },
    });
    let renderer!: ReactTestRenderer;

    act(() => {
      renderer = create(<ReadAloudButton ownerKey={OWNER} getText={() => "Configured."} />);
    });
    act(() => {
      button(renderer, "Read aloud").props.onClick();
    });

    expect(synthesis.speak).toHaveBeenCalledWith(
      expect.objectContaining({ voice, rate: 1.5, volume: 0.4 }),
    );

    act(() => {
      renderer.unmount();
    });
  });

  it("auto starts when given a completion signal", () => {
    const synthesis = installSpeechSynthesis();
    let renderer!: ReactTestRenderer;

    act(() => {
      renderer = create(
        <ReadAloudButton ownerKey={OWNER} getText={() => "Do not speak yet."} />,
      );
    });
    expect(synthesis.speak).not.toHaveBeenCalled();

    act(() => {
      renderer.update(
        <ReadAloudButton
          ownerKey={OWNER}
          autoStart
          getText={() => "Speak automatically."}
        />,
      );
    });

    expect(synthesis.speak).toHaveBeenCalledOnce();
    expect(button(renderer, "Pause read aloud")).toBeDefined();
    act(() => renderer.unmount());
  });

  it("shows pause and stop after auto start survives a remount", () => {
    const synthesis = installSpeechSynthesis();
    let renderer!: ReactTestRenderer;

    act(() => {
      renderer = create(
        <ReadAloudButton
          ownerKey={OWNER}
          autoStart
          getText={() => "Speak automatically."}
        />,
      );
    });
    expect(synthesis.speak).toHaveBeenCalledOnce();

    act(() => {
      renderer.unmount();
      renderer = create(
        <ReadAloudButton
          ownerKey={OWNER}
          autoStart
          getText={() => "Speak automatically."}
        />,
      );
    });

    expect(button(renderer, "Pause read aloud")).toBeDefined();
    expect(button(renderer, "Stop read aloud")).toBeDefined();
    act(() => renderer.unmount());
  });

  it("speaks plain response text and supports pause, resume, and stop", () => {
    const synthesis = installSpeechSynthesis();
    let renderer!: ReactTestRenderer;

    act(() => {
      renderer = create(
        <ReadAloudButton ownerKey={OWNER} getText={() => "A short answer."} />,
      );
    });

    act(() => {
      button(renderer, "Read aloud").props.onClick();
    });

    expect(synthesis.speak).toHaveBeenCalledOnce();
    expect(synthesis.speak).toHaveBeenCalledWith(
      expect.objectContaining({ text: "A short answer." }),
    );

    act(() => {
      button(renderer, "Pause read aloud").props.onClick();
    });
    expect(synthesis.pause).toHaveBeenCalledOnce();

    act(() => {
      button(renderer, "Resume read aloud").props.onClick();
    });
    expect(synthesis.resume).toHaveBeenCalledOnce();

    act(() => {
      button(renderer, "Stop read aloud").props.onClick();
    });
    expect(synthesis.cancel).toHaveBeenCalledOnce();
    expect(button(renderer, "Read aloud").props.title).toBe("Read aloud");
    expect(buttons(renderer, "Stop read aloud")).toHaveLength(0);

    act(() => {
      renderer.unmount();
    });
  });

  it("cancels the previous response before speaking another one", () => {
    vi.useFakeTimers();
    const synthesis = installSpeechSynthesis();
    let renderer!: ReactTestRenderer;

    act(() => {
      renderer = create(
        <>
          <ReadAloudButton ownerKey="a" getText={() => "First response."} />
          <ReadAloudButton ownerKey="b" getText={() => "Second response."} />
        </>,
      );
    });

    act(() => {
      buttons(renderer, "Read aloud")[0].props.onClick();
    });
    act(() => {
      buttons(renderer, "Read aloud").at(-1)!.props.onClick();
    });

    expect(synthesis.cancel).toHaveBeenCalledOnce();
    expect(synthesis.speak).toHaveBeenCalledTimes(1);

    act(() => {
      vi.advanceTimersByTime(10);
    });

    expect(synthesis.speak).toHaveBeenCalledTimes(2);

    act(() => {
      renderer.unmount();
    });
    vi.useRealTimers();
  });

  it("cancels a pending start when Stop is clicked before speech begins", () => {
    vi.useFakeTimers();
    const synthesis = installSpeechSynthesis();
    let renderer!: ReactTestRenderer;

    act(() => {
      renderer = create(
        <>
          <ReadAloudButton ownerKey="a" getText={() => "First response."} />
          <ReadAloudButton ownerKey="b" getText={() => "Second response."} />
        </>,
      );
    });

    act(() => {
      buttons(renderer, "Read aloud")[0].props.onClick();
    });
    act(() => {
      buttons(renderer, "Read aloud").at(-1)!.props.onClick();
    });

    act(() => {
      button(renderer, "Stop read aloud").props.onClick();
    });

    act(() => {
      vi.advanceTimersByTime(10);
    });

    expect(synthesis.speak).toHaveBeenCalledTimes(1);
    act(() => renderer.unmount());
    vi.useRealTimers();
  });

  it("does not start speech after stop cancels a pending message switch", () => {
    vi.useFakeTimers();
    const synthesis = installSpeechSynthesis();
    let renderer!: ReactTestRenderer;

    act(() => {
      renderer = create(
        <>
          <ReadAloudButton ownerKey="a" getText={() => "First response."} />
          <ReadAloudButton ownerKey="b" getText={() => "Second response."} />
        </>,
      );
    });

    act(() => {
      buttons(renderer, "Read aloud")[0].props.onClick();
    });
    act(() => {
      buttons(renderer, "Read aloud").at(-1)!.props.onClick();
    });
    expect(synthesis.speak).toHaveBeenCalledTimes(1);

    act(() => {
      stopReadAloudPlayback();
      vi.advanceTimersByTime(10);
    });

    expect(synthesis.speak).toHaveBeenCalledTimes(1);
    act(() => renderer.unmount());
    vi.useRealTimers();
  });

  it("stops playback when stopReadAloudPlayback is called", () => {
    const synthesis = installSpeechSynthesis();
    let renderer!: ReactTestRenderer;

    act(() => {
      renderer = create(
        <ReadAloudButton ownerKey={OWNER} getText={() => "Stop me."} />,
      );
    });
    act(() => {
      button(renderer, "Read aloud").props.onClick();
    });

    act(() => {
      stopReadAloudPlayback();
    });

    expect(synthesis.cancel).toHaveBeenCalledOnce();
    expect(button(renderer, "Read aloud")).toBeDefined();
    act(() => renderer.unmount());
  });

  it("stops playback when Stop is clicked", () => {
    const synthesis = installSpeechSynthesis();
    let renderer!: ReactTestRenderer;

    act(() => {
      renderer = create(
        <ReadAloudButton ownerKey={OWNER} getText={() => "Stop me."} />,
      );
    });
    act(() => {
      button(renderer, "Read aloud").props.onClick();
    });
    act(() => {
      button(renderer, "Stop read aloud").props.onClick();
    });

    expect(synthesis.cancel).toHaveBeenCalledOnce();
    expect(button(renderer, "Read aloud")).toBeDefined();
    act(() => renderer.unmount());
  });

  it("does not cancel playback when the control remounts during speech", () => {
    const synthesis = installSpeechSynthesis();
    let renderer!: ReactTestRenderer;

    act(() => {
      renderer = create(
        <ReadAloudButton ownerKey={OWNER} getText={() => "Leaving chat."} />,
      );
    });
    act(() => {
      button(renderer, "Read aloud").props.onClick();
    });
    act(() => {
      renderer.unmount();
      renderer = create(
        <ReadAloudButton ownerKey={OWNER} getText={() => "Leaving chat."} />,
      );
    });

    expect(synthesis.cancel).not.toHaveBeenCalled();
    expect(button(renderer, "Pause read aloud")).toBeDefined();
  });

  it("returns to idle when the speech engine rejects playback", () => {
    const synthesis = installSpeechSynthesis();
    synthesis.speak.mockImplementation(() => {
      throw new Error("speech unavailable");
    });
    let renderer!: ReactTestRenderer;

    act(() => {
      renderer = create(
        <ReadAloudButton ownerKey={OWNER} getText={() => "Try speaking."} />,
      );
    });
    act(() => {
      button(renderer, "Read aloud").props.onClick();
    });

    expect(button(renderer, "Read aloud").props.title).toBe("Read aloud");
    expect(buttons(renderer, "Stop read aloud")).toHaveLength(0);

    act(() => {
      renderer.unmount();
    });
  });

  it("disables the control when speech synthesis is unavailable", () => {
    let renderer!: ReactTestRenderer;

    act(() => {
      renderer = create(
        <ReadAloudButton ownerKey={OWNER} getText={() => "Unavailable."} />,
      );
    });

    expect(button(renderer, "Read aloud").props.disabled).toBe(true);

    act(() => {
      renderer.unmount();
    });
  });

  it("disables the control when the utterance constructor is unavailable", () => {
    vi.stubGlobal("window", {
      speechSynthesis: {
        speak: vi.fn(),
        pause: vi.fn(),
        resume: vi.fn(),
        cancel: vi.fn(),
      },
    });
    let renderer!: ReactTestRenderer;

    act(() => {
      renderer = create(
        <ReadAloudButton ownerKey={OWNER} getText={() => "Unavailable."} />,
      );
    });

    expect(button(renderer, "Read aloud").props.disabled).toBe(true);

    act(() => {
      renderer.unmount();
    });
  });
});
