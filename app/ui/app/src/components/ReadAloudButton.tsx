// Single global text-to-speech session; UI state syncs through the store below so
// message remounts (for example after chat refetch) still show pause/stop controls.
import {
  PauseIcon,
  PlayIcon,
  SpeakerWaveIcon,
  StopIcon,
} from "@heroicons/react/24/outline";
import { useSettings } from "@/hooks/useSettings";
import { useCallback, useEffect, useRef, useSyncExternalStore } from "react";

type PlaybackState = "idle" | "playing" | "paused";

type ActivePlayback = {
  ownerKey: string;
  synthesis: SpeechSynthesis;
  utterance: SpeechSynthesisUtterance;
};

type PlaybackSnapshot = {
  ownerKey: string;
  state: PlaybackState;
};

let activePlayback: ActivePlayback | null = null;
let playbackSnapshot: PlaybackSnapshot | null = null;
const playbackListeners = new Set<() => void>();
const autoStartedOwnerKeys = new Set<string>();
let speakGeneration = 0;
let pendingSpeak: { ownerKey: string; generation: number } | null = null;

/** WebKit often drops speak() if it runs in the same turn as cancel(). */
const SPEAK_AFTER_CANCEL_MS = 10;

function emitPlaybackChange() {
  playbackListeners.forEach((listener) => listener());
}

function subscribePlayback(listener: () => void) {
  playbackListeners.add(listener);
  return () => playbackListeners.delete(listener);
}

function getPlaybackState(ownerKey: string): PlaybackState {
  if (pendingSpeak?.ownerKey === ownerKey) {
    return "playing";
  }
  if (!playbackSnapshot || playbackSnapshot.ownerKey !== ownerKey) {
    return "idle";
  }
  return playbackSnapshot.state;
}

function setPlaybackSnapshot(next: PlaybackSnapshot | null) {
  playbackSnapshot = next;
  emitPlaybackChange();
}

function getSpeechSynthesis(): SpeechSynthesis | null {
  if (
    typeof window === "undefined" ||
    !("speechSynthesis" in window) ||
    typeof SpeechSynthesisUtterance === "undefined"
  ) {
    return null;
  }

  return window.speechSynthesis;
}

function invalidatePendingSpeak() {
  speakGeneration += 1;
}

function cancelActiveSynthesis() {
  if (!activePlayback) {
    return;
  }

  try {
    activePlayback.synthesis.cancel();
  } catch {
    /* ignored */
  }
}

function tearDownPlayback(options: { clearAutoStarted: "all" } | { ownerKey: string }) {
  invalidatePendingSpeak();
  pendingSpeak = null;
  cancelActiveSynthesis();
  activePlayback = null;
  playbackSnapshot = null;
  if ("clearAutoStarted" in options) {
    autoStartedOwnerKeys.clear();
  } else {
    autoStartedOwnerKeys.delete(options.ownerKey);
  }
  emitPlaybackChange();
}

function clearActivePlayback(playback: ActivePlayback) {
  if (activePlayback?.utterance !== playback.utterance) {
    return;
  }

  activePlayback = null;
  autoStartedOwnerKeys.delete(playback.ownerKey);
  if (playbackSnapshot?.ownerKey === playback.ownerKey) {
    setPlaybackSnapshot(null);
  }
}

export function stopReadAloudPlayback() {
  tearDownPlayback({ clearAutoStarted: "all" });
}

/** @internal test helper */
export function resetReadAloudPlaybackForTests() {
  stopReadAloudPlayback();
  speakGeneration = 0;
}

export default function ReadAloudButton({
  ownerKey,
  autoStart = false,
  getText,
}: {
  ownerKey: string;
  autoStart?: boolean;
  getText: () => string;
}) {
  const playbackState = useSyncExternalStore(
    subscribePlayback,
    () => getPlaybackState(ownerKey),
    () => getPlaybackState(ownerKey),
  );
  const playbackRef = useRef<ActivePlayback | null>(null);
  const startRef = useRef<() => void>(() => {});
  const { settings } = useSettings();
  const speechSynthesis = getSpeechSynthesis();
  const supported = speechSynthesis !== null;

  const start = useCallback(() => {
    const synthesis = getSpeechSynthesis();
    const text = getText().trim();
    if (!synthesis || !text) {
      return;
    }

    const generation = ++speakGeneration;

    const beginSpeak = () => {
      if (generation !== speakGeneration) {
        return;
      }

      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = Math.min(2, Math.max(0.5, settings.speechRate));
      utterance.volume = Math.min(1, Math.max(0, settings.speechVolume));
      if (settings.speechVoice) {
        utterance.voice =
          synthesis
            .getVoices()
            .find((voice) => voice.voiceURI === settings.speechVoice) ?? null;
      }

      const playback: ActivePlayback = {
        ownerKey,
        synthesis,
        utterance,
      };

      utterance.onstart = () => {
        setPlaybackSnapshot({ ownerKey, state: "playing" });
      };
      utterance.onend = () => clearActivePlayback(playback);
      utterance.onerror = () => clearActivePlayback(playback);

      playbackRef.current = playback;
      activePlayback = playback;
      setPlaybackSnapshot({ ownerKey, state: "playing" });
      try {
        synthesis.speak(utterance);
      } catch {
        clearActivePlayback(playback);
      }
    };

    const busy =
      activePlayback !== null || synthesis.speaking || synthesis.pending;
    if (busy) {
      const previous = activePlayback;
      synthesis.cancel();
      activePlayback = null;
      if (previous) {
        autoStartedOwnerKeys.delete(previous.ownerKey);
        if (playbackSnapshot?.ownerKey === previous.ownerKey) {
          setPlaybackSnapshot(null);
        }
      }
      // WebKit needs a gap after cancel(); speakGeneration guards stale timers.
      pendingSpeak = { ownerKey, generation };
      emitPlaybackChange();
      globalThis.setTimeout(() => {
        pendingSpeak = null;
        emitPlaybackChange();
        beginSpeak();
      }, SPEAK_AFTER_CANCEL_MS);
      return;
    }

    beginSpeak();
  }, [
    getText,
    ownerKey,
    settings.speechRate,
    settings.speechVoice,
    settings.speechVolume,
  ]);

  startRef.current = start;

  useEffect(() => {
    if (!autoStart || autoStartedOwnerKeys.has(ownerKey)) {
      return;
    }

    autoStartedOwnerKeys.add(ownerKey);
    startRef.current();
  }, [autoStart, ownerKey]);

  useEffect(() => {
    const playback = activePlayback;
    if (!playback || playback.ownerKey !== ownerKey) {
      return;
    }

    playbackRef.current = playback;
  }, [ownerKey, playbackState]);

  const pauseOrResume = () => {
    const playback = playbackRef.current;
    if (
      !playback ||
      activePlayback?.utterance !== playback.utterance ||
      playback.ownerKey !== ownerKey
    ) {
      start();
      return;
    }

    if (playbackState === "playing") {
      playback.synthesis.pause();
      setPlaybackSnapshot({ ownerKey, state: "paused" });
      return;
    }

    playback.synthesis.resume();
    setPlaybackSnapshot({ ownerKey, state: "playing" });
  };

  const stop = () => {
    if (
      pendingSpeak?.ownerKey === ownerKey &&
      pendingSpeak.generation === speakGeneration
    ) {
      tearDownPlayback({ ownerKey });
      return;
    }

    const playback = playbackRef.current;
    if (
      !playback ||
      activePlayback?.utterance !== playback.utterance ||
      playback.ownerKey !== ownerKey
    ) {
      return;
    }

    tearDownPlayback({ ownerKey });
  };

  const isPlaying = playbackState === "playing";
  const title = isPlaying
    ? "Pause read aloud"
    : playbackState === "paused"
      ? "Resume read aloud"
      : "Read aloud";

  return (
    <div className="flex items-center gap-1">
      <button
        type="button"
        className="h-7 w-7 px-1 py-0.5 text-xs cursor-pointer rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800 flex items-center justify-center text-neutral-500 dark:text-neutral-400 disabled:cursor-not-allowed disabled:opacity-50"
        onClick={pauseOrResume}
        disabled={!supported}
        title={title}
        aria-label={title}
      >
        {isPlaying ? (
          <PauseIcon className="h-5 w-5" />
        ) : playbackState === "paused" ? (
          <PlayIcon className="h-5 w-5" />
        ) : (
          <SpeakerWaveIcon className="h-5 w-5" />
        )}
      </button>
      {playbackState !== "idle" && (
        <button
          type="button"
          className="h-7 w-7 px-1 py-0.5 text-xs cursor-pointer rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800 flex items-center justify-center text-neutral-500 dark:text-neutral-400"
          onClick={stop}
          title="Stop read aloud"
          aria-label="Stop read aloud"
        >
          <StopIcon className="h-5 w-5" />
        </button>
      )}
    </div>
  );
}
