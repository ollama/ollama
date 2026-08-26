import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, create, type ReactTestRenderer } from "react-test-renderer";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const apiMocks = vi.hoisted(() => ({
  disconnectUser: vi.fn(),
  fetchConnectUrl: vi.fn(),
  fetchUser: vi.fn(),
}));

vi.mock("@/api", () => apiMocks);

import {
  ACCOUNT_CONNECTION_POLL_INTERVAL_MS,
  ACCOUNT_CONNECTION_TIMEOUT_MS,
  UserProvider,
  useUser,
} from "./useUser";

type UserState = ReturnType<typeof useUser>;

function UserStateHarness({
  onRender,
}: {
  onRender: (state: UserState) => void;
}) {
  onRender(useUser());
  return null;
}

async function flushUpdates() {
  await Promise.resolve();
  await Promise.resolve();
  await vi.advanceTimersByTimeAsync(0);
}

describe("useUser account connection", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    vi.stubGlobal("window", {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      setInterval: globalThis.setInterval,
      clearInterval: vi.fn(globalThis.clearInterval),
      setTimeout: globalThis.setTimeout,
      clearTimeout: vi.fn(globalThis.clearTimeout),
      open: vi.fn(),
    });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it("keeps polling after an early signed-out response and shares the authenticated user", async () => {
    const authenticatedUser = {
      id: "user-id",
      name: "Test User",
      email: "test@example.com",
    };
    apiMocks.fetchConnectUrl.mockResolvedValue(
      "https://ollama.com/connect?name=MacBook&key=public-key&launch=true",
    );
    apiMocks.fetchUser
      .mockResolvedValueOnce(null)
      .mockResolvedValueOnce(null)
      .mockResolvedValue(authenticatedUser);

    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false, gcTime: Infinity } },
    });
    let initiatingState!: UserState;
    let observingState!: UserState;
    let renderer: ReactTestRenderer | undefined;
    const renderHarness = (showInitiator: boolean) => (
      <QueryClientProvider client={queryClient}>
        <UserProvider>
          {showInitiator && (
            <UserStateHarness
              key="initiator"
              onRender={(nextState) => (initiatingState = nextState)}
            />
          )}
          <UserStateHarness
            key="observer"
            onRender={(nextState) => (observingState = nextState)}
          />
        </UserProvider>
      </QueryClientProvider>
    );

    try {
      await act(async () => {
        renderer = create(renderHarness(true));
        await flushUpdates();
      });

      expect(initiatingState.isAuthenticated).toBe(false);
      expect(observingState.isAuthenticated).toBe(false);

      await act(async () => {
        await initiatingState.connectUser();
        await flushUpdates();
      });

      expect(window.open).toHaveBeenCalledWith(
        "https://ollama.com/connect?name=MacBook&key=public-key&launch=true",
        "_blank",
      );
      expect(initiatingState.isConnecting).toBe(true);
      expect(observingState.isConnecting).toBe(true);
      expect(apiMocks.fetchUser).toHaveBeenCalledTimes(2);

      await act(async () => {
        renderer?.update(renderHarness(false));
        await flushUpdates();
      });

      await act(async () => {
        await vi.advanceTimersByTimeAsync(ACCOUNT_CONNECTION_POLL_INTERVAL_MS);
        await flushUpdates();
      });

      expect(apiMocks.fetchUser).toHaveBeenCalledTimes(3);
      expect(observingState.user).toEqual(authenticatedUser);
      expect(observingState.isAuthenticated).toBe(true);
      expect(observingState.isConnecting).toBe(false);
      expect(observingState.connectionError).toBeNull();
      expect(window.clearInterval).toHaveBeenCalled();
      expect(window.removeEventListener).toHaveBeenCalledWith(
        "focus",
        expect.any(Function),
      );
    } finally {
      await act(async () => {
        renderer?.unmount();
        await flushUpdates();
      });
      queryClient.clear();
    }
  });

  it("stops a connection attempt after the bounded timeout", async () => {
    apiMocks.fetchConnectUrl.mockResolvedValue(
      "https://ollama.com/connect?name=MacBook&key=public-key&launch=true",
    );
    apiMocks.fetchUser.mockResolvedValue(null);

    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false, gcTime: Infinity } },
    });
    let state!: UserState;
    let renderer: ReactTestRenderer | undefined;

    try {
      await act(async () => {
        renderer = create(
          <QueryClientProvider client={queryClient}>
            <UserProvider>
              <UserStateHarness onRender={(nextState) => (state = nextState)} />
            </UserProvider>
          </QueryClientProvider>,
        );
        await flushUpdates();
      });

      await act(async () => {
        await state.connectUser();
        await flushUpdates();
      });

      expect(state.isConnecting).toBe(true);

      await act(async () => {
        await vi.advanceTimersByTimeAsync(ACCOUNT_CONNECTION_TIMEOUT_MS);
        await flushUpdates();
      });

      expect(state.isConnecting).toBe(false);
      expect(state.connectionError).toBe(
        "Connection is taking longer than expected. Please try again.",
      );

      const fetchCountAtTimeout = apiMocks.fetchUser.mock.calls.length;
      await act(async () => {
        await vi.advanceTimersByTimeAsync(ACCOUNT_CONNECTION_POLL_INTERVAL_MS);
        await flushUpdates();
      });
      expect(apiMocks.fetchUser).toHaveBeenCalledTimes(fetchCountAtTimeout);
    } finally {
      await act(async () => {
        renderer?.unmount();
        await flushUpdates();
      });
      queryClient.clear();
    }
  });
});
