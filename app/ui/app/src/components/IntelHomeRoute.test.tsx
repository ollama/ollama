import { type ComponentType } from "react";
import { act, create, type ReactTestRenderer } from "react-test-renderer";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as api from "@/api";
import { Settings } from "@/gotypes";
import { CURRENT_ONBOARDING_VERSION } from "@/lib/onboarding";
import { Route } from "@/routes/index";

const mocks = vi.hoisted(() => ({ navigate: vi.fn() }));
vi.mock("@tanstack/react-router", async (importOriginal) =>
  Object.assign(
    {},
    await importOriginal<typeof import("@tanstack/react-router")>(),
    { useNavigate: () => mocks.navigate },
  ),
);
vi.mock("@/components/LoadingScreen", () => ({ default: () => null }));

let renderer: ReactTestRenderer | undefined;
let client: QueryClient;

beforeEach(() => {
  vi.useFakeTimers();
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  client = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: Infinity } },
  });
  vi.spyOn(api, "fetchUser").mockRejectedValue(new Error("offline"));
});

afterEach(async () => {
  await act(async () => renderer?.unmount());
  renderer = undefined;
  client.clear();
  vi.useRealTimers();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  mocks.navigate.mockReset();
});

async function renderHome() {
  const Home = Route.options.component as ComponentType;
  await act(async () => {
    renderer = create(
      <QueryClientProvider client={client}>
        <Home />
      </QueryClientProvider>,
    );
  });
}

describe("Intel home bootstrap", () => {
  it("falls back at five seconds despite failed settings polling and user lookup", async () => {
    vi.spyOn(api, "getSettings").mockRejectedValue(new Error("offline"));
    await renderHome();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(4999);
    });
    expect(mocks.navigate).not.toHaveBeenCalled();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1);
    });
    expect(mocks.navigate).toHaveBeenCalledExactlyOnceWith({
      to: "/c/$chatId",
      params: { chatId: "new" },
      replace: true,
    });
  });

  it("uses fresh settings before redirecting and cancels the later fallback", async () => {
    client.setQueryData(["settings"], {
      settings: new Settings({ OnboardingVersion: CURRENT_ONBOARDING_VERSION }),
    });
    vi.spyOn(api, "getSettings").mockResolvedValue({
      settings: new Settings({ OnboardingVersion: 0 }),
    } as Awaited<ReturnType<typeof api.getSettings>>);
    await renderHome();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1);
    });
    expect(mocks.navigate).toHaveBeenCalledExactlyOnceWith({
      to: "/onboarding",
      replace: true,
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(6000);
    });
    expect(mocks.navigate).toHaveBeenCalledTimes(1);
  });

  it("cleans up the fallback when unmounted with settings still pending", async () => {
    vi.spyOn(api, "getSettings").mockImplementation(
      () => new Promise(() => {}),
    );
    await renderHome();
    await act(async () => renderer?.unmount());
    renderer = undefined;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(6000);
    });
    expect(mocks.navigate).not.toHaveBeenCalled();
  });
});
