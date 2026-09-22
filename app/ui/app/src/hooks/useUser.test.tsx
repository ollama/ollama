import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, create } from "react-test-renderer";
import { afterEach, expect, it, vi } from "vitest";
import { fetchUser } from "@/api";
import { useUser, userQueryOptions } from "./useUser";

vi.mock("@/api", () => ({
  fetchUser: vi.fn(),
  fetchConnectUrl: vi.fn(),
  disconnectUser: vi.fn(),
}));
afterEach(() => vi.unstubAllGlobals());

it("shares startup account loading with Settings and reuses the fresh result", async () => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  let resolve!: (value: null) => void;
  vi.mocked(fetchUser).mockImplementation(
    () =>
      new Promise((done) => {
        resolve = done;
      }),
  );
  const client = new QueryClient();
  function Account() {
    const { isLoading } = useUser();
    return <span>{isLoading ? "Loading" : "Ready"}</span>;
  }
  const element = (
    <QueryClientProvider client={client}>
      <Account />
    </QueryClientProvider>
  );
  let renderer: ReturnType<typeof create> | undefined;
  try {
    const preload = client.prefetchQuery(userQueryOptions);
    await act(async () => {
      renderer = create(element);
    });
    expect(fetchUser).toHaveBeenCalledOnce();
    await act(async () => {
      resolve(null);
      await preload;
    });
    await act(async () => {
      renderer!.unmount();
    });
    await act(async () => {
      renderer = create(element);
    });
    expect(fetchUser).toHaveBeenCalledOnce();
    expect(renderer!.root.findByType("span").children).toEqual(["Ready"]);
  } finally {
    await act(async () => renderer?.unmount());
    client.clear();
  }
});
