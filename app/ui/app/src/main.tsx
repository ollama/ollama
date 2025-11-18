import { StrictMode } from "react";
import ReactDOM from "react-dom/client";
import { RouterProvider, createRouter } from "@tanstack/react-router";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { routeTree } from "./routeTree.gen";
import { fetchUser } from "./api";
import { StreamingProvider } from "./contexts/StreamingContext";
import { User } from "@/gotypes";

declare global {
  interface Window {
    __initialUserDataPromise?: Promise<User | null>;
  }
}

const queryClient = new QueryClient({
  defaultOptions: {
    mutations: {
      networkMode: "always", // Run mutations regardless of network state
    },
    queries: {
      networkMode: "always", // Allow queries even when offline (local server)
    },
  },
});

// Poll for server readiness, then fetch user data
// This ensures initialDataLoaded isn't set to true until we have actual data or timeout
const initializeUserData = async () => {
  const maxAttempts = 20; // 20 attempts * 500ms = 10 seconds max
  const retryDelay = 500;

  // First, wait for server to be ready
  for (let i = 0; i < maxAttempts; i++) {
    try {
      const versionResponse = await fetch(
        `${window.location.origin}/api/version`,
        { method: "HEAD" },
      );
      if (versionResponse.ok) {
        break;
      }
    } catch (error) {
      console.error("Failed to fetch version:", error);
    }

    if (i < maxAttempts - 1) {
      await new Promise((resolve) => setTimeout(resolve, retryDelay));
    }
  }

  // Now fetch user data
  try {
    const userData = await fetchUser();
    queryClient.setQueryData(["user"], userData);
    return userData;
  } catch (error) {
    console.error("Failed to fetch user data:", error);
    queryClient.setQueryData(["user"], null);
    return null;
  }
};

// Start initialization and track the promise
const initialUserDataPromise = initializeUserData();

// Export the promise so hooks can await it
window.__initialUserDataPromise = initialUserDataPromise;

const router = createRouter({
  routeTree,
  context: { queryClient },
});

// Register the router instance for type safety
declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}

const rootElement = document.getElementById("root")!;
if (!rootElement.innerHTML) {
  const root = ReactDOM.createRoot(rootElement);
  root.render(
    <StrictMode>
      <QueryClientProvider client={queryClient}>
        <StreamingProvider>
          <RouterProvider router={router} />
        </StreamingProvider>
      </QueryClientProvider>
    </StrictMode>,
  );
}
