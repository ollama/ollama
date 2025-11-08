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

// Track initial user data fetch
let initialUserDataPromise: Promise<User | null> | null = null;

// Initialize user data on app startup
const initializeUserData = async () => {
  try {
    const userData = await fetchUser();
    queryClient.setQueryData(["user"], userData);
    return userData;
  } catch (error) {
    console.error("Error initializing user data:", error);
    queryClient.setQueryData(["user"], null);
    return null;
  }
};

// Start initialization immediately and track the promise
initialUserDataPromise = initializeUserData();

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
