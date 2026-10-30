import { StrictMode } from "react";
import ReactDOM from "react-dom/client";
import { RouterProvider, createRouter } from "@tanstack/react-router";
import { QueryClientProvider } from "@tanstack/react-query";
import { routeTree } from "./routeTree.gen";
import { queryClient } from "./lib/queryClient";
import { userQueryOptions } from "./hooks/useUser";
import { StreamingProvider } from "./contexts/StreamingContext";

void queryClient.prefetchQuery(userQueryOptions);

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
