import { QueryClient } from "@tanstack/react-query";

export const queryClient = new QueryClient({
  defaultOptions: {
    mutations: { networkMode: "always" },
    queries: { networkMode: "always" },
  },
});
