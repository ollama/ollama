import { useSyncExternalStore } from "react";

export type Theme = "light" | "dark";

const mql = window.matchMedia("(prefers-color-scheme: dark)");

function subscribe(callback: () => void): () => void {
  mql.addEventListener("change", callback);
  return () => mql.removeEventListener("change", callback);
}

function getSnapshot(): Theme {
  return mql.matches ? "dark" : "light";
}

export function useTheme(): Theme {
  return useSyncExternalStore(subscribe, getSnapshot);
}
