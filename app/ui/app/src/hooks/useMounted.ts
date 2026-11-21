import { useEffect, useState } from "react";

/**
 * Returns false during the component's first render and true after the
 * first animation frame has painted. Use it to suppress mount-time CSS
 * transitions: render without transition classes while `mounted` is false,
 * so a component whose initial state is applied asynchronously does not
 * animate from a default state to its real state on load (#12954).
 */
export function useMounted(): boolean {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    // requestAnimationFrame runs before paint; the nested rAF guarantees the
    // first frame has actually painted before transitions are enabled.
    const handle = requestAnimationFrame(() => {
      requestAnimationFrame(() => setMounted(true));
    });
    return () => cancelAnimationFrame(handle);
  }, []);

  return mounted;
}
