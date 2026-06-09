import { useEffect } from "react";

export default function AppReady() {
  useEffect(() => {
    window.ready?.();
  }, []);

  return null;
}
