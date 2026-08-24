export function isWindowsPlatform(): boolean {
  if (typeof window !== "undefined" && window.OLLAMA_PLATFORM) {
    return window.OLLAMA_PLATFORM === "windows";
  }

  return (
    typeof navigator !== "undefined" &&
    navigator.platform.toLowerCase().includes("win")
  );
}
