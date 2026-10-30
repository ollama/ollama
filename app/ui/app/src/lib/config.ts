// API configuration
const DEV_API_URL = "http://127.0.0.1:3001";

// Base URL for fetch API calls (can be relative in production)
export const API_BASE = import.meta.env.DEV ? DEV_API_URL : "";

// Full host URL for Ollama client (needs full origin in production)
export const OLLAMA_HOST = import.meta.env.DEV
  ? DEV_API_URL
  : window.location.origin;

export const OLLAMA_DOT_COM =
  import.meta.env.VITE_OLLAMA_DOT_COM_URL || "https://ollama.com";
