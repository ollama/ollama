import { Ollama } from "ollama/browser";

let _ollamaClient: Ollama | null = null;

export const ollamaClient = new Proxy({} as Ollama, {
  get(_target, prop) {
    if (!_ollamaClient) {
      const host = import.meta.env.DEV
        ? "http://127.0.0.1:3001"
        : window.location.origin;

      _ollamaClient = new Ollama({
        host,
      });
    }
    const value = _ollamaClient[prop as keyof Ollama];
    return typeof value === "function" ? value.bind(_ollamaClient) : value;
  },
});
