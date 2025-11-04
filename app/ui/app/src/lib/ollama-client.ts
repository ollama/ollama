import { Ollama } from "ollama/browser";

let _ollamaClient: Ollama | null = null;

export const ollamaClient = new Proxy({} as Ollama, {
  get(_target, prop) {
    if (!_ollamaClient) {
      _ollamaClient = new Ollama({
        host: window.location.origin,
      });
    }
    const value = _ollamaClient[prop as keyof Ollama];
    return typeof value === "function" ? value.bind(_ollamaClient) : value;
  },
});
