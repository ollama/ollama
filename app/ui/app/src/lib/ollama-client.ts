import { Ollama } from "ollama/browser";
import { OLLAMA_HOST } from "./config";

let _ollamaClient: Ollama | null = null;

export const ollamaClient = new Proxy({} as Ollama, {
  get(_target, prop) {
    if (!_ollamaClient) {
      _ollamaClient = new Ollama({
        host: OLLAMA_HOST,
      });
    }
    const value = _ollamaClient[prop as keyof Ollama];
    return typeof value === "function" ? value.bind(_ollamaClient) : value;
  },
});
