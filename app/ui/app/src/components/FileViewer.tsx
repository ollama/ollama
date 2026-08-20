import { useEffect, useMemo, useState } from "react";
import { XMarkIcon, AtSymbolIcon } from "@heroicons/react/24/outline";
import type { BundledLanguage, ThemedToken, ThemeRegistrationAny } from "shiki";
import CopyButton from "./CopyButton";
import { useProjectFileContent } from "@/hooks/useProject";
import { highlighter, highlighterPromise } from "@/lib/highlighter";

// Extensions mapped to the languages loaded in lib/highlighter; anything else
// renders as plain text
const LANGUAGE_BY_EXTENSION: Record<string, BundledLanguage> = {
  js: "javascript",
  mjs: "javascript",
  cjs: "javascript",
  jsx: "jsx",
  ts: "typescript",
  mts: "typescript",
  cts: "typescript",
  tsx: "tsx",
  py: "python",
  sh: "bash",
  bash: "bash",
  zsh: "shell",
  json: "json",
  html: "html",
  htm: "html",
  css: "css",
  go: "go",
  rs: "rust",
  java: "java",
  c: "c",
  h: "c",
  cpp: "cpp",
  cc: "cpp",
  hpp: "cpp",
  sql: "sql",
  swift: "swift",
  yaml: "yaml",
  yml: "yaml",
  md: "markdown",
  markdown: "markdown",
};

// lib/highlighter registers these themes by name, but shiki only types the
// names it bundles itself
const LIGHT_THEME = "one-light" as unknown as ThemeRegistrationAny;
const DARK_THEME = "one-dark" as unknown as ThemeRegistrationAny;

// Highlighting every token of a very large file blocks the UI, so past this
// size the viewer falls back to plain text
const MAX_HIGHLIGHT_CHARS = 100 * 1024;

function languageOf(path: string): BundledLanguage | null {
  const name = path.slice(path.lastIndexOf("/") + 1);
  const dot = name.lastIndexOf(".");
  if (dot <= 0) return null;
  return LANGUAGE_BY_EXTENSION[name.slice(dot + 1).toLowerCase()] ?? null;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function CodeLines({
  lines,
  fallback,
}: {
  lines: ThemedToken[][] | undefined;
  fallback: string;
}) {
  if (!lines) return <>{fallback}</>;
  return (
    <>
      {lines.map((tokens, i) => (
        <div key={i} className="min-h-5">
          {tokens.map((token, j) => (
            <span key={j} style={{ color: token.color }}>
              {token.content}
            </span>
          ))}
        </div>
      ))}
    </>
  );
}

// FileViewer previews a file of the active project in a modal, with syntax
// highlighting for code, inline rendering for images, and a shortcut to
// mention the file in the chat.
export function FileViewer({
  path,
  onClose,
  onMention,
}: {
  path: string;
  onClose: () => void;
  onMention: (path: string) => void;
}) {
  const { file, isLoading, error } = useProjectFileContent(path);
  const [highlighterReady, setHighlighterReady] = useState(!!highlighter);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        onClose();
      }
    };
    window.addEventListener("keydown", onKeyDown, true);
    return () => window.removeEventListener("keydown", onKeyDown, true);
  }, [onClose]);

  // The highlighter loads asynchronously at boot; re-render once it's ready
  useEffect(() => {
    if (highlighter) return;
    let cancelled = false;
    highlighterPromise.then(() => {
      if (!cancelled) setHighlighterReady(true);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const content = file?.binary ? "" : (file?.content ?? "");
  const language = useMemo(() => languageOf(path), [path]);

  const tokens = useMemo(() => {
    if (!highlighterReady || !highlighter) return null;
    if (!language || !content || content.length > MAX_HIGHLIGHT_CHARS) {
      return null;
    }
    try {
      return {
        light: highlighter.codeToTokensBase(content, {
          lang: language,
          theme: LIGHT_THEME,
        }),
        dark: highlighter.codeToTokensBase(content, {
          lang: language,
          theme: DARK_THEME,
        }),
      };
    } catch {
      // unknown language for the loaded highlighter: render as plain text
      return null;
    }
  }, [content, language, highlighterReady]);

  const lineCount = useMemo(
    () => (content ? content.split("\n").length : 0),
    [content],
  );

  const slash = path.lastIndexOf("/");
  const name = slash === -1 ? path : path.slice(slash + 1);
  const dir = slash === -1 ? "" : path.slice(0, slash);
  const imageSrc =
    file?.mimeType && file.content
      ? `data:${file.mimeType};base64,${file.content}`
      : null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-6"
      onClick={onClose}
    >
      <div
        className="flex max-h-[85vh] w-full max-w-4xl flex-col overflow-hidden rounded-2xl border border-neutral-200 bg-white shadow-xl dark:border-neutral-700 dark:bg-neutral-900"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex flex-none items-center gap-3 border-b border-neutral-200 px-4 py-3 dark:border-neutral-800">
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-semibold text-neutral-800 dark:text-neutral-100">
              {name}
            </p>
            <p
              className="truncate text-xs text-neutral-500 dark:text-neutral-400"
              title={path}
            >
              {dir || "."}
              {file ? ` · ${formatBytes(file.size)}` : ""}
              {file?.truncated ? " · preview truncated" : ""}
            </p>
          </div>
          <button
            type="button"
            onClick={() => onMention(path)}
            title="Mention this file in the chat"
            className="flex items-center gap-1 rounded-lg px-2 py-1 text-xs text-neutral-600 hover:bg-neutral-100 dark:text-neutral-300 dark:hover:bg-neutral-800 cursor-pointer"
          >
            <AtSymbolIcon className="h-4 w-4" />
            Mention
          </button>
          {content && (
            <CopyButton
              content={content}
              className="text-neutral-500 dark:text-neutral-400"
              title="Copy file contents"
            />
          )}
          <button
            type="button"
            onClick={onClose}
            title="Close"
            className="rounded-lg p-1 text-neutral-400 hover:bg-neutral-100 hover:text-neutral-600 dark:hover:bg-neutral-800 dark:hover:text-neutral-300 cursor-pointer"
          >
            <XMarkIcon className="h-4 w-4" />
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-auto bg-neutral-50 dark:bg-neutral-800/40">
          {isLoading && (
            <p className="p-4 text-sm text-neutral-400">Loading file…</p>
          )}
          {error && <p className="p-4 text-sm text-red-500">{error}</p>}
          {!isLoading && !error && file && (
            <>
              {imageSrc ? (
                <div className="flex h-full items-center justify-center p-4">
                  <img
                    src={imageSrc}
                    alt={name}
                    className="max-h-[70vh] max-w-full object-contain"
                  />
                </div>
              ) : file.binary ? (
                <p className="p-4 text-sm text-neutral-500 dark:text-neutral-400">
                  Binary file — no preview available.
                </p>
              ) : content === "" ? (
                <p className="p-4 text-sm text-neutral-400">Empty file.</p>
              ) : (
                <div className="flex min-w-0 font-mono text-[13px] leading-5">
                  <div className="flex-none select-none py-3 pl-4 pr-3 text-right text-neutral-400 dark:text-neutral-600">
                    {Array.from({ length: lineCount }, (_, i) => (
                      <div key={i}>{i + 1}</div>
                    ))}
                  </div>
                  <div className="min-w-0 flex-1 overflow-x-auto py-3 pr-4">
                    <pre className="m-0 whitespace-pre dark:hidden">
                      <code>
                        <CodeLines lines={tokens?.light} fallback={content} />
                      </code>
                    </pre>
                    <pre className="m-0 hidden whitespace-pre dark:block">
                      <code>
                        <CodeLines lines={tokens?.dark} fallback={content} />
                      </code>
                    </pre>
                  </div>
                </div>
              )}
              {file.truncated && (
                <p className="px-4 pb-3 text-xs text-neutral-400">
                  Showing the first {formatBytes(content.length)} of the file.
                </p>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
