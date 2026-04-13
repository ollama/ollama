import React from "react";
import { Streamdown, defaultRemarkPlugins } from "streamdown";
import remarkCitationParser from "@/utils/remarkCitationParser";
import CopyButton from "./CopyButton";
import type { BundledLanguage } from "shiki";
import type { ThemeRegistrationAny, ThemedToken } from "@shikijs/types";
import { highlighter, THEME_LIGHT, THEME_DARK } from "@/lib/highlighter";
import { useTheme } from "@/hooks/useTheme";

interface StreamingMarkdownContentProps {
  content: string;
  isStreaming?: boolean;
  size?: "sm" | "md" | "lg";
  browserToolResult?: { page_stack: string[] };
  deferRendering?: boolean;
}

// Helper to extract text from React nodes
const extractText = (node: React.ReactNode): string => {
  if (typeof node === "string") return node;
  if (typeof node === "number") return String(node);
  if (!node) return "";
  if (React.isValidElement(node)) {
    const props = node.props as { children?: React.ReactNode };
    if (props.children) {
      return extractText(props.children);
    }
  }
  if (Array.isArray(node)) {
    return node.map(extractText).join("");
  }
  return "";
};

const CodeBlock = React.memo(
  ({ children }: React.HTMLAttributes<HTMLPreElement>) => {
    const theme = useTheme();
    const [, forceRender] = React.useReducer((x: number) => x + 1, 0);
    const visibleRef = React.useRef(false);

    // Callback ref: set up IntersectionObserver once when the DOM node mounts
    const observerRef = React.useRef<IntersectionObserver | null>(null);
    const setBlockRef = React.useCallback((node: HTMLDivElement | null) => {
      if (observerRef.current) {
        observerRef.current.disconnect();
        observerRef.current = null;
      }
      if (!node || visibleRef.current) return;

      observerRef.current = new IntersectionObserver(
        ([entry]) => {
          if (entry.isIntersecting) {
            visibleRef.current = true;
            observerRef.current?.disconnect();
            observerRef.current = null;
            forceRender();
          }
        },
        { rootMargin: "200px" },
      );
      observerRef.current.observe(node);
    }, [forceRender]);

    // Extract code and language from children
    const codeElement = children as React.ReactElement<{
      className?: string;
      children: React.ReactNode;
    }>;
    const language =
      codeElement.props.className?.replace(/language-/, "") || "";
    const codeText = extractText(codeElement.props.children);

    // Lazy per-theme token cache: tokenize active theme only, cache both for instant theme switches
    const cacheRef = React.useRef<{
      key: string;
      light: ThemedToken[][] | null;
      dark: ThemedToken[][] | null;
    }>({ key: "", light: null, dark: null });

    const tokens = React.useMemo(() => {
      if (!visibleRef.current || !highlighter) return null;
      // Skip tokenization for unregistered languages to avoid Shiki throwing per code block
      if (language && !highlighter.getLoadedLanguages().includes(language)) return null;

      const cacheKey = `${codeText}:${language}`;
      if (cacheRef.current.key !== cacheKey) {
        cacheRef.current = { key: cacheKey, light: null, dark: null };
      }

      const cache = cacheRef.current;
      if (cache[theme]) return cache[theme];

      try {
        const shikiTheme = theme === "dark" ? THEME_DARK : THEME_LIGHT;
        cache[theme] = highlighter.codeToTokensBase(codeText, {
          lang: language as BundledLanguage,
          theme: shikiTheme as ThemeRegistrationAny,
        });
        return cache[theme];
      } catch (error) {
        console.error("Failed to highlight code:", error);
        return null;
      }
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [visibleRef.current, codeText, language, theme]);

    return (
      <div ref={setBlockRef} className="relative bg-neutral-100 dark:bg-neutral-800 rounded-2xl overflow-hidden my-6">
        <div className="flex select-none">
          {language && (
            <div className="text-[13px] text-neutral-500 dark:text-neutral-400 font-mono px-4 py-2">
              {language}
            </div>
          )}
          <CopyButton
            content={codeText}
            showLabels={true}
            className="copy-button text-neutral-500 dark:text-neutral-400 bg-neutral-100 dark:bg-neutral-800 ml-auto"
          />
        </div>
        <pre className={`m-0 text-sm overflow-x-auto p-4 ${theme === "dark" ? "bg-neutral-800" : "bg-neutral-100"}`}>
          <code className="font-mono text-sm">
            {tokens
              ? tokens.map((line: ThemedToken[], i: number) => (
                  <React.Fragment key={i}>
                    {line.map((token: ThemedToken, j: number) => (
                      <span
                        key={j}
                        style={{
                          color: token.color,
                        }}
                      >
                        {token.content}
                      </span>
                    ))}
                    {i < tokens.length - 1 && "\n"}
                  </React.Fragment>
                ))
              : codeText}
          </code>
        </pre>
      </div>
    );
  },
);

const remarkPlugins = [
  defaultRemarkPlugins.gfm,
  defaultRemarkPlugins.math,
  remarkCitationParser,
];

const StreamingMarkdownContent: React.FC<StreamingMarkdownContentProps> =
  React.memo(({ content, isStreaming = false, size, browserToolResult, deferRendering = false }) => {
    const [, forceRender] = React.useReducer((x: number) => x + 1, 0);
    const readyRef = React.useRef(!deferRendering);
    const observerRef = React.useRef<IntersectionObserver | null>(null);

    const setDeferRef = React.useCallback(
      (node: HTMLDivElement | null) => {
        if (observerRef.current) {
          observerRef.current.disconnect();
          observerRef.current = null;
        }
        if (!node || readyRef.current) return;
        observerRef.current = new IntersectionObserver(
          ([entry]) => {
            if (entry.isIntersecting) {
              readyRef.current = true;
              observerRef.current?.disconnect();
              observerRef.current = null;
              forceRender();
            }
          },
          { rootMargin: "200px" },
        );
        observerRef.current.observe(node);
      },
      [forceRender],
    );

    const showStreamdown = readyRef.current;

    return (
      <div
        ref={deferRendering && !showStreamdown ? setDeferRef : undefined}
        className={`
          max-w-full
          ${size === "sm" ? "prose-sm" : size === "lg" ? "prose-lg" : ""}
          prose
          prose-neutral
          prose-headings:font-semibold
          prose:text-neutral-800
          prose-strong:font-semibold
          prose-ul:marker:text-neutral-700
          prose-li:marker:text-neutral-700
          prose-headings:text-neutral-800
          prose-pre:bg-transparent
          prose-pre:rounded-xl
          prose-pre:text-neutral-800
          prose-pre:font-normal
          prose-pre:my-0
          prose-pre:max-w-full
          prose-pre:pt-1
          [&_table]:border-collapse
          [&_table]:w-full
          [&_table]:border
          [&_table]:border-neutral-200
          [&_table]:rounded-lg
          [&_table]:overflow-hidden
          [&_th]:px-3
          [&_th]:py-2
          [&_th]:text-left
          [&_th]:font-semibold
          [&_th]:border-b
          [&_th]:border-r
          [&_th]:border-neutral-200
          [&_th:last-child]:border-r-0
          [&_td]:px-3
          [&_td]:py-2
          [&_td]:border-r
          [&_td]:border-neutral-200
          [&_td:last-child]:border-r-0
          [&_tbody_tr:not(:last-child)_td]:border-b
          [&_code:not(pre_code)]:text-neutral-700
          [&_code:not(pre_code)]:bg-neutral-100
          [&_code:not(pre_code)]:font-normal
          [&_code:not(pre_code)]:px-1.5
          [&_code:not(pre_code)]:py-0.5
          [&_code:not(pre_code)]:rounded-md
          [&_code:not(pre_code)]:text-[90%]
          [&_code:not(pre_code)]:before:hidden
          [&_code:not(pre_code)]:after:hidden
          dark:prose-invert
          dark:prose:text-neutral-200
          dark:prose-pre:bg-none
          dark:prose-headings:text-neutral-200
          dark:prose-strong:text-neutral-200
          dark:prose-pre:text-neutral-200
          dark:prose:pre:text-neutral-200
          dark:[&_table]:border-neutral-700
          dark:[&_thead]:bg-neutral-800
          dark:[&_th]:border-neutral-700
          dark:[&_td]:border-neutral-700
          dark:[&_code:not(pre_code)]:text-neutral-200
          dark:[&_code:not(pre_code)]:bg-neutral-800
          dark:[&_code:not(pre_code)]:font-normal
          dark:prose-ul:marker:text-neutral-300
          dark:prose-li:marker:text-neutral-300
          break-words
        `}
      >
        {showStreamdown ? (
          <StreamingMarkdownErrorBoundary
            content={content}
            isStreaming={isStreaming}
          >
            <Streamdown
              parseIncompleteMarkdown={isStreaming}
              isAnimating={isStreaming}
              remarkPlugins={remarkPlugins}
              controls={false}
              components={{
                pre: CodeBlock,
                code: ({
                  children,
                  className,
                }: React.HTMLAttributes<HTMLElement>) => (
                  <code className={className}>{children}</code>
                ),
                table: ({
                  children,
                  ...props
                }: React.HTMLAttributes<HTMLTableElement>) => (
                  <div className="overflow-x-auto max-w-full">
                    <table
                      {...props}
                      className="border-collapse w-full border border-neutral-200 dark:border-neutral-700 rounded-lg overflow-hidden"
                    >
                      {children}
                    </table>
                  </div>
                ),
                // @ts-expect-error: custom citation type
                "ol-citation": ({
                  cursor,
                }: {
                  cursor: number;
                  start: number;
                  end: number;
                }) => {
                  const pageStack = browserToolResult?.page_stack;
                  const hasValidPage = pageStack && cursor < pageStack.length;
                  const pageUrl = hasValidPage ? pageStack[cursor] : null;

                  const getPageTitle = (url: string) => {
                    if (url.startsWith("search_results_")) {
                      const searchTerm = url.substring("search_results_".length);
                      return `Search: ${searchTerm}`;
                    }
                    try {
                      const urlObj = new URL(url);
                      return urlObj.hostname;
                    } catch {
                      return url;
                    }
                  };

                  const citationElement = (
                    <span className="text-xs text-neutral-500 dark:text-neutral-400 bg-neutral-100 dark:bg-neutral-800 rounded-full px-2 py-1 ml-1">
                      [{cursor}]
                    </span>
                  );

                  if (pageUrl && pageUrl.startsWith("http")) {
                    return (
                      <a
                        href={pageUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center hover:opacity-80 transition-opacity no-underline"
                        title={getPageTitle(pageUrl)}
                      >
                        {citationElement}
                      </a>
                    );
                  }

                  return citationElement;
                },
              }}
            >
              {content}
            </Streamdown>
          </StreamingMarkdownErrorBoundary>
        ) : (
          <div className="whitespace-pre-wrap text-sm text-neutral-800 dark:text-neutral-200">
            {content}
          </div>
        )}
      </div>
    );
  });

interface StreamingMarkdownErrorBoundaryProps {
  content: string;
  children: React.ReactNode;
  isStreaming: boolean;
}

// Sometimes remark will throw errors, particularly when rendering math. We add
// this fallback to show the plain text content if there's an error, and then we
// retry rendering the content if the content changes OR if we change our
// streaming state (because we render things differently when in streaming mode
// v. not, so for some cases we'll automatically recover once streaming is over)
//
// This should not be relied on for anything known to be broken (any known
// errors should be fixed!), but it's necessary to not break the full UI because
// of some bad markdown
class StreamingMarkdownErrorBoundary extends React.Component<
  StreamingMarkdownErrorBoundaryProps,
  { hasError: boolean }
> {
  constructor(props: StreamingMarkdownErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  componentDidUpdate(prevProps: StreamingMarkdownErrorBoundaryProps) {
    if (
      prevProps.isStreaming !== this.props.isStreaming ||
      prevProps.content !== this.props.content
    ) {
      this.setState({ hasError: false });
    }
  }

  static getDerivedStateFromError(/*_error: Error*/) {
    return { hasError: true };
  }

  componentDidCatch(error: Error, info: { componentStack: string }) {
    console.error(
      "StreamingMarkdownContent: caught rendering error",
      error,
      info,
    );
  }

  render() {
    if (this.state.hasError) {
      // TODO(drifkin): render this more nicely so it's not so jarring. For
      // example, probably want to render newlines, etc. But let's not get too
      // fancy because then we'll end up needing an ErrorBoundaryErrorBoundary
      // :upside_down_face:
      return <div>{this.props.content}</div>;
    }

    return this.props.children;
  }
}

export default StreamingMarkdownContent;
