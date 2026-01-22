import React from "react";
import { Streamdown, defaultRemarkPlugins } from "streamdown";
import remarkCitationParser from "@/utils/remarkCitationParser";

interface StreamingMarkdownContentProps {
  content: string;
  isStreaming?: boolean;
  browserToolResult?: any; // TODO: proper type
}

const StreamingMarkdownContent: React.FC<StreamingMarkdownContentProps> = ({
  content,
  isStreaming = false,
  browserToolResult,
}) => {
    // Build the remark plugins array - keep default GFM and Math, add citations
    const remarkPlugins = React.useMemo(() => {
      return [
        defaultRemarkPlugins.gfm,
        defaultRemarkPlugins.math,
        defaultRemarkPlugins.cjkFriendly,
        defaultRemarkPlugins.cjkFriendlyGfmStrikethrough,
        remarkCitationParser,
      ];
    }, []);

    return (
      <div className="max-w-full break-words">
        <StreamingMarkdownErrorBoundary
          content={content}
          isStreaming={isStreaming}
        >
          <Streamdown
            isAnimating={isStreaming}
            remarkPlugins={remarkPlugins}
            controls={false}
            components={{
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
      </div>
    );
};

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
