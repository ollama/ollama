import { Message as MessageType, ToolCall, File } from "@/gotypes";
import Thinking from "./Thinking";
import StreamingMarkdownContent from "./StreamingMarkdownContent";
import { ImageThumbnail } from "./ImageThumbnail";
import { isImageFile } from "@/utils/imageUtils";
import CopyButton from "./CopyButton";
import React, { useState, useMemo, useRef } from "react";

const Message = React.memo(
  ({
    message,
    onEditMessage,
    messageIndex,
    isStreaming,
    isFaded,
    browserToolResult,
    lastToolQuery,
  }: {
    message: MessageType;
    onEditMessage?: (content: string, index: number) => void;
    messageIndex?: number;
    isStreaming: boolean;
    isFaded?: boolean;
    // TODO(drifkin): this type isn't right
    browserToolResult?: BrowserToolResult;
    lastToolQuery?: string;
  }) => {
    if (message.role === "user") {
      return (
        <UserMessage
          message={message}
          onEditMessage={onEditMessage}
          messageIndex={messageIndex}
          isFaded={isFaded}
        />
      );
    } else {
      return (
        <OtherRoleMessage
          message={message}
          isStreaming={isStreaming}
          isFaded={isFaded}
          browserToolResult={browserToolResult}
          lastToolQuery={lastToolQuery}
        />
      );
    }
  },
  (prevProps, nextProps) => {
    return (
      prevProps.message === nextProps.message &&
      prevProps.onEditMessage === nextProps.onEditMessage &&
      prevProps.messageIndex === nextProps.messageIndex &&
      prevProps.isStreaming === nextProps.isStreaming &&
      prevProps.isFaded === nextProps.isFaded &&
      prevProps.browserToolResult === nextProps.browserToolResult
    );
  },
);

export default Message;

// TODO(drifkin): fill in more (or generate from go types)
type BrowserToolResult = {
  page_stack: string[];
};

type BrowserToolContent = {
  cursor: number;
  title: string;
  url: string;
  startingLine: number;
  totalLines: number;
  lines: string[];
};

// Example:
/*
[0] Devon Rifkin(search_results_Devon Rifkin)
**viewing lines [0 - 134] of 167**

L0:
L1: 
L2: URL: 
L3: # Search Results
*/

function processBrowserToolContent(content: string): BrowserToolContent {
  const lines = content.split("\n");

  const firstLine = lines[0];
  // For a first line like the following:
  // [0] Page Title(search_results_Query)
  // we want to extract:
  // - cursor: 0
  // - title: Page Title
  // - url: search_results_Query

  // use a regex to extract the cursor, title and URL, all in one shot. It's okay if the page title has parens in it, the very last parens should be the URL
  const firstLineMatch = firstLine.match(/^\[(\d+)\]\s+(.+)\(([^)]+)\)$/);

  const cursor = firstLineMatch ? parseInt(firstLineMatch[1], 10) : 0;
  const title = firstLineMatch ? firstLineMatch[2].trim() : "";
  const url = firstLineMatch ? firstLineMatch[3] : "";

  // Parse the viewing lines info from the second line
  // Example: **viewing lines [0 - 134] of 167**
  const viewingLineMatch = lines[1]?.match(
    /\*\*viewing lines \[(\d+) - (\d+)\] of (\d+)\*\*/,
  );
  const startingLine = viewingLineMatch ? parseInt(viewingLineMatch[1], 10) : 0;
  let totalLines = viewingLineMatch ? parseInt(viewingLineMatch[3], 10) : 0;

  // TEMP(drifkin): waiting for a fix from parth, for now making it so we make
  // sure the total lines is at least as much as the ending line number + 1
  const endingLine = viewingLineMatch ? parseInt(viewingLineMatch[2], 10) : 0;
  totalLines = Math.max(totalLines, endingLine + 1);

  // Extract the actual content lines (skip first 2 lines and empty line 3)
  const contentLines = lines.slice(3).filter((line) => line.startsWith("L"));
  // remove the L<number>: prefix with a regex
  const contentLinesWithoutPrefix = contentLines.map((line) =>
    line.replace(/^L(\d+):\s*/, ""),
  );

  return {
    cursor,
    title,
    url,
    startingLine,
    totalLines,
    lines: contentLinesWithoutPrefix,
  };
}

function BrowserToolResult({
  content,
}: {
  toolResult: BrowserToolResult;
  content: string;
}) {
  const [isCollapsed, setIsCollapsed] = React.useState(true);
  const processedContent = useMemo(
    () => processBrowserToolContent(content),
    [content],
  );

  let urlToUse: string | null = null;
  if (processedContent.url.startsWith("http")) {
    urlToUse = processedContent.url;
  }

  const isSearchResults =
    /^search_results_/i.test(processedContent.url) ||
    /_search$/i.test(processedContent.url);

  return (
    <div
      className={`flex flex-col w-full ${!isCollapsed ? "text-neutral-800 dark:text-neutral-200" : "text-neutral-600 dark:text-neutral-400"}
         hover:text-neutral-800
        dark:hover:text-neutral-200 transition-colors`}
    >
      <div
        className="flex cursor-pointer group/browser self-start relative"
        onClick={() => setIsCollapsed(!isCollapsed)}
      >
        {/* Browser icon */}
        <svg
          className={`w-10 absolute -left-0.5 top-1 transition-opacity ${
            isCollapsed ? "opacity-100" : "opacity-0"
          } group-hover/browser:opacity-0 fill-current will-change-opacity`}
          xmlns="http://www.w3.org/2000/svg"
        >
          <path d="M17.9297 8.96484C17.9297 13.9131 13.9131 17.9297 8.96484 17.9297C4.0166 17.9297 0 13.9131 0 8.96484C0 4.0166 4.0166 0 8.96484 0C13.9131 0 17.9297 4.0166 17.9297 8.96484ZM8.52608 0.544075C7.9195 0.648681 7.34943 0.980368 6.84016 1.49718C5.65243 1.83219 4.58058 2.44328 3.70023 3.25921C3.62549 3.21723 3.56076 3.16926 3.49805 3.12012C3.24316 2.91797 2.90918 2.8916 2.68066 3.11133C2.46094 3.33105 2.44336 3.68262 2.68945 3.91113C2.76044 3.97523 2.83584 4.03812 2.91591 4.09957C1.95192 5.28662 1.33642 6.76661 1.2196 8.38477H1.05469C0.738281 8.38477 0.474609 8.64844 0.474609 8.96484C0.474609 9.28125 0.738281 9.53613 1.05469 9.53613H1.21922C1.33511 11.1701 1.95962 12.6636 2.93835 13.8571C2.84984 13.9239 2.76702 13.9925 2.68945 14.0625C2.44336 14.2822 2.46094 14.6426 2.68066 14.8623C2.90918 15.082 3.24316 15.0557 3.49805 14.8535C3.56925 14.7972 3.64306 14.7424 3.72762 14.6944C4.60071 15.4974 5.6608 16.0989 6.83417 16.431C7.32996 16.9376 7.88339 17.2682 8.47128 17.3845C8.56963 17.5572 8.75351 17.6748 8.96484 17.6748C9.1767 17.6748 9.36492 17.5566 9.46571 17.3832C10.0509 17.2655 10.6018 16.9355 11.0955 16.431C12.2689 16.0989 13.329 15.4974 14.2021 14.6944C14.2866 14.7424 14.3604 14.7972 14.4316 14.8535C14.6865 15.0557 15.0205 15.082 15.249 14.8623C15.4688 14.6426 15.4863 14.2822 15.2402 14.0625C15.1627 13.9925 15.0798 13.9239 14.9913 13.8571C15.9701 12.6636 16.5946 11.1701 16.7105 9.53613H17.0508C17.3672 9.53613 17.6309 9.28125 17.6309 8.96484C17.6309 8.64844 17.3672 8.38477 17.0508 8.38477H16.7101C16.5933 6.76661 15.9778 5.28661 15.0138 4.09957C15.0939 4.03812 15.1692 3.97523 15.2402 3.91113C15.4863 3.68262 15.4688 3.33105 15.249 3.11133C15.0205 2.8916 14.6865 2.91797 14.4316 3.12012C14.3689 3.16926 14.3042 3.21723 14.2295 3.25921C13.3491 2.44328 12.2773 1.83219 11.0895 1.49718C10.582 0.982111 10.014 0.650921 9.40974 0.545145C9.30275 0.416705 9.14207 0.333984 8.96484 0.333984C8.78813 0.333984 8.63061 0.416228 8.52608 0.544075Z" />
          <path d="M8.39345 16.2322V17.0946C8.39345 17.1996 8.42151 17.2988 8.47104 17.3841C7.17953 17.1251 6.0523 15.8435 5.33988 13.9209C5.68987 13.8012 6.07289 13.7058 6.48098 13.6307C6.9693 14.9474 7.64728 15.9065 8.39345 16.2322ZM12.5896 13.9209C11.8787 15.8395 10.7546 17.1198 9.46632 17.3819C9.5163 17.2971 9.54482 17.1987 9.54482 17.0946V16.229C10.2875 15.899 10.9621 14.9423 11.4485 13.6307C11.8566 13.7058 12.2396 13.8012 12.5896 13.9209ZM6.14535 12.5477C5.74273 12.6298 5.35968 12.7296 5.00217 12.8459C4.73645 11.8577 4.57374 10.7411 4.54053 9.53605H5.71744C5.75386 10.6089 5.90649 11.6345 6.14535 12.5477ZM12.9273 12.8459C12.5698 12.7296 12.1868 12.6298 11.7841 12.5477C12.023 11.6345 12.1756 10.6089 12.212 9.53605H13.389C13.3557 10.7411 13.193 11.8577 12.9273 12.8459ZM6.13265 5.42334C5.9038 6.32028 5.75639 7.32624 5.71828 8.38469H4.54152C4.57689 7.19848 4.73601 6.09939 4.9947 5.12525C5.35068 5.24129 5.73201 5.341 6.13265 5.42334ZM13.388 8.38469H12.2112C12.1731 7.32624 12.0257 6.32028 11.7968 5.42334C12.1975 5.341 12.5788 5.24129 12.9348 5.12525C13.1935 6.09939 13.3526 7.19848 13.388 8.38469ZM8.39345 0.913986V1.69612C7.63864 2.02298 6.95359 2.99375 6.46426 4.3367C6.05787 4.26121 5.67698 4.16517 5.32878 4.04549C6.0504 2.07603 7.2039 0.775083 8.52864 0.541016C8.44416 0.642235 8.39345 0.772865 8.39345 0.913986ZM12.6007 4.04549C12.2525 4.16517 11.8716 4.26121 11.4652 4.3367C10.9778 2.99892 10.2961 2.03048 9.54482 1.69935V0.913986C9.54482 0.77311 9.49255 0.642688 9.40642 0.541528C10.7288 0.778686 11.8801 2.07875 12.6007 4.04549Z" />
          <path d="M8.96484 17.6748C9.28125 17.6748 9.54492 17.4111 9.54492 17.0947V0.914062C9.54492 0.597656 9.28125 0.333984 8.96484 0.333984C8.64844 0.333984 8.39355 0.597656 8.39355 0.914062V17.0947C8.39355 17.4111 8.64844 17.6748 8.96484 17.6748ZM3.49805 14.8535C4.67578 13.9219 6.56543 13.4209 8.96484 13.4209C11.3643 13.4209 13.2539 13.9219 14.4316 14.8535C14.6865 15.0557 15.0205 15.082 15.249 14.8623C15.4688 14.6426 15.4863 14.2822 15.2402 14.0625C14.0625 12.999 11.6719 12.2695 8.96484 12.2695C6.25781 12.2695 3.86719 12.999 2.68945 14.0625C2.44336 14.2822 2.46094 14.6426 2.68066 14.8623C2.90918 15.082 3.24316 15.0557 3.49805 14.8535ZM1.05469 9.53613H17.0508C17.3672 9.53613 17.6309 9.28125 17.6309 8.96484C17.6309 8.64844 17.3672 8.38477 17.0508 8.38477H1.05469C0.738281 8.38477 0.474609 8.64844 0.474609 8.96484C0.474609 9.28125 0.738281 9.53613 1.05469 9.53613ZM8.96484 5.7041C11.6719 5.7041 14.0625 4.97461 15.2402 3.91113C15.4863 3.68262 15.4688 3.33105 15.249 3.11133C15.0205 2.8916 14.6865 2.91797 14.4316 3.12012C13.2539 4.04297 11.3643 4.55273 8.96484 4.55273C6.56543 4.55273 4.67578 4.04297 3.49805 3.12012C3.24316 2.91797 2.90918 2.8916 2.68066 3.11133C2.46094 3.33105 2.44336 3.68262 2.68945 3.91113C3.86719 4.97461 6.25781 5.7041 8.96484 5.7041Z" />
        </svg>

        {/* Arrow */}
        <svg
          className={`h-4 w-4 absolute top-1.5 transition-all ${
            isCollapsed
              ? "-rotate-90 opacity-0 group-hover/browser:opacity-100"
              : "rotate-0 opacity-100"
          } will-change-[opacity,transform]`}
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <polyline points="6 9 12 15 18 9"></polyline>
        </svg>

        <div className="ml-6">
          {isSearchResults ? (
            <span>
              {(() => {
                const term = /^search_results_/i.test(processedContent.url)
                  ? processedContent.url.replace(/^search_results_/i, "")
                  : processedContent.url.replace(/_search$/i, "");
                return (
                  <>
                    Search results for <InlineSearchTerm term={term} />
                  </>
                );
              })()}
            </span>
          ) : (
            <InlineSearchTerm term={processedContent.title} />
          )}
          {urlToUse != null && (
            <span className="text-neutral-500 text-sm ml-2 break-all">
              ({urlToUse})
            </span>
          )}
          <span className="text-neutral-500 text-sm ml-2">
            (lines {processedContent.startingLine}-
            {processedContent.startingLine + processedContent.lines.length - 1}{" "}
            of {processedContent.totalLines})
          </span>
        </div>
      </div>
      <div
        className={`text-xs text-neutral-500 dark:text-neutral-500 rounded-md overflow-y-auto
          transition-[max-height,opacity] duration-300 ease-in-out ml-6 mt-2`}
        style={{
          maxHeight: isCollapsed ? "0px" : "20rem",
          opacity: isCollapsed ? 0 : 1,
        }}
      >
        <div className="transition-transform duration-300 opacity-75">
          <div className="overflow-x-auto">
            {processedContent.lines.map((line, index) => {
              const lineNumber = processedContent.startingLine + index;
              return (
                <div
                  key={index}
                  className="flex whitespace-nowrap text-xs h-[2em] font-mono"
                >
                  <div className="w-10 text-right pr-2 text-neutral-500 flex-shrink-0 border-r border-neutral-200 dark:border-neutral-700">
                    {lineNumber}
                  </div>
                  <div className="pl-2 pr-4">{line}</div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

function ToolRoleContent({
  message,
  browserToolResult,
  lastToolQuery,
}: {
  message: MessageType;
  browserToolResult?: BrowserToolResult;
  lastToolQuery?: string;
}) {
  const content = message.content;
  const rawToolResult = (message as any).tool_result;
  const toolName = (message as any).tool_name || (message as any).toolName;
  const [isCollapsed, setIsCollapsed] = useState(true);

  if (browserToolResult && typeof browserToolResult === "object") {
    return (
      <BrowserToolResult toolResult={browserToolResult} content={content} />
    );
  }
  return (
    // collapsable tool result with raw json
    <div className="space-y-2">
      {content && !rawToolResult && (
        <pre className="text-xs whitespace-pre-wrap overflow-x-auto bg-neutral-100 dark:bg-neutral-800 text-neutral-800 dark:text-neutral-200 p-2 rounded-md max-h-40">
          <code>{content}</code>
        </pre>
      )}

      {rawToolResult && (
        <div className="flex flex-col w-full text-neutral-600 dark:text-neutral-400 relative select-text hover:text-neutral-800 dark:hover:text-neutral-200 transition-colors">
          <div
            className="flex cursor-pointer group/browser self-start relative"
            onClick={() => setIsCollapsed(!isCollapsed)}
          >
            {/* Globe icon */}
            <svg
              className="w-10 absolute -left-0.5 top-1 fill-current transition-opacity group-hover/browser:opacity-0"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path d="M17.9297 8.96484C17.9297 13.9131 13.9131 17.9297 8.96484 17.9297C4.0166 17.9297 0 13.9131 0 8.96484C0 4.0166 4.0166 0 8.96484 0C13.9131 0 17.9297 4.0166 17.9297 8.96484ZM8.52608 0.544075C7.9195 0.648681 7.34943 0.980368 6.84016 1.49718C5.65243 1.83219 4.58058 2.44328 3.70023 3.25921C3.62549 3.21723 3.56076 3.16926 3.49805 3.12012C3.24316 2.91797 2.90918 2.8916 2.68066 3.11133C2.46094 3.33105 2.44336 3.68262 2.68945 3.91113C2.76044 3.97523 2.83584 4.03812 2.91591 4.09957C1.95192 5.28662 1.33642 6.76661 1.2196 8.38477H1.05469C0.738281 8.38477 0.474609 8.64844 0.474609 8.96484C0.474609 9.28125 0.738281 9.53613 1.05469 9.53613H1.21922C1.33511 11.1701 1.95962 12.6636 2.93835 13.8571C2.84984 13.9239 2.76702 13.9925 2.68945 14.0625C2.44336 14.2822 2.46094 14.6426 2.68066 14.8623C2.90918 15.082 3.24316 15.0557 3.49805 14.8535C3.56925 14.7972 3.64306 14.7424 3.72762 14.6944C4.60071 15.4974 5.6608 16.0989 6.83417 16.431C7.32996 16.9376 7.88339 17.2682 8.47128 17.3845C8.56963 17.5572 8.75351 17.6748 8.96484 17.6748C9.1767 17.6748 9.36492 17.5566 9.46571 17.3832C10.0509 17.2655 10.6018 16.9355 11.0955 16.431C12.2689 16.0989 13.329 15.4974 14.2021 14.6944C14.2866 14.7424 14.3604 14.7972 14.4316 14.8535C14.6865 15.0557 15.0205 15.082 15.249 14.8623C15.4688 14.6426 15.4863 14.2822 15.2402 14.0625C15.1627 13.9925 15.0798 13.9239 14.9913 13.8571C15.9701 12.6636 16.5946 11.1701 16.7105 9.53613H17.0508C17.3672 9.53613 17.6309 9.28125 17.6309 8.96484C17.6309 8.64844 17.3672 8.38477 17.0508 8.38477H16.7101C16.5933 6.76661 15.9778 5.28661 15.0138 4.09957C15.0939 4.03812 15.1692 3.97523 15.2402 3.91113C15.4863 3.68262 15.4688 3.33105 15.249 3.11133C15.0205 2.8916 14.6865 2.91797 14.4316 3.12012C14.3689 3.16926 14.3042 3.21723 14.2295 3.25921C13.3491 2.44328 12.2773 1.83219 11.0895 1.49718C10.582 0.982111 10.014 0.650921 9.40974 0.545145C9.30275 0.416705 9.14207 0.333984 8.96484 0.333984C8.78813 0.333984 8.63061 0.416228 8.52608 0.544075Z" />
              <path d="M8.39345 16.2322V17.0946C8.39345 17.1996 8.42151 17.2988 8.47104 17.3841C7.17953 17.1251 6.0523 15.8435 5.33988 13.9209C5.68987 13.8012 6.07289 13.7058 6.48098 13.6307C6.9693 14.9474 7.64728 15.9065 8.39345 16.2322ZM12.5896 13.9209C11.8787 15.8395 10.7546 17.1198 9.46632 17.3819C9.5163 17.2971 9.54482 17.1987 9.54482 17.0946V16.229C10.2875 15.899 10.9621 14.9423 11.4485 13.6307C11.8566 13.7058 12.2396 13.8012 12.5896 13.9209ZM6.14535 12.5477C5.74273 12.6298 5.35968 12.7296 5.00217 12.8459C4.73645 11.8577 4.57374 10.7411 4.54053 9.53605H5.71744C5.75386 10.6089 5.90649 11.6345 6.14535 12.5477ZM12.9273 12.8459C12.5698 12.7296 12.1868 12.6298 11.7841 12.5477C12.023 11.6345 12.1756 10.6089 12.212 9.53605H13.389C13.3557 10.7411 13.193 11.8577 12.9273 12.8459ZM6.13265 5.42334C5.9038 6.32028 5.75639 7.32624 5.71828 8.38469H4.54152C4.57689 7.19848 4.73601 6.09939 4.9947 5.12525C5.35068 5.24129 5.73201 5.341 6.13265 5.42334ZM13.388 8.38469H12.2112C12.1731 7.32624 12.0257 6.32028 11.7968 5.42334C12.1975 5.341 12.5788 5.24129 12.9348 5.12525C13.1935 6.09939 13.3526 7.19848 13.388 8.38469ZM8.39345 0.913986V1.69612C7.63864 2.02298 6.95359 2.99375 6.46426 4.3367C6.05787 4.26121 5.67698 4.16517 5.32878 4.04549C6.0504 2.07603 7.2039 0.775083 8.52864 0.541016C8.44416 0.642235 8.39345 0.772865 8.39345 0.913986ZM12.6007 4.04549C12.2525 4.16517 11.8716 4.26121 11.4652 4.3367C10.9778 2.99892 10.2961 2.03048 9.54482 1.69935V0.913986C9.54482 0.77311 9.49255 0.642688 9.40642 0.541528C10.7288 0.778686 11.8801 2.07875 12.6007 4.04549Z" />
              <path d="M8.96484 17.6748C9.28125 17.6748 9.54492 17.4111 9.54492 17.0947V0.914062C9.54492 0.597656 9.28125 0.333984 8.96484 0.333984C8.64844 0.333984 8.39355 0.597656 8.39355 0.914062V17.0947C8.39355 17.4111 8.64844 17.6748 8.96484 17.6748ZM3.49805 14.8535C4.67578 13.9219 6.56543 13.4209 8.96484 13.4209C11.3643 13.4209 13.2539 13.9219 14.4316 14.8535C14.6865 15.0557 15.0205 15.082 15.249 14.8623C15.4688 14.6426 15.4863 14.2822 15.2402 14.0625C14.0625 12.999 11.6719 12.2695 8.96484 12.2695C6.25781 12.2695 3.86719 12.999 2.68945 14.0625C2.44336 14.2822 2.46094 14.6426 2.68066 14.8623C2.90918 15.082 3.24316 15.0557 3.49805 14.8535ZM1.05469 9.53613H17.0508C17.3672 9.53613 17.6309 9.28125 17.6309 8.96484C17.6309 8.64844 17.3672 8.38477 17.0508 8.38477H1.05469C0.738281 8.38477 0.474609 8.64844 0.474609 8.96484C0.474609 9.28125 0.738281 9.53613 1.05469 9.53613ZM8.96484 5.7041C11.6719 5.7041 14.0625 4.97461 15.2402 3.91113C15.4863 3.68262 15.4688 3.33105 15.249 3.11133C15.0205 2.8916 14.6865 2.91797 14.4316 3.12012C13.2539 4.04297 11.3643 4.55273 8.96484 4.55273C6.56543 4.55273 4.67578 4.04297 3.49805 3.12012C3.24316 2.91797 2.90918 2.8916 2.68066 3.11133C2.46094 3.33105 2.44336 3.68262 2.68945 3.91113C3.86719 4.97461 6.25781 5.7041 8.96484 5.7041Z" />
            </svg>
            {/* Arrow */}
            <svg
              className={`h-4 w-4 absolute top-1.5 transition-all opacity-0 group-hover/browser:opacity-100 ${
                isCollapsed ? "-rotate-90" : "rotate-0"
              } will-change-[opacity,transform]`}
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <polyline points="6 9 12 15 18 9"></polyline>
            </svg>

            <div
              className={`${toolName === "web_search" || toolName === "web_fetch" ? "ml-6" : "ml-6"}`}
            >
              <span>
                {toolName === "web_search"
                  ? (() => {
                      const q =
                        (lastToolQuery && lastToolQuery.trim()) ||
                        (rawToolResult &&
                        typeof rawToolResult === "object" &&
                        typeof (rawToolResult as any).query === "string"
                          ? (rawToolResult as any).query.trim()
                          : "");
                      return q ? (
                        <>
                          Search results for <InlineSearchTerm term={q} />
                        </>
                      ) : (
                        "Web search results"
                      );
                    })()
                  : toolName === "web_fetch"
                    ? (() => {
                        const u =
                          (lastToolQuery && lastToolQuery.trim()) ||
                          (rawToolResult &&
                          typeof rawToolResult === "object" &&
                          typeof (rawToolResult as any).url === "string"
                            ? (rawToolResult as any).url
                            : "");
                        return u ? (
                          <>
                            Fetch results for{" "}
                            <span className="break-all">{u}</span>
                          </>
                        ) : (
                          "Web fetch results"
                        );
                      })()
                    : "Raw tool result"}
              </span>
            </div>
          </div>

          <div
            className="text-xs text-neutral-500 dark:text-neutral-500 rounded-md overflow-y-auto transition-[max-height,opacity] duration-300 ease-in-out ml-6 mt-2"
            style={{
              maxHeight: isCollapsed ? "0px" : "20rem",
              opacity: isCollapsed ? 0 : 1,
            }}
          >
            <pre
              id="raw-json-tool-result"
              className="text-xs overflow-x-auto bg-neutral-50 dark:bg-neutral-900 text-neutral-800 dark:text-neutral-200 p-2 rounded-md border border-neutral-200 dark:border-neutral-700"
            >
              <code>
                {typeof rawToolResult === "string"
                  ? rawToolResult
                  : JSON.stringify(rawToolResult, null, 2)}
              </code>
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}

function InlineSearchTerm({ term }: { term: string }) {
  return (
    // <span className="font-bold before:content-['\201C'] after:content-['\201D']">
    <span className="font-medium">{term}</span>
  );
}

function cursorToPageText(
  cursor: number,
  browserToolResult: BrowserToolResult | undefined,
): string {
  if (browserToolResult) {
    let page = browserToolResult.page_stack[cursor];
    if (page) {
      if (page.startsWith("search_results_")) {
        const searchTerm = page.replace(/^search_results_/, "");
        page = `Search results for "${searchTerm}"`;
      }
      return page;
    }
    return page || "Unknown page";
  }

  if (cursor === undefined) {
    console.warn("cursor is undefined");
    return "Page";
  }

  return `Page #${cursor}`;
}

function cursorToPage(
  cursor: number,
  browserToolResult: BrowserToolResult | undefined,
) {
  const pageText = cursorToPageText(cursor, browserToolResult);

  return (
    <span className="font-medium text-sm text-neutral-500 border border-neutral-200 dark:border-neutral-700 rounded-md px-1 py-0.5">
      {pageText}
    </span>
  );
}

// TODO(drifkin): pull out into another file
function BrowserToolCallDisplay({
  toolCall,
  browserToolResult,
}: {
  toolCall: ToolCall;
  browserToolResult?: BrowserToolResult;
}) {
  const args = JSON.parse(toolCall.function.arguments);
  if (toolCall.function.name === "browser.search") {
    const query = args.query;
    return (
      <div className="text-neutral-600 dark:text-neutral-400 relative mb-3 select-text">
        <svg
          className="fill-current h-4 absolute top-1"
          viewBox="0 0 18 18"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path d="M0 7.01367C0 10.8809 3.14648 14.0273 7.01367 14.0273C8.54297 14.0273 9.94043 13.5352 11.0918 12.709L15.416 17.042C15.6182 17.2441 15.8818 17.3408 16.1631 17.3408C16.7607 17.3408 17.1738 16.8926 17.1738 16.3037C17.1738 16.0225 17.0684 15.7676 16.8838 15.583L12.5859 11.2588C13.4912 10.0811 14.0273 8.61328 14.0273 7.01367C14.0273 3.14648 10.8809 0 7.01367 0C3.14648 0 0 3.14648 0 7.01367ZM1.50293 7.01367C1.50293 3.97266 3.97266 1.50293 7.01367 1.50293C10.0547 1.50293 12.5244 3.97266 12.5244 7.01367C12.5244 10.0547 10.0547 12.5244 7.01367 12.5244C3.97266 12.5244 1.50293 10.0547 1.50293 7.01367Z" />
        </svg>
        <div className="ml-6">
          Searching for <InlineSearchTerm term={query} />
          &#8230;
        </div>
      </div>
    );
  } else if (toolCall.function.name === "browser.open") {
    const cursor = args.cursor;
    const id = args.id;
    const idAllNumeric = !isNaN(Number(id));

    if (idAllNumeric) {
      return (
        <div className="text-neutral-600 dark:text-neutral-400 relative mb-3 select-text">
          <svg
            className="fill-current h-4 absolute top-1"
            viewBox="0 0 18 18"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path d="M0 7.01367C0 10.8809 3.14648 14.0273 7.01367 14.0273C8.54297 14.0273 9.94043 13.5352 11.0918 12.709L15.416 17.042C15.6182 17.2441 15.8818 17.3408 16.1631 17.3408C16.7607 17.3408 17.1738 16.8926 17.1738 16.3037C17.1738 16.0225 17.0684 15.7676 16.8838 15.583L12.5859 11.2588C13.4912 10.0811 14.0273 8.61328 14.0273 7.01367C14.0273 3.14648 10.8809 0 7.01367 0C3.14648 0 0 3.14648 0 7.01367ZM1.50293 7.01367C1.50293 3.97266 3.97266 1.50293 7.01367 1.50293C10.0547 1.50293 12.5244 3.97266 12.5244 7.01367C12.5244 10.0547 10.0547 12.5244 7.01367 12.5244C3.97266 12.5244 1.50293 10.0547 1.50293 7.01367Z" />
          </svg>
          <div className="ml-6">
            Opening link #{id} from {cursorToPage(cursor, browserToolResult)}
          </div>
        </div>
      );
    } else {
      const loc = args.loc;
      return (
        <div className="text-neutral-600 dark:text-neutral-400 relative mb-3 select-text">
          <svg
            className="fill-current h-4 absolute top-1"
            viewBox="0 0 18 18"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path d="M0 7.01367C0 10.8809 3.14648 14.0273 7.01367 14.0273C8.54297 14.0273 9.94043 13.5352 11.0918 12.709L15.416 17.042C15.6182 17.2441 15.8818 17.3408 16.1631 17.3408C16.7607 17.3408 17.1738 16.8926 17.1738 16.3037C17.1738 16.0225 17.0684 15.7676 16.8838 15.583L12.5859 11.2588C13.4912 10.0811 14.0273 8.61328 14.0273 7.01367C14.0273 3.14648 10.8809 0 7.01367 0C3.14648 0 0 3.14648 0 7.01367ZM1.50293 7.01367C1.50293 3.97266 3.97266 1.50293 7.01367 1.50293C10.0547 1.50293 12.5244 3.97266 12.5244 7.01367C12.5244 10.0547 10.0547 12.5244 7.01367 12.5244C3.97266 12.5244 1.50293 10.0547 1.50293 7.01367Z" />
          </svg>
          <div className="ml-6">
            {loc
              ? `Scrolling to line ${loc} on ${cursorToPageText(
                  cursor,
                  browserToolResult,
                )}`
              : `Scrolling`}
          </div>
        </div>
      );
    }
  } else if (toolCall.function.name === "browser.find") {
    const cursor = args.cursor;
    const pattern = args.pattern;

    return (
      <div className="text-neutral-600 dark:text-neutral-400 relative mb-3 select-text">
        <svg
          className="fill-current h-4 absolute top-1"
          viewBox="0 0 18 18"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path d="M0 7.01367C0 10.8809 3.14648 14.0273 7.01367 14.0273C8.54297 14.0273 9.94043 13.5352 11.0918 12.709L15.416 17.042C15.6182 17.2441 15.8818 17.3408 16.1631 17.3408C16.7607 17.3408 17.1738 16.8926 17.1738 16.3037C17.1738 16.0225 17.0684 15.7676 16.8838 15.583L12.5859 11.2588C13.4912 10.0811 14.0273 8.61328 14.0273 7.01367C14.0273 3.14648 10.8809 0 7.01367 0C3.14648 0 0 3.14648 0 7.01367ZM1.50293 7.01367C1.50293 3.97266 3.97266 1.50293 7.01367 1.50293C10.0547 1.50293 12.5244 3.97266 12.5244 7.01367C12.5244 10.0547 10.0547 12.5244 7.01367 12.5244C3.97266 12.5244 1.50293 10.0547 1.50293 7.01367Z" />
        </svg>
        <div className="ml-6">
          Searching for <InlineSearchTerm term={pattern} /> on{" "}
          {cursorToPage(cursor, browserToolResult)}
        </div>
      </div>
    );
  }

  return (
    <div>
      <code>name: {toolCall.function.name}</code>
      <pre>
        <code>args: {toolCall.function.arguments}</code>
      </pre>
    </div>
  );
}

function ToolCallDisplay({
  toolCall,
  browserToolResult,
}: {
  toolCall: ToolCall;
  browserToolResult?: BrowserToolResult;
}) {
  const [isCollapsed, setIsCollapsed] = React.useState(true);

  // frontend tool call display for web_search
  if (toolCall.function.name === "web_search") {
    let args: Record<string, unknown> | null = null;
    try {
      args = JSON.parse(toolCall.function.arguments) as Record<string, unknown>;
    } catch (e) {
      args = null;
    }
    const query = args && typeof args.query === "string" ? args.query : "";
    return (
      <div className="text-neutral-600 dark:text-neutral-400 relative select-text">
        {/*  Magnifying Glass Icon */}
        <svg
          className="fill-current h-4 absolute top-1"
          viewBox="0 0 18 18"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path d="M0 7.01367C0 10.8809 3.14648 14.0273 7.01367 14.0273C8.54297 14.0273 9.94043 13.5352 11.0918 12.709L15.416 17.042C15.6182 17.2441 15.8818 17.3408 16.1631 17.3408C16.7607 17.3408 17.1738 16.8926 17.1738 16.3037C17.1738 16.0225 17.0684 15.7676 16.8838 15.583L12.5859 11.2588C13.4912 10.0811 14.0273 8.61328 14.0273 7.01367C14.0273 3.14648 10.8809 0 7.01367 0C3.14648 0 0 3.14648 0 7.01367ZM1.50293 7.01367C1.50293 3.97266 3.97266 1.50293 7.01367 1.50293C10.0547 1.50293 12.5244 3.97266 12.5244 7.01367C12.5244 10.0547 10.0547 12.5244 7.01367 12.5244C3.97266 12.5244 1.50293 10.0547 1.50293 7.01367Z" />
        </svg>
        <div className="ml-6">
          Searching for <InlineSearchTerm term={query} />
          &#8230;
        </div>
      </div>
    );
  }

  if (toolCall.function.name === "web_fetch") {
    let args: Record<string, unknown> | null = null;
    try {
      args = JSON.parse(toolCall.function.arguments) as Record<string, unknown>;
    } catch (e) {
      args = null;
    }
    const url = args && typeof args.url === "string" ? args.url : "";
    return (
      <div className="text-neutral-600 dark:text-neutral-400 relative select-text">
        {/* Magnifying Glass Icon */}
        <svg
          className="fill-current h-4 absolute top-1"
          viewBox="0 0 18 18"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path d="M0 7.01367C0 10.8809 3.14648 14.0273 7.01367 14.0273C8.54297 14.0273 9.94043 13.5352 11.0918 12.709L15.416 17.042C15.6182 17.2441 15.8818 17.3408 16.1631 17.3408C16.7607 17.3408 17.1738 16.8926 17.1738 16.3037C17.1738 16.0225 17.0684 15.7676 16.8838 15.583L12.5859 11.2588C13.4912 10.0811 14.0273 8.61328 14.0273 7.01367C14.0273 3.14648 10.8809 0 7.01367 0C3.14648 0 0 3.14648 0 7.01367ZM1.50293 7.01367C1.50293 3.97266 3.97266 1.50293 7.01367 1.50293C10.0547 1.50293 12.5244 3.97266 12.5244 7.01367C12.5244 10.0547 10.0547 12.5244 7.01367 12.5244C3.97266 12.5244 1.50293 10.0547 1.50293 7.01367Z" />
        </svg>
        <div className="ml-6">
          Fetching for <InlineSearchTerm term={url} />
          &#8230;
        </div>
      </div>
    );
  }

  if (!toolCall.function.name.startsWith("browser.")) {
    let preview = "";
    // preview from the tool's JSON arguments.
    try {
      const argsObj = JSON.parse(toolCall.function.arguments) as Record<
        string,
        unknown
      >;
      const preferredKey = [
        "query",
        "url",
        "pattern",
        "id",
        "file",
        "path",
      ].find((k) => Object.prototype.hasOwnProperty.call(argsObj, k));
      if (preferredKey && typeof (argsObj as any)[preferredKey] === "string") {
        preview = String((argsObj as any)[preferredKey]);
      }
    } catch (err) {
      console.error(
        "Failed to parse toolCall.function.arguments in Message.tsx:",
        err,
      );
    }

    return (
      <div className="text-neutral-600 dark:text-neutral-400 relative select-text">
        <svg
          className="h-4 w-4 absolute top-1.5"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
        </svg>
        <div className="ml-6">
          Calling <span className="font-mono">{toolCall.function.name}</span>
          {preview ? (
            <>
              : <InlineSearchTerm term={preview} />
            </>
          ) : null}
          &#8230;
        </div>
      </div>
    );
  }

  if (toolCall.function.name.startsWith("browser.")) {
    return (
      <BrowserToolCallDisplay
        toolCall={toolCall}
        browserToolResult={browserToolResult}
      />
    );
  }

  let parsedArgs = null;
  try {
    parsedArgs = JSON.parse(toolCall.function.arguments);
  } catch {
    parsedArgs = toolCall.function.arguments;
  }

  // Create a compact preview of arguments as a string
  const getArgsPreview = () => {
    if (!parsedArgs || typeof parsedArgs !== "object") {
      return parsedArgs ? String(parsedArgs) : "";
    }

    const argPairs = Object.entries(parsedArgs)
      .map(([key, value]) => {
        let displayValue;
        if (typeof value === "string") {
          displayValue = `"${value}"`;
        } else if (typeof value === "object") {
          displayValue = JSON.stringify(value);
        } else {
          displayValue = String(value);
        }
        return `${key}=${displayValue}`;
      })
      .join(", ");

    return argPairs;
  };

  return (
    <div
      className={`flex flex-col w-full ${!isCollapsed ? "text-neutral-800 dark:text-neutral-200" : "text-neutral-600 dark:text-neutral-400"}
         hover:text-neutral-800
        dark:hover:text-neutral-200 transition-colors`}
    >
      <div
        className="flex items-center cursor-pointer group/tool self-start relative"
        onClick={() => setIsCollapsed(!isCollapsed)}
      >
        {/* Tool icon */}
        <svg
          className={`w-3 absolute left-0 top-1/2 -translate-y-1/2 transition-opacity ${
            isCollapsed ? "opacity-100" : "opacity-0"
          } group-hover/tool:opacity-0 fill-current will-change-opacity`}
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
        </svg>
        {/* Arrow */}
        <svg
          className={`h-4 w-4 absolute transition-all ${
            isCollapsed
              ? "-rotate-90 opacity-0 group-hover/tool:opacity-100"
              : "rotate-0 opacity-100"
          } will-change-[opacity,transform]`}
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <polyline points="6 9 12 15 18 9"></polyline>
        </svg>

        <h3 className="ml-6 font-mono text-sm">
          <span className="font-semibold">{toolCall.function.name}</span>
          {isCollapsed && parsedArgs && (
            <span className="text-neutral-500 dark:text-neutral-500 ml-1">
              ({getArgsPreview()})
            </span>
          )}
          <span className="text-neutral-500 dark:text-neutral-500 ml-2 text-xs">
            {toolCall.type}
          </span>
        </h3>
      </div>
      <div
        className={`text-xs text-neutral-500 dark:text-neutral-500 rounded-md
          transition-[max-height,opacity] duration-300 ease-in-out ml-6 mt-2`}
        style={{
          maxHeight: isCollapsed ? "0px" : "40rem",
          opacity: isCollapsed ? 0 : 1,
        }}
      >
        <div className="transition-transform duration-300 opacity-75">
          {parsedArgs && (
            <div className="mb-4">
              <div className="text-xs font-semibold text-neutral-600 dark:text-neutral-400 mb-1">
                Arguments:
              </div>
              <pre className="text-xs bg-neutral-100 dark:bg-neutral-800 p-2 rounded overflow-x-auto">
                <code className="text-neutral-800 dark:text-neutral-200">
                  {typeof parsedArgs === "object"
                    ? JSON.stringify(parsedArgs, null, 2)
                    : parsedArgs}
                </code>
              </pre>
            </div>
          )}

          {toolCall.function.result && (
            <div>
              <div className="text-xs font-semibold text-neutral-600 dark:text-neutral-400 mb-1">
                Result:
              </div>
              <pre className="text-xs bg-neutral-100 dark:bg-neutral-800 p-2 rounded overflow-x-auto max-h-40">
                <code className="text-neutral-800 dark:text-neutral-200">
                  {typeof toolCall.function.result === "object"
                    ? JSON.stringify(toolCall.function.result, null, 2)
                    : toolCall.function.result}
                </code>
              </pre>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function UserMessage({
  message,
  onEditMessage,
  messageIndex,
  isFaded,
}: {
  message: MessageType;
  onEditMessage?: (content: string, index: number) => void;
  messageIndex?: number;
  isFaded?: boolean;
}) {
  const handleEdit = () => {
    if (onEditMessage && messageIndex !== undefined) {
      onEditMessage(message.content, messageIndex);
    }
  };

  return (
    <div
      className={`flex flex-col transition-opacity duration-300 ${isFaded ? "opacity-50" : "opacity-100"}`}
    >
      {/* Show image attachments above the message background */}
      {message.attachments && message.attachments.length > 0 && (
        <div className="flex gap-2 mb-2 overflow-x-auto justify-end max-w-md self-end">
          {message.attachments
            .filter((attachment: File) => isImageFile(attachment.filename))
            .map((attachment: File, index: number) => (
              <div key={`image-attachment-${index}`} className="flex-shrink-0">
                <ImageThumbnail
                  image={attachment}
                  className="w-16 h-16 object-cover rounded-md"
                />
              </div>
            ))}
        </div>
      )}

      <div className="message-container mb-8 max-w-md self-end">
        <div
          className="message rounded-3xl bg-neutral-100 px-4 py-2 leading-normal
                    dark:bg-neutral-700 dark:text-white group/message relative"
          data-role="user"
        >
          {/* Show non-image attachments inside the message */}
          {message.attachments &&
            message.attachments.some(
              (attachment: File) => !isImageFile(attachment.filename),
            ) && (
              <div className="flex gap-2 mb-2 overflow-x-auto">
                {message.attachments
                  .filter(
                    (attachment: File) => !isImageFile(attachment.filename),
                  )
                  .map((attachment: File, index: number) => (
                    <div
                      key={`file-attachment-${index}`}
                      className="flex items-center gap-2 py-1 px-2 rounded-lg bg-neutral-50 dark:bg-neutral-600/50 transition-colors flex-shrink-0"
                    >
                      <svg
                        className="w-3 h-3 text-neutral-400 dark:text-neutral-500 flex-shrink-0"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={1.5}
                          d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                        />
                      </svg>
                      <span className="text-xs text-neutral-600 dark:text-neutral-400 max-w-[120px] truncate">
                        {attachment.filename}
                      </span>
                    </div>
                  ))}
              </div>
            )}

          <div className="message-content whitespace-pre-line break-words">
            {message.content}
          </div>

          {/* Edit button */}
          <button
            type="button"
            className={`edit-button absolute -bottom-5 right-1 text-xs
                     ${
                       isFaded
                         ? "opacity-30"
                         : "opacity-0 group-hover/message:opacity-100 text-neutral-500 hover:text-neutral-700 dark:text-neutral-400 dark:hover:text-neutral-200 cursor-pointer"
                     }`}
            onClick={isFaded ? undefined : handleEdit}
          >
            edit
          </button>
        </div>
      </div>
    </div>
  );
}

function OtherRoleMessage({
  message,
  isStreaming,
  isFaded,
  browserToolResult,
  lastToolQuery,
}: {
  message: MessageType;
  previousMessage?: MessageType;
  isStreaming: boolean;
  isFaded?: boolean;
  // TODO(drifkin): this type isn't right
  browserToolResult?: BrowserToolResult;
  lastToolQuery?: string;
}) {
  const messageRef = useRef<HTMLDivElement>(null);

  return (
    <div
      className={`flex mb-8 flex-col transition-opacity duration-300 space-y-4 ${isFaded ? "opacity-50" : "opacity-100"}`}
    >
      <div className="flex-1 flex flex-col justify-start relative group max-w-none text-wrap break-words">
        {/* Thinking area */}
        {message.thinking && (
          <Thinking
            thinking={message.thinking}
            startTime={message.thinkingTimeStart}
            endTime={message.thinkingTimeEnd}
          />
        )}

        {/* Only render content div if there's actual content to show */}
        {(() => {
          // Skip rendering content div for tool messages with structured tool_calls
          if (
            message.role === "tool" &&
            message.tool_calls &&
            message.tool_calls.length > 0
          ) {
            return null;
          }

          if (
            message.role !== "tool" &&
            (!message.content || !message.content.trim())
          ) {
            return null;
          }

          // Render appropriate content
          return (
            <div
              className="max-w-full prose dark:prose-invert assistant-message-content break-words"
              id="message-container"
              ref={messageRef}
            >
              {message.role === "tool" ? (
                <ToolRoleContent
                  message={message}
                  browserToolResult={browserToolResult}
                  lastToolQuery={lastToolQuery}
                />
              ) : (
                <StreamingMarkdownContent
                  content={message.content}
                  isStreaming={isStreaming}
                  browserToolResult={browserToolResult as BrowserToolResult}
                />
              )}
            </div>
          );
        })()}
      </div>

      {message.tool_calls && message.tool_calls.length > 0 && (
        <div>
          {message.tool_calls.map((toolCall: ToolCall, index: number) => (
            <ToolCallDisplay
              key={index}
              toolCall={toolCall}
              browserToolResult={browserToolResult}
            />
          ))}
        </div>
      )}

      {message.tool_call && (
        <ToolCallDisplay
          toolCall={message.tool_call}
          browserToolResult={browserToolResult}
        />
      )}

      {!isStreaming &&
        message.role === "assistant" &&
        message.content &&
        message.content.trim() &&
        (!message.tool_calls || message.tool_calls.length === 0) &&
        !message.tool_call && (
          <div className="-ml-1">
            <CopyButton
              content={message.content || ""}
              copyRef={messageRef as React.RefObject<HTMLElement>}
              removeClasses={["copy-button"]}
              size="md"
              showLabels={false}
              className="copy-button z-10 text-neutral-500 dark:text-neutral-400"
              title="Copy"
            />
          </div>
        )}
    </div>
  );
}
