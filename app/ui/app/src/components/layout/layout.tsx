import { Link } from "@tanstack/react-router";
import { ChatIcon } from "@/components/ChatIcon";
import { isWindowsPlatform } from "@/lib/platform";
import { useState } from "react";

let sessionSidebarOpen = false;

const DEFAULT_SIDEBAR_WIDTH = 192;
const MIN_SIDEBAR_WIDTH = 160;
const MAX_SIDEBAR_WIDTH = 480;
const SIDEBAR_KEYBOARD_STEP = 16;

let sessionSidebarWidth = DEFAULT_SIDEBAR_WIDTH;

function clampSidebarWidth(width: number) {
  return Math.min(MAX_SIDEBAR_WIDTH, Math.max(MIN_SIDEBAR_WIDTH, width));
}

export function SidebarLayout({
  sidebar,
  title,
  children,
}: React.PropsWithChildren<{
  sidebar: React.ReactNode;
  title?: string;
}>) {
  const [sidebarOpen, setSidebarOpen] = useState(sessionSidebarOpen);
  const [sidebarWidth, setSidebarWidth] = useState(sessionSidebarWidth);
  const [isResizingSidebar, setIsResizingSidebar] = useState(false);
  const isWindows = isWindowsPlatform();

  const toggleSidebar = () => {
    sessionSidebarOpen = !sidebarOpen;
    setSidebarOpen(sessionSidebarOpen);
  };

  const updateSidebarWidth = (width: number) => {
    sessionSidebarWidth = clampSidebarWidth(width);
    setSidebarWidth(sessionSidebarWidth);
  };

  const handleResizePointerDown = (
    event: React.PointerEvent<HTMLDivElement>,
  ) => {
    if (event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    setIsResizingSidebar(true);
  };

  const handleResizePointerMove = (
    event: React.PointerEvent<HTMLDivElement>,
  ) => {
    if (!isResizingSidebar) return;
    updateSidebarWidth(event.clientX);
  };

  const stopResizingSidebar = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    setIsResizingSidebar(false);
  };

  const handleResizeKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    let nextWidth: number | undefined;
    switch (event.key) {
      case "ArrowLeft":
        nextWidth = sidebarWidth - SIDEBAR_KEYBOARD_STEP;
        break;
      case "ArrowRight":
        nextWidth = sidebarWidth + SIDEBAR_KEYBOARD_STEP;
        break;
      case "Home":
        nextWidth = MIN_SIDEBAR_WIDTH;
        break;
      case "End":
        nextWidth = MAX_SIDEBAR_WIDTH;
        break;
    }
    if (nextWidth === undefined) return;
    event.preventDefault();
    updateSidebarWidth(nextWidth);
  };

  const sidebarControlsLeft = isWindows
    ? 8
    : sidebarOpen
      ? sidebarWidth - 52
      : 80;

  return (
    <div
      className={`flex h-screen w-full overflow-hidden dark:bg-neutral-900 ${isResizingSidebar ? "cursor-col-resize select-none" : ""}`}
    >
      <div
        className={`absolute flex mx-2 py-2 z-20 items-center text-neutral-500 dark:text-neutral-400 ${isResizingSidebar ? "" : "transition-[left] duration-375"}`}
        style={{ left: sidebarControlsLeft }}
      >
        <button
          onClick={toggleSidebar}
          onMouseDown={(e) => {
            e.stopPropagation();
          }}
          className="h-9 w-9 flex items-center justify-center rounded-full hover:bg-neutral-100 dark:hover:bg-neutral-700/75 cursor-pointer"
          aria-label={sidebarOpen ? "Hide sidebar" : "Show sidebar"}
          title={sidebarOpen ? "Hide sidebar" : "Show sidebar"}
        >
          <svg
            className="h-5 w-5 fill-current"
            viewBox="0 0 24 19"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path d="M7.76132 16.6344H9.58103V1.59842H7.76132V16.6344ZM4.20898 18.2316H19.124C21.6518 18.2316 23.1293 16.6963 23.1293 14.0209V4.2205C23.1293 1.54512 21.6518 0.00351715 19.124 0.00351715H4.20898C1.54336 0.00351715 0 1.54512 0 4.2205V14.0209C0 16.6963 1.54336 18.2316 4.20898 18.2316ZM4.31191 16.3184C2.79628 16.3184 1.91327 15.4434 1.91327 13.926V4.31542C1.91327 2.79979 2.79628 1.91678 4.31191 1.91678H18.8174C20.333 1.91678 21.216 2.79979 21.216 4.31542V13.926C21.216 15.4434 20.333 16.3184 18.8174 16.3184H4.31191ZM5.85116 5.50038C6.1951 5.50038 6.49217 5.20507 6.49217 4.87968C6.49217 4.54628 6.1951 4.25722 5.85116 4.25722H3.8412C3.49725 4.25722 3.20819 4.54628 3.20819 4.87968C3.20819 5.20507 3.49725 5.50038 3.8412 5.50038H5.85116ZM5.85116 8.1158C6.1951 8.1158 6.49217 7.82049 6.49217 7.4871C6.49217 7.1537 6.1951 6.8744 5.85116 6.8744H3.8412C3.49725 6.8744 3.20819 7.1537 3.20819 7.4871C3.20819 7.82049 3.49725 8.1158 3.8412 8.1158H5.85116ZM5.85116 10.725C6.1951 10.725 6.49217 10.4439 6.49217 10.1105C6.49217 9.77713 6.1951 9.48983 5.85116 9.48983H3.8412C3.49725 9.48983 3.20819 9.77713 3.20819 10.1105C3.20819 10.4439 3.49725 10.725 3.8412 10.725H5.85116Z" />
          </svg>
        </button>
        {!title && (
          <Link
            to="/c/$chatId"
            params={{ chatId: "new" }}
            title="New chat"
            className={`flex ml-1 items-center justify-center rounded-full transition-opacity duration-375 h-9 w-9 hover:bg-neutral-100 dark:hover:bg-neutral-700 ${
              sidebarOpen ? "opacity-0 pointer-events-none" : "opacity-100"
            }`}
          >
            <ChatIcon />
          </Link>
        )}
      </div>
      <div
        className={`relative flex max-h-screen flex-none flex-col ${isResizingSidebar ? "" : "transition-[width] duration-300"} ${
          sidebarOpen
            ? "border-r border-neutral-200 bg-neutral-50 dark:border-neutral-800 dark:bg-neutral-950/40"
            : "w-0"
        }`}
        style={sidebarOpen ? { width: sidebarWidth } : undefined}
      >
        <div
          onDoubleClick={() => window.doubleClick && window.doubleClick()}
          onMouseDown={() => window.drag && window.drag()}
          className="flex-none h-13 w-full"
        ></div>
        {sidebarOpen && sidebar}
        {sidebarOpen && (
          <div
            role="separator"
            aria-label="Resize sidebar"
            aria-orientation="vertical"
            aria-valuemin={MIN_SIDEBAR_WIDTH}
            aria-valuemax={MAX_SIDEBAR_WIDTH}
            aria-valuenow={sidebarWidth}
            aria-valuetext={`${sidebarWidth} pixels`}
            tabIndex={0}
            className="absolute inset-y-0 right-0 z-30 w-2 translate-x-full cursor-col-resize touch-none focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-neutral-400"
            onPointerDown={handleResizePointerDown}
            onPointerMove={handleResizePointerMove}
            onPointerUp={stopResizingSidebar}
            onPointerCancel={stopResizingSidebar}
            onLostPointerCapture={() => setIsResizingSidebar(false)}
            onKeyDown={handleResizeKeyDown}
          />
        )}
      </div>
      <main className="flex min-w-0 flex-1 flex-col transition-all duration-300">
        <div
          className={`h-13 z-10 flex w-full flex-none items-center bg-white dark:bg-neutral-900 ${title ? "" : isWindows ? "xl:hidden" : "xl:fixed xl:bg-transparent xl:dark:bg-transparent"}`}
          onDoubleClick={() => window.doubleClick && window.doubleClick()}
          onMouseDown={() => window.drag && window.drag()}
        >
          {title && (
            <h1
              className={`${sidebarOpen ? "pl-6" : isWindows ? "pl-16" : "pl-36"} transition-[padding-left] duration-300 font-rounded text-md font-medium dark:text-white`}
            >
              {title}
            </h1>
          )}
        </div>
        {children}
      </main>
    </div>
  );
}
