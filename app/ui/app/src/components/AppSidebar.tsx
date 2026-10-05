import { Link } from "@/components/ui/link";
import { ChatIcon } from "@/components/ChatIcon";
import { Cog6ToothIcon, RectangleGroupIcon } from "@heroicons/react/24/outline";

type AppSection = "apps" | "chat" | "settings";

export function AppNavigation({ current }: { current: AppSection }) {
  const itemClass = (section: AppSection) =>
    `flex w-full items-center gap-3 rounded-lg px-2 py-2 text-left text-sm text-neutral-700 hover:bg-neutral-100 dark:text-neutral-100 dark:hover:bg-neutral-800 ${
      current === section ? "bg-neutral-100 dark:bg-neutral-800" : ""
    }`;

  return (
    <div className="flex flex-col gap-0.5">
      <Link to="/connect" className={itemClass("apps")} draggable={false}>
        <RectangleGroupIcon className="h-5 w-5 stroke-current" />
        <span className="truncate">Apps</span>
      </Link>
      <Link
        to="/c/$chatId"
        params={{ chatId: "new" }}
        mask={{ to: "/" }}
        className={itemClass("chat")}
        draggable={false}
      >
        <ChatIcon />
        <span className="truncate">Chat</span>
      </Link>
      <Link to="/settings" className={itemClass("settings")} draggable={false}>
        <Cog6ToothIcon className="h-5 w-5 stroke-current" />
        <span className="truncate">Settings</span>
      </Link>
    </div>
  );
}

export function AppSidebar({ current }: { current: AppSection }) {
  return (
    <nav className="flex flex-1 flex-col px-4 pb-4 select-none">
      <AppNavigation current={current} />
    </nav>
  );
}
