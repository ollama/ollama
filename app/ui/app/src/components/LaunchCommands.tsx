import CopyButton from "@/components/CopyButton";
import { useSettings } from "@/hooks/useSettings";
import launchIntegrations from "@/data/launch-integrations.json";

interface LaunchCommand {
  id: string;
  name: string;
  command: string;
  description: string;
  icon?: string;
  darkIcon?: string;
  iconClassName?: string;
}
const LAUNCH_COMMANDS: LaunchCommand[] = launchIntegrations;

function LaunchIcon({ item }: { item: LaunchCommand }) {
  if (item.icon) {
    if (item.darkIcon) {
      return (
        <picture>
          <source
            srcSet={item.darkIcon}
            media="(prefers-color-scheme: dark)"
          />
          <img
            src={item.icon}
            alt=""
            className={`${item.iconClassName ?? "h-8 w-8"} rounded-sm`}
          />
        </picture>
      );
    }

    return (
      <img
        src={item.icon}
        alt=""
        className={`${item.iconClassName ?? "h-8 w-8"} rounded-sm`}
      />
    );
  }

  return null;
}

export default function LaunchCommands() {
  const isWindows = navigator.platform.toLowerCase().includes("win");
  const { setSettings } = useSettings();

  const renderCommandCard = (item: LaunchCommand) => (
    <div key={item.command} className="w-full text-left">
      <div className="flex items-start gap-4 sm:gap-5">
        <div
          aria-hidden="true"
          className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-xl overflow-hidden ${
            item.icon
              ? "border border-neutral-200 bg-white dark:border-neutral-700 dark:bg-neutral-900"
              : "border border-neutral-200 bg-neutral-100 dark:border-neutral-700 dark:bg-neutral-800"
          }`}
        >
          <LaunchIcon item={item} />
        </div>

        <div className="min-w-0 flex-1">
          <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
            {item.name}
          </span>

          <p className="mt-0.5 text-xs text-neutral-500 dark:text-neutral-400">
            {item.description}
          </p>

          <div className="mt-2 flex items-center gap-2 rounded-xl bg-neutral-50 px-3 py-2 dark:bg-neutral-800">
            <code className="min-w-0 flex-1 truncate text-xs text-neutral-600 dark:text-neutral-300">
              {item.command}
            </code>
            <CopyButton
              content={item.command}
              size="md"
              title="Copy command to clipboard"
              className="text-neutral-500 hover:bg-neutral-200/60 hover:text-neutral-700 dark:text-neutral-400 dark:hover:bg-neutral-700/70 dark:hover:text-neutral-200"
              onCopy={() => {
                setSettings({ LastHomeView: item.id }).catch(() => {});
              }}
            />
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <main className="relative flex h-screen w-full flex-col">
      <section
        className={`relative min-h-0 flex-1 overflow-y-auto overscroll-contain ${
          isWindows ? "xl:pt-4" : "xl:pt-8"
        }`}
      >
        <div className="mx-auto w-full max-w-[730px] px-4 pb-20 pt-4 sm:px-6 sm:pb-24 sm:pt-6 lg:px-8 lg:pb-28 lg:pt-8">
          <h1 className="text-xl font-semibold text-neutral-900 dark:text-neutral-100">
            Launch
          </h1>

          <p className="mt-1 text-sm text-neutral-500 dark:text-neutral-400">
            External integrations from{" "}
            <code className="rounded bg-neutral-100 px-1.5 py-0.5 text-xs dark:bg-neutral-800">
              ollama launch
            </code>
            , without the chat entry.
          </p>

          <div className="mt-6 grid gap-7">
            {LAUNCH_COMMANDS.map(renderCommandCard)}
          </div>
        </div>
      </section>
    </main>
  );
}
