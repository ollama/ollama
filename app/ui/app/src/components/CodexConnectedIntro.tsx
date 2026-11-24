import {
  Dialog,
  DialogPanel,
  DialogTitle,
  Description,
} from "@headlessui/react";

export function CodexConnectedIntro({ onDone }: { onDone: () => void }) {
  return (
    <Dialog open onClose={() => {}} className="relative z-50">
      <div
        className="claude-connected-backdrop fixed inset-0 bg-black/20 dark:bg-black/50"
        aria-hidden="true"
      />
      <div className="fixed inset-0 flex items-center justify-center overflow-y-auto p-6">
        <DialogPanel className="claude-connected-dialog relative max-h-full w-full max-w-md overflow-y-auto rounded-2xl bg-white font-sans shadow-2xl ring-1 ring-black/10 dark:bg-neutral-800 dark:ring-white/10">
          <img
            src="/chatgpt-connected.png"
            alt="Ollama models alongside OpenAI models in the ChatGPT Codex model picker"
            width={1172}
            height={1084}
            className="h-auto w-full object-contain"
            draggable={false}
          />
          <div className="p-6">
            <DialogTitle className="font-rounded text-lg font-medium leading-6 text-neutral-950 dark:text-neutral-100">
              Use Ollama models in ChatGPT
            </DialogTitle>
            <Description className="mt-2 text-[13px] leading-5 text-neutral-500 dark:text-neutral-400">
              Click Continue to open ChatGPT. In Codex mode, choose an Ollama
              model from the model picker for your task.
            </Description>
            <div className="mt-5 flex justify-end">
              <button
                type="button"
                data-autofocus
                onClick={onDone}
                className="rounded-full bg-neutral-100 px-6 py-2 text-sm font-normal text-neutral-950 transition-colors hover:bg-neutral-200 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-500 dark:bg-white dark:hover:bg-neutral-100"
              >
                Continue
              </button>
            </div>
          </div>
        </DialogPanel>
      </div>
    </Dialog>
  );
}
