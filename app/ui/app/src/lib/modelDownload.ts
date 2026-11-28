import { Model } from "@/gotypes";

export type PullEvent = {
  status: string;
  digest?: string;
  total?: number;
  completed?: number;
  done?: boolean;
};

export type DownloadStatus =
  | "idle"
  | "starting"
  | "downloading"
  | "completed"
  | "error";

export interface DownloadState {
  status: DownloadStatus;
  statusText?: string;
  completed?: number;
  total?: number;
  error?: string;
}

export const IDLE_DOWNLOAD: DownloadState = { status: "idle" };

export function isInstalled(model: Model): boolean {
  return model.digest !== undefined && model.digest !== "";
}

export function applyPullEvent(
  previous: DownloadState,
  event: PullEvent,
): DownloadState {
  if (event.done || event.status === "success") {
    return { status: "completed" };
  }

  return {
    status: "downloading",
    statusText: event.status,
    completed: event.completed ?? previous.completed,
    total: event.total ?? previous.total,
  };
}

const K = 1024;
const SIZES = ["B", "KB", "MB", "GB", "TB"];

export function formatBytes(bytes: number, unit?: string): string {
  let i: number;
  if (unit) {
    i = SIZES.indexOf(unit);
    if (i < 0) i = 0;
  } else {
    i = bytes === 0 ? 0 : Math.floor(Math.log(bytes) / Math.log(K));
  }

  const decimals = SIZES[i] === "GB" || SIZES[i] === "TB" ? 1 : 0;
  return `${(bytes / Math.pow(K, i)).toFixed(decimals)} ${SIZES[i]}`;
}

// Progress of the layer currently being pulled, not of the model as a whole:
// the pull stream reports total/completed per digest, so the percentage must
// never be presented as overall download progress.
export function layerProgressText(state: DownloadState): string | undefined {
  const { completed, total } = state;
  if (completed === undefined || total === undefined || total <= 0) {
    return undefined;
  }

  const unitIndex = Math.max(
    0,
    Math.min(SIZES.length - 1, Math.floor(Math.log(total) / Math.log(K))),
  );
  const unit = SIZES[unitIndex];
  const percentage = Math.floor((completed / total) * 100);
  return `${formatBytes(completed, unit)} / ${formatBytes(total, unit)} (${percentage}%)`;
}
