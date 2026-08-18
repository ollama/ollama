import { useMemo, useState } from "react";
import {
  ChevronRightIcon,
  ChevronDownIcon,
  FolderIcon,
  DocumentIcon,
  ArrowPathIcon,
  XMarkIcon,
} from "@heroicons/react/24/outline";
import { useProject, useProjectFiles } from "@/hooks/useProject";
import { ProjectFile } from "@/gotypes";

interface TreeNode {
  name: string;
  path: string;
  isDir: boolean;
  children: TreeNode[];
}

function buildTree(files: ProjectFile[]): TreeNode[] {
  const root: TreeNode = { name: "", path: "", isDir: true, children: [] };
  const nodes = new Map<string, TreeNode>([["", root]]);

  // Files come sorted by path, so parents always appear before children
  for (const file of files) {
    const idx = file.path.lastIndexOf("/");
    const parentPath = idx === -1 ? "" : file.path.slice(0, idx);
    const name = idx === -1 ? file.path : file.path.slice(idx + 1);

    const node: TreeNode = {
      name,
      path: file.path,
      isDir: file.isDir,
      children: [],
    };

    // Parent may be missing if its dir entry was filtered out; skip orphans
    const parent = nodes.get(parentPath);
    if (!parent) continue;

    parent.children.push(node);
    if (file.isDir) {
      nodes.set(file.path, node);
    }
  }

  const sortChildren = (node: TreeNode) => {
    node.children.sort((a, b) => {
      if (a.isDir !== b.isDir) return a.isDir ? -1 : 1;
      return a.name.localeCompare(b.name);
    });
    node.children.forEach(sortChildren);
  };
  sortChildren(root);

  return root.children;
}

function TreeEntry({
  node,
  depth,
  expanded,
  onToggle,
  onFileClick,
}: {
  node: TreeNode;
  depth: number;
  expanded: Set<string>;
  onToggle: (path: string) => void;
  onFileClick: (path: string) => void;
}) {
  const isExpanded = expanded.has(node.path);

  return (
    <>
      <button
        type="button"
        onClick={() =>
          node.isDir ? onToggle(node.path) : onFileClick(node.path)
        }
        title={node.path}
        className="flex w-full items-center gap-1 rounded-md px-2 py-1 text-left text-sm text-neutral-700 hover:bg-neutral-200/60 dark:text-neutral-300 dark:hover:bg-neutral-700/60 cursor-pointer"
        style={{ paddingLeft: `${depth * 12 + 8}px` }}
      >
        {node.isDir ? (
          <>
            {isExpanded ? (
              <ChevronDownIcon className="h-3 w-3 flex-shrink-0 text-neutral-400" />
            ) : (
              <ChevronRightIcon className="h-3 w-3 flex-shrink-0 text-neutral-400" />
            )}
            <FolderIcon className="h-4 w-4 flex-shrink-0 text-neutral-400" />
          </>
        ) : (
          <DocumentIcon className="ml-4 h-4 w-4 flex-shrink-0 text-neutral-400" />
        )}
        <span className="truncate">{node.name}</span>
      </button>
      {node.isDir &&
        isExpanded &&
        node.children.map((child) => (
          <TreeEntry
            key={child.path}
            node={child}
            depth={depth + 1}
            expanded={expanded}
            onToggle={onToggle}
            onFileClick={onFileClick}
          />
        ))}
    </>
  );
}

export function ProjectPanel() {
  const { project, closeProject } = useProject();
  const { files, truncated, isLoading, refresh } = useProjectFiles();
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [isRefreshing, setIsRefreshing] = useState(false);

  const tree = useMemo(() => buildTree(files), [files]);

  if (!project) return null;

  const handleToggle = (path: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  };

  const handleFileClick = (path: string) => {
    window.dispatchEvent(
      new CustomEvent("project:mention-file", { detail: { path } }),
    );
  };

  const handleRefresh = async () => {
    setIsRefreshing(true);
    try {
      await refresh();
    } finally {
      setIsRefreshing(false);
    }
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex items-center justify-between px-3 pb-1">
        <span
          className="truncate text-xs font-semibold uppercase tracking-wide text-neutral-500 dark:text-neutral-400"
          title={project.root}
        >
          {project.name}
        </span>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={handleRefresh}
            title="Refresh file list"
            className="rounded-md p-1 text-neutral-400 hover:bg-neutral-200/60 hover:text-neutral-600 dark:hover:bg-neutral-700/60 dark:hover:text-neutral-300 cursor-pointer"
          >
            <ArrowPathIcon
              className={`h-3.5 w-3.5 ${isRefreshing ? "animate-spin" : ""}`}
            />
          </button>
          <button
            type="button"
            onClick={() => closeProject()}
            title="Close project"
            className="rounded-md p-1 text-neutral-400 hover:bg-neutral-200/60 hover:text-neutral-600 dark:hover:bg-neutral-700/60 dark:hover:text-neutral-300 cursor-pointer"
          >
            <XMarkIcon className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-1 pb-2">
        {isLoading ? (
          <p className="px-3 py-2 text-sm text-neutral-400">Loading files…</p>
        ) : (
          <>
            {tree.map((node) => (
              <TreeEntry
                key={node.path}
                node={node}
                depth={0}
                expanded={expanded}
                onToggle={handleToggle}
                onFileClick={handleFileClick}
              />
            ))}
            {truncated && (
              <p className="px-3 py-2 text-xs text-neutral-400">
                File list truncated — project is very large.
              </p>
            )}
          </>
        )}
      </div>
    </div>
  );
}
