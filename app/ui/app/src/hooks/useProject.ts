import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useCallback, useMemo } from "react";
import {
  getProject,
  openProject,
  closeProject,
  getProjectFiles,
  getProjectFile,
} from "@/api";
import { ProjectFile } from "@/gotypes";

// Approximate context budget for @-mentioned files. Must stay in sync with
// maxMentionTotalBytes in app/ui/project.go.
export const MENTION_TOTAL_BYTES = 128 * 1024;

export function useProject() {
  const queryClient = useQueryClient();

  const { data: project, isLoading } = useQuery({
    queryKey: ["project"],
    queryFn: getProject,
  });

  const invalidate = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ["project"] });
    queryClient.invalidateQueries({ queryKey: ["projectFiles"] });
  }, [queryClient]);

  const openMutation = useMutation({
    mutationFn: openProject,
    onSuccess: invalidate,
  });

  const closeMutation = useMutation({
    mutationFn: closeProject,
    onSuccess: invalidate,
  });

  // Opens the native folder picker and activates the selected folder
  const openProjectDialog = useCallback(async () => {
    const path = await window.webview?.selectWorkingDirectory();
    if (!path) return null;
    return openMutation.mutateAsync(path);
  }, [openMutation]);

  const openProjectPath = useCallback(
    (path: string) => openMutation.mutateAsync(path),
    [openMutation],
  );

  const close = useCallback(() => closeMutation.mutateAsync(), [closeMutation]);

  const hasProject = !!project?.root;

  return useMemo(
    () => ({
      project: hasProject ? project : null,
      recentProjects: project?.recent ?? [],
      isLoading,
      openProjectDialog,
      openProjectPath,
      closeProject: close,
      isOpening: openMutation.isPending,
    }),
    [
      hasProject,
      project,
      isLoading,
      openProjectDialog,
      openProjectPath,
      close,
      openMutation.isPending,
    ],
  );
}

export function useProjectFiles() {
  const { project } = useProject();
  const queryClient = useQueryClient();

  const { data, isLoading } = useQuery({
    queryKey: ["projectFiles", project?.root],
    queryFn: () => getProjectFiles(),
    enabled: !!project?.root,
    staleTime: Infinity,
  });

  const refresh = useCallback(async () => {
    if (!project?.root) return;
    const refreshed = await getProjectFiles(true);
    queryClient.setQueryData(["projectFiles", project.root], refreshed);
  }, [project?.root, queryClient]);

  const files: ProjectFile[] = useMemo(() => data?.files ?? [], [data?.files]);

  // Set of non-directory paths for O(1) mention validation
  const filePathSet = useMemo(() => {
    const set = new Set<string>();
    for (const f of files) {
      if (!f.isDir) set.add(f.path);
    }
    return set;
  }, [files]);

  // Map of path -> size for context budget estimation
  const fileSizes = useMemo(() => {
    const map = new Map<string, number>();
    for (const f of files) {
      if (!f.isDir) map.set(f.path, f.size);
    }
    return map;
  }, [files]);

  return useMemo(
    () => ({
      files,
      filePathSet,
      fileSizes,
      truncated: data?.truncated ?? false,
      isLoading,
      refresh,
    }),
    [files, filePathSet, fileSizes, data?.truncated, isLoading, refresh],
  );
}

// useProjectFileContent loads a single file of the active project for
// previewing. Pass null to skip the request.
export function useProjectFileContent(path: string | null) {
  const { project } = useProject();

  const { data, isLoading, error } = useQuery({
    queryKey: ["projectFile", project?.root, path],
    queryFn: () => getProjectFile(path!),
    enabled: !!project?.root && !!path,
    // files change on disk, so don't serve a stale preview on reopen
    staleTime: 0,
    gcTime: 0,
    retry: false,
  });

  return useMemo(
    () => ({
      file: data ?? null,
      isLoading,
      error: error instanceof Error ? error.message : null,
    }),
    [data, isLoading, error],
  );
}
