//go:build !windows

package llm

func adjustPlatformLibraryPaths(paths, _ []string) []string {
	return paths
}
