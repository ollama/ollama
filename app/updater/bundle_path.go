package updater

import (
	"fmt"
	"path/filepath"
	"strings"
)

// updateArchiveRoot is the only directory a macOS update zip may contain.
// Release zips are made with `ditto -c -k --norsrc --keepParent Ollama.app`,
// so every entry is Ollama.app or a file inside it.
const updateArchiveRoot = "Ollama.app"

type bundleEntryScope int

const (
	bundleEntryRelative bundleEntryScope = iota
	bundleEntryWithArchiveRoot
)

// bundleEntryPath maps a zip entry onto root. With bundleEntryWithArchiveRoot
// the entry must be Ollama.app itself or a path inside it. That is the check
// verifyDownload uses before it will extract a download. Paths that escape
// the archive are rejected.
func bundleEntryPath(root, name string, scope bundleEntryScope) (string, error) {
	cleanName := filepath.Clean(filepath.FromSlash(name))
	if !filepath.IsLocal(cleanName) {
		return "", fmt.Errorf("bundle contains invalid path: %s", name)
	}
	if scope == bundleEntryWithArchiveRoot && cleanName != updateArchiveRoot &&
		!strings.HasPrefix(cleanName, updateArchiveRoot+string(osPathSeparator())) {
		return "", fmt.Errorf("bundle contains invalid path: %s", name)
	}
	return filepath.Join(root, cleanName), nil
}

func osPathSeparator() string {
	return string(filepath.Separator)
}
