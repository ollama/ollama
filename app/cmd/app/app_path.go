package main

import (
	"path/filepath"
	"strings"
)

func bundleInApplications(bundlePath, systemWidePath, homeDir string) bool {
	applicationsDirs := []string{filepath.Dir(systemWidePath)}
	if homeDir != "" {
		applicationsDirs = append(applicationsDirs, filepath.Join(homeDir, "Applications"))
	}

	for _, dir := range applicationsDirs {
		rel, err := filepath.Rel(dir, bundlePath)
		if err != nil || rel == "." || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
			continue
		}
		return true
	}
	return false
}
