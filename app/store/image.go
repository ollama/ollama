//go:build windows || darwin

package store

import (
	"path/filepath"
)

// ImgDir returns the directory for cached chat images, which DeleteChat cleans up.
func (s *Store) ImgDir() string {
	dbPath := s.DBPath
	if dbPath == "" {
		dbPath = defaultDBPath
	}
	storeDir := filepath.Dir(dbPath)
	return filepath.Join(storeDir, "cache", "images")
}
