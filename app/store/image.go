//go:build windows || darwin

package store

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type Image struct {
	Filename string `json:"filename"`
	Path     string `json:"path"`
	Size     int64  `json:"size,omitempty"`
	MimeType string `json:"mime_type,omitempty"`
}

// Bytes loads image data from disk for a given ImageData reference
func (i *Image) Bytes() ([]byte, error) {
	return ImgBytes(i.Path)
}

// ImgBytes reads image data from the specified file path
func ImgBytes(path string) ([]byte, error) {
	if path == "" {
		return nil, fmt.Errorf("empty image path")
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read image file %s: %w", path, err)
	}

	return data, nil
}

// ImgDir returns the directory path for storing images for a specific chat
func (s *Store) ImgDir() string {
	dbPath := s.DBPath
	if dbPath == "" {
		dbPath = defaultDBPath
	}
	storeDir := filepath.Dir(dbPath)
	return filepath.Join(storeDir, "cache", "images")
}

// ImgToFile saves image data to disk and returns ImageData reference
func (s *Store) ImgToFile(chatID string, imageBytes []byte, filename, mimeType string) (Image, error) {
	baseImageDir := s.ImgDir()
	if err := os.MkdirAll(baseImageDir, 0o755); err != nil {
		return Image{}, fmt.Errorf("create base image directory: %w", err)
	}

	// Root prevents path traversal issues
	root, err := os.OpenRoot(baseImageDir)
	if err != nil {
		return Image{}, fmt.Errorf("open image root directory: %w", err)
	}
	defer root.Close()

	// Create chat-specific subdirectory within the root
	chatDir := sanitize(chatID)
	if err := root.Mkdir(chatDir, 0o755); err != nil && !os.IsExist(err) {
		return Image{}, fmt.Errorf("create chat directory: %w", err)
	}

	// Generate a unique filename to avoid conflicts
	// Use hash of content + original filename for uniqueness
	hash := sha256.Sum256(imageBytes)
	hashStr := hex.EncodeToString(hash[:])[:16] // Use first 16 chars of hash

	// Extract file extension from original filename or mime type
	ext := filepath.Ext(filename)
	if ext == "" {
		switch mimeType {
		case "image/jpeg":
			ext = ".jpg"
		case "image/png":
			ext = ".png"
		case "image/webp":
			ext = ".webp"
		default:
			ext = ".img"
		}
	}

	// Create unique filename: hash + original name + extension
	baseFilename := sanitize(strings.TrimSuffix(filename, ext))
	uniqueFilename := fmt.Sprintf("%s_%s%s", hashStr, baseFilename, ext)
	relativePath := filepath.Join(chatDir, uniqueFilename)
	file, err := root.Create(relativePath)
	if err != nil {
		return Image{}, fmt.Errorf("create image file: %w", err)
	}
	defer file.Close()

	if _, err := file.Write(imageBytes); err != nil {
		return Image{}, fmt.Errorf("write image data: %w", err)
	}

	return Image{
		Filename: uniqueFilename,
		Path:     filepath.Join(baseImageDir, relativePath),
		Size:     int64(len(imageBytes)),
		MimeType: mimeType,
	}, nil
}

// sanitize removes unsafe characters from filenames
func sanitize(filename string) string {
	// Convert to safe characters only
	safe := strings.Map(func(r rune) rune {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '-' {
			return r
		}
		return '_'
	}, filename)

	// Clean up and validate
	safe = strings.Trim(safe, "_")
	if safe == "" {
		return "image"
	}
	return safe
}
