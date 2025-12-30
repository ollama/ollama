package server

import (
	"archive/tar"
	"compress/gzip"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/types/model"
)

// MediaTypeMCP is the media type for MCP server layers in manifests.
const MediaTypeMCP = "application/vnd.ollama.image.mcp"

// GetMCPsPath returns the path to the extracted MCPs cache directory.
// If digest is empty, returns the mcps directory itself.
// If digest is provided, returns the path to the extracted MCP for that digest.
func GetMCPsPath(digest string) (string, error) {
	// only accept actual sha256 digests
	pattern := "^sha256[:-][0-9a-fA-F]{64}$"
	re := regexp.MustCompile(pattern)

	if digest != "" && !re.MatchString(digest) {
		return "", ErrInvalidDigestFormat
	}

	digest = strings.ReplaceAll(digest, ":", "-")
	path := filepath.Join(envconfig.Models(), "mcps", digest)
	dirPath := filepath.Dir(path)
	if digest == "" {
		dirPath = path
	}

	if err := os.MkdirAll(dirPath, 0o755); err != nil {
		return "", fmt.Errorf("%w: ensure path elements are traversable", err)
	}

	return path, nil
}

// ExtractMCPBlob extracts an MCP tar.gz blob to the mcps cache.
// The blob is expected to be at the blobs path for the given digest.
// Returns the path to the extracted MCP directory.
func ExtractMCPBlob(digest string) (string, error) {
	// Get the blob path
	blobPath, err := GetBlobsPath(digest)
	if err != nil {
		return "", fmt.Errorf("getting blob path: %w", err)
	}

	// Get the extraction path
	mcpPath, err := GetMCPsPath(digest)
	if err != nil {
		return "", fmt.Errorf("getting mcp path: %w", err)
	}

	// Check if already extracted (look for any file)
	entries, err := os.ReadDir(mcpPath)
	if err == nil && len(entries) > 0 {
		return mcpPath, nil
	}

	// Open the blob
	f, err := os.Open(blobPath)
	if err != nil {
		return "", fmt.Errorf("opening blob: %w", err)
	}
	defer f.Close()

	// Create gzip reader
	gzr, err := gzip.NewReader(f)
	if err != nil {
		return "", fmt.Errorf("creating gzip reader: %w", err)
	}
	defer gzr.Close()

	// Create tar reader
	tr := tar.NewReader(gzr)

	// Create the mcp directory
	if err := os.MkdirAll(mcpPath, 0o755); err != nil {
		return "", fmt.Errorf("creating mcp directory: %w", err)
	}

	// Extract files
	for {
		header, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", fmt.Errorf("reading tar: %w", err)
		}

		// Clean the name and ensure it doesn't escape the target directory
		name := filepath.Clean(header.Name)
		if strings.HasPrefix(name, "..") {
			return "", fmt.Errorf("invalid path in archive: %s", header.Name)
		}

		target := filepath.Join(mcpPath, name)

		// Verify the target is within mcpPath
		if !strings.HasPrefix(target, filepath.Clean(mcpPath)+string(os.PathSeparator)) && target != filepath.Clean(mcpPath) {
			return "", fmt.Errorf("path escapes mcp directory: %s", header.Name)
		}

		switch header.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0o755); err != nil {
				return "", fmt.Errorf("creating directory: %w", err)
			}
		case tar.TypeReg:
			// Ensure parent directory exists
			if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
				return "", fmt.Errorf("creating parent directory: %w", err)
			}

			outFile, err := os.OpenFile(target, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, os.FileMode(header.Mode))
			if err != nil {
				return "", fmt.Errorf("creating file: %w", err)
			}

			if _, err := io.Copy(outFile, tr); err != nil {
				outFile.Close()
				return "", fmt.Errorf("writing file: %w", err)
			}
			outFile.Close()
		}
	}

	return mcpPath, nil
}

// CreateMCPLayer creates an MCP layer from a local directory.
// The directory can optionally contain an mcp.json or package.json file.
// Returns the created layer.
func CreateMCPLayer(mcpDir string) (Layer, error) {
	// Verify directory exists
	info, err := os.Stat(mcpDir)
	if err != nil {
		return Layer{}, fmt.Errorf("mcp directory not found: %w", err)
	}
	if !info.IsDir() {
		return Layer{}, fmt.Errorf("mcp path is not a directory: %s", mcpDir)
	}

	// Create a temporary file for the tar.gz
	blobsPath, err := GetBlobsPath("")
	if err != nil {
		return Layer{}, fmt.Errorf("getting blobs path: %w", err)
	}

	tmpFile, err := os.CreateTemp(blobsPath, "mcp-*.tar.gz")
	if err != nil {
		return Layer{}, fmt.Errorf("creating temp file: %w", err)
	}
	tmpPath := tmpFile.Name()
	defer func() {
		tmpFile.Close()
		os.Remove(tmpPath)
	}()

	// Create gzip writer
	gzw := gzip.NewWriter(tmpFile)
	defer gzw.Close()

	// Create tar writer
	tw := tar.NewWriter(gzw)
	defer tw.Close()

	// Walk the mcp directory and add files to tar
	err = filepath.Walk(mcpDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Get relative path
		relPath, err := filepath.Rel(mcpDir, path)
		if err != nil {
			return err
		}

		// Skip the root directory itself
		if relPath == "." {
			return nil
		}

		// Create tar header
		header, err := tar.FileInfoHeader(info, "")
		if err != nil {
			return err
		}
		header.Name = relPath

		if err := tw.WriteHeader(header); err != nil {
			return err
		}

		// Write file contents if it's a regular file
		if !info.IsDir() {
			f, err := os.Open(path)
			if err != nil {
				return err
			}
			defer f.Close()

			if _, err := io.Copy(tw, f); err != nil {
				return err
			}
		}

		return nil
	})
	if err != nil {
		return Layer{}, fmt.Errorf("creating tar archive: %w", err)
	}

	// Close writers to flush
	if err := tw.Close(); err != nil {
		return Layer{}, fmt.Errorf("closing tar writer: %w", err)
	}
	if err := gzw.Close(); err != nil {
		return Layer{}, fmt.Errorf("closing gzip writer: %w", err)
	}
	if err := tmpFile.Close(); err != nil {
		return Layer{}, fmt.Errorf("closing temp file: %w", err)
	}

	// Open the temp file for reading
	tmpFile, err = os.Open(tmpPath)
	if err != nil {
		return Layer{}, fmt.Errorf("reopening temp file: %w", err)
	}
	defer tmpFile.Close()

	// Create the layer (this will compute the digest and move to blobs)
	layer, err := NewLayer(tmpFile, MediaTypeMCP)
	if err != nil {
		return Layer{}, fmt.Errorf("creating layer: %w", err)
	}

	// Extract the mcp to the cache so it's ready to use
	if _, err := ExtractMCPBlob(layer.Digest); err != nil {
		return Layer{}, fmt.Errorf("extracting mcp: %w", err)
	}

	return layer, nil
}

// IsLocalMCPPath checks if an MCP reference looks like a local path.
// Local paths are explicitly prefixed with /, ./, ../, or ~.
func IsLocalMCPPath(name string) bool {
	return strings.HasPrefix(name, "/") ||
		strings.HasPrefix(name, "./") ||
		strings.HasPrefix(name, "../") ||
		strings.HasPrefix(name, "~")
}

// MCPNamespace is the namespace used for standalone MCPs in the registry.
const MCPNamespace = "mcp"

// IsMCPReference checks if a name refers to an MCP (has mcp/ prefix).
func IsMCPReference(name string) bool {
	name = strings.ReplaceAll(name, string(os.PathSeparator), "/")
	parts := strings.Split(name, "/")

	// mcp/name or mcp/name:tag
	if len(parts) >= 1 && parts[0] == MCPNamespace {
		return true
	}
	// namespace/mcp/name (e.g., myuser/mcp/websearch)
	if len(parts) >= 2 && parts[1] == MCPNamespace {
		return true
	}
	return false
}

// ParseMCPName parses an MCP reference string into a model.Name.
// The Kind field is set to "mcp".
func ParseMCPName(name string) model.Name {
	n := model.ParseName(name)

	// If Kind wasn't set (old format without mcp/), set it
	if n.Kind == "" {
		n.Kind = MCPNamespace
	}

	return n
}

// GetMCPManifestPath returns the path to the MCP manifest file.
func GetMCPManifestPath(n model.Name) (string, error) {
	if n.Model == "" {
		return "", fmt.Errorf("mcp name is required")
	}

	// Ensure Kind is set
	if n.Kind == "" {
		n.Kind = MCPNamespace
	}

	path := filepath.Join(
		envconfig.Models(),
		"manifests",
		n.Filepath(),
	)

	return path, nil
}
