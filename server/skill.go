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

// MediaTypeSkill is the media type for skill layers in manifests.
const MediaTypeSkill = "application/vnd.ollama.image.skill"

// GetSkillsPath returns the path to the extracted skills cache directory.
// If digest is empty, returns the skills directory itself.
// If digest is provided, returns the path to the extracted skill for that digest.
func GetSkillsPath(digest string) (string, error) {
	// only accept actual sha256 digests
	pattern := "^sha256[:-][0-9a-fA-F]{64}$"
	re := regexp.MustCompile(pattern)

	if digest != "" && !re.MatchString(digest) {
		return "", ErrInvalidDigestFormat
	}

	digest = strings.ReplaceAll(digest, ":", "-")
	path := filepath.Join(envconfig.Models(), "skills", digest)
	dirPath := filepath.Dir(path)
	if digest == "" {
		dirPath = path
	}

	if err := os.MkdirAll(dirPath, 0o755); err != nil {
		return "", fmt.Errorf("%w: ensure path elements are traversable", err)
	}

	return path, nil
}

// ExtractSkillBlob extracts a skill tar.gz blob to the skills cache.
// The blob is expected to be at the blobs path for the given digest.
// Returns the path to the extracted skill directory.
func ExtractSkillBlob(digest string) (string, error) {
	// Get the blob path
	blobPath, err := GetBlobsPath(digest)
	if err != nil {
		return "", fmt.Errorf("getting blob path: %w", err)
	}

	// Get the extraction path
	skillPath, err := GetSkillsPath(digest)
	if err != nil {
		return "", fmt.Errorf("getting skill path: %w", err)
	}

	// Check if already extracted
	if _, err := os.Stat(filepath.Join(skillPath, "SKILL.md")); err == nil {
		return skillPath, nil
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

	// Create the skill directory
	if err := os.MkdirAll(skillPath, 0o755); err != nil {
		return "", fmt.Errorf("creating skill directory: %w", err)
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

		target := filepath.Join(skillPath, name)

		// Verify the target is within skillPath
		if !strings.HasPrefix(target, filepath.Clean(skillPath)+string(os.PathSeparator)) && target != filepath.Clean(skillPath) {
			return "", fmt.Errorf("path escapes skill directory: %s", header.Name)
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

	return skillPath, nil
}

// CreateSkillLayer creates a skill layer from a local directory.
// The directory must contain a SKILL.md file.
// Returns the created layer.
func CreateSkillLayer(skillDir string) (Layer, error) {
	// Verify SKILL.md exists
	skillMdPath := filepath.Join(skillDir, "SKILL.md")
	if _, err := os.Stat(skillMdPath); err != nil {
		return Layer{}, fmt.Errorf("skill directory must contain SKILL.md: %w", err)
	}

	// Create a temporary file for the tar.gz
	blobsPath, err := GetBlobsPath("")
	if err != nil {
		return Layer{}, fmt.Errorf("getting blobs path: %w", err)
	}

	tmpFile, err := os.CreateTemp(blobsPath, "skill-*.tar.gz")
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

	// Walk the skill directory and add files to tar
	err = filepath.Walk(skillDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Get relative path
		relPath, err := filepath.Rel(skillDir, path)
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
	layer, err := NewLayer(tmpFile, MediaTypeSkill)
	if err != nil {
		return Layer{}, fmt.Errorf("creating layer: %w", err)
	}

	// Extract the skill to the cache so it's ready to use
	if _, err := ExtractSkillBlob(layer.Digest); err != nil {
		return Layer{}, fmt.Errorf("extracting skill: %w", err)
	}

	return layer, nil
}

// IsLocalSkillPath checks if a skill reference looks like a local path.
// Local paths are explicitly prefixed with /, ./, ../, or ~.
// Registry references like "skill/calculator:1.0.0" should NOT be treated as local paths.
func IsLocalSkillPath(name string) bool {
	// Local paths are explicitly indicated by path prefixes
	return strings.HasPrefix(name, "/") ||
		strings.HasPrefix(name, "./") ||
		strings.HasPrefix(name, "../") ||
		strings.HasPrefix(name, "~")
}

// SkillNamespace is the namespace used for standalone skills in the registry.
const SkillNamespace = "skill"

// IsSkillReference checks if a name refers to a skill (has skill/ prefix).
func IsSkillReference(name string) bool {
	// Check for skill/ prefix (handles both "skill/foo" and "registry/skill/foo")
	name = strings.ReplaceAll(name, string(os.PathSeparator), "/")
	parts := strings.Split(name, "/")

	// skill/name or skill/name:tag
	if len(parts) >= 1 && parts[0] == SkillNamespace {
		return true
	}
	// namespace/skill/name (e.g., myuser/skill/calc) - not a skill ref
	// registry/skill/name (e.g., registry.ollama.ai/skill/calc)
	if len(parts) >= 2 && parts[1] == SkillNamespace {
		return true
	}
	return false
}

// ParseSkillName parses a skill reference string into a model.Name.
// The Kind field is set to "skill".
// Examples:
//   - "calculator" -> library/skill/calculator:latest
//   - "myname/calculator" -> myname/skill/calculator:latest
//   - "myname/skill/calculator:1.0.0" -> myname/skill/calculator:1.0.0
func ParseSkillName(name string) model.Name {
	// Use the standard parser which now handles Kind
	n := model.ParseName(name)

	// If Kind wasn't set (old format without skill/), set it
	if n.Kind == "" {
		n.Kind = SkillNamespace
	}

	return n
}

// SkillDisplayName returns a user-friendly display name for a skill.
func SkillDisplayName(n model.Name) string {
	return n.DisplayShortest()
}

// GetSkillManifestPath returns the path to the skill manifest file.
// Uses the 5-part structure: host/namespace/kind/model/tag
func GetSkillManifestPath(n model.Name) (string, error) {
	if n.Model == "" {
		return "", fmt.Errorf("skill name is required")
	}

	// Ensure Kind is set
	if n.Kind == "" {
		n.Kind = SkillNamespace
	}

	path := filepath.Join(
		envconfig.Models(),
		"manifests",
		n.Filepath(),
	)

	return path, nil
}
