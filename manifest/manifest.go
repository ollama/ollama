package manifest

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/ollama/ollama/types/model"
)

var blobFilenamePattern = regexp.MustCompile(`^sha256-[0-9a-fA-F]{64}$`)

type Manifest struct {
	SchemaVersion int     `json:"schemaVersion"`
	MediaType     string  `json:"mediaType"`
	Config        Layer   `json:"config"`
	Layers        []Layer `json:"layers"`

	filepath string
	fi       os.FileInfo
	digest   string
	name     model.Name
}

func (m *Manifest) Size() (size int64) {
	for _, layer := range append(m.Layers, m.Config) {
		size += layer.Size
	}

	return
}

func (m *Manifest) Digest() string {
	return m.digest
}

func (m *Manifest) BlobDigest() string {
	if m.digest == "" {
		return ""
	}

	return "sha256:" + m.digest
}

func (m *Manifest) FileInfo() os.FileInfo {
	return m.fi
}

// ReadConfigJSON reads and unmarshals a config layer as JSON.
func (m *Manifest) ReadConfigJSON(configPath string, v any) error {
	for _, layer := range m.Layers {
		if layer.MediaType == "application/vnd.ollama.image.json" && layer.Name == configPath {
			blobPath, err := BlobsPath(layer.Digest)
			if err != nil {
				return err
			}
			data, err := os.ReadFile(blobPath)
			if err != nil {
				return err
			}
			return json.Unmarshal(data, v)
		}
	}
	return fmt.Errorf("config %q not found in manifest", configPath)
}

func (m *Manifest) Remove() error {
	return removeNamedManifestPaths(m.name)
}

func (m *Manifest) RemoveLayers() error {
	ms, err := Manifests(true)
	if err != nil {
		return err
	}

	// Build set of digests still in use by other manifests
	inUse := make(map[string]struct{})
	for _, other := range ms {
		if other.BlobDigest() != "" {
			inUse[other.BlobDigest()] = struct{}{}
		}
		for _, layer := range append(other.Layers, other.Config) {
			if layer.Digest != "" {
				inUse[layer.Digest] = struct{}{}
			}
		}
	}

	digests := make([]string, 0, len(m.Layers)+2)
	digests = append(digests, m.BlobDigest())
	for _, layer := range m.Layers {
		digests = append(digests, layer.Digest)
	}
	digests = append(digests, m.Config.Digest)

	// Remove manifest and layer blobs not used by any other manifest
	for _, digest := range digests {
		if digest == "" {
			continue
		}
		if _, used := inUse[digest]; used {
			continue
		}
		blob, err := BlobsPath(digest)
		if err != nil {
			return err
		}
		if err := os.Remove(blob); os.IsNotExist(err) {
			slog.Debug("blob does not exist", "digest", digest)
		} else if err != nil {
			return err
		}
	}

	return nil
}

func ParseNamedManifest(n model.Name) (*Manifest, error) {
	if !n.IsFullyQualified() {
		return nil, model.Unqualified(n)
	}

	p, root, err := resolveManifestPath(n)
	if err != nil {
		return nil, err
	}

	return parseManifestFile(normalizeLogicalName(n), p, root)
}

func ReadManifestData(n model.Name) ([]byte, error) {
	if !n.IsFullyQualified() {
		return nil, model.Unqualified(n)
	}

	p, root, err := resolveManifestPath(n)
	if err != nil {
		return nil, err
	}

	f, _, err := OpenVerifiedManifest(p, root)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	return io.ReadAll(f)
}

func parseManifestFile(name model.Name, path, root string) (*Manifest, error) {
	var m Manifest
	f, digest, err := OpenVerifiedManifest(path, root)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}

	if err := json.NewDecoder(f).Decode(&m); err != nil {
		return nil, err
	}

	m.filepath = path
	m.fi = fi
	m.digest = digest
	m.name = name

	return &m, nil
}

func WriteManifest(name model.Name, config Layer, layers []Layer) error {
	m := Manifest{
		SchemaVersion: 2,
		MediaType:     "application/vnd.docker.distribution.manifest.v2+json",
		Config:        config,
		Layers:        layers,
	}

	var b bytes.Buffer
	if err := json.NewEncoder(&b).Encode(m); err != nil {
		return err
	}

	return WriteManifestData(name, b.Bytes())
}

// WriteManifestData stores raw manifest bytes as a content-addressed blob and
// updates the v2 named manifest path to reference that blob. Any legacy named
// manifest for the same model is removed after the v2 write succeeds.
func WriteManifestData(name model.Name, data []byte) error {
	if !name.IsFullyQualified() {
		return model.Unqualified(name)
	}

	digest, err := writeManifestBlob(data)
	if err != nil {
		return err
	}

	if err := LinkManifest(name, digest); err != nil {
		return err
	}

	return removeLegacyManifestPaths(name)
}

// LinkManifest updates the v2 named manifest path to reference an existing
// manifest blob. It prefers symlinks, then hardlinks, then a byte-for-byte copy
// for filesystems that do not support links.
func LinkManifest(name model.Name, digest string) error {
	if !name.IsFullyQualified() {
		return model.Unqualified(name)
	}

	manifestPath, err := V2PathForName(name)
	if err != nil {
		return err
	}
	blobPath, err := BlobsPath(digest)
	if err != nil {
		return err
	}
	if _, err := os.Stat(blobPath); err != nil {
		return err
	}
	if err := checkBlobDigest(blobPath, digest); err != nil {
		return err
	}

	if err := os.MkdirAll(filepath.Dir(manifestPath), 0o755); err != nil {
		return err
	}
	if err := os.Remove(manifestPath); err != nil && !os.IsNotExist(err) {
		return err
	}

	if rel, err := filepath.Rel(filepath.Dir(manifestPath), blobPath); err == nil {
		if err := os.Symlink(rel, manifestPath); err == nil {
			return nil
		}
	}

	if err := os.Link(blobPath, manifestPath); err == nil {
		return nil
	}

	return copyManifestFile(blobPath, manifestPath)
}

func writeManifestBlob(data []byte) (string, error) {
	sum := sha256.Sum256(data)
	digest := fmt.Sprintf("sha256:%x", sum)

	blobPath, err := BlobsPath(digest)
	if err != nil {
		return "", err
	}
	if existing, err := os.ReadFile(blobPath); err == nil && bytes.Equal(existing, data) {
		return digest, nil
	}

	blobs, err := BlobsPath("")
	if err != nil {
		return "", err
	}
	temp, err := os.CreateTemp(blobs, "sha256-")
	if err != nil {
		return "", err
	}
	tempName := temp.Name()
	defer os.Remove(tempName)

	if _, err := temp.Write(data); err != nil {
		temp.Close()
		return "", err
	}
	if err := temp.Close(); err != nil {
		return "", err
	}
	if err := os.Chmod(tempName, 0o644); err != nil {
		return "", err
	}
	if err := os.Rename(tempName, blobPath); err != nil {
		if err := os.Remove(blobPath); err != nil && !os.IsNotExist(err) {
			return "", err
		}
		if err := os.Rename(tempName, blobPath); err != nil {
			return "", err
		}
	}

	return digest, nil
}

func copyManifestFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	temp, err := os.CreateTemp(filepath.Dir(dst), ".manifest-*")
	if err != nil {
		return err
	}
	tempName := temp.Name()
	defer os.Remove(tempName)

	if _, err := io.Copy(temp, in); err != nil {
		temp.Close()
		return err
	}
	if err := temp.Close(); err != nil {
		return err
	}
	if err := os.Chmod(tempName, 0o644); err != nil {
		return err
	}

	return os.Rename(tempName, dst)
}

// OpenVerifiedManifest opens a named manifest path rooted under root. Symlinks must resolve to a
// blob whose basename is sha256-<hex> and whose bytes hash to that digest.
// Regular-file manifests are treated as legacy/copy fallback manifests and are
// opened without mutating the local store.
func OpenVerifiedManifest(path, root string) (*os.File, string, error) {
	resolvedRoot, err := filepath.EvalSymlinks(root)
	if err != nil {
		return nil, "", err
	}

	info, err := os.Lstat(path)
	if err != nil {
		return nil, "", err
	}

	target, err := evalAbs(path)
	if err != nil {
		return nil, "", err
	}

	if info.Mode()&os.ModeSymlink != 0 {
		base := filepath.Base(target)
		if !blobFilenamePattern.MatchString(base) {
			return nil, "", fmt.Errorf("manifest symlink target %q is not a sha256 blob", target)
		}

		digest := strings.ToLower(strings.TrimPrefix(base, "sha256-"))
		blobPath, err := BlobsPath("sha256:" + digest)
		if err != nil {
			return nil, "", err
		}
		if !sameFile(target, blobPath) {
			return nil, "", fmt.Errorf("manifest symlink target %q does not match blob %q", target, blobPath)
		}

		f, err := os.Open(path)
		if err != nil {
			return nil, "", err
		}
		if err := checkBlobDigestReader(f, "sha256:"+digest); err != nil {
			f.Close()
			return nil, "", err
		}
		if _, err := f.Seek(0, io.SeekStart); err != nil {
			f.Close()
			return nil, "", err
		}

		return f, digest, nil
	}

	if !pathWithin(target, resolvedRoot) {
		return nil, "", fmt.Errorf("manifest path %q resolves outside manifest directory", path)
	}

	f, err := os.Open(path)
	if err != nil {
		return nil, "", err
	}

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		f.Close()
		return nil, "", err
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		f.Close()
		return nil, "", err
	}
	digest := fmt.Sprintf("%x", h.Sum(nil))

	return f, digest, nil
}

// MigrateManifestLinks moves legacy named manifests into manifests-v2. This is currently unwired but
// will be added in the future.
func MigrateManifestLinks() (int, error) {
	manifests, err := Path()
	if err != nil {
		return 0, err
	}

	// TODO(mxyng): use something less brittle
	matches, err := filepath.Glob(filepath.Join(manifests, "*", "*", "*", "*"))
	if err != nil {
		return 0, err
	}

	var migrated int
	for _, match := range matches {
		fi, err := os.Stat(match)
		if err != nil {
			return migrated, err
		}
		if fi.IsDir() {
			continue
		}

		rel, err := filepath.Rel(manifests, match)
		if err != nil {
			return migrated, fmt.Errorf("%s %w", match, err)
		}

		n := model.ParseNameFromFilepath(rel)
		if !n.IsFullyQualified() {
			slog.Warn("bad manifest name", "path", rel)
			continue
		}

		data, err := readManifestPath(match, manifests)
		if err != nil {
			return migrated, err
		}
		if err := WriteManifestData(normalizeLogicalName(n), data); err != nil {
			return migrated, err
		}
		migrated++
	}

	return migrated, nil
}

func readManifestPath(path, root string) ([]byte, error) {
	f, _, err := OpenVerifiedManifest(path, root)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	return io.ReadAll(f)
}

func pathWithin(path, root string) bool {
	rel, err := filepath.Rel(root, path)
	return err == nil && rel != "." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) && rel != ".."
}

func evalAbs(path string) (string, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	return filepath.EvalSymlinks(abs)
}

func sameFile(a, b string) bool {
	ai, err := os.Stat(a)
	if err != nil {
		return false
	}
	bi, err := os.Stat(b)
	if err != nil {
		return false
	}
	return os.SameFile(ai, bi)
}

func checkBlobDigest(path, digest string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	return checkBlobDigestReader(f, digest)
}

func checkBlobDigestReader(r io.Reader, digest string) error {
	h := sha256.New()
	if _, err := io.Copy(h, r); err != nil {
		return err
	}

	got := fmt.Sprintf("sha256:%x", h.Sum(nil))
	if got != strings.ToLower(strings.Replace(digest, "-", ":", 1)) {
		return errors.New("digest mismatch")
	}

	return nil
}

func Manifests(continueOnError bool) (map[model.Name]*Manifest, error) {
	ms := make(map[model.Name]*Manifest)

	manifestsV2, err := V2Path()
	if err != nil {
		return nil, err
	}
	if err := collectManifests(ms, manifestsV2, continueOnError); err != nil {
		return nil, err
	}

	manifests, err := Path()
	if err != nil {
		return nil, err
	}
	if err := collectManifests(ms, manifests, continueOnError); err != nil {
		return nil, err
	}

	return ms, nil
}

func collectManifests(ms map[model.Name]*Manifest, root string, continueOnError bool) error {
	// TODO(mxyng): use something less brittle
	matches, err := filepath.Glob(filepath.Join(root, "*", "*", "*", "*"))
	if err != nil {
		return err
	}

	for _, match := range matches {
		fi, err := os.Lstat(match)
		if err != nil {
			return err
		}

		if !fi.IsDir() {
			rel, err := filepath.Rel(root, match)
			if err != nil {
				if !continueOnError {
					return fmt.Errorf("%s %w", match, err)
				}
				slog.Warn("bad filepath", "path", match, "error", err)
				continue
			}

			n := model.ParseNameFromFilepath(rel)
			if !n.IsValid() {
				if !continueOnError {
					return fmt.Errorf("invalid manifest name: %s", rel)
				}
				slog.Warn("bad manifest name", "path", rel)
				continue
			}

			n = normalizeLogicalName(n)
			if _, ok := ms[n]; ok {
				continue
			}

			m, err := parseManifestFile(n, match, root)
			if err != nil {
				if !continueOnError {
					return fmt.Errorf("%s %w", n, err)
				}
				slog.Warn("bad manifest", "name", n, "error", err)
				continue
			}

			ms[n] = m
		}
	}

	return nil
}
