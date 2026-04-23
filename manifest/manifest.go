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
	"runtime"
	"strings"

	"github.com/ollama/ollama/types/model"
)

var (
	blobFilenamePattern = regexp.MustCompile(`^sha256-[0-9a-fA-F]{64}$`)

	// ErrNoCompatibleManifest is returned when a manifest list does not contain
	// a child manifest for the requested runner.
	ErrNoCompatibleManifest = errors.New("no compatible manifest found")
)

const (
	MediaTypeManifest     = "application/vnd.docker.distribution.manifest.v2+json"
	MediaTypeManifestList = "application/vnd.ollama.manifest.list.v2+json"

	RunnerMLX      = "mlx"
	RunnerOllama   = "ollama"
	RunnerLlamaCPP = "llamacpp"

	FormatSafetensors = "safetensors"
	FormatGGUF        = "gguf"
)

type Manifest struct {
	SchemaVersion int        `json:"schemaVersion"`
	MediaType     string     `json:"mediaType"`
	Config        Layer      `json:"config"`
	Layers        []Layer    `json:"layers"`
	Runner        string     `json:"runner,omitempty"`
	Format        string     `json:"format,omitempty"`
	Manifests     []Manifest `json:"manifests,omitempty"`

	filepath       string
	fi             os.FileInfo
	digest         string
	selectedDigest string
	name           model.Name
}

func (m Manifest) isReference() bool {
	return m.MediaType != MediaTypeManifestList && m.digest != "" && m.Config.Digest == "" && len(m.Layers) == 0
}

func (m Manifest) MarshalJSON() ([]byte, error) {
	if m.MediaType == MediaTypeManifestList {
		return json.Marshal(struct {
			SchemaVersion int        `json:"schemaVersion"`
			MediaType     string     `json:"mediaType"`
			Manifests     []Manifest `json:"manifests"`
		}{
			SchemaVersion: m.SchemaVersion,
			MediaType:     m.MediaType,
			Manifests:     m.Manifests,
		})
	}

	if m.isReference() {
		return json.Marshal(struct {
			MediaType string `json:"mediaType"`
			Digest    string `json:"digest"`
			Runner    string `json:"runner,omitempty"`
			Format    string `json:"format,omitempty"`
		}{
			MediaType: m.MediaType,
			Digest:    m.BlobDigest(),
			Runner:    m.Runner,
			Format:    m.Format,
		})
	}

	return json.Marshal(struct {
		SchemaVersion int     `json:"schemaVersion"`
		MediaType     string  `json:"mediaType"`
		Config        Layer   `json:"config"`
		Layers        []Layer `json:"layers"`
		Runner        string  `json:"runner,omitempty"`
		Format        string  `json:"format,omitempty"`
	}{
		SchemaVersion: m.SchemaVersion,
		MediaType:     m.MediaType,
		Config:        m.Config,
		Layers:        m.Layers,
		Runner:        m.Runner,
		Format:        m.Format,
	})
}

func (m *Manifest) UnmarshalJSON(data []byte) error {
	var raw struct {
		SchemaVersion int        `json:"schemaVersion"`
		MediaType     string     `json:"mediaType"`
		Config        Layer      `json:"config"`
		Layers        []Layer    `json:"layers"`
		Runner        string     `json:"runner,omitempty"`
		Format        string     `json:"format,omitempty"`
		Manifests     []Manifest `json:"manifests,omitempty"`
		Digest        string     `json:"digest,omitempty"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}

	*m = Manifest{
		SchemaVersion: raw.SchemaVersion,
		MediaType:     raw.MediaType,
		Config:        raw.Config,
		Layers:        raw.Layers,
		Runner:        raw.Runner,
		Format:        raw.Format,
		Manifests:     raw.Manifests,
	}
	if raw.Digest != "" {
		digest, err := canonicalBlobDigest(raw.Digest)
		if err != nil {
			return err
		}
		m.digest = strings.TrimPrefix(digest, "sha256:")
	}

	return nil
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

// SelectedDigest returns the digest of the runnable manifest selected from a
// manifest list. For non-list manifests, it is the same as Digest.
func (m *Manifest) SelectedDigest() string {
	if m.selectedDigest != "" {
		return m.selectedDigest
	}

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

// RemoveNamed removes a named manifest object and then removes any blobs that
// were reachable only from that object.
func RemoveNamed(n model.Name) error {
	candidates, err := ReferencedBlobDigestsForName(n)
	if err != nil {
		return err
	}

	if err := removeNamedManifestPaths(n); err != nil {
		return err
	}

	_, err = RemoveUnreferencedBlobs(candidates...)
	return err
}

// ReferencedBlobDigestsForName returns the blob digests reachable from the
// named manifest object as stored on disk. Manifest lists include the parent
// list blob, every child manifest blob, and each child's config and layer blobs.
func ReferencedBlobDigestsForName(n model.Name) ([]string, error) {
	if !n.IsFullyQualified() {
		return nil, model.Unqualified(n)
	}

	p, root, err := resolveManifestPath(n)
	if err != nil {
		return nil, err
	}

	data, _, digest, err := readVerifiedManifest(p, root)
	if err != nil {
		return nil, err
	}

	return referencedBlobDigestsForData(digest, data)
}

// RetainedBlobDigests returns the complete set of blob digests reachable from
// all named manifest objects in the local store.
func RetainedBlobDigests() (map[string]struct{}, error) {
	refs, err := namedManifestRefs(true)
	if err != nil {
		return nil, err
	}

	retained := make(map[string]struct{})
	for n, ref := range refs {
		data, _, digest, err := readVerifiedManifest(ref.path, ref.root)
		if err != nil {
			slog.Warn("bad manifest", "name", n, "error", err)
			continue
		}

		digests, err := referencedBlobDigestsForData(digest, data)
		if err != nil {
			slog.Warn("bad manifest", "name", n, "error", err)
			continue
		}

		for _, digest := range digests {
			retained[digest] = struct{}{}
		}
	}

	return retained, nil
}

func referencedBlobDigestsForData(manifestDigest string, data []byte) ([]string, error) {
	var digests []string
	seen := make(map[string]struct{})

	add := func(digest string) error {
		if digest == "" {
			return nil
		}

		digest, err := canonicalBlobDigest(digest)
		if err != nil {
			return err
		}
		if _, ok := seen[digest]; ok {
			return nil
		}
		seen[digest] = struct{}{}
		digests = append(digests, digest)

		return nil
	}
	addManifest := func(m *Manifest) error {
		for _, layer := range append(m.Layers, m.Config) {
			if err := add(layer.Digest); err != nil {
				return err
			}
		}

		return nil
	}

	if err := add(blobDigest(manifestDigest)); err != nil {
		return nil, err
	}

	m, err := parseManifest(data)
	if err != nil {
		return nil, err
	}
	if m.MediaType == MediaTypeManifestList {
		for i := range m.Manifests {
			child := &m.Manifests[i]
			if err := add(child.BlobDigest()); err != nil {
				return nil, err
			}
			if child.isReference() {
				resolved, err := parseManifestBlob(child.BlobDigest())
				if err != nil {
					return nil, err
				}
				child = resolved
			}
			if err := addManifest(child); err != nil {
				return nil, err
			}
		}

		return digests, nil
	}

	if err := addManifest(m); err != nil {
		return nil, err
	}

	return digests, nil
}

// RemoveUnreferencedBlobs removes candidate blob digests that are not reachable
// from any current manifest. It returns the number of blobs removed.
func RemoveUnreferencedBlobs(candidates ...string) (int, error) {
	inUse, err := RetainedBlobDigests()
	if err != nil {
		return 0, err
	}

	var removed int
	seen := make(map[string]struct{})
	for _, digest := range candidates {
		if digest == "" {
			continue
		}
		digest, err = canonicalBlobDigest(digest)
		if err != nil {
			return removed, err
		}
		if _, ok := seen[digest]; ok {
			continue
		}
		seen[digest] = struct{}{}
		if _, used := inUse[digest]; used {
			continue
		}

		blob, err := BlobsPath(digest)
		if err != nil {
			return removed, err
		}
		if err := os.Remove(blob); os.IsNotExist(err) {
			slog.Debug("blob does not exist", "digest", digest)
		} else if err != nil {
			return removed, err
		} else {
			removed++
		}
	}

	return removed, nil
}

func ParseNamedManifest(n model.Name) (*Manifest, error) {
	return parseNamedManifest(n, runnerPreferences())
}

// ParseNamedManifestForRunner returns the named manifest selected for runner.
// If the named object is a manifest list, runner must match one child entry.
func ParseNamedManifestForRunner(n model.Name, runner string) (*Manifest, error) {
	return parseNamedManifest(n, runnerPreferencesFor(runner))
}

func parseNamedManifest(n model.Name, preferences []string) (*Manifest, error) {
	if !n.IsFullyQualified() {
		return nil, model.Unqualified(n)
	}

	p, root, err := resolveManifestPath(n)
	if err != nil {
		return nil, err
	}

	return parseManifestFile(normalizeLogicalName(n), p, root, preferences)
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

// ReadSelectedManifestData returns the runnable manifest bytes for a named
// model. If the named object is a manifest list, it returns the selected child
// manifest according to local runner preferences.
func ReadSelectedManifestData(n model.Name) ([]byte, error) {
	if !n.IsFullyQualified() {
		return nil, model.Unqualified(n)
	}

	p, root, err := resolveManifestPath(n)
	if err != nil {
		return nil, err
	}

	data, _, _, err := readVerifiedManifest(p, root)
	if err != nil {
		return nil, err
	}

	m, err := parseManifest(data)
	if err != nil {
		return nil, err
	}
	if m.MediaType == MediaTypeManifestList {
		child, err := selectManifest(m.Manifests)
		if err != nil {
			return nil, err
		}
		return json.Marshal(child)
	}

	return data, nil
}

func parseManifestFile(name model.Name, path, root string, preferences []string) (*Manifest, error) {
	data, fi, digest, err := readVerifiedManifest(path, root)
	if err != nil {
		return nil, err
	}

	return parseManifestData(name, path, fi, digest, data, preferences)
}

func parseManifestData(name model.Name, path string, fi os.FileInfo, digest string, data []byte, preferences []string) (*Manifest, error) {
	m, err := parseManifest(data)
	if err != nil {
		return nil, err
	}
	if m.MediaType == MediaTypeManifestList {
		child, err := selectManifestWithPreferences(m.Manifests, preferences)
		if err != nil {
			return nil, err
		}
		selectedDigest := child.digest
		child.filepath = path
		child.fi = fi
		child.digest = digest
		child.selectedDigest = selectedDigest
		child.name = name
		return child, nil
	}

	if len(preferences) == 1 && m.Runner != "" && !strings.EqualFold(m.Runner, preferences[0]) {
		return nil, fmt.Errorf("%w for runners: %s", ErrNoCompatibleManifest, preferences[0])
	}

	m.filepath = path
	m.fi = fi
	m.digest = digest
	m.name = name

	return m, nil
}

func selectManifest(manifests []Manifest) (*Manifest, error) {
	return selectManifestWithPreferences(manifests, runnerPreferences())
}

func selectManifestWithPreferences(manifests []Manifest, preferences []string) (*Manifest, error) {
	for _, runner := range preferences {
		for i := range manifests {
			if manifests[i].MediaType != "" && manifests[i].MediaType != MediaTypeManifest {
				continue
			}
			if strings.EqualFold(manifests[i].Runner, runner) {
				child := manifests[i]
				if child.isReference() {
					childDigest := child.digest
					resolved, err := parseManifestBlob(child.BlobDigest())
					if err != nil {
						return nil, err
					}
					if resolved.Runner == "" {
						resolved.Runner = child.Runner
					}
					if resolved.Format == "" {
						resolved.Format = child.Format
					}
					if resolved.digest == "" {
						resolved.digest = childDigest
					}
					child = *resolved
				}
				return &child, nil
			}
		}
	}

	return nil, fmt.Errorf("%w for runners: %s", ErrNoCompatibleManifest, strings.Join(preferences, ", "))
}

func runnerPreferences() []string {
	if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
		return []string{RunnerMLX, RunnerOllama, RunnerLlamaCPP}
	}

	return []string{RunnerOllama, RunnerLlamaCPP, RunnerMLX}
}

func runnerPreferencesFor(runner string) []string {
	runner = strings.ToLower(strings.TrimSpace(runner))
	if runner == "" {
		return runnerPreferences()
	}

	return []string{runner}
}

func parseManifest(data []byte) (*Manifest, error) {
	var m Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	if m.MediaType == MediaTypeManifestList {
		for i := range m.Manifests {
			if m.Manifests[i].MediaType == MediaTypeManifestList {
				return nil, errors.New("nested manifest lists are not supported")
			}
			if m.Manifests[i].isReference() {
				continue
			}

			canonical, err := json.Marshal(m.Manifests[i])
			if err != nil {
				return nil, err
			}
			sum := sha256.Sum256(canonical)
			m.Manifests[i].digest = fmt.Sprintf("%x", sum)
		}
	}

	return &m, nil
}

func readManifestBlob(digest string) ([]byte, error) {
	digest, err := canonicalBlobDigest(digest)
	if err != nil {
		return nil, err
	}

	blobPath, err := BlobsPath(digest)
	if err != nil {
		return nil, err
	}

	data, err := os.ReadFile(blobPath)
	if err != nil {
		return nil, err
	}
	if err := checkBlobDigestReader(bytes.NewReader(data), digest); err != nil {
		return nil, err
	}

	return data, nil
}

func parseManifestBlob(digest string) (*Manifest, error) {
	data, err := readManifestBlob(digest)
	if err != nil {
		return nil, err
	}

	return parseManifest(data)
}

func canonicalBlobDigest(digest string) (string, error) {
	if _, err := BlobsPath(digest); err != nil {
		return "", err
	}

	digest = strings.ToLower(strings.Replace(digest, "-", ":", 1))
	return digest, nil
}

func blobDigest(digest string) string {
	if digest == "" {
		return ""
	}

	digest = strings.ToLower(digest)
	if strings.HasPrefix(digest, "sha256:") || strings.HasPrefix(digest, "sha256-") {
		return digest
	}

	return "sha256:" + digest
}

func WriteManifest(name model.Name, config Layer, layers []Layer) error {
	return WriteManifestWithMetadata(name, config, layers, "", "")
}

// WriteManifestWithMetadata stores a single runnable manifest with optional
// runner and weight format metadata.
func WriteManifestWithMetadata(name model.Name, config Layer, layers []Layer, runner, format string) error {
	m := Manifest{
		SchemaVersion: 2,
		MediaType:     MediaTypeManifest,
		Config:        config,
		Layers:        layers,
		Runner:        runner,
		Format:        format,
	}

	var b bytes.Buffer
	if err := json.NewEncoder(&b).Encode(m); err != nil {
		return err
	}

	return WriteManifestData(name, b.Bytes())
}

// NewManifestReference returns a manifest-list entry that points at an existing
// child manifest blob.
func NewManifestReference(digest, runner, format string) (Manifest, error) {
	digest, err := canonicalBlobDigest(digest)
	if err != nil {
		return Manifest{}, err
	}

	return Manifest{
		MediaType: MediaTypeManifest,
		Runner:    runner,
		Format:    format,
		digest:    strings.TrimPrefix(digest, "sha256:"),
	}, nil
}

// WriteManifestBlob stores raw manifest bytes as a content-addressed blob.
func WriteManifestBlob(data []byte) (string, error) {
	return writeManifestBlob(data)
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
	data, _, _, err := readVerifiedManifest(path, root)
	return data, err
}

func readVerifiedManifest(path, root string) ([]byte, os.FileInfo, string, error) {
	f, digest, err := OpenVerifiedManifest(path, root)
	if err != nil {
		return nil, nil, "", err
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return nil, nil, "", err
	}

	data, err := io.ReadAll(f)
	if err != nil {
		return nil, nil, "", err
	}

	return data, fi, digest, nil
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
	refs, err := namedManifestRefs(continueOnError)
	if err != nil {
		return nil, err
	}

	ms := make(map[model.Name]*Manifest, len(refs))
	for n, ref := range refs {
		m, err := parseManifestFile(n, ref.path, ref.root, runnerPreferences())
		if err != nil {
			if !continueOnError {
				return nil, fmt.Errorf("%s %w", n, err)
			}
			slog.Warn("bad manifest", "name", n, "error", err)
			continue
		}

		ms[n] = m
	}
	return ms, nil
}

type namedManifestRef struct {
	path string
	root string
}

func namedManifestRefs(continueOnError bool) (map[model.Name]namedManifestRef, error) {
	refs := make(map[model.Name]namedManifestRef)

	manifestsV2, err := V2Path()
	if err != nil {
		return nil, err
	}
	if err := collectManifestRefs(refs, manifestsV2, continueOnError); err != nil {
		return nil, err
	}

	manifests, err := Path()
	if err != nil {
		return nil, err
	}
	if err := collectManifestRefs(refs, manifests, continueOnError); err != nil {
		return nil, err
	}

	return refs, nil
}

func collectManifestRefs(refs map[model.Name]namedManifestRef, root string, continueOnError bool) error {
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
			if _, ok := refs[n]; ok {
				continue
			}

			refs[n] = namedManifestRef{path: match, root: root}
		}
	}

	return nil
}
