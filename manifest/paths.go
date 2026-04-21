package manifest

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/types/model"
)

var (
	ErrAmbiguousDigestPrefix = errors.New("ambiguous digest prefix")
	ErrInvalidDigestFormat   = errors.New("invalid digest format")
	digestPrefixPattern      = regexp.MustCompile(`^[0-9a-fA-F]{12,64}$`)
)

const (
	legacyDirName     = "manifests"
	v2DirName         = "manifests-v2"
	defaultPublicHost = "registry.ollama.ai"
	v2CanonicalHost   = "ollama.com"
	defaultNamespace  = "library"
	defaultTag        = "latest"
)

func Path() (string, error) {
	return manifestPath(legacyDirName)
}

func V2Path() (string, error) {
	return manifestPath(v2DirName)
}

func manifestPath(dir string) (string, error) {
	path := filepath.Join(envconfig.Models(), dir)
	if err := os.MkdirAll(path, 0o755); err != nil {
		return "", fmt.Errorf("%w: ensure path elements are traversable", err)
	}

	return path, nil
}

// PathForName returns the path to the manifest file for a specific model name.
func PathForName(n model.Name) (string, error) {
	return LegacyPathForName(n)
}

func LegacyPathForName(n model.Name) (string, error) {
	if !n.IsValid() {
		return "", os.ErrNotExist
	}

	manifests, err := Path()
	if err != nil {
		return "", err
	}

	return filepath.Join(manifests, n.Filepath()), nil
}

func V2PathForName(n model.Name) (string, error) {
	if !n.IsValid() {
		return "", os.ErrNotExist
	}

	manifests, err := V2Path()
	if err != nil {
		return "", err
	}

	return filepath.Join(manifests, canonicalV2Name(n).Filepath()), nil
}

func ResolvePathForName(n model.Name) (string, error) {
	path, _, err := resolveManifestPath(n)
	return path, err
}

func resolveManifestPath(n model.Name) (string, string, error) {
	if !n.IsValid() {
		return "", "", os.ErrNotExist
	}

	if digest, ok, err := ResolveDigestReferenceName(n); err != nil {
		return "", "", err
	} else if ok {
		blobPath, err := BlobsPath(digest)
		if err != nil {
			return "", "", err
		}
		if _, err := os.Lstat(blobPath); err == nil {
			root, err := BlobsPath("")
			return blobPath, root, err
		} else if !os.IsNotExist(err) {
			return "", "", err
		}

		return "", "", os.ErrNotExist
	}

	v2Path, err := V2PathForName(n)
	if err != nil {
		return "", "", err
	}
	if _, err := os.Lstat(v2Path); err == nil {
		root, err := V2Path()
		return v2Path, root, err
	} else if !os.IsNotExist(err) {
		return "", "", err
	}

	legacyRoot, err := Path()
	if err != nil {
		return "", "", err
	}
	for _, legacyName := range legacyNameCandidates(n) {
		legacyPath := filepath.Join(legacyRoot, legacyName.Filepath())
		if _, err := os.Lstat(legacyPath); err == nil {
			return legacyPath, legacyRoot, nil
		} else if !os.IsNotExist(err) {
			return "", "", err
		}
	}

	return "", "", os.ErrNotExist
}

func removeNamedManifestPaths(n model.Name) error {
	candidates := legacyNameCandidates(n)
	paths := make([]string, 0, 1+len(candidates))

	v2Path, err := V2PathForName(n)
	if err != nil {
		return err
	}
	paths = append(paths, v2Path)

	for _, legacyName := range candidates {
		legacyPath, err := LegacyPathForName(legacyName)
		if err != nil {
			return err
		}
		paths = append(paths, legacyPath)
	}

	for _, path := range paths {
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			return err
		}
	}

	return pruneManifestRoots()
}

func removeLegacyManifestPaths(n model.Name) error {
	for _, legacyName := range legacyNameCandidates(n) {
		legacyPath, err := LegacyPathForName(legacyName)
		if err != nil {
			return err
		}
		if err := os.Remove(legacyPath); err != nil && !os.IsNotExist(err) {
			return err
		}
	}

	legacyRoot, err := Path()
	if err != nil {
		return err
	}

	if err := PruneDirectory(legacyRoot); err != nil && !os.IsNotExist(err) {
		return err
	}

	return nil
}

func pruneManifestRoots() error {
	roots := []func() (string, error){Path, V2Path}
	for _, rootFn := range roots {
		root, err := rootFn()
		if err != nil {
			return err
		}
		if err := PruneDirectory(root); err != nil && !os.IsNotExist(err) {
			return err
		}
	}

	return nil
}

// normalizeLogicalName maps any public host to the legacy default
// so that map keys use a single identity regardless of on-disk host.
func normalizeLogicalName(n model.Name) model.Name {
	if isDefaultPublicHost(n.Host) {
		n.Host = defaultPublicHost
	}

	return n
}

// canonicalV2Name maps any public host to the v2 canonical host
// for use in manifests-v2/ on-disk paths.
func canonicalV2Name(n model.Name) model.Name {
	if isDefaultPublicHost(n.Host) {
		n.Host = v2CanonicalHost
	}

	return n
}

func legacyNameCandidates(n model.Name) []model.Name {
	names := []model.Name{n}
	if !isDefaultPublicHost(n.Host) {
		return names
	}

	alt := n
	switch {
	case strings.EqualFold(n.Host, defaultPublicHost):
		alt.Host = v2CanonicalHost
	default:
		alt.Host = defaultPublicHost
	}

	return append(names, alt)
}

func isDefaultPublicHost(host string) bool {
	return strings.EqualFold(host, defaultPublicHost) || strings.EqualFold(host, v2CanonicalHost)
}

// DigestReference parses a bare local manifest blob reference in sha256-<hex>
// filename form. It deliberately rejects paths and registry digest syntax; this
// form identifies blobs already present in the local Ollama model store.
func DigestReference(ref string) (string, bool) {
	ref = strings.TrimSpace(ref)
	if !blobFilenamePattern.MatchString(ref) {
		return "", false
	}

	return "sha256:" + strings.ToLower(strings.TrimPrefix(ref, "sha256-")), true
}

// ResolveDigestReference resolves a local manifest blob reference in sha256-hex,
// sha256:hex, or bare hex prefix form.
func ResolveDigestReference(ref string) (string, bool, error) {
	ref = strings.TrimSpace(ref)
	ref = strings.ToLower(strings.Replace(ref, "-", ":", 1))
	ref = strings.TrimPrefix(ref, "sha256:")
	if !digestPrefixPattern.MatchString(ref) {
		return "", false, nil
	}
	if len(ref) == 64 {
		return "sha256:" + ref, true, nil
	}

	digest, err := resolveBlobDigestPrefix(ref)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", false, nil
		}
		return "", true, err
	}
	return digest, true, nil
}

func IsDigestReference(ref string) bool {
	_, ok := DigestReference(ref)
	return ok
}

// DigestReferenceName returns the local blob digest for names that are the
// default model-name expansion of a digest reference.
func DigestReferenceName(n model.Name) (string, bool) {
	digest, ok, err := ResolveDigestReferenceName(n)
	if err != nil {
		return "", false
	}
	return digest, ok
}

func ResolveDigestReferenceName(n model.Name) (string, bool, error) {
	if !n.IsFullyQualified() {
		return "", false, nil
	}
	if !isDefaultPublicHost(n.Host) || !strings.EqualFold(n.Namespace, defaultNamespace) || !strings.EqualFold(n.Tag, defaultTag) {
		return "", false, nil
	}

	return ResolveDigestReference(n.Model)
}

// IsDigestReferenceName reports whether n names a local manifest blob instead
// of a model tag.
func IsDigestReferenceName(n model.Name) bool {
	_, ok := DigestReferenceName(n)
	return ok
}

func resolveBlobDigestPrefix(prefix string) (string, error) {
	root, err := BlobsPath("")
	if err != nil {
		return "", err
	}
	matches, err := filepath.Glob(filepath.Join(root, "sha256-"+prefix+"*"))
	if err != nil {
		return "", err
	}

	var digest string
	for _, match := range matches {
		base := filepath.Base(match)
		if !blobFilenamePattern.MatchString(base) {
			continue
		}
		if digest != "" {
			return "", fmt.Errorf("%w %q", ErrAmbiguousDigestPrefix, prefix)
		}
		digest = "sha256:" + strings.TrimPrefix(base, "sha256-")
	}
	if digest == "" {
		return "", os.ErrNotExist
	}

	return digest, nil
}

func BlobsPath(digest string) (string, error) {
	// only accept actual sha256 digests
	pattern := "^sha256[:-][0-9a-fA-F]{64}$"
	re := regexp.MustCompile(pattern)

	if digest != "" && !re.MatchString(digest) {
		return "", ErrInvalidDigestFormat
	}

	digest = strings.ReplaceAll(digest, ":", "-")
	path := filepath.Join(envconfig.Models(), "blobs", digest)
	dirPath := filepath.Dir(path)
	if digest == "" {
		dirPath = path
	}

	if err := os.MkdirAll(dirPath, 0o755); err != nil {
		return "", fmt.Errorf("%w: ensure path elements are traversable", err)
	}

	return path, nil
}

// PruneDirectory removes empty directories recursively.
func PruneDirectory(path string) error {
	info, err := os.Lstat(path)
	if err != nil {
		return err
	}

	if info.IsDir() && info.Mode()&os.ModeSymlink == 0 {
		entries, err := os.ReadDir(path)
		if err != nil {
			return err
		}

		for _, entry := range entries {
			if err := PruneDirectory(filepath.Join(path, entry.Name())); err != nil {
				return err
			}
		}

		entries, err = os.ReadDir(path)
		if err != nil {
			return err
		}

		if len(entries) > 0 {
			return nil
		}

		return os.Remove(path)
	}

	return nil
}
