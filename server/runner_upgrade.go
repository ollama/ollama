package server

import (
	"context"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

const (
	// A slow registry must not slow down a run.
	upgradeProbeTimeout = 2 * time.Second

	// How long a "no list upstream" answer is trusted, so a model that will
	// never gain a variant is probed rarely rather than on every request.
	upgradeProbeTTL = 24 * time.Hour
)

var upgradeProbes sync.Map // manifest digest -> time.Time of the last negative probe

// runnerUpgradeAvailable reports whether the registry has replaced name with a
// manifest list offering a runner this host prefers over the local one. It
// answers only for RunnerAuto, so a request naming a runner, or none at all,
// never reaches the registry.
func runnerUpgradeAvailable(ctx context.Context, n model.Name, requested string, m *Model) bool {
	if !strings.EqualFold(requested, manifest.RunnerAuto) {
		return false
	}
	// A local list already resolved to this host's preference, so there is
	// nothing better to fetch.
	if preferred := manifest.PreferredRunner(); preferred == "" || strings.EqualFold(m.Runner, preferred) {
		return false
	}
	if !n.IsFullyQualified() || manifest.IsDigestReferenceName(n) {
		return false
	}

	digest := m.ManifestDigest
	if digest == "" {
		return false
	}
	if at, ok := upgradeProbes.Load(digest); ok && time.Since(at.(time.Time)) < upgradeProbeTTL {
		return false
	}

	if servesManifestList(ctx, n) {
		return true
	}
	upgradeProbes.Store(digest, time.Now())
	return false
}

// servesManifestList reports whether the registry now serves n as a manifest list.
func servesManifestList(ctx context.Context, n model.Name) bool {
	ctx, cancel := context.WithTimeout(ctx, upgradeProbeTimeout)
	defer cancel()

	u := n.BaseURL().JoinPath("v2", n.DisplayNamespaceModel(), "manifests", n.Tag)
	req, err := http.NewRequestWithContext(ctx, http.MethodHead, u.String(), nil)
	if err != nil {
		return false
	}
	req.Header.Set("Accept", strings.Join([]string{manifest.MediaTypeManifestList, manifest.MediaTypeManifest}, ", "))

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return false
	}

	mediaType, _, _ := strings.Cut(resp.Header.Get("Content-Type"), ";")
	return strings.EqualFold(strings.TrimSpace(mediaType), manifest.MediaTypeManifestList)
}
