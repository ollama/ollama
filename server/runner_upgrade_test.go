package server

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

// listServer answers HEAD with mediaType and counts the probes it receives.
func listServer(t *testing.T, mediaType string) (model.Name, *atomic.Int64) {
	t.Helper()

	var probes atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		probes.Add(1)
		w.Header().Set("Content-Type", mediaType)
	}))
	t.Cleanup(server.Close)

	n := model.ParseName(strings.TrimPrefix(server.URL, "http://") + "/library/upgrade-me:latest")
	n.ProtocolScheme = "http"
	return n, &probes
}

func ggmlModel(digest string) *Model {
	return &Model{Runner: manifest.RunnerGGML, ManifestDigest: digest}
}

func TestRunnerUpgradeAvailable(t *testing.T) {
	if manifest.PreferredRunner() == manifest.RunnerGGML {
		t.Skip("host prefers ggml, so a ggml model is already the preference")
	}

	t.Run("auto finds a list upstream", func(t *testing.T) {
		upgradeProbes.Clear()
		n, probes := listServer(t, manifest.MediaTypeManifestList)

		if !runnerUpgradeAvailable(t.Context(), n, manifest.RunnerAuto, ggmlModel("sha256:a")) {
			t.Fatal("runnerUpgradeAvailable() = false, want true")
		}
		if got := probes.Load(); got != 1 {
			t.Errorf("probes = %d, want 1", got)
		}
	})

	t.Run("a named runner never probes", func(t *testing.T) {
		upgradeProbes.Clear()
		n, probes := listServer(t, manifest.MediaTypeManifestList)

		for _, runner := range []string{"", manifest.RunnerGGML, manifest.RunnerMLX} {
			if runnerUpgradeAvailable(t.Context(), n, runner, ggmlModel("sha256:b")) {
				t.Errorf("runnerUpgradeAvailable(%q) = true, want false", runner)
			}
		}
		if got := probes.Load(); got != 0 {
			t.Errorf("probes = %d, want 0", got)
		}
	})

	t.Run("a plain manifest upstream is remembered", func(t *testing.T) {
		upgradeProbes.Clear()
		n, probes := listServer(t, manifest.MediaTypeManifest)

		for range 3 {
			if runnerUpgradeAvailable(t.Context(), n, manifest.RunnerAuto, ggmlModel("sha256:c")) {
				t.Fatal("runnerUpgradeAvailable() = true, want false")
			}
		}
		if got := probes.Load(); got != 1 {
			t.Errorf("probes = %d, want 1 — the negative answer should be remembered", got)
		}
	})

	t.Run("a model already on the preferred runner never probes", func(t *testing.T) {
		upgradeProbes.Clear()
		n, probes := listServer(t, manifest.MediaTypeManifestList)

		m := &Model{Runner: manifest.PreferredRunner(), ManifestDigest: "sha256:d"}
		if runnerUpgradeAvailable(t.Context(), n, manifest.RunnerAuto, m) {
			t.Error("runnerUpgradeAvailable() = true, want false")
		}
		if got := probes.Load(); got != 0 {
			t.Errorf("probes = %d, want 0", got)
		}
	})
}
