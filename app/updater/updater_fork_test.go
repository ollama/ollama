//go:build windows || darwin

package updater

import (
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"
	"time"

	"github.com/ollama/ollama/app/store"
	"github.com/ollama/ollama/app/version"
)

func TestForkBuildTag(t *testing.T) {
	for _, tt := range []struct{ version, want string }{
		{"0.35.1-b11325-glm5next-unload-9-g6c178a3-dirty", "0.35.1-b11325-glm5next-unload"},
		{"0.35.1-b11325-glm5next-unload-0-g6c178a3", "0.35.1-b11325-glm5next-unload"},
		{"v0.35.1-b11325-glm5next-unload-12-gabcdef0", "0.35.1-b11325-glm5next-unload"},
		{"0.35.1-b11325-glm5next-unload", "0.35.1-b11325-glm5next-unload"},
		{"0.35.1-b11325-glm5next-unload-dirty", "0.35.1-b11325-glm5next-unload"},
		{"0.35.1-rc0-3-g1234567", "0.35.1-rc0"},
		{"0.0.0", "0.0.0"},
		{"6c178a3-dirty", "6c178a3"},
	} {
		if got := forkBuildTag(tt.version); got != tt.want {
			t.Errorf("forkBuildTag(%q) = %q, want %q", tt.version, got, tt.want)
		}
	}
}

func TestGitHubReleaseIsUpdate(t *testing.T) {
	installed := time.Date(2026, 10, 7, 16, 24, 0, 0, time.UTC)
	before := installed.Add(-time.Hour)
	after := installed.Add(time.Hour)
	const current = "0.35.1-b11325-glm5next-unload-9-g6c178a3-dirty"

	for _, tt := range []struct {
		name      string
		tag       string
		current   string
		published time.Time
		want      bool
	}{
		{"same tag as build", "v0.35.1-b11325-glm5next-unload", current, after, false},
		{"exact tag build", "v0.35.1-b11325-glm5next-unload", "0.35.1-b11325-glm5next-unload-0-g1234567", after, false},
		{"older suffix published before install", "v0.35.1-b11325-glm5next", current, before, false},
		{"prefix of build tag published later", "v0.35.1-b11325-glm5next", current, after, true},
		{"new suffix published later", "v0.35.1-b11325-glm5next-keepalive", current, after, true},
		{"new suffix published before install", "v0.35.1-b11325-glm5next-keepalive", current, before, false},
		{"new suffix without publish time", "v0.35.1-b11325-glm5next-keepalive", current, time.Time{}, true},
		{"newer patch", "v0.35.2-b11325-glm5next", current, before, true},
		{"newer minor", "v0.40.0-b11400-glm5next", current, before, true},
		{"newer llama rev", "v0.35.1-b11400-glm5next", current, before, true},
		{"older patch", "v0.35.0-b11325-glm5next", current, after, false},
		{"older llama rev", "v0.35.1-b11000-glm5next", current, after, false},
		{"numeric not lexical", "v0.35.10-b11325-x", "0.35.9-b11325-x-1-gabcdef0", before, true},
		{"dev build", "v0.35.1-b11325-glm5next-unload", "0.0.0", before, true},
		{"empty tag", "", current, after, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := githubReleaseIsUpdate(tt.tag, tt.current, tt.published, installed); got != tt.want {
				t.Errorf("githubReleaseIsUpdate(%q, %q) = %v, want %v", tt.tag, tt.current, got, tt.want)
			}
		})
	}
}

func TestCheckForUpdateGitHubRelease(t *testing.T) {
	ext := ".zip"
	if runtime.GOOS == "windows" {
		ext = ".exe"
	}
	installed := time.Date(2026, 10, 7, 16, 24, 0, 0, time.UTC)

	oldURL, oldVersion, oldBuildTime := UpdateCheckURLBase, version.Version, currentBuildTime
	t.Cleanup(func() {
		UpdateCheckURLBase, version.Version, currentBuildTime = oldURL, oldVersion, oldBuildTime
	})
	version.Version = "0.35.1-b11325-glm5next-unload-9-g6c178a3-dirty"
	currentBuildTime = func() time.Time { return installed }

	var body []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/repos/maxkongerskov/ollama/releases/latest" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write(body)
	}))
	defer server.Close()
	UpdateCheckURLBase = server.URL + "/repos/maxkongerskov/ollama/releases/latest"

	updater := &Updater{Store: &store.Store{DBPath: filepath.Join(t.TempDir(), "test.db")}}
	defer updater.Store.Close()

	asset := func(tag string) testAsset {
		return testAsset{"Ollama-" + tag[1:] + "-" + runtime.GOOS + "-" + runtime.GOARCH + ext, server.URL + "/dl/" + tag + ext}
	}

	// Same release as the running build: not offered again.
	tag := "v0.35.1-b11325-glm5next-unload"
	body = releaseJSON(t, tag, installed.Add(-time.Hour), asset(tag))
	if ok, _ := updater.checkForUpdate(t.Context()); ok {
		t.Fatal("same release offered as update")
	}

	// DMG-only release: nothing to install.
	tag = "v0.35.1-b11325-glm5next-keepalive"
	body = releaseJSON(t, tag, installed.Add(time.Hour), testAsset{"Ollama-x-darwin-arm64.dmg", server.URL + "/dl/x.dmg"})
	if ok, _ := updater.checkForUpdate(t.Context()); ok {
		t.Fatal("dmg-only release offered as update")
	}

	// Newer release with a zip/exe: offered.
	body = releaseJSON(t, tag, installed.Add(time.Hour), asset(tag))
	ok, resp := updater.checkForUpdate(t.Context())
	if !ok {
		t.Fatal("expected newer fork release to be offered")
	}
	if resp.UpdateVersion != "0.35.1-b11325-glm5next-keepalive" || resp.UpdateURL != asset(tag).URL {
		t.Fatalf("unexpected update response %+v", resp)
	}

	// After installing it, the running build reports that tag: not re-offered.
	version.Version = "0.35.1-b11325-glm5next-keepalive-0-gabcdef0"
	currentBuildTime = func() time.Time { return installed.Add(2 * time.Hour) }
	if ok, _ := updater.checkForUpdate(t.Context()); ok {
		t.Fatal("installed release offered again")
	}
}
