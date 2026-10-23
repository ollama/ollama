//go:build windows || darwin

package updater

import (
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/ollama/ollama/app/store"
	"github.com/ollama/ollama/app/version"
)

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
	signed := func(a testAsset) []testAsset {
		return withSignatures([]testAsset{a})
	}

	// Same release as the running build: not offered again.
	tag := "v0.35.1-b11325-glm5next-unload"
	body = releaseJSON(t, tag, installed.Add(-time.Hour), signed(asset(tag))...)
	if ok, _ := updater.checkForUpdate(t.Context()); ok {
		t.Fatal("same release offered as update")
	}

	// DMG-only release: nothing to install.
	tag = "v0.35.1-b11325-glm5next-keepalive"
	body = releaseJSON(t, tag, installed.Add(time.Hour), testAsset{"Ollama-x-darwin-arm64.dmg", server.URL + "/dl/x.dmg"})
	if ok, _ := updater.checkForUpdate(t.Context()); ok {
		t.Fatal("dmg-only release offered as update")
	}

	// A zip without a signature is not offered.
	body = releaseJSON(t, tag, installed.Add(time.Hour), asset(tag))
	if ok, _ := updater.checkForUpdate(t.Context()); ok {
		t.Fatal("unsigned zip offered as update")
	}

	// Newer release with a signed zip/exe: offered.
	body = releaseJSON(t, tag, installed.Add(time.Hour), signed(asset(tag))...)
	ok, resp := updater.checkForUpdate(t.Context())
	if !ok {
		t.Fatal("expected newer fork release to be offered")
	}
	if resp.UpdateVersion != "0.35.1-b11325-glm5next-keepalive" || resp.UpdateURL != asset(tag).URL || resp.SignatureURL != asset(tag).URL+".ed25519" {
		t.Fatalf("unexpected update response %+v", resp)
	}

	// After installing it, the running build reports that tag: not re-offered.
	version.Version = "0.35.1-b11325-glm5next-keepalive-0-gabcdef0"
	currentBuildTime = func() time.Time { return installed.Add(2 * time.Hour) }
	if ok, _ := updater.checkForUpdate(t.Context()); ok {
		t.Fatal("installed release offered again")
	}
}
