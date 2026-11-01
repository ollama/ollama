//go:build (windows || darwin) && updater_live

package updater

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/app/store"
	"github.com/ollama/ollama/app/version"
)

// TestLiveAppUpdate exercises the production update endpoint and downloads the
// current OS update artifact. It is intentionally excluded from normal test
// runs because it depends on ollama.com and downloads a release artifact.
//
// Run with:
//
//	go test -tags updater_live -run TestLiveAppUpdate ./app/updater
func TestLiveAppUpdate(t *testing.T) {
	const spoofedVersion = "0.20.0"

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	oldUpdateStageDir := UpdateStageDir
	oldUpdateDownloaded := UpdateDownloaded
	oldVersion := version.Version
	defer func() {
		UpdateStageDir = oldUpdateStageDir
		UpdateDownloaded = oldUpdateDownloaded
		version.Version = oldVersion
	}()

	version.Version = spoofedVersion

	switch runtime.GOOS {
	case "windows":
		t.Setenv("LOCALAPPDATA", t.TempDir())
	case "darwin":
	default:
		t.Fatalf("unsupported updater live test OS %q", runtime.GOOS)
	}
	expectedFilename := liveExpectedFilename()

	UpdateStageDir = filepath.Join(t.TempDir(), "updates")
	UpdateDownloaded = false
	verifyCalled := false
	restoreVerification := wrapLiveUpdateVerification(&verifyCalled)
	defer restoreVerification()

	updater := &Updater{Store: &store.Store{DBPath: filepath.Join(t.TempDir(), "db.sqlite")}}
	defer updater.Store.Close()

	available, updateResp := updater.checkForUpdate(ctx)
	if !available {
		t.Fatalf("expected production update check to offer an update for spoofed version %s", spoofedVersion)
	}
	if updateResp.UpdateURL == "" {
		t.Fatal("production update response did not include a download URL")
	}
	t.Logf("production update version=%q url=%q", updateResp.UpdateVersion, updateResp.UpdateURL)

	if err := updater.DownloadNewRelease(ctx, updateResp); err != nil {
		if runtime.GOOS == "windows" && strings.Contains(err.Error(), "install.ps1 does not support cache-only mode") {
			if !verifyCalled {
				t.Fatal("production install.ps1 lacked cache-only support before signature verification ran")
			}
			t.Skipf("production install.ps1 has not shipped app cache-only support yet: %v", err)
		}
		t.Fatalf("download production update: %v", err)
	}

	staged := liveStagedUpdate()
	if staged == "" {
		t.Fatal("production update was not staged")
	}
	t.Logf("staged production update at %s", staged)

	assertPathInsideDir(t, UpdateStageDir, staged)
	if filepath.Base(staged) != expectedFilename {
		t.Fatalf("expected staged %s update filename to be %q, got %q", runtime.GOOS, expectedFilename, filepath.Base(staged))
	}
	expectedExt := filepath.Ext(expectedFilename)
	if filepath.Ext(staged) != expectedExt {
		t.Fatalf("expected staged %s update to be a %s artifact, got %s", runtime.GOOS, expectedExt, staged)
	}

	info, err := os.Stat(staged)
	if err != nil {
		t.Fatalf("stat staged update: %v", err)
	}
	if info.Size() == 0 {
		t.Fatal("staged production update is empty")
	}

	if !verifyCalled {
		t.Fatal("DownloadNewRelease did not verify the staged update")
	}
	t.Logf("production updater download path verified staged %s update", runtime.GOOS)
}

func assertPathInsideDir(t *testing.T, dir, name string) {
	t.Helper()

	dir, err := filepath.Abs(dir)
	if err != nil {
		t.Fatal(err)
	}
	name, err = filepath.Abs(name)
	if err != nil {
		t.Fatal(err)
	}

	rel, err := filepath.Rel(dir, name)
	if err != nil {
		t.Fatal(err)
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) || filepath.IsAbs(rel) {
		t.Fatalf("staged update escaped update stage dir: %s", name)
	}
}
