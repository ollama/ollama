package updater

import (
	"archive/tar"
	"archive/zip"
	"bytes"
	"compress/gzip"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/klauspost/compress/zstd"
)

func TestCompareVersions(t *testing.T) {
	tests := []struct {
		current string
		latest  string
		want    int
		wantErr bool
	}{
		{"0.35.0", "v0.35.1", -1, false},
		{"v0.35.0", "v0.35.1", -1, false},
		{"0.35.1", "0.35.1", 0, false},
		{"v0.35.1", "v0.35.1", 0, false},
		{"0.36.0", "0.35.1", 1, false},
		{"0.0.0", "v0.35.1", -1, false},
		{"0.0.0-dev", "v0.35.1", -1, false},
		{"v0.35.1-rc1", "v0.35.1", -1, false},
		{"v0.35.1", "v0.35.1-rc1", 1, false},
		{"invalid", "v0.35.1", 0, true},
		{"v0.35.1", "invalid", 0, true},
	}

	for _, tt := range tests {
		got, err := CompareVersions(tt.current, tt.latest)
		if (err != nil) != tt.wantErr {
			t.Errorf("CompareVersions(%q, %q) error = %v, wantErr %v", tt.current, tt.latest, err, tt.wantErr)
			continue
		}
		if !tt.wantErr && got != tt.want {
			t.Errorf("CompareVersions(%q, %q) = %d, want %d", tt.current, tt.latest, got, tt.want)
		}
	}
}

func TestSelectAsset(t *testing.T) {
	assets := []ReleaseAsset{
		{Name: "install.sh", DownloadURL: "https://example.com/install.sh", Size: 100},
		{Name: "ollama-linux-amd64-rocm.tar.zst", DownloadURL: "https://example.com/rocm.tar.zst", Size: 250},
		{Name: "ollama-linux-amd64.tar.zst", DownloadURL: "https://example.com/ollama-linux-amd64.tar.zst", Size: 200},
		{Name: "ollama-linux-arm64-jetpack5.tar.zst", DownloadURL: "https://example.com/jetpack5.tar.zst", Size: 220},
		{Name: "ollama-linux-arm64.tar.zst", DownloadURL: "https://example.com/ollama-linux-arm64.tar.zst", Size: 210},
		{Name: "Ollama-darwin.zip", DownloadURL: "https://example.com/Ollama-darwin.zip", Size: 300},
		{Name: "OllamaSetup.exe", DownloadURL: "https://example.com/OllamaSetup.exe", Size: 400},
		{Name: "ollama-windows-arm64.zip", DownloadURL: "https://example.com/ollama-windows-arm64.zip", Size: 410},
	}

	tests := []struct {
		goos     string
		goarch   string
		wantName string
	}{
		{"linux", "amd64", "ollama-linux-amd64.tar.zst"},
		{"linux", "arm64", "ollama-linux-arm64.tar.zst"},
		{"darwin", "arm64", "Ollama-darwin.zip"},
		{"darwin", "amd64", "Ollama-darwin.zip"},
		{"windows", "amd64", "OllamaSetup.exe"},
		{"windows", "arm64", "ollama-windows-arm64.zip"},
	}

	for _, tt := range tests {
		selected := SelectAsset(assets, tt.goos, tt.goarch)
		if selected == nil {
			t.Errorf("SelectAsset for %s/%s returned nil", tt.goos, tt.goarch)
			continue
		}
		if selected.Name != tt.wantName {
			t.Errorf("SelectAsset for %s/%s = %s, want %s", tt.goos, tt.goarch, selected.Name, tt.wantName)
		}
	}
}

func TestCheckWithMockServer(t *testing.T) {
	releaseJSON := gitHubRelease{
		TagName:     "v0.35.1",
		Name:        "Release v0.35.1",
		HTMLURL:     "https://github.com/ollama/ollama/releases/tag/v0.35.1",
		Body:        "Fixes and improvements",
		PublishedAt: time.Now(),
		Assets: []gitHubAsset{
			{Name: "ollama-linux-amd64.tar.zst", BrowserDownloadURL: "https://example.com/ollama-linux-amd64.tar.zst", Size: 1024},
		},
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(releaseJSON)
	}))
	defer server.Close()

	ctx := context.Background()
	info, err := Check(ctx, CheckOptions{
		CurrentVersion: "0.35.0",
		OS:             "linux",
		Arch:           "amd64",
		URL:            server.URL,
	})
	if err != nil {
		t.Fatalf("Check failed: %v", err)
	}

	if !info.UpdateAvailable {
		t.Errorf("expected UpdateAvailable=true, got false")
	}
	if info.LatestVersion != "v0.35.1" {
		t.Errorf("expected LatestVersion=v0.35.1, got %s", info.LatestVersion)
	}
	if info.Asset == nil || info.Asset.Name != "ollama-linux-amd64.tar.zst" {
		t.Errorf("expected asset ollama-linux-amd64.tar.zst, got %+v", info.Asset)
	}
}

func TestCheckUpToDate(t *testing.T) {
	releaseJSON := gitHubRelease{
		TagName: "v0.35.1",
		Name:    "v0.35.1",
		HTMLURL: "https://github.com/ollama/ollama/releases/tag/v0.35.1",
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(releaseJSON)
	}))
	defer server.Close()

	ctx := context.Background()
	info, err := Check(ctx, CheckOptions{
		CurrentVersion: "0.35.1",
		OS:             "linux",
		Arch:           "amd64",
		URL:            server.URL,
	})
	if err != nil {
		t.Fatalf("Check failed: %v", err)
	}

	if info.UpdateAvailable {
		t.Errorf("expected UpdateAvailable=false when versions match")
	}
}

func TestPull(t *testing.T) {
	content := []byte("fake-update-binary-contents-12345")

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "33")
		_, _ = w.Write(content)
	}))
	defer server.Close()

	tempDir := t.TempDir()
	release := &ReleaseInfo{
		CurrentVersion:  "0.35.0",
		LatestVersion:   "v0.35.1",
		UpdateAvailable: true,
		Asset: &ReleaseAsset{
			Name:        "ollama-test.bin",
			DownloadURL: server.URL + "/download",
			Size:        int64(len(content)),
		},
	}

	var progressCalled int32
	opts := PullOptions{
		Dir: tempDir,
		ProgressFn: func(downloaded, total int64) {
			atomic.AddInt32(&progressCalled, 1)
		},
	}

	result, err := Pull(context.Background(), release, opts)
	if err != nil {
		t.Fatalf("Pull failed: %v", err)
	}

	if result.AlreadyExisted {
		t.Errorf("expected AlreadyExisted=false on first pull")
	}
	if result.Size != int64(len(content)) {
		t.Errorf("expected size %d, got %d", len(content), result.Size)
	}

	saved, err := os.ReadFile(result.FilePath)
	if err != nil {
		t.Fatalf("read downloaded file: %v", err)
	}
	if !bytes.Equal(saved, content) {
		t.Errorf("content mismatch")
	}

	if atomic.LoadInt32(&progressCalled) == 0 {
		t.Errorf("expected progressFn to be called at least once")
	}

	// Pull again: should return AlreadyExisted = true
	result2, err := Pull(context.Background(), release, opts)
	if err != nil {
		t.Fatalf("Second pull failed: %v", err)
	}
	if !result2.AlreadyExisted {
		t.Errorf("expected AlreadyExisted=true on second pull")
	}
}

func TestInstallTarGz(t *testing.T) {
	tempDir := t.TempDir()
	tarGzPath := filepath.Join(tempDir, "update.tar.gz")

	// Create test tar.gz
	f, err := os.Create(tarGzPath)
	if err != nil {
		t.Fatal(err)
	}
	gw := gzip.NewWriter(f)
	tw := tar.NewWriter(gw)

	fileContent := []byte("ollama executable test")
	hdr := &tar.Header{
		Name: "bin/ollama",
		Mode: 0o755,
		Size: int64(len(fileContent)),
	}
	if err := tw.WriteHeader(hdr); err != nil {
		t.Fatal(err)
	}
	if _, err := tw.Write(fileContent); err != nil {
		t.Fatal(err)
	}
	tw.Close()
	gw.Close()
	f.Close()

	destDir := filepath.Join(tempDir, "install_root")
	if err := Install(tarGzPath, destDir); err != nil {
		t.Fatalf("Install failed: %v", err)
	}

	installedFile := filepath.Join(destDir, "bin", "ollama")
	data, err := os.ReadFile(installedFile)
	if err != nil {
		t.Fatalf("read installed file: %v", err)
	}
	if !bytes.Equal(data, fileContent) {
		t.Errorf("content mismatch: got %q, want %q", string(data), string(fileContent))
	}
}

func TestInstallTarZst(t *testing.T) {
	tempDir := t.TempDir()
	tarZstPath := filepath.Join(tempDir, "update.tar.zst")

	f, err := os.Create(tarZstPath)
	if err != nil {
		t.Fatal(err)
	}
	zw, err := zstd.NewWriter(f)
	if err != nil {
		t.Fatal(err)
	}
	tw := tar.NewWriter(zw)

	fileContent := []byte("ollama zstd test executable")
	hdr := &tar.Header{
		Name: "bin/ollama",
		Mode: 0o755,
		Size: int64(len(fileContent)),
	}
	if err := tw.WriteHeader(hdr); err != nil {
		t.Fatal(err)
	}
	if _, err := tw.Write(fileContent); err != nil {
		t.Fatal(err)
	}
	tw.Close()
	zw.Close()
	f.Close()

	destDir := filepath.Join(tempDir, "install_zst_root")
	if err := Install(tarZstPath, destDir); err != nil {
		t.Fatalf("Install tar.zst failed: %v", err)
	}

	installedFile := filepath.Join(destDir, "bin", "ollama")
	data, err := os.ReadFile(installedFile)
	if err != nil {
		t.Fatalf("read installed file: %v", err)
	}
	if !bytes.Equal(data, fileContent) {
		t.Errorf("content mismatch: got %q, want %q", string(data), string(fileContent))
	}
}

func TestInstallTarZstWithWrapperScript(t *testing.T) {
	tempDir := t.TempDir()
	tarZstPath := filepath.Join(tempDir, "update.tar.zst")

	f, err := os.Create(tarZstPath)
	if err != nil {
		t.Fatal(err)
	}
	zw, err := zstd.NewWriter(f)
	if err != nil {
		t.Fatal(err)
	}
	tw := tar.NewWriter(zw)

	binaryContent := []byte("\x7fELF-new-ollama-binary")
	hdr := &tar.Header{
		Name: "bin/ollama",
		Mode: 0o755,
		Size: int64(len(binaryContent)),
	}
	if err := tw.WriteHeader(hdr); err != nil {
		t.Fatal(err)
	}
	if _, err := tw.Write(binaryContent); err != nil {
		t.Fatal(err)
	}
	tw.Close()
	zw.Close()
	f.Close()

	destDir := filepath.Join(tempDir, "install_root")
	binDir := filepath.Join(destDir, "bin")
	if err := os.MkdirAll(binDir, 0o755); err != nil {
		t.Fatal(err)
	}

	wrapperScript := []byte("#!/usr/bin/env bash\nREAL_BIN=\"${OLLAMA_REAL_BIN:-/usr/local/bin/ollama.bin}\"\nexec \"$REAL_BIN\" \"$@\"\n")
	if err := os.WriteFile(filepath.Join(binDir, "ollama"), wrapperScript, 0o755); err != nil {
		t.Fatal(err)
	}

	if err := Install(tarZstPath, destDir); err != nil {
		t.Fatalf("Install tar.zst failed: %v", err)
	}

	// Wrapper script must still exist and be intact
	scriptData, err := os.ReadFile(filepath.Join(binDir, "ollama"))
	if err != nil {
		t.Fatalf("read wrapper script: %v", err)
	}
	if !bytes.Equal(scriptData, wrapperScript) {
		t.Errorf("wrapper script was overwritten!")
	}

	// Real binary must be extracted to ollama.bin
	realBinData, err := os.ReadFile(filepath.Join(binDir, "ollama.bin"))
	if err != nil {
		t.Fatalf("read real binary: %v", err)
	}
	if !bytes.Equal(realBinData, binaryContent) {
		t.Errorf("real binary content mismatch: got %q, want %q", string(realBinData), string(binaryContent))
	}
}

func TestInstallZip(t *testing.T) {
	tempDir := t.TempDir()
	zipPath := filepath.Join(tempDir, "update.zip")

	f, err := os.Create(zipPath)
	if err != nil {
		t.Fatal(err)
	}
	zw := zip.NewWriter(f)

	fileContent := []byte("zip binary test")
	w, err := zw.Create("Ollama.app/Contents/MacOS/Ollama")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := w.Write(fileContent); err != nil {
		t.Fatal(err)
	}
	zw.Close()
	f.Close()

	destDir := filepath.Join(tempDir, "app_dest")
	if err := Install(zipPath, destDir); err != nil {
		t.Fatalf("Install zip failed: %v", err)
	}

	installed := filepath.Join(destDir, "Ollama.app", "Contents", "MacOS", "Ollama")
	data, err := os.ReadFile(installed)
	if err != nil {
		t.Fatalf("read installed zip file: %v", err)
	}
	if !bytes.Equal(data, fileContent) {
		t.Errorf("content mismatch: got %q, want %q", string(data), string(fileContent))
	}
}

func TestInstallZipPathTraversalPrevented(t *testing.T) {
	tempDir := t.TempDir()
	zipPath := filepath.Join(tempDir, "malicious.zip")

	f, err := os.Create(zipPath)
	if err != nil {
		t.Fatal(err)
	}
	zw := zip.NewWriter(f)

	w, err := zw.Create("../escape.txt")
	if err != nil {
		t.Fatal(err)
	}
	_, _ = w.Write([]byte("bad"))
	zw.Close()
	f.Close()

	destDir := filepath.Join(tempDir, "dest")
	err = Install(zipPath, destDir)
	if err == nil {
		t.Fatalf("expected error on path traversal in zip, got nil")
	}
}

func TestIsRCVersion(t *testing.T) {
	tests := []struct {
		v    string
		want bool
	}{
		{"v0.40.0-rc2", true},
		{"0.40.0-rc0", true},
		{"v0.40.0-rc.1", true},
		{"v0.35.1", false},
		{"0.35.1", false},
		{"v0.40.0", false},
		{"v0.40.0-dev", false},
		{"", false},
	}
	for _, tt := range tests {
		got := IsRCVersion(tt.v)
		if got != tt.want {
			t.Errorf("IsRCVersion(%q) = %v, want %v", tt.v, got, tt.want)
		}
	}
}

func TestFindHighestTag(t *testing.T) {
	tags := []string{
		"v0.34.0",
		"v0.35.0",
		"v0.35.1",
		"v0.40.0-rc0",
		"v0.40.0-rc1",
		"v0.40.0-rc2",
	}

	highestStable := findHighestTag(tags, false)
	if highestStable != "v0.35.1" {
		t.Errorf("findHighestTag(tags, false) = %q, want v0.35.1", highestStable)
	}

	highestRC := findHighestTag(tags, true)
	if highestRC != "v0.40.0-rc2" {
		t.Errorf("findHighestTag(tags, true) = %q, want v0.40.0-rc2", highestRC)
	}
}

func TestCheckRC(t *testing.T) {
	releaseJSON := gitHubRelease{
		TagName:     "v0.40.0-rc1",
		Name:        "Release v0.40.0-rc1",
		HTMLURL:     "https://github.com/ollama/ollama/releases/tag/v0.40.0-rc1",
		Body:        "Pre-release fixes",
		PublishedAt: time.Now(),
		Assets: []gitHubAsset{
			{Name: "ollama-linux-amd64.tar.zst", BrowserDownloadURL: "https://example.com/ollama-linux-amd64.tar.zst", Size: 1024},
		},
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(releaseJSON)
	}))
	defer server.Close()

	ctx := context.Background()
	info, err := Check(ctx, CheckOptions{
		CurrentVersion: "0.35.1",
		OS:             "linux",
		Arch:           "amd64",
		URL:            server.URL,
		IncludeRC:      true,
	})
	if err != nil {
		t.Fatalf("Check failed: %v", err)
	}

	if !info.UpdateAvailable {
		t.Errorf("expected UpdateAvailable=true, got false")
	}
	if !info.IsRC {
		t.Errorf("expected IsRC=true, got false")
	}
	if info.LatestVersion != "v0.40.0-rc1" {
		t.Errorf("expected LatestVersion=v0.40.0-rc1, got %s", info.LatestVersion)
	}
}

func TestResolveBinaryTarget(t *testing.T) {
	tempDir := t.TempDir()

	// 1. Regular binary
	normalBin := filepath.Join(tempDir, "ollama")
	if err := os.WriteFile(normalBin, []byte("\x7fELFfakebinary"), 0o755); err != nil {
		t.Fatal(err)
	}
	if got := resolveBinaryTarget(normalBin); got != normalBin {
		t.Errorf("resolveBinaryTarget(normalBin) = %q, want %q", got, normalBin)
	}

	// 2. Wrapper script referencing ollama.bin
	scriptPath := filepath.Join(tempDir, "script", "ollama")
	if err := os.MkdirAll(filepath.Dir(scriptPath), 0o755); err != nil {
		t.Fatal(err)
	}
	scriptContent := `#!/usr/bin/env bash
REAL_BIN="${OLLAMA_REAL_BIN:-/usr/local/bin/ollama.bin}"
exec "$REAL_BIN" "$@"
`
	if err := os.WriteFile(scriptPath, []byte(scriptContent), 0o755); err != nil {
		t.Fatal(err)
	}
	wantBin := filepath.Join(filepath.Dir(scriptPath), "ollama.bin")
	if got := resolveBinaryTarget(scriptPath); got != wantBin {
		t.Errorf("resolveBinaryTarget(scriptPath) = %q, want %q", got, wantBin)
	}

	// 3. Symlink
	symlinkOllama := filepath.Join(tempDir, "symlink_dir", "ollama")
	if err := os.MkdirAll(filepath.Dir(symlinkOllama), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(normalBin, symlinkOllama); err != nil {
		t.Fatal(err)
	}
	if got := resolveBinaryTarget(symlinkOllama); got != normalBin {
		t.Errorf("resolveBinaryTarget(symlinkOllama) = %q, want %q", got, normalBin)
	}
}

