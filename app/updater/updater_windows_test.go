//go:build windows

package updater

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	appversion "github.com/ollama/ollama/app/version"
)

type fakeInstallScriptProcess struct {
	released atomic.Bool
}

func (p *fakeInstallScriptProcess) Release() error {
	p.released.Store(true)
	return nil
}

func resetWindowsUpdaterState(t *testing.T) {
	t.Helper()

	oldUpdateStageDir := UpdateStageDir
	oldRunningInstaller := runningInstaller
	oldUpgradeLogFile := UpgradeLogFile
	oldInstallScriptInstallerLogFile := installScriptInstallerLogFile
	oldInstallScriptInstallerLogFileGlob := installScriptInstallerLogFileGlob
	oldUpgradeMarkerFile := UpgradeMarkerFile
	oldVerifyDownload := VerifyDownload
	oldVerifyInstallScriptSignature := verifyInstallScriptSignature
	oldRunInstallScriptCacheOnly := runInstallScriptCacheOnly
	oldStartInstallScriptInstall := startInstallScriptInstall
	oldExitAfterStartingUpgrade := exitAfterStartingUpgrade
	oldUpdateDownloaded := UpdateDownloaded
	oldAppVersion := appversion.Version
	t.Cleanup(func() {
		UpdateStageDir = oldUpdateStageDir
		runningInstaller = oldRunningInstaller
		UpgradeLogFile = oldUpgradeLogFile
		installScriptInstallerLogFile = oldInstallScriptInstallerLogFile
		installScriptInstallerLogFileGlob = oldInstallScriptInstallerLogFileGlob
		UpgradeMarkerFile = oldUpgradeMarkerFile
		VerifyDownload = oldVerifyDownload
		verifyInstallScriptSignature = oldVerifyInstallScriptSignature
		runInstallScriptCacheOnly = oldRunInstallScriptCacheOnly
		startInstallScriptInstall = oldStartInstallScriptInstall
		exitAfterStartingUpgrade = oldExitAfterStartingUpgrade
		UpdateDownloaded = oldUpdateDownloaded
		appversion.Version = oldAppVersion
	})
}

func stageInstallScript(t *testing.T, version string) string {
	t.Helper()
	return stageInstallScriptBody(t, version, `$CacheOnly = $env:OLLAMA_CACHE_ONLY -eq "1"`)
}

func stageInstallScriptBody(t *testing.T, _ string, body string) string {
	t.Helper()

	stageDir := filepath.Join(UpdateStageDir, "etag")
	if err := os.MkdirAll(stageDir, 0o755); err != nil {
		t.Fatal(err)
	}
	script := filepath.Join(stageDir, installScriptName)
	if err := os.WriteFile(script, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return script
}

func markInstallScriptStageReady(t *testing.T, script, targetVersion string) {
	t.Helper()
	if err := writeInstallScriptStageReady(filepath.Dir(script), targetVersion); err != nil {
		t.Fatal(err)
	}
}

func envValue(env []string, key string) (string, bool) {
	prefix := strings.ToUpper(key) + "="
	for _, entry := range env {
		if strings.HasPrefix(strings.ToUpper(entry), prefix) {
			return entry[len(prefix):], true
		}
	}
	return "", false
}

func TestWithWindowsPowerShellModulePathReplacesAmbientValue(t *testing.T) {
	env := withWindowsPowerShellModulePath([]string{
		"Path=C:\\Windows",
		"PSModulePath=C:\\user-controlled",
		"psmodulepath=C:\\also-user-controlled",
	})

	var count int
	var modulePath string
	for _, entry := range env {
		key, value, ok := strings.Cut(entry, "=")
		if ok && strings.EqualFold(key, "PSModulePath") {
			count++
			modulePath = value
		}
	}
	if count != 1 {
		t.Fatalf("PSModulePath entries = %d, want 1 in %v", count, env)
	}
	if strings.Contains(strings.ToLower(modulePath), "user-controlled") {
		t.Fatalf("PSModulePath should not preserve ambient user-controlled paths: %s", modulePath)
	}
	lowerModulePath := strings.ToLower(modulePath)
	if !strings.Contains(lowerModulePath, "windowspowershell") || !strings.Contains(lowerModulePath, "modules") {
		t.Fatalf("PSModulePath = %s, want Windows PowerShell module path", modulePath)
	}
}

func TestValidateWindowsPowerShellPathReportsMissingPath(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "powershell.exe")
	err := validateWindowsPowerShellPath(missing)
	if err == nil {
		t.Fatal("expected missing Windows PowerShell path error")
	}
	if !strings.Contains(err.Error(), "Windows PowerShell not found") {
		t.Fatalf("error = %q, want clear Windows PowerShell not found message", err)
	}
	if !strings.Contains(err.Error(), missing) {
		t.Fatalf("error = %q, want missing path", err)
	}
}

func TestValidateWindowsPowerShellPathRejectsDirectory(t *testing.T) {
	dir := t.TempDir()
	err := validateWindowsPowerShellPath(dir)
	if err == nil {
		t.Fatal("expected directory path error")
	}
	if !strings.Contains(err.Error(), "Windows PowerShell path is a directory") {
		t.Fatalf("error = %q, want directory path message", err)
	}
}

func TestPowerShellSignatureVerificationScriptEncodesTargetPath(t *testing.T) {
	filename := `C:\Temp\install'; throw 'oops.ps1`
	script := powerShellSignatureVerificationScript(filename)
	if strings.Contains(script, filename) {
		t.Fatalf("verification script should not contain the raw target path:\n%s", script)
	}
	if strings.Contains(script, "$args[0]") {
		t.Fatalf("verification script should not depend on PowerShell argument binding:\n%s", script)
	}
	if !strings.Contains(script, "Get-AuthenticodeSignature -LiteralPath $target") {
		t.Fatalf("verification script should use LiteralPath:\n%s", script)
	}
}

func TestInstallScriptURLUsesUpdateDirectory(t *testing.T) {
	got, err := installScriptURL(UpdateResponse{
		UpdateURL: "https://example.com/releases/v0.21.0/OllamaSetup.exe?ignored=1",
	})
	if err != nil {
		t.Fatal(err)
	}
	if got != "https://example.com/releases/v0.21.0/install.ps1" {
		t.Fatalf("install script URL = %s", got)
	}
}

func TestInstallScriptURLAcceptsDirectInstallScriptURL(t *testing.T) {
	updateResp := UpdateResponse{
		UpdateURL:     "https://example.com/download/install.ps1",
		UpdateVersion: "v0.21.0",
	}
	got, err := installScriptURL(updateResp)
	if err != nil {
		t.Fatal(err)
	}
	if got != "https://example.com/download/install.ps1" {
		t.Fatalf("install script URL = %s", got)
	}
	if version := normalizedUpdateVersion(updateResp); version != "0.21.0" {
		t.Fatalf("version = %q, want 0.21.0", version)
	}
}

func TestInstallScriptURLRejectsInvalidUpdateURLs(t *testing.T) {
	for _, rawURL := range []string{
		"",
		"not-a-url",
		"/download/OllamaSetup.exe",
		"://missing-scheme",
		"ftp://example.com/download/OllamaSetup.exe",
		"https://user@example.com/download/OllamaSetup.exe",
	} {
		t.Run(rawURL, func(t *testing.T) {
			if got, err := installScriptURL(UpdateResponse{UpdateURL: rawURL}); err == nil {
				t.Fatalf("installScriptURL(%q) = %q, want error", rawURL, got)
			}
		})
	}
}

func TestNormalizedUpdateVersionDoesNotInferFromURL(t *testing.T) {
	updateResp := UpdateResponse{
		UpdateURL: "https://example.com/releases/download/v0.24.0/OllamaSetup.exe",
	}
	if version := normalizedUpdateVersion(updateResp); version != "" {
		t.Fatalf("version = %q, want empty when update response omits version", version)
	}
}

func TestNormalizedSemverRequiresFullVersion(t *testing.T) {
	for _, value := range []string{"", "1", "1.2", "latest"} {
		if got := normalizedSemver(value); got != "" {
			t.Errorf("normalizedSemver(%q) = %q, want empty", value, got)
		}
	}
}

func TestDownloadInstallScriptReleaseRejectsInvalidNonemptyVersion(t *testing.T) {
	updater := &Updater{}
	err := updater.downloadInstallScriptRelease(t.Context(), UpdateResponse{
		UpdateURL:     "https://example.com/releases/download/latest/OllamaSetup.exe",
		UpdateVersion: "latest",
	})
	if err == nil || !strings.Contains(err.Error(), "invalid update version") {
		t.Fatalf("error = %v, want invalid update version", err)
	}
}

func TestNewInstallScriptCacheOnlyCommandSetsVersionOutput(t *testing.T) {
	workingDir := t.TempDir()
	t.Setenv("OLLAMA_CACHE_VERSION_FILE", `C:\untrusted\cache-version`)

	command := newInstallScriptCommand("install.ps1", workingDir, "", installScriptModeCacheOnly)
	got, ok := envValue(command.Env, "OLLAMA_CACHE_VERSION_FILE")
	if !ok || got != filepath.Join(workingDir, stagedCacheVersionFile) {
		t.Fatalf("OLLAMA_CACHE_VERSION_FILE = %q, %v", got, ok)
	}
}

func TestVerifyDownloadRejectsUnsignedInstallScript(t *testing.T) {
	resetWindowsUpdaterState(t)

	t.Setenv("LOCALAPPDATA", t.TempDir())
	UpdateStageDir = t.TempDir()
	verifyInstallScriptSignature = func(filename string) error {
		return fmt.Errorf("signature verification failed")
	}
	script := stageInstallScript(t, "")
	markInstallScriptStageReady(t, script, "0.21.0")

	err := verifyDownload()
	if err == nil || !strings.Contains(err.Error(), "signature verification failed") {
		t.Fatalf("expected signature verification failure, got %v", err)
	}
}

func TestDoUpgradeAtStartupRejectsUnsignedInstallScript(t *testing.T) {
	resetWindowsUpdaterState(t)

	t.Setenv("LOCALAPPDATA", t.TempDir())
	UpdateStageDir = t.TempDir()
	runDir := t.TempDir()
	runningInstaller = filepath.Join(runDir, "OllamaSetup.exe")
	UpgradeLogFile = filepath.Join(runDir, "upgrade.log")
	UpgradeMarkerFile = filepath.Join(runDir, "upgraded")
	VerifyDownload = verifyDownload
	verifyInstallScriptSignature = func(filename string) error {
		return fmt.Errorf("signature verification failed")
	}

	script := stageInstallScript(t, "")
	markInstallScriptStageReady(t, script, "0.21.0")
	startInstallScriptInstall = func(command installScriptCommand) (installScriptProcess, error) {
		t.Fatal("installer should not start after signature verification fails")
		return nil, nil
	}

	err := DoUpgradeAtStartup()
	if err == nil || !strings.Contains(err.Error(), "signature verification failed") {
		t.Fatalf("expected signature verification failure, got %v", err)
	}
	if _, err := os.Stat(script); !os.IsNotExist(err) {
		t.Fatalf("unsigned staged install script was not removed after verification failure: %v", err)
	}
}

func TestVerifyDownloadRejectsInstallScriptWithoutCacheOnlySupport(t *testing.T) {
	resetWindowsUpdaterState(t)

	t.Setenv("LOCALAPPDATA", t.TempDir())
	UpdateStageDir = t.TempDir()
	VerifyDownload = verifyDownload
	verifyInstallScriptSignature = func(filename string) error {
		return nil
	}
	script := stageInstallScriptBody(t, "0.21.0", "# signed old install.ps1")
	markInstallScriptStageReady(t, script, "0.21.0")

	err := VerifyDownload()
	if err == nil || !strings.Contains(err.Error(), "does not support cache-only mode") {
		t.Fatalf("expected cache-only support failure, got %v", err)
	}
}

func TestDownloadNewReleaseCleansStagedScriptWhenSignatureFails(t *testing.T) {
	resetWindowsUpdaterState(t)

	UpdateStageDir = t.TempDir()
	UpdateDownloaded = false
	t.Setenv("LOCALAPPDATA", t.TempDir())

	verifyInstallScriptSignature = func(filename string) error {
		if filepath.Base(filename) != installScriptName {
			t.Fatalf("expected to verify install.ps1, got %s", filename)
		}
		return fmt.Errorf("bad signature")
	}
	runInstallScriptCacheOnly = func(ctx context.Context, scriptPath, cacheDir, version string) (string, error) {
		t.Fatal("cache-only phase should not run after signature failure")
		return "", nil
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v0.21.0/install.ps1" {
			t.Errorf("unexpected request path %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.Header().Set("ETag", `"script-etag"`)
		w.WriteHeader(http.StatusOK)
		if r.Method == http.MethodGet {
			_, _ = w.Write([]byte(`$CacheOnly = $env:OLLAMA_CACHE_ONLY -eq "1"`))
		}
	}))
	defer server.Close()

	updater := &Updater{}
	err := updater.DownloadNewRelease(t.Context(), UpdateResponse{
		UpdateURL:     server.URL + "/v0.21.0/OllamaSetup.exe",
		UpdateVersion: "v0.21.0",
	})
	if err == nil || !strings.Contains(err.Error(), "install.ps1 signature verification failed") {
		t.Fatalf("expected signature verification failure, got %v", err)
	}
	if UpdateDownloaded {
		t.Fatal("UpdateDownloaded should remain false after signature failure")
	}
	entries, err := os.ReadDir(UpdateStageDir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("staged script cache was not cleaned up: %v", entries)
	}
}

func TestDownloadNewReleaseDoesNotStagePartialInstallScript(t *testing.T) {
	resetWindowsUpdaterState(t)

	UpdateStageDir = t.TempDir()
	UpdateDownloaded = false

	verifyInstallScriptSignature = func(filename string) error {
		t.Fatal("partial install.ps1 should not be verified")
		return nil
	}
	runInstallScriptCacheOnly = func(ctx context.Context, scriptPath, cacheDir, version string) (string, error) {
		t.Fatal("partial install.ps1 should not run")
		return "", nil
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v0.21.0/install.ps1" {
			t.Errorf("unexpected request path %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.Header().Set("ETag", `"script-etag"`)
		switch r.Method {
		case http.MethodHead:
			w.WriteHeader(http.StatusOK)
		case http.MethodGet:
			w.Header().Set("Content-Length", "1024")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`$CacheOnly = $env:OLLAMA_CACHE_ONLY -eq "1"`))
			if flusher, ok := w.(http.Flusher); ok {
				flusher.Flush()
			}
		default:
			t.Errorf("unexpected method %s", r.Method)
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
	}))
	defer server.Close()

	updater := &Updater{}
	err := updater.DownloadNewRelease(t.Context(), UpdateResponse{
		UpdateURL:     server.URL + "/v0.21.0/OllamaSetup.exe",
		UpdateVersion: "v0.21.0",
	})
	if err == nil {
		t.Fatal("expected partial install.ps1 download to fail")
	}
	if UpdateDownloaded {
		t.Fatal("UpdateDownloaded should remain false after a partial install.ps1 download")
	}
	if getStagedInstallScript() != "" {
		t.Fatal("partial install.ps1 should not be considered pending")
	}
}

func TestDownloadNewReleaseRejectsOversizedInstallScript(t *testing.T) {
	resetWindowsUpdaterState(t)

	UpdateStageDir = t.TempDir()
	verifyInstallScriptSignature = func(string) error {
		t.Fatal("oversized install.ps1 should not be verified")
		return nil
	}
	runInstallScriptCacheOnly = func(context.Context, string, string, string) (string, error) {
		t.Fatal("oversized install.ps1 should not run")
		return "", nil
	}

	content := strings.Repeat("x", maxInstallScriptSize+1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v0.21.0/install.ps1" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("ETag", `"oversized"`)
		w.Header().Set("Content-Length", fmt.Sprint(len(content)))
		if r.Method != http.MethodHead {
			_, _ = io.WriteString(w, content)
		}
	}))
	defer server.Close()

	updater := &Updater{}
	err := updater.DownloadNewRelease(t.Context(), UpdateResponse{
		UpdateURL:     server.URL + "/v0.21.0/OllamaSetup.exe",
		UpdateVersion: "0.21.0",
	})
	if err == nil || !strings.Contains(err.Error(), "exceeds maximum size") {
		t.Fatalf("expected oversized install.ps1 error, got %v", err)
	}
	if entries, readErr := os.ReadDir(UpdateStageDir); readErr != nil || len(entries) != 0 {
		t.Fatalf("staging dir after oversized download = %v, %v; want empty", entries, readErr)
	}
}

func TestDownloadNewReleaseUsesCompletedCachedInstallScript(t *testing.T) {
	resetWindowsUpdaterState(t)

	UpdateStageDir = t.TempDir()
	UpdateDownloaded = false
	t.Setenv("LOCALAPPDATA", t.TempDir())

	var getRequested atomic.Bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v0.21.0/install.ps1" {
			t.Errorf("unexpected request path %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.Header().Set("ETag", `"script-etag"`)
		switch r.Method {
		case http.MethodHead:
			w.WriteHeader(http.StatusOK)
		case http.MethodGet:
			getRequested.Store(true)
			w.WriteHeader(http.StatusOK)
		default:
			t.Errorf("unexpected method %s", r.Method)
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
	}))
	defer server.Close()

	updateResp := UpdateResponse{
		UpdateURL:     server.URL + "/v0.21.0/OllamaSetup.exe",
		UpdateVersion: "v0.21.0",
	}
	stageKey := strings.Join([]string{
		`"script-etag"`,
		"0.21.0",
		updateResp.UpdateURL,
	}, "\n")
	scriptPath, err := updateStagePath(UpdateStageDir, stageKey, installScriptName)
	if err != nil {
		t.Fatal(err)
	}
	cacheDir := filepath.Dir(scriptPath)
	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(scriptPath, []byte(`$CacheOnly = $env:OLLAMA_CACHE_ONLY -eq "1"`), 0o644); err != nil {
		t.Fatal(err)
	}
	markInstallScriptStageReady(t, scriptPath, "0.21.0")

	verifyInstallScriptSignature = func(filename string) error {
		if filename != scriptPath {
			t.Fatalf("verified script = %s, want %s", filename, scriptPath)
		}
		return nil
	}
	runInstallScriptCacheOnly = func(ctx context.Context, scriptPath, cacheDir, version string) (string, error) {
		t.Fatal("cache-only phase should not rerun for completed cached update")
		return "", nil
	}

	updater := &Updater{}
	if err := updater.DownloadNewRelease(t.Context(), updateResp); err != nil {
		t.Fatal(err)
	}
	if getRequested.Load() {
		t.Fatal("completed cached install.ps1 should not be downloaded again")
	}
	if !UpdateDownloaded {
		t.Fatal("UpdateDownloaded should be true for completed cached update")
	}
}

func TestDownloadNewReleaseRerunsCacheOnlyForNewVersionWhenInstallScriptETagUnchanged(t *testing.T) {
	resetWindowsUpdaterState(t)

	UpdateStageDir = t.TempDir()
	UpdateDownloaded = false
	t.Setenv("LOCALAPPDATA", t.TempDir())
	var scriptGetCount atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v0.22.0/install.ps1" {
			t.Errorf("unexpected request path %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.Header().Set("ETag", `"same-script-etag"`)
		switch r.Method {
		case http.MethodHead:
			w.WriteHeader(http.StatusOK)
		case http.MethodGet:
			scriptGetCount.Add(1)
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`$CacheOnly = $env:OLLAMA_CACHE_ONLY -eq "1"`))
		default:
			t.Errorf("unexpected method %s", r.Method)
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
	}))
	defer server.Close()

	oldUpdateResp := UpdateResponse{
		UpdateURL:     server.URL + "/v0.21.0/OllamaSetup.exe",
		UpdateVersion: "v0.21.0",
	}
	oldStageKey := strings.Join([]string{
		`"same-script-etag"`,
		"0.21.0",
		oldUpdateResp.UpdateURL,
	}, "\n")
	oldScriptPath, err := updateStagePath(UpdateStageDir, oldStageKey, installScriptName)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(oldScriptPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(oldScriptPath, []byte(`$CacheOnly = $env:OLLAMA_CACHE_ONLY -eq "1"`), 0o644); err != nil {
		t.Fatal(err)
	}
	markInstallScriptStageReady(t, oldScriptPath, "0.21.0")

	verifyInstallScriptSignature = func(filename string) error {
		return nil
	}
	var cacheOnlyRuns atomic.Int32
	var recordedVersion string
	runInstallScriptCacheOnly = func(ctx context.Context, scriptPath, cacheDir, version string) (string, error) {
		cacheOnlyRuns.Add(1)
		recordedVersion = version
		if scriptPath == oldScriptPath {
			t.Fatal("new update should not reuse the old staged script path")
		}
		return "0.22.0", nil
	}

	newUpdateResp := UpdateResponse{
		UpdateURL:     server.URL + "/v0.22.0/OllamaSetup.exe",
		UpdateVersion: "v0.22.0",
	}
	updater := &Updater{}
	if err := updater.DownloadNewRelease(t.Context(), newUpdateResp); err != nil {
		t.Fatal(err)
	}

	if scriptGetCount.Load() != 1 {
		t.Fatalf("expected install.ps1 to be staged for the new target, got %d GETs", scriptGetCount.Load())
	}
	if cacheOnlyRuns.Load() != 1 {
		t.Fatalf("cache-only should rerun for the new target even with an existing payload, got %d", cacheOnlyRuns.Load())
	}
	if recordedVersion != "0.22.0" {
		t.Fatalf("version = %q, want 0.22.0", recordedVersion)
	}
	if !UpdateDownloaded {
		t.Fatal("UpdateDownloaded should be true after recaching the newer target")
	}
	if _, err := os.Stat(oldScriptPath); !os.IsNotExist(err) {
		t.Fatalf("old staged script should be removed before recaching newer target: %v", err)
	}
}

func TestDownloadNewReleaseStagesInstallScriptAndRunsCacheOnly(t *testing.T) {
	resetWindowsUpdaterState(t)

	UpdateStageDir = t.TempDir()
	UpdateDownloaded = false
	t.Setenv("LOCALAPPDATA", t.TempDir())
	var exeGetRequested atomic.Bool
	var scriptHeadCount atomic.Int32
	var scriptGetCount atomic.Int32
	var recordedScript string
	var recordedCache string
	var recordedVersion string
	verifyInstallScriptSignature = func(filename string) error {
		if filepath.Base(filename) != installScriptName {
			t.Fatalf("expected to verify install.ps1, got %s", filename)
		}
		return nil
	}
	runInstallScriptCacheOnly = func(ctx context.Context, scriptPath, cacheDir, version string) (string, error) {
		recordedScript = scriptPath
		recordedCache = cacheDir
		recordedVersion = version
		return "0.21.0", nil
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v0.21.0/install.ps1":
			w.Header().Set("ETag", `"script-etag"`)
			switch r.Method {
			case http.MethodHead:
				scriptHeadCount.Add(1)
				w.WriteHeader(http.StatusOK)
			case http.MethodGet:
				scriptGetCount.Add(1)
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(`$CacheOnly = $env:OLLAMA_CACHE_ONLY -eq "1"`))
			default:
				t.Errorf("unexpected method %s", r.Method)
				w.WriteHeader(http.StatusMethodNotAllowed)
			}
		case "/v0.21.0/OllamaSetup.exe":
			w.Header().Set("ETag", `"exe-etag"`)
			switch r.Method {
			case http.MethodHead:
				t.Errorf("app should let install.ps1 check the installer ETag")
				w.WriteHeader(http.StatusMethodNotAllowed)
			case http.MethodGet:
				exeGetRequested.Store(true)
				w.WriteHeader(http.StatusOK)
			default:
				t.Errorf("unexpected method %s", r.Method)
				w.WriteHeader(http.StatusMethodNotAllowed)
			}
		default:
			t.Errorf("unexpected request path %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()

	updater := &Updater{}
	err := updater.DownloadNewRelease(t.Context(), UpdateResponse{
		UpdateURL:     server.URL + "/v0.21.0/OllamaSetup.exe",
		UpdateVersion: "v0.21.0",
	})
	if err != nil {
		t.Fatal(err)
	}

	if exeGetRequested.Load() {
		t.Fatal("app cache phase should let install.ps1 download OllamaSetup.exe")
	}
	if scriptHeadCount.Load() != 1 || scriptGetCount.Load() != 1 {
		t.Fatalf("expected one HEAD and one GET for install.ps1, got HEAD=%d GET=%d", scriptHeadCount.Load(), scriptGetCount.Load())
	}
	if recordedScript == "" || filepath.Base(recordedScript) != installScriptName {
		t.Fatalf("cache-only script not recorded correctly: %s", recordedScript)
	}
	if recordedCache != filepath.Dir(recordedScript) {
		t.Fatalf("cache dir = %s, want %s", recordedCache, filepath.Dir(recordedScript))
	}
	if recordedVersion != "0.21.0" {
		t.Fatalf("version = %q, want 0.21.0", recordedVersion)
	}
	if _, err := os.Stat(filepath.Join(recordedCache, installScriptName)); err != nil {
		t.Fatalf("final install.ps1 missing after cache-only success: %v", err)
	}
	if _, err := os.Stat(filepath.Join(recordedCache, stagedCacheReadyFile)); err != nil {
		t.Fatalf("cache-ready marker missing after cache-only success: %v", err)
	}
	if !UpdateDownloaded {
		t.Fatal("UpdateDownloaded should be true after staged script download succeeds")
	}
}

func TestDownloadNewReleaseRejectsInstallScriptWithoutCacheOnlySupport(t *testing.T) {
	resetWindowsUpdaterState(t)

	UpdateStageDir = t.TempDir()
	UpdateDownloaded = false

	var downloadOnlyRan atomic.Bool
	verifyInstallScriptSignature = func(filename string) error {
		if filepath.Base(filename) != installScriptName {
			t.Fatalf("expected to verify install.ps1, got %s", filename)
		}
		return nil
	}
	runInstallScriptCacheOnly = func(ctx context.Context, scriptPath, cacheDir, version string) (string, error) {
		downloadOnlyRan.Store(true)
		return "0.21.0", nil
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v0.21.0/install.ps1":
			w.Header().Set("ETag", `"old-script-etag"`)
			w.WriteHeader(http.StatusOK)
			if r.Method == http.MethodGet {
				_, _ = w.Write([]byte("# signed old install.ps1"))
			}
		default:
			t.Errorf("unexpected request path %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()

	updater := &Updater{}
	err := updater.DownloadNewRelease(t.Context(), UpdateResponse{
		UpdateURL:     server.URL + "/v0.21.0/OllamaSetup.exe",
		UpdateVersion: "v0.21.0",
	})
	if err == nil || !strings.Contains(err.Error(), "install.ps1 does not support cache-only mode") {
		t.Fatalf("expected unsupported install.ps1 error, got %v", err)
	}

	if downloadOnlyRan.Load() {
		t.Fatal("install.ps1 cache-only phase should not run without support marker")
	}
	if getStagedInstallScript() != "" {
		t.Fatal("unsupported install.ps1 should not remain staged")
	}
	if UpdateDownloaded {
		t.Fatal("UpdateDownloaded should stay false when install.ps1 lacks cache-only support")
	}
}

func TestDownloadNewReleaseDoesNotMarkDownloadedWhenCacheOnlyFails(t *testing.T) {
	resetWindowsUpdaterState(t)

	UpdateStageDir = t.TempDir()
	UpdateDownloaded = false

	verifyInstallScriptSignature = func(filename string) error {
		return nil
	}
	runInstallScriptCacheOnly = func(ctx context.Context, scriptPath, cacheDir, version string) (string, error) {
		return "", fmt.Errorf("download failed")
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v0.21.0/OllamaSetup.exe" && r.Method == http.MethodHead {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		if r.URL.Path != "/v0.21.0/install.ps1" {
			t.Errorf("unexpected request path %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.Header().Set("ETag", `"script-etag"`)
		w.WriteHeader(http.StatusOK)
		if r.Method == http.MethodGet {
			_, _ = w.Write([]byte(`$CacheOnly = $env:OLLAMA_CACHE_ONLY -eq "1"`))
		}
	}))
	defer server.Close()

	updater := &Updater{}
	err := updater.DownloadNewRelease(t.Context(), UpdateResponse{
		UpdateURL:     server.URL + "/v0.21.0/OllamaSetup.exe",
		UpdateVersion: "v0.21.0",
	})
	if err == nil || !strings.Contains(err.Error(), "download failed") {
		t.Fatalf("expected download phase failure, got %v", err)
	}
	if UpdateDownloaded {
		t.Fatal("UpdateDownloaded should remain false when cache-only fails")
	}
	if getStagedInstallScript() != "" {
		t.Fatal("incomplete staged update should not be pending")
	}
	entries, err := os.ReadDir(UpdateStageDir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("failed cache-only stage was not cleaned up: %v", entries)
	}
}

func TestDownloadNewReleaseRejectsPayloadVersionMismatch(t *testing.T) {
	resetWindowsUpdaterState(t)
	UpdateStageDir = t.TempDir()
	UpdateDownloaded = false

	verifyInstallScriptSignature = func(filename string) error { return nil }
	runInstallScriptCacheOnly = func(ctx context.Context, scriptPath, cacheDir, version string) (string, error) {
		return "0.22.0", nil
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v0.21.0/install.ps1" {
			t.Errorf("unexpected request path %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.Header().Set("ETag", `"script-etag"`)
		w.WriteHeader(http.StatusOK)
		if r.Method == http.MethodGet {
			_, _ = w.Write([]byte(`$CacheOnly = $env:OLLAMA_CACHE_ONLY -eq "1"`))
		}
	}))
	defer server.Close()

	updater := &Updater{}
	err := updater.DownloadNewRelease(t.Context(), UpdateResponse{
		UpdateURL:     server.URL + "/v0.21.0/OllamaSetup.exe",
		UpdateVersion: "v0.21.0",
	})
	if err == nil || !strings.Contains(err.Error(), "cached installer version 0.22.0 does not match update version 0.21.0") {
		t.Fatalf("error = %v, want payload version mismatch", err)
	}
	if UpdateDownloaded {
		t.Fatal("UpdateDownloaded should remain false after a version mismatch")
	}
	entries, err := os.ReadDir(UpdateStageDir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("mismatched update stage was not cleaned up: %v", entries)
	}
}

func TestDownloadNewReleaseMarksReadyAfterCacheOnlySucceeds(t *testing.T) {
	resetWindowsUpdaterState(t)

	UpdateStageDir = t.TempDir()
	UpdateDownloaded = false
	t.Setenv("LOCALAPPDATA", t.TempDir())

	verifyInstallScriptSignature = func(filename string) error {
		return nil
	}
	var requestedVersion string
	runInstallScriptCacheOnly = func(ctx context.Context, scriptPath, cacheDir, version string) (string, error) {
		requestedVersion = version
		return "0.21.0", nil
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v0.21.0/install.ps1" {
			t.Errorf("unexpected request path %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.Header().Set("ETag", `"script-etag"`)
		w.WriteHeader(http.StatusOK)
		if r.Method == http.MethodGet {
			_, _ = w.Write([]byte(`$CacheOnly = $env:OLLAMA_CACHE_ONLY -eq "1"`))
		}
	}))
	defer server.Close()

	updater := &Updater{}
	err := updater.DownloadNewRelease(t.Context(), UpdateResponse{
		UpdateURL: server.URL + "/v0.21.0/OllamaSetup.exe",
	})
	if err != nil {
		t.Fatalf("cache-only success should be enough for app readiness: %v", err)
	}
	if !UpdateDownloaded {
		t.Fatal("UpdateDownloaded should be true after cache-only succeeds")
	}
	if requestedVersion != "" {
		t.Fatalf("cache-only requested version = %q, want latest", requestedVersion)
	}
	if getStagedInstallScript() == "" {
		t.Fatal("staged install.ps1 should be ready after cache-only succeeds")
	}
	staged := getStagedInstallScript()
	stagedVersion, err := readInstallScriptStageVersion(filepath.Dir(staged))
	if err != nil {
		t.Fatal(err)
	}
	if stagedVersion != "0.21.0" {
		t.Fatalf("staged version = %q, want 0.21.0", stagedVersion)
	}
}

func TestIsUpdatePendingRequiresReadyInstallScript(t *testing.T) {
	resetWindowsUpdaterState(t)

	UpdateStageDir = t.TempDir()
	appversion.Version = "0.20.0"
	stageDir := filepath.Join(UpdateStageDir, "etag")
	if err := os.MkdirAll(stageDir, 0o755); err != nil {
		t.Fatal(err)
	}
	script := filepath.Join(stageDir, installScriptName)
	if err := os.WriteFile(script, []byte("script"), 0o644); err != nil {
		t.Fatal(err)
	}

	if IsUpdatePending() {
		t.Fatal("update should not be pending without a cache-ready marker")
	}
	markInstallScriptStageReady(t, script, "0.21.0")
	if !IsUpdatePending() {
		t.Fatal("update should be pending with a staged script and cache-ready marker")
	}
}

func TestIsUpdatePendingDiscardsStagesAlreadySatisfiedByRunningApp(t *testing.T) {
	for _, test := range []struct {
		name           string
		currentVersion string
		targetVersion  string
	}{
		{name: "same release", currentVersion: "0.21.0", targetVersion: "0.21.0"},
		{name: "newer release", currentVersion: "0.22.0", targetVersion: "0.21.0"},
		{name: "release supersedes prerelease", currentVersion: "0.21.0", targetVersion: "0.21.0-rc1"},
		{name: "same precedence with build metadata", currentVersion: "0.21.0+new", targetVersion: "0.21.0+old"},
	} {
		t.Run(test.name, func(t *testing.T) {
			resetWindowsUpdaterState(t)
			UpdateStageDir = t.TempDir()
			appversion.Version = test.currentVersion

			script := stageInstallScript(t, test.targetVersion)
			markInstallScriptStageReady(t, script, test.targetVersion)
			if IsUpdatePending() {
				t.Fatalf("running version %s should satisfy staged version %s", test.currentVersion, test.targetVersion)
			}
			if _, err := os.Stat(filepath.Dir(script)); !os.IsNotExist(err) {
				t.Fatalf("obsolete stage was not removed: %v", err)
			}
		})
	}
}

func TestIsUpdatePendingKeepsNewerStage(t *testing.T) {
	resetWindowsUpdaterState(t)
	UpdateStageDir = t.TempDir()
	appversion.Version = "0.21.0-rc1"

	script := stageInstallScript(t, "0.21.0")
	markInstallScriptStageReady(t, script, "0.21.0")
	if !IsUpdatePending() {
		t.Fatal("release update should remain pending for an older prerelease app")
	}
	if _, err := os.Stat(script); err != nil {
		t.Fatalf("newer staged update was removed: %v", err)
	}
}

func TestIsUpdatePendingDiscardsInvalidStageMetadata(t *testing.T) {
	resetWindowsUpdaterState(t)
	UpdateStageDir = t.TempDir()
	appversion.Version = "0.20.0"

	script := stageInstallScript(t, "")
	readyFile := filepath.Join(filepath.Dir(script), stagedCacheReadyFile)
	if err := os.WriteFile(readyFile, []byte("1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if IsUpdatePending() {
		t.Fatal("stage with invalid version metadata should not be pending")
	}
	if _, err := os.Stat(filepath.Dir(script)); !os.IsNotExist(err) {
		t.Fatalf("invalid stage was not removed: %v", err)
	}
}

func TestIsUpdatePendingKeepsStageWhenRunningVersionIsInvalid(t *testing.T) {
	resetWindowsUpdaterState(t)
	UpdateStageDir = t.TempDir()
	appversion.Version = "development"

	script := stageInstallScript(t, "0.21.0")
	markInstallScriptStageReady(t, script, "0.21.0")
	if !IsUpdatePending() {
		t.Fatal("unknown running version should not discard a valid staged update")
	}
}

func TestDoUpgradeAtStartupRunsInstallScriptInstallCached(t *testing.T) {
	resetWindowsUpdaterState(t)

	t.Setenv("OLLAMA_CACHE_ONLY", "1")
	t.Setenv("OLLAMA_DEBUG", "1")
	t.Setenv("OLLAMA_INSTALL_CACHED", "1")
	t.Setenv("OLLAMA_INSTALL_DIR", "C:\\bad")
	t.Setenv("OLLAMA_UNINSTALL", "1")
	t.Setenv("OLLAMA_VERSION", "bad-version")
	t.Setenv("PSModulePath", "C:\\user-controlled")
	t.Setenv("LOCALAPPDATA", t.TempDir())

	UpdateStageDir = t.TempDir()
	runDir := t.TempDir()
	UpgradeLogFile = filepath.Join(runDir, "upgrade.log")
	UpgradeMarkerFile = filepath.Join(runDir, "upgraded")
	VerifyDownload = verifyDownload
	verifyInstallScriptSignature = func(filename string) error {
		return nil
	}

	script := stageInstallScript(t, "0.21.0")
	markInstallScriptStageReady(t, script, "0.21.0")
	var recorded installScriptCommand
	process := &fakeInstallScriptProcess{}
	startInstallScriptInstall = func(command installScriptCommand) (installScriptProcess, error) {
		recorded = command
		return process, nil
	}
	var exitCode *int
	exitAfterStartingUpgrade = func(code int) {
		exitCode = &code
	}

	if err := DoUpgradeAtStartup(); err != nil {
		t.Fatal(err)
	}

	if exitCode == nil || *exitCode != 0 {
		t.Fatalf("exit code = %v, want 0", exitCode)
	}
	if !process.released.Load() {
		t.Fatal("install script process was not released")
	}
	if recorded.Program != windowsPowerShellPath() {
		t.Fatalf("program = %s", recorded.Program)
	}
	if strings.Join(recorded.Args, " ") != fmt.Sprintf("-NoProfile -ExecutionPolicy %s -File %s", installScriptExecutionPolicy, script) {
		t.Fatalf("unexpected args: %v", recorded.Args)
	}
	if recorded.Dir != filepath.Dir(script) {
		t.Fatalf("dir = %s, want %s", recorded.Dir, filepath.Dir(script))
	}
	if recorded.CreationFlags&windowsCreateNoWindow == 0 {
		t.Fatalf("creation flags = %#x, want CREATE_NO_WINDOW", recorded.CreationFlags)
	}
	if got, ok := envValue(recorded.Env, "OLLAMA_INSTALL_CACHED"); !ok || got != "1" {
		t.Fatalf("OLLAMA_INSTALL_CACHED = %q, %v", got, ok)
	}
	if _, ok := envValue(recorded.Env, "OLLAMA_CACHE_ONLY"); ok {
		t.Fatal("OLLAMA_CACHE_ONLY should not be passed to install-cached phase")
	}
	if _, ok := envValue(recorded.Env, "OLLAMA_DEBUG"); ok {
		t.Fatal("OLLAMA_DEBUG should not be passed to install.ps1")
	}
	if _, ok := envValue(recorded.Env, "OLLAMA_INSTALL_DIR"); ok {
		t.Fatal("OLLAMA_INSTALL_DIR should not be passed to install.ps1")
	}
	if _, ok := envValue(recorded.Env, "OLLAMA_UNINSTALL"); ok {
		t.Fatal("OLLAMA_UNINSTALL should not be passed to install.ps1")
	}
	if _, ok := envValue(recorded.Env, "OLLAMA_VERSION"); ok {
		t.Fatal("OLLAMA_VERSION should not be passed to install-cached phase")
	}
	modulePath, ok := envValue(recorded.Env, "PSModulePath")
	if !ok {
		t.Fatal("PSModulePath should be set for Windows PowerShell")
	}
	if strings.Contains(strings.ToLower(modulePath), "user-controlled") {
		t.Fatalf("PSModulePath should not preserve ambient user-controlled paths: %s", modulePath)
	}
	if !strings.Contains(strings.ToLower(modulePath), "windowspowershell") {
		t.Fatalf("PSModulePath = %s, want Windows PowerShell module path", modulePath)
	}
	if _, err := os.Stat(UpgradeMarkerFile); err != nil {
		t.Fatalf("upgrade marker missing: %v", err)
	}
	if _, err := os.Stat(filepath.Join(filepath.Dir(script), stagedCacheReadyFile)); !os.IsNotExist(err) {
		t.Fatalf("cache-ready marker should be consumed before launching install.ps1: %v", err)
	}
}

func TestDoUpgradeInteractiveShowsInstallerProgress(t *testing.T) {
	resetWindowsUpdaterState(t)

	t.Setenv("LOCALAPPDATA", t.TempDir())
	UpdateStageDir = t.TempDir()
	UpgradeMarkerFile = filepath.Join(t.TempDir(), "upgraded")
	VerifyDownload = verifyDownload
	verifyInstallScriptSignature = func(filename string) error { return nil }

	script := stageInstallScript(t, "0.21.0")
	markInstallScriptStageReady(t, script, "0.21.0")
	var recorded installScriptCommand
	startInstallScriptInstall = func(command installScriptCommand) (installScriptProcess, error) {
		recorded = command
		return &fakeInstallScriptProcess{}, nil
	}
	exitAfterStartingUpgrade = func(int) {}

	if err := DoUpgrade(true); err != nil {
		t.Fatal(err)
	}
	want := fmt.Sprintf("-NoProfile -ExecutionPolicy %s -File %s -ShowProgress", installScriptExecutionPolicy, script)
	if strings.Join(recorded.Args, " ") != want {
		t.Fatalf("unexpected args: %v", recorded.Args)
	}
}

func TestDoUpgradeAtStartupIgnoresInstallScriptWithoutCacheReady(t *testing.T) {
	resetWindowsUpdaterState(t)

	UpdateStageDir = t.TempDir()
	t.Setenv("LOCALAPPDATA", t.TempDir())
	UpgradeMarkerFile = filepath.Join(t.TempDir(), "upgraded")
	VerifyDownload = verifyDownload
	verifyInstallScriptSignature = func(filename string) error {
		return nil
	}
	stageInstallScript(t, "")

	startInstallScriptInstall = func(command installScriptCommand) (installScriptProcess, error) {
		t.Fatal("installer should not start without cache-ready")
		return nil, nil
	}
	exitAfterStartingUpgrade = func(code int) {
		t.Fatalf("upgrade should not exit without cache-ready, got %d", code)
	}

	err := DoUpgradeAtStartup()
	if err == nil || !strings.Contains(err.Error(), "failed to lookup downloads") {
		t.Fatalf("expected incomplete staged update to be ignored, got %v", err)
	}
}

func TestDoUpgradeAtStartupClearsCacheReadyWhenInstallScriptStartFails(t *testing.T) {
	resetWindowsUpdaterState(t)

	t.Setenv("LOCALAPPDATA", t.TempDir())
	UpdateStageDir = t.TempDir()
	runDir := t.TempDir()
	UpgradeMarkerFile = filepath.Join(runDir, "upgraded")
	VerifyDownload = verifyDownload
	verifyInstallScriptSignature = func(filename string) error {
		return nil
	}

	script := stageInstallScript(t, "0.21.0")
	markInstallScriptStageReady(t, script, "0.21.0")
	startInstallScriptInstall = func(command installScriptCommand) (installScriptProcess, error) {
		return nil, fmt.Errorf("start failed")
	}
	exitAfterStartingUpgrade = func(code int) {
		t.Fatalf("upgrade should not exit when install.ps1 fails to start, got %d", code)
	}

	err := DoUpgradeAtStartup()
	if err == nil || !strings.Contains(err.Error(), "unable to start install.ps1 upgrade") {
		t.Fatalf("expected install.ps1 start failure, got %v", err)
	}
	if _, err := os.Stat(filepath.Join(filepath.Dir(script), stagedCacheReadyFile)); !os.IsNotExist(err) {
		t.Fatalf("cache-ready marker should be consumed after start failure: %v", err)
	}
	if _, err := os.Stat(UpgradeMarkerFile); !os.IsNotExist(err) {
		t.Fatalf("upgrade marker should be removed after start failure: %v", err)
	}
	if IsUpdatePending() {
		t.Fatal("failed install.ps1 launch should not stay pending without a refreshed cache")
	}
}

func TestDoPostUpgradeCleanupRemovesStagedUpdateAndMarkers(t *testing.T) {
	resetWindowsUpdaterState(t)

	localAppData := t.TempDir()
	t.Setenv("LOCALAPPDATA", localAppData)
	UpdateStageDir = t.TempDir()
	runDir := t.TempDir()
	runningInstaller = filepath.Join(runDir, Installer)
	UpgradeMarkerFile = filepath.Join(runDir, "upgraded")

	script := stageInstallScript(t, "0.21.0")
	markInstallScriptStageReady(t, script, "0.21.0")
	if err := os.WriteFile(runningInstaller, []byte("running"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(UpgradeMarkerFile), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(UpgradeMarkerFile, []byte("1"), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := DoPostUpgradeCleanup(); err != nil {
		t.Fatal(err)
	}

	if _, err := os.Stat(script); !os.IsNotExist(err) {
		t.Fatalf("staged install script was not removed: %v", err)
	}
	if _, err := os.Stat(UpgradeMarkerFile); !os.IsNotExist(err) {
		t.Fatalf("upgrade marker was not removed: %v", err)
	}
	if _, err := os.Stat(runningInstaller); !os.IsNotExist(err) {
		t.Fatalf("running installer was not removed: %v", err)
	}
}

func TestDoPostUpgradeCleanupLogsInstallerFailure(t *testing.T) {
	resetWindowsUpdaterState(t)

	localAppData := t.TempDir()
	t.Setenv("LOCALAPPDATA", localAppData)
	UpdateStageDir = t.TempDir()
	runDir := filepath.Join(localAppData, "Ollama")
	runningInstaller = filepath.Join(runDir, Installer)
	UpgradeLogFile = filepath.Join(runDir, "upgrade.log")
	installScriptInstallerLogFile = filepath.Join(runDir, "OllamaSetup.log")
	installScriptInstallerLogFileGlob = filepath.Join(runDir, "OllamaInstaller-*.log")
	UpgradeMarkerFile = filepath.Join(runDir, "upgraded")

	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(UpgradeMarkerFile, []byte("1"), 0o644); err != nil {
		t.Fatal(err)
	}
	markerTime := time.Now().Add(-2 * time.Minute)
	if err := os.Chtimes(UpgradeMarkerFile, markerTime, markerTime); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(installScriptInstallerLogFile, []byte("Installation process failed."), 0o644); err != nil {
		t.Fatal(err)
	}

	var logs bytes.Buffer
	previousLogger := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug})))
	t.Cleanup(func() {
		slog.SetDefault(previousLogger)
	})

	if err := DoPostUpgradeCleanup(); err != nil {
		t.Fatal(err)
	}

	logText := logs.String()
	if !strings.Contains(logText, "Windows installer reported upgrade failure") {
		t.Fatalf("expected installer failure warning, got logs:\n%s", logText)
	}
	if !strings.Contains(logText, installScriptInstallerLogFile) {
		t.Fatalf("expected installer log path in warning, got logs:\n%s", logText)
	}
}

func TestDoPostUpgradeCleanupLogsMSIInstallerFailure(t *testing.T) {
	resetWindowsUpdaterState(t)

	localAppData := t.TempDir()
	t.Setenv("LOCALAPPDATA", localAppData)
	UpdateStageDir = t.TempDir()
	runDir := filepath.Join(localAppData, "Ollama")
	runningInstaller = filepath.Join(runDir, Installer)
	UpgradeLogFile = filepath.Join(runDir, "upgrade.log")
	installScriptInstallerLogFile = filepath.Join(runDir, "OllamaSetup.log")
	installScriptInstallerLogFileGlob = filepath.Join(runDir, "OllamaInstaller-*.log")
	UpgradeMarkerFile = filepath.Join(runDir, "upgraded")

	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(UpgradeMarkerFile, []byte("1"), 0o644); err != nil {
		t.Fatal(err)
	}
	markerTime := time.Now().Add(-2 * time.Minute)
	if err := os.Chtimes(UpgradeMarkerFile, markerTime, markerTime); err != nil {
		t.Fatal(err)
	}
	msiLog := filepath.Join(runDir, "OllamaInstaller-install-ollama-core.log")
	if err := os.WriteFile(msiLog, []byte("Action ended: InstallFinalize. Return value 3."), 0o644); err != nil {
		t.Fatal(err)
	}

	var logs bytes.Buffer
	previousLogger := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug})))
	t.Cleanup(func() {
		slog.SetDefault(previousLogger)
	})

	if err := DoPostUpgradeCleanup(); err != nil {
		t.Fatal(err)
	}

	logText := logs.String()
	if !strings.Contains(logText, "Windows installer reported upgrade failure") {
		t.Fatalf("expected installer failure warning, got logs:\n%s", logText)
	}
	if !strings.Contains(logText, msiLog) {
		t.Fatalf("expected MSI log path in warning, got logs:\n%s", logText)
	}
}

func TestWindowsInstallerLogIndicatesFailureIgnoresStaleAndSuccessfulLogs(t *testing.T) {
	resetWindowsUpdaterState(t)

	logFile := filepath.Join(t.TempDir(), "OllamaSetup.log")
	if err := os.WriteFile(logFile, []byte("Installation process succeeded."), 0o644); err != nil {
		t.Fatal(err)
	}
	failed, err := windowsInstallerLogIndicatesFailure(logFile, time.Time{})
	if err != nil {
		t.Fatal(err)
	}
	if failed {
		t.Fatal("successful installer log should not be reported as failure")
	}

	if err := os.WriteFile(logFile, []byte("Installation process failed."), 0o644); err != nil {
		t.Fatal(err)
	}
	staleCutoff := time.Now().Add(2 * time.Minute)
	failed, err = windowsInstallerLogIndicatesFailure(logFile, staleCutoff)
	if err != nil {
		t.Fatal(err)
	}
	if failed {
		t.Fatal("stale installer log should not be reported as failure")
	}
}

func TestIsInstallerRunning(t *testing.T) {
	resetWindowsUpdaterState(t)

	oldInstaller := Installer
	defer func() {
		Installer = oldInstaller
	}()

	slog.SetLogLoggerLevel(slog.LevelDebug)
	Installer = "go.exe"
	if !isInstallerRunning() {
		t.Fatal("not running")
	}
}
