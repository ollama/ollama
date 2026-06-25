//go:build windows && updater_integration

package updater

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"golang.org/x/sys/windows/registry"
)

func TestWindowsInstallScriptUpdaterEndToEnd(t *testing.T) {
	if os.Getenv("OLLAMA_TEST_SANDBOX") != "1" {
		t.Fatal("refusing to run destructive updater integration test outside Windows Sandbox")
	}

	installerPath := os.Getenv("OLLAMA_TEST_UPDATER_INSTALLER")
	installScriptPath := os.Getenv("OLLAMA_TEST_UPDATER_INSTALL_SCRIPT")
	targetVersion := os.Getenv("OLLAMA_TEST_UPDATER_VERSION")
	if installerPath == "" || installScriptPath == "" || targetVersion == "" {
		t.Fatal("OLLAMA_TEST_UPDATER_INSTALLER, OLLAMA_TEST_UPDATER_INSTALL_SCRIPT, and OLLAMA_TEST_UPDATER_VERSION are required")
	}

	installScript, err := os.ReadFile(installScriptPath)
	if err != nil {
		t.Fatal(err)
	}
	const downloadBase = `$DownloadBaseURL = "https://ollama.com/download"`
	if strings.Count(string(installScript), downloadBase) != 1 {
		t.Fatalf("install.ps1 contains %d download base assignments, want 1", strings.Count(string(installScript), downloadBase))
	}

	server := newWindowsUpdaterTestServer(t, installerPath, installScript, downloadBase)
	defer server.Close()

	localAppData := os.Getenv("LOCALAPPDATA")
	appDataDir := filepath.Join(localAppData, "Ollama")
	UpdateStageDir = filepath.Join(appDataDir, "updates_v2")
	UpgradeLogFile = filepath.Join(appDataDir, "upgrade.log")
	installScriptInstallerLogFile = filepath.Join(appDataDir, "OllamaSetup.log")
	runningInstaller = filepath.Join(appDataDir, Installer)
	UpgradeMarkerFile = filepath.Join(appDataDir, "upgraded")
	UpdateDownloaded = false

	exitCode := make(chan int, 1)
	previousExit := exitAfterStartingUpgrade
	exitAfterStartingUpgrade = func(code int) { exitCode <- code }
	t.Cleanup(func() { exitAfterStartingUpgrade = previousExit })

	updater := &Updater{}
	err = updater.DownloadNewRelease(context.Background(), UpdateResponse{
		UpdateURL:     server.URL + "/OllamaSetup.exe",
		UpdateVersion: targetVersion,
	})
	if err != nil {
		t.Fatalf("stage update: %v", err)
	}
	if !UpdateDownloaded || !IsUpdatePending() {
		t.Fatal("update was not marked ready after the cache-only phase")
	}
	if matches, err := filepath.Glob(filepath.Join(appDataDir, "install_cache", "*", "OllamaSetup.exe")); err != nil || len(matches) != 1 {
		t.Fatalf("cached installers = %v, %v; want exactly one", matches, err)
	}

	if err := DoUpgradeAtStartup(); err != nil {
		t.Fatalf("start cached update: %v", err)
	}
	select {
	case code := <-exitCode:
		if code != 0 {
			t.Fatalf("updater exit code = %d, want 0", code)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("updater did not request app exit after starting install.ps1")
	}

	deadline := time.Now().Add(5 * time.Minute)
	for time.Now().Before(deadline) {
		installedVersion, installed := windowsUpdaterInstalledVersion()
		cachedInstallers, cacheErr := filepath.Glob(filepath.Join(appDataDir, "install_cache", "*", "OllamaSetup.exe"))
		if installed && installedVersion == targetVersion && cacheErr == nil && len(cachedInstallers) == 0 {
			return
		}
		time.Sleep(time.Second)
	}

	logData, _ := os.ReadFile(installScriptInstallerLogFile)
	installedVersion, _ := windowsUpdaterInstalledVersion()
	t.Fatalf("update did not finish: installed version %q, installer log:\n%s", installedVersion, logData)
}

func newWindowsUpdaterTestServer(t *testing.T, installerPath string, installScript []byte, downloadBase string) *httptest.Server {
	t.Helper()

	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/install.ps1":
			content := strings.Replace(string(installScript), downloadBase, fmt.Sprintf(`$DownloadBaseURL = "%s"`, server.URL), 1)
			w.Header().Set("Content-Type", "text/plain; charset=utf-8")
			w.Header().Set("ETag", `"app-updater-install-script"`)
			w.Header().Set("Content-Length", fmt.Sprint(len(content)))
			if r.Method != http.MethodHead {
				_, _ = io.WriteString(w, content)
			}
		case "/OllamaSetup.exe":
			file, err := os.Open(installerPath)
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			defer file.Close()
			info, err := file.Stat()
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			w.Header().Set("ETag", `"app-updater-installer"`)
			http.ServeContent(w, r, info.Name(), info.ModTime(), file)
		default:
			http.NotFound(w, r)
		}
	}))
	return server
}

func windowsUpdaterInstalledVersion() (string, bool) {
	const uninstallKey = `Software\Microsoft\Windows\CurrentVersion\Uninstall\{44E83376-CE68-45EB-8FC1-393500EB558C}_is1`
	key, err := registry.OpenKey(registry.CURRENT_USER, uninstallKey, registry.QUERY_VALUE)
	if err != nil {
		return "", false
	}
	defer key.Close()
	version, _, err := key.GetStringValue("DisplayVersion")
	return strings.TrimSpace(version), err == nil
}
