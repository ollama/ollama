package version

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/ollama/ollama/auth"
)

var updateCheckURLBase = "https://ollama.com"

// CheckForUpdate calls the ollama.com update API and reports whether a
// newer version is available.
func CheckForUpdate(ctx context.Context) (bool, error) {
	requestURL, err := url.Parse(updateCheckURLBase + "/api/update")
	if err != nil {
		return false, fmt.Errorf("parse update URL: %w", err)
	}

	query := requestURL.Query()
	query.Add("os", runtime.GOOS)
	query.Add("arch", runtime.GOARCH)
	query.Add("version", Version)
	requestURL.RawQuery = query.Encode()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, requestURL.String(), nil)
	if err != nil {
		return false, fmt.Errorf("create request: %w", err)
	}

	_ = auth.SignRequest(ctx, req)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return false, fmt.Errorf("update check request: %w", err)
	}
	defer resp.Body.Close()

	return resp.StatusCode == http.StatusOK, nil
}

func cacheFilePath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".ollama", "update"), nil
}

// CacheAvailableUpdate creates the update marker file.
func CacheAvailableUpdate() error {
	path, err := cacheFilePath()
	if err != nil {
		return err
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	return f.Close()
}

// HasCachedUpdate reports whether a non-stale update marker exists.
func HasCachedUpdate() bool {
	path, err := cacheFilePath()
	if err != nil {
		return false
	}

	fi, err := os.Stat(path)
	if err != nil {
		return false
	}

	return time.Since(fi.ModTime()) <= 24*time.Hour
}

// ClearCachedUpdate removes the update marker file.
func ClearCachedUpdate() error {
	path, err := cacheFilePath()
	if err != nil {
		return err
	}

	err = os.Remove(path)
	if os.IsNotExist(err) {
		return nil
	}
	return err
}

func IsOfficialInstall() bool {
	exe, err := os.Executable()
	if err != nil {
		return false
	}

	exe, err = filepath.EvalSymlinks(exe)
	if err != nil {
		return false
	}

	switch runtime.GOOS {
	case "windows":
		localAppData := os.Getenv("LOCALAPPDATA")
		if localAppData == "" {
			return false
		}
		return strings.HasPrefix(strings.ToLower(exe), strings.ToLower(filepath.Join(localAppData, "Programs", "Ollama")+string(filepath.Separator)))
	case "darwin":
		return strings.HasPrefix(exe, "/Applications/Ollama.app/")
	default:
		dir := filepath.Dir(exe)
		return dir == "/usr/local/bin" || dir == "/usr/bin" || dir == "/bin"
	}
}

// DoUpdate downloads and runs the platform-appropriate install script.
func DoUpdate(force bool) error {
	if !force && !IsOfficialInstall() {
		return fmt.Errorf("ollama appears to be installed through a package manager. Please update it using your package manager")
	}

	var scriptURL, tmpPattern, shell string
	switch runtime.GOOS {
	case "windows":
		scriptURL = "https://ollama.com/install.ps1"
		tmpPattern = "ollama-install-*.ps1"
		shell = "powershell"
	default:
		scriptURL = "https://ollama.com/install.sh"
		tmpPattern = "ollama-install-*.sh"
		shell = "sh"
	}

	resp, err := http.Get(scriptURL)
	if err != nil {
		return fmt.Errorf("download install script: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download install script: status %d", resp.StatusCode)
	}

	tmpFile, err := os.CreateTemp("", tmpPattern)
	if err != nil {
		return fmt.Errorf("create temp file: %w", err)
	}
	defer os.Remove(tmpFile.Name())

	if _, err := io.Copy(tmpFile, resp.Body); err != nil {
		tmpFile.Close()
		return fmt.Errorf("write install script: %w", err)
	}
	tmpFile.Close()

	cmd := exec.Command(shell, tmpFile.Name())
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// IsLocalHost reports whether the configured Ollama host points to the
// local machine.
func IsLocalHost(host *url.URL) bool {
	hostname := host.Hostname()
	switch hostname {
	case "", "127.0.0.1", "localhost", "::1", "0.0.0.0":
		return true
	}

	if ip := net.ParseIP(hostname); ip != nil {
		return ip.IsLoopback()
	}

	return false
}
