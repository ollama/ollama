package cmd

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/version"
)

func TestUpdateCheckCommand(t *testing.T) {
	releaseJSON := map[string]any{
		"tag_name":     "v99.0.0",
		"name":         "Release v99.0.0",
		"html_url":     "https://github.com/ollama/ollama/releases/tag/v99.0.0",
		"published_at": time.Now().Format(time.RFC3339),
		"assets": []map[string]any{
			{
				"name":                 "ollama-linux-amd64.tar.zst",
				"browser_download_url": "https://example.com/download/ollama-linux-amd64.tar.zst",
				"size":                 1024,
			},
		},
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(releaseJSON)
	}))
	defer server.Close()

	// Ensure no running Ollama server intercepts version resolution
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:0")

	cli := NewCLI()
	var stdout, stderr bytes.Buffer
	cli.SetOut(&stdout)
	cli.SetErr(&stderr)
	cli.SetArgs([]string{"update", "check", "--url", server.URL})

	if err := cli.Execute(); err != nil {
		t.Fatalf("update check command failed: %v", err)
	}
}

func TestUpdatePullCommand(t *testing.T) {
	fakeContent := []byte("new-ollama-binary-content-data")

	fileServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "30")
		_, _ = w.Write(fakeContent)
	}))
	defer fileServer.Close()

	releaseJSON := map[string]any{
		"tag_name": "v99.0.0",
		"html_url": "https://github.com/ollama/ollama/releases/tag/v99.0.0",
		"assets": []map[string]any{
			{
				"name":                 "ollama-linux-amd64.tar.zst",
				"browser_download_url": fileServer.URL + "/ollama-linux-amd64.tar.zst",
				"size":                 int64(len(fakeContent)),
			},
			{
				"name":                 "ollama-linux-arm64.tar.zst",
				"browser_download_url": fileServer.URL + "/ollama-linux-arm64.tar.zst",
				"size":                 int64(len(fakeContent)),
			},
			{
				"name":                 "Ollama-darwin.zip",
				"browser_download_url": fileServer.URL + "/Ollama-darwin.zip",
				"size":                 int64(len(fakeContent)),
			},
			{
				"name":                 "OllamaSetup.exe",
				"browser_download_url": fileServer.URL + "/OllamaSetup.exe",
				"size":                 int64(len(fakeContent)),
			},
		},
	}

	metaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(releaseJSON)
	}))
	defer metaServer.Close()

	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:0")

	tempDir := t.TempDir()

	cli := NewCLI()
	cli.SetArgs([]string{"update", "pull", "--url", metaServer.URL, "--dir", tempDir})

	if err := cli.Execute(); err != nil {
		t.Fatalf("update pull command failed: %v", err)
	}

	files, err := os.ReadDir(tempDir)
	if err != nil {
		t.Fatalf("read tempDir: %v", err)
	}
	if len(files) == 0 {
		t.Fatalf("expected downloaded update file in %s", tempDir)
	}

	downloadedPath := filepath.Join(tempDir, files[0].Name())
	content, err := os.ReadFile(downloadedPath)
	if err != nil {
		t.Fatalf("read downloaded file: %v", err)
	}
	if !bytes.Equal(content, fakeContent) {
		t.Errorf("content mismatch")
	}
}

func TestUpdatePullUpToDateRequiresForce(t *testing.T) {
	oldVer := version.Version
	version.Version = "99.0.0"
	defer func() { version.Version = oldVer }()

	// Release version matching current version
	releaseJSON := map[string]any{
		"tag_name": "v99.0.0",
		"html_url": "https://github.com/ollama/ollama/releases/tag/v99.0.0",
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(releaseJSON)
	}))
	defer server.Close()

	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:0")

	cli := NewCLI()
	cli.SetArgs([]string{"update", "pull", "--url", server.URL})

	if err := cli.Execute(); err != nil {
		t.Fatalf("expected pull without update to succeed gracefully: %v", err)
	}
}

func TestPullCommandWithUpdateFlag(t *testing.T) {
	fakeContent := []byte("binary-payload-for-pull-flag")

	fileServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(fakeContent)
	}))
	defer fileServer.Close()

	releaseJSON := map[string]any{
		"tag_name": "v99.0.0",
		"assets": []map[string]any{
			{
				"name":                 "ollama-linux-amd64.tar.zst",
				"browser_download_url": fileServer.URL + "/update.bin",
				"size":                 int64(len(fakeContent)),
			},
			{
				"name":                 "ollama-linux-arm64.tar.zst",
				"browser_download_url": fileServer.URL + "/update.bin",
				"size":                 int64(len(fakeContent)),
			},
			{
				"name":                 "Ollama-darwin.zip",
				"browser_download_url": fileServer.URL + "/update.bin",
				"size":                 int64(len(fakeContent)),
			},
			{
				"name":                 "OllamaSetup.exe",
				"browser_download_url": fileServer.URL + "/update.bin",
				"size":                 int64(len(fakeContent)),
			},
		},
	}

	metaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(releaseJSON)
	}))
	defer metaServer.Close()

	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:0")

	tempDir := t.TempDir()

	cli := NewCLI()
	cli.SetArgs([]string{"pull", "--update", "--url", metaServer.URL, "--dir", tempDir})

	if err := cli.Execute(); err != nil {
		t.Fatalf("ollama pull --update failed: %v", err)
	}

	files, err := os.ReadDir(tempDir)
	if err != nil || len(files) == 0 {
		t.Fatalf("expected downloaded file in %s", tempDir)
	}
}

func TestPullCommandWithUpdateArg(t *testing.T) {
	fakeContent := []byte("binary-payload-for-pull-arg")

	fileServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(fakeContent)
	}))
	defer fileServer.Close()

	releaseJSON := map[string]any{
		"tag_name": "v99.0.0",
		"assets": []map[string]any{
			{
				"name":                 "ollama-linux-amd64.tar.zst",
				"browser_download_url": fileServer.URL + "/update.bin",
				"size":                 int64(len(fakeContent)),
			},
			{
				"name":                 "ollama-linux-arm64.tar.zst",
				"browser_download_url": fileServer.URL + "/update.bin",
				"size":                 int64(len(fakeContent)),
			},
			{
				"name":                 "Ollama-darwin.zip",
				"browser_download_url": fileServer.URL + "/update.bin",
				"size":                 int64(len(fakeContent)),
			},
			{
				"name":                 "OllamaSetup.exe",
				"browser_download_url": fileServer.URL + "/update.bin",
				"size":                 int64(len(fakeContent)),
			},
		},
	}

	metaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(releaseJSON)
	}))
	defer metaServer.Close()

	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:0")

	tempDir := t.TempDir()

	cli := NewCLI()
	cli.SetArgs([]string{"pull", "update", "--url", metaServer.URL, "--dir", tempDir})

	if err := cli.Execute(); err != nil {
		t.Fatalf("ollama pull update failed: %v", err)
	}
}

func TestUpdateRootCommandFlags(t *testing.T) {
	releaseJSON := map[string]any{
		"tag_name": "v99.0.0",
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(releaseJSON)
	}))
	defer server.Close()

	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:0")

	cli := NewCLI()
	cli.SetArgs([]string{"update", "--check", "--url", server.URL})

	if err := cli.Execute(); err != nil {
		t.Fatalf("ollama update --check failed: %v", err)
	}
}

func TestUpdateConflictingFlags(t *testing.T) {
	cli := NewCLI()
	cli.SetArgs([]string{"update", "--check", "--pull"})

	err := cli.Execute()
	if err == nil {
		t.Fatalf("expected error for conflicting --check and --pull flags, got nil")
	}
	if !strings.Contains(err.Error(), "both --check and --pull") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestCheckUpdateFlagOnRoot(t *testing.T) {
	releaseJSON := map[string]any{
		"tag_name": "v99.0.0",
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(releaseJSON)
	}))
	defer server.Close()

	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:0")

	cli := NewCLI()
	cli.SetArgs([]string{"--check-update"})

	// Set context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	_ = cli.ExecuteContext(ctx)
}

func TestUpdateCheckWithRC(t *testing.T) {
	releaseJSON := map[string]any{
		"tag_name":     "v99.0.0-rc2",
		"name":         "Release v99.0.0-rc2",
		"html_url":     "https://github.com/ollama/ollama/releases/tag/v99.0.0-rc2",
		"published_at": time.Now().Format(time.RFC3339),
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(releaseJSON)
	}))
	defer server.Close()

	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:0")

	for _, flag := range []string{"--rc", "--prerelease"} {
		cli := NewCLI()
		var stdout, stderr bytes.Buffer
		cli.SetOut(&stdout)
		cli.SetErr(&stderr)
		cli.SetArgs([]string{"update", "check", flag, "--url", server.URL})

		if err := cli.Execute(); err != nil {
			t.Fatalf("update check %s failed: %v", flag, err)
		}
	}
}

func TestPullWithRC(t *testing.T) {
	fakeContent := []byte("payload-for-rc-pull")

	fileServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(fakeContent)
	}))
	defer fileServer.Close()

	releaseJSON := map[string]any{
		"tag_name": "v99.0.0-rc2",
		"assets": []map[string]any{
			{
				"name":                 "ollama-linux-amd64.tar.zst",
				"browser_download_url": fileServer.URL + "/update.bin",
				"size":                 int64(len(fakeContent)),
			},
			{
				"name":                 "ollama-linux-arm64.tar.zst",
				"browser_download_url": fileServer.URL + "/update.bin",
				"size":                 int64(len(fakeContent)),
			},
			{
				"name":                 "Ollama-darwin.zip",
				"browser_download_url": fileServer.URL + "/update.bin",
				"size":                 int64(len(fakeContent)),
			},
			{
				"name":                 "OllamaSetup.exe",
				"browser_download_url": fileServer.URL + "/update.bin",
				"size":                 int64(len(fakeContent)),
			},
		},
	}

	metaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(releaseJSON)
	}))
	defer metaServer.Close()

	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:0")

	tempDir := t.TempDir()

	cli := NewCLI()
	cli.SetArgs([]string{"pull", "--update", "--rc", "--url", metaServer.URL, "--dir", tempDir})

	if err := cli.Execute(); err != nil {
		t.Fatalf("ollama pull --update --rc failed: %v", err)
	}

	files, err := os.ReadDir(tempDir)
	if err != nil || len(files) == 0 {
		t.Fatalf("expected downloaded file in %s", tempDir)
	}
}
