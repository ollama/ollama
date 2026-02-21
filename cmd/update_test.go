package cmd

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/version"
)

func runUpdateCmd(t *testing.T, serverURL string, checkOnly bool) (string, error) {
	t.Helper()

	origURL := updateCheckURL
	updateCheckURL = serverURL
	t.Cleanup(func() { updateCheckURL = origURL })

	var buf bytes.Buffer
	origOutput := updateOutput
	updateOutput = &buf
	t.Cleanup(func() { updateOutput = origOutput })

	cmd := &cobra.Command{}
	cmd.SetContext(t.Context())
	cmd.Flags().Bool("check", false, "")
	if checkOnly {
		if err := cmd.Flags().Set("check", "true"); err != nil {
			t.Fatal(err)
		}
	}

	err := UpdateHandler(cmd, nil)
	return buf.String(), err
}

func TestUpdateHandler_AlreadyUpToDate(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))
	defer srv.Close()

	out, err := runUpdateCmd(t, srv.URL, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "already up to date") {
		t.Errorf("expected 'already up to date' in output, got: %q", out)
	}
	if !strings.Contains(out, version.Version) {
		t.Errorf("expected current version %q in output, got: %q", version.Version, out)
	}
}

func TestUpdateHandler_CheckFlag(t *testing.T) {
	const newVersion = "9.9.9"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]string{
			"url":     "https://github.com/ollama/ollama/releases/download/v" + newVersion + "/ollama-linux-amd64.tgz",
			"version": newVersion,
		}); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	}))
	defer srv.Close()

	out, err := runUpdateCmd(t, srv.URL, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, newVersion) {
		t.Errorf("expected new version %q in output, got: %q", newVersion, out)
	}
	if !strings.Contains(out, version.Version) {
		t.Errorf("expected current version %q in output, got: %q", version.Version, out)
	}
}

func TestUpdateHandler_VersionFromURL(t *testing.T) {
	// Server returns a response with no "version" field — version must be extracted from URL
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]string{
			"url": "https://github.com/ollama/ollama/releases/download/v8.8.8/ollama-linux-amd64.tgz",
		}); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	}))
	defer srv.Close()

	out, err := runUpdateCmd(t, srv.URL, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "v8.8.8") {
		t.Errorf("expected version extracted from URL in output, got: %q", out)
	}
}

func TestUpdateHandler_ServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	_, err := runUpdateCmd(t, srv.URL, false)
	if err == nil {
		t.Fatal("expected error for 500 response, got nil")
	}
	if !strings.Contains(err.Error(), "500") {
		t.Errorf("expected status code in error message, got: %v", err)
	}
}

func TestUpdateHandler_QueryParams(t *testing.T) {
	var gotOS, gotArch, gotVersion string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotOS = r.URL.Query().Get("os")
		gotArch = r.URL.Query().Get("arch")
		gotVersion = r.URL.Query().Get("version")
		w.WriteHeader(http.StatusNoContent)
	}))
	defer srv.Close()

	_, err := runUpdateCmd(t, srv.URL, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if gotOS == "" {
		t.Error("expected 'os' query param to be set")
	}
	if gotArch == "" {
		t.Error("expected 'arch' query param to be set")
	}
	if gotVersion != version.Version {
		t.Errorf("expected version %q, got %q", version.Version, gotVersion)
	}
}
