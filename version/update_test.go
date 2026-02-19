package version

import (
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestCheckForUpdate(t *testing.T) {
	t.Run("update available", func(t *testing.T) {
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Query().Get("os") == "" || r.URL.Query().Get("arch") == "" || r.URL.Query().Get("version") == "" {
				t.Error("missing expected query parameters")
			}
			w.WriteHeader(http.StatusOK)
		}))
		defer ts.Close()

		old := updateCheckURLBase
		updateCheckURLBase = ts.URL
		defer func() { updateCheckURLBase = old }()

		available, err := CheckForUpdate(context.Background())
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !available {
			t.Fatal("expected update to be available")
		}
	})

	t.Run("up to date", func(t *testing.T) {
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusNoContent)
		}))
		defer ts.Close()

		old := updateCheckURLBase
		updateCheckURLBase = ts.URL
		defer func() { updateCheckURLBase = old }()

		available, err := CheckForUpdate(context.Background())
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if available {
			t.Fatal("expected no update available")
		}
	})

	t.Run("network error", func(t *testing.T) {
		old := updateCheckURLBase
		updateCheckURLBase = "http://localhost:1"
		defer func() { updateCheckURLBase = old }()

		ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
		defer cancel()

		_, err := CheckForUpdate(ctx)
		if err == nil {
			t.Fatal("expected error for unreachable server")
		}
	})
}

func TestCacheRoundTrip(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("HOME", tmp)
	os.MkdirAll(filepath.Join(tmp, ".ollama"), 0o755)

	if err := CacheAvailableUpdate(); err != nil {
		t.Fatalf("cache write: %v", err)
	}

	if !HasCachedUpdate() {
		t.Fatal("expected cached update to be present")
	}

	if err := ClearCachedUpdate(); err != nil {
		t.Fatalf("cache clear: %v", err)
	}

	if HasCachedUpdate() {
		t.Fatal("expected no cached update after clear")
	}
}

func TestHasCachedUpdateStale(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("HOME", tmp)
	os.MkdirAll(filepath.Join(tmp, ".ollama"), 0o755)

	if err := CacheAvailableUpdate(); err != nil {
		t.Fatalf("cache write: %v", err)
	}

	// Backdate the file to make it stale
	path := filepath.Join(tmp, ".ollama", "update")
	staleTime := time.Now().Add(-25 * time.Hour)
	os.Chtimes(path, staleTime, staleTime)

	if HasCachedUpdate() {
		t.Fatal("expected no cached update for stale file")
	}
}

func TestIsLocalHost(t *testing.T) {
	tests := []struct {
		host  string
		local bool
	}{
		{"http://127.0.0.1:11434", true},
		{"http://localhost:11434", true},
		{"http://[::1]:11434", true},
		{"http://0.0.0.0:11434", true},
		{"http://remote.example.com:11434", false},
		{"http://192.168.1.100:11434", false},
	}

	for _, tt := range tests {
		t.Run(tt.host, func(t *testing.T) {
			u, err := url.Parse(tt.host)
			if err != nil {
				t.Fatalf("parse URL: %v", err)
			}
			if got := IsLocalHost(u); got != tt.local {
				t.Errorf("IsLocalHost(%s) = %v, want %v", tt.host, got, tt.local)
			}
		})
	}
}
