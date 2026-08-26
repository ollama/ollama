//go:build windows || darwin

package ui

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"golang.org/x/crypto/ssh"
)

func TestFeatureFlagsTypedValues(t *testing.T) {
	server := &Server{
		featureFlags: newFeatureFlagService(
			func(_ context.Context, key string) (any, error) {
				switch key {
				case "enabled":
					return true, nil
				case "mode":
					return "compact", nil
				case "wrong-type":
					return "yes", nil
				default:
					return nil, errors.New("unavailable")
				}
			},
			func() (bool, error) { return false, nil },
		),
	}

	if got := server.FeatureFlagBool(t.Context(), "enabled", false); !got {
		t.Fatal("FeatureFlagBool() = false, want true")
	}
	if got := server.FeatureFlagString(t.Context(), "mode", "standard"); got != "compact" {
		t.Fatalf("FeatureFlagString() = %q, want compact", got)
	}
	if got := server.FeatureFlagBool(t.Context(), "wrong-type", false); got {
		t.Fatal("FeatureFlagBool() accepted a string value")
	}
	if got := server.FeatureFlagString(t.Context(), "missing", "standard"); got != "standard" {
		t.Fatalf("FeatureFlagString() = %q, want fallback", got)
	}
}

func TestFeatureFlagsResolveOncePerSession(t *testing.T) {
	var calls atomic.Int32
	started := make(chan struct{})
	release := make(chan struct{})
	server := &Server{
		featureFlags: newFeatureFlagService(
			func(context.Context, string) (any, error) {
				if calls.Add(1) == 1 {
					close(started)
				}
				<-release
				return true, nil
			},
			func() (bool, error) { return false, nil },
		),
	}

	const callers = 20
	results := make(chan bool, callers)
	var wg sync.WaitGroup
	for range callers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			results <- server.FeatureFlagBool(t.Context(), "shared", false)
		}()
	}
	<-started
	close(release)
	wg.Wait()
	close(results)
	for result := range results {
		if !result {
			t.Fatal("FeatureFlagBool() = false, want true")
		}
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("remote calls = %d, want 1", got)
	}
}

func TestFeatureFlagsCacheFailures(t *testing.T) {
	for _, tt := range []struct {
		name          string
		cloudDisabled func() (bool, error)
	}{
		{name: "cloud off", cloudDisabled: func() (bool, error) { return true, nil }},
		{name: "cloud status unavailable", cloudDisabled: func() (bool, error) { return false, errors.New("unavailable") }},
		{name: "remote unavailable", cloudDisabled: func() (bool, error) { return false, nil }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var calls atomic.Int32
			server := &Server{
				featureFlags: newFeatureFlagService(
					func(context.Context, string) (any, error) {
						calls.Add(1)
						return nil, errors.New("unavailable")
					},
					tt.cloudDisabled,
				),
			}

			if got := server.FeatureFlagBool(t.Context(), "enabled", true); !got {
				t.Fatal("first call did not return its fallback")
			}
			if got := server.FeatureFlagBool(t.Context(), "enabled", false); !got {
				t.Fatal("second call did not return the session-stable fallback")
			}
			wantCalls := int32(1)
			if tt.name != "remote unavailable" {
				wantCalls = 0
			}
			if got := calls.Load(); got != wantCalls {
				t.Fatalf("remote calls = %d, want %d", got, wantCalls)
			}
		})
	}
}

func TestFeatureFlagsRejectInvalidKeysWithoutFetching(t *testing.T) {
	var calls atomic.Int32
	server := &Server{
		featureFlags: newFeatureFlagService(
			func(context.Context, string) (any, error) {
				calls.Add(1)
				return true, nil
			},
			func() (bool, error) { return false, nil },
		),
	}

	for _, key := range []string{"", "flags/all", "has space", strings.Repeat("x", 129)} {
		if got := server.FeatureFlagBool(t.Context(), key, false); got {
			t.Fatalf("FeatureFlagBool(%q) = true, want fallback", key)
		}
	}
	if got := calls.Load(); got != 0 {
		t.Fatalf("remote calls = %d, want 0", got)
	}
}

func TestFeatureFlagLocalAPI(t *testing.T) {
	server := &Server{
		Dev: true,
		featureFlags: newFeatureFlagService(
			func(_ context.Context, key string) (any, error) {
				if key == "enabled" {
					return true, nil
				}
				return "compact", nil
			},
			func() (bool, error) { return false, nil },
		),
	}

	for _, tt := range []struct {
		path string
		want any
	}{
		{path: "/api/v1/feature-flags/enabled?type=boolean&default=false", want: true},
		{path: "/api/v1/feature-flags/mode?type=string&default=standard", want: "compact"},
	} {
		rr := httptest.NewRecorder()
		server.Handler().ServeHTTP(rr, httptest.NewRequest(http.MethodGet, tt.path, nil))
		if rr.Code != http.StatusOK {
			t.Fatalf("%s status = %d, want %d", tt.path, rr.Code, http.StatusOK)
		}
		var response struct {
			Value any `json:"value"`
		}
		if err := json.NewDecoder(rr.Body).Decode(&response); err != nil {
			t.Fatal(err)
		}
		if response.Value != tt.want {
			t.Fatalf("%s value = %v, want %v", tt.path, response.Value, tt.want)
		}
	}
}

func TestFetchFeatureFlag(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	writeFeatureFlagTestKey(t, home)

	remote := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/api/app/feature-flags/enabled" {
			t.Fatalf("request = %s %s, want signed feature lookup", r.Method, r.URL.Path)
		}
		verifyFeatureFlagRequest(t, r)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"value":true}`)
	}))
	defer remote.Close()

	previous := OllamaDotCom
	OllamaDotCom = remote.URL
	t.Cleanup(func() { OllamaDotCom = previous })

	value, err := (&Server{}).fetchFeatureFlag(t.Context(), "enabled")
	if err != nil {
		t.Fatal(err)
	}
	if value != true {
		t.Fatalf("value = %v, want true", value)
	}
}

func TestFetchFeatureFlagRejectsUnexpectedResponses(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	writeFeatureFlagTestKey(t, home)

	for _, body := range []string{
		`{"value":null}`,
		`{"value":1}`,
		`{"value":{"enabled":true}}`,
		`{"value":true,"reason":"forced"}`,
		`{"value":true}{"value":false}`,
	} {
		t.Run(body, func(t *testing.T) {
			remote := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				fmt.Fprint(w, body)
			}))
			defer remote.Close()

			previous := OllamaDotCom
			OllamaDotCom = remote.URL
			defer func() { OllamaDotCom = previous }()

			if _, err := (&Server{}).fetchFeatureFlag(t.Context(), "enabled"); err == nil {
				t.Fatal("fetchFeatureFlag() accepted an unexpected response")
			}
		})
	}
}

func writeFeatureFlagTestKey(t *testing.T, home string) {
	t.Helper()
	_, privateKey, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	block, err := ssh.MarshalPrivateKey(privateKey, "")
	if err != nil {
		t.Fatal(err)
	}
	keyPath := filepath.Join(home, ".ollama", "id_ed25519")
	if err := os.MkdirAll(filepath.Dir(keyPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(keyPath, pem.EncodeToMemory(block), 0o600); err != nil {
		t.Fatal(err)
	}
}

func verifyFeatureFlagRequest(t *testing.T, req *http.Request) {
	t.Helper()
	keyData, signatureData, ok := strings.Cut(req.Header.Get("Authorization"), ":")
	if !ok {
		t.Fatal("request is missing its public-key signature")
	}
	keyData = strings.TrimPrefix(keyData, "Bearer ")
	publicKeyData, err := base64.StdEncoding.DecodeString(keyData)
	if err != nil {
		t.Fatal(err)
	}
	publicKey, err := ssh.ParsePublicKey(publicKeyData)
	if err != nil {
		t.Fatal(err)
	}
	signature, err := base64.StdEncoding.DecodeString(signatureData)
	if err != nil {
		t.Fatal(err)
	}
	challenge := []byte(req.Method + "," + req.URL.RequestURI())
	if err := publicKey.Verify(challenge, &ssh.Signature{Format: publicKey.Type(), Blob: signature}); err != nil {
		t.Fatal(err)
	}
}
