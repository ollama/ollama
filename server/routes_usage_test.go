package server

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"encoding/pem"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	internalcloud "github.com/ollama/ollama/internal/cloud"
	"golang.org/x/crypto/ssh"
)

func TestUsageHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	t.Setenv("OLLAMA_NO_CLOUD", "")
	t.Setenv("OLLAMA_AUTH", "1")
	writeTestPrivateKey(t)

	startsAt := time.Date(2026, time.June, 29, 0, 0, 0, 0, time.UTC)
	endsAt := time.Date(2026, time.July, 27, 0, 0, 0, 0, time.UTC)

	tests := []struct {
		name           string
		upstreamStatus int
		upstreamBody   any
		wantStatus     int
		wantError      string
		wantSigninURL  string
	}{
		{
			name:           "success",
			upstreamStatus: http.StatusOK,
			upstreamBody: api.UsageResponse{
				Activity: api.UsageActivity{
					Cost: "0.00709",
					Period: api.UsagePeriod{
						Type:       "last_4_weeks",
						StartingAt: startsAt,
						EndingAt:   endsAt,
					},
					Models: []api.UsageActivityModel{{Name: "qwen3-coder:480b", RequestCount: 1, Cost: "0.00709"}},
				},
				Limits: api.UsageLimits{
					Session: api.UsageLimit{Usage: 0.006, Models: []api.UsageLimitModel{{Name: "qwen3-coder:480b", RequestCount: 2}}},
					Weekly:  api.UsageLimit{Usage: 0.002, Models: []api.UsageLimitModel{{Name: "web search", RequestCount: 1}}},
				},
			},
			wantStatus: http.StatusOK,
		},
		{
			name:           "not signed in",
			upstreamStatus: http.StatusUnauthorized,
			upstreamBody:   map[string]string{"error": "unauthorized", "signin_url": "https://ollama.com/connect/test"},
			wantStatus:     http.StatusUnauthorized,
			wantError:      "unauthorized",
			wantSigninURL:  "https://ollama.com/connect/test",
		},
		{
			name:           "suspended account",
			upstreamStatus: http.StatusForbidden,
			upstreamBody:   map[string]string{"error": "account suspended"},
			wantStatus:     http.StatusForbidden,
			wantError:      "account suspended",
		},
		{
			name:           "upstream unavailable",
			upstreamStatus: http.StatusInternalServerError,
			upstreamBody:   map[string]string{"error": "internal error"},
			wantStatus:     http.StatusServiceUnavailable,
			wantError:      "account unavailable",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodGet || r.URL.Path != "/api/usage" {
					t.Errorf("unexpected upstream request to %s %s", r.Method, r.URL.Path)
					http.NotFound(w, r)
					return
				}
				if got := r.Header.Get("Authorization"); got == "" {
					t.Error("upstream request is missing Authorization header")
				}
				if got := r.URL.Query().Get("ts"); got == "" {
					t.Error("upstream request is missing authentication timestamp")
				}

				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(tt.upstreamStatus)
				if err := json.NewEncoder(w).Encode(tt.upstreamBody); err != nil {
					t.Fatal(err)
				}
			}))
			defer upstream.Close()

			originalBaseURL := usageBaseURL
			usageBaseURL = upstream.URL
			t.Cleanup(func() { usageBaseURL = originalBaseURL })

			server := &Server{}
			router, err := server.GenerateRoutes()
			if err != nil {
				t.Fatal(err)
			}

			request := httptest.NewRequest(http.MethodGet, "/api/usage", nil)
			response := httptest.NewRecorder()
			router.ServeHTTP(response, request)

			if response.Code != tt.wantStatus {
				t.Fatalf("status = %d, want %d (%s)", response.Code, tt.wantStatus, response.Body.String())
			}

			if tt.wantStatus == http.StatusOK {
				var usage api.UsageResponse
				if err := json.Unmarshal(response.Body.Bytes(), &usage); err != nil {
					t.Fatal(err)
				}
				if usage.Activity.Cost != "0.00709" || len(usage.Activity.Models) != 1 || usage.Activity.Models[0].Name != "qwen3-coder:480b" || usage.Limits.Session.Usage != 0.006 || usage.Limits.Weekly.Usage != 0.002 {
					t.Errorf("usage = %#v, want upstream usage", usage)
				}
				return
			}

			var body struct {
				Error     string `json:"error"`
				SigninURL string `json:"signin_url"`
			}
			if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil {
				t.Fatal(err)
			}
			if body.Error != tt.wantError {
				t.Errorf("error = %q, want %q", body.Error, tt.wantError)
			}
			if body.SigninURL != tt.wantSigninURL {
				t.Errorf("signin URL = %q, want %q", body.SigninURL, tt.wantSigninURL)
			}
		})
	}
}

func writeTestPrivateKey(t *testing.T) {
	t.Helper()

	_, privateKey, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	privateKeyBlock, err := ssh.MarshalPrivateKey(privateKey, "")
	if err != nil {
		t.Fatal(err)
	}

	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	keyPath := filepath.Join(home, ".ollama", "id_ed25519")
	if err := os.MkdirAll(filepath.Dir(keyPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(keyPath, pem.EncodeToMemory(privateKeyBlock), 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestUsageHandlerCloudDisabled(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	t.Setenv("OLLAMA_NO_CLOUD", "1")

	var upstreamRequests atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamRequests.Add(1)
		http.Error(w, "unexpected request", http.StatusInternalServerError)
	}))
	defer upstream.Close()

	originalBaseURL := usageBaseURL
	usageBaseURL = upstream.URL
	t.Cleanup(func() { usageBaseURL = originalBaseURL })

	server := &Server{}
	router, err := server.GenerateRoutes()
	if err != nil {
		t.Fatal(err)
	}

	request := httptest.NewRequest(http.MethodGet, "/api/usage", nil)
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)

	if response.Code != http.StatusForbidden {
		t.Fatalf("status = %d, want %d (%s)", response.Code, http.StatusForbidden, response.Body.String())
	}
	if got, want := response.Body.String(), `{"error":"`+internalcloud.DisabledError(cloudErrUsageUnavailable)+`"}`; got != want {
		t.Fatalf("body = %q, want %q", got, want)
	}
	if got := upstreamRequests.Load(); got != 0 {
		t.Fatalf("upstream requests = %d, want 0", got)
	}
}
