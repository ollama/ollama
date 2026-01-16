package server

import (
	"context"
	"strings"
	"testing"
	"time"
)

func TestGetAuthorizationTokenRejectsCrossDomain(t *testing.T) {
	tests := []struct {
		realm        string
		originalHost string
		wantMismatch bool
	}{
		{"https://example.com/token", "example.com", false},
		{"https://example.com/token", "other.com", true},
		{"https://example.com/token", "localhost:8000", true},
		{"https://localhost:5000/token", "localhost:5000", false},
		{"https://localhost:5000/token", "localhost:6000", true},
	}

	for _, tt := range tests {
		t.Run(tt.originalHost, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
			defer cancel()

			challenge := registryChallenge{Realm: tt.realm, Service: "test", Scope: "repo:x:pull"}
			_, err := getAuthorizationToken(ctx, challenge, tt.originalHost)

			isMismatch := err != nil && strings.Contains(err.Error(), "does not match")
			if tt.wantMismatch && !isMismatch {
				t.Errorf("expected domain mismatch error, got: %v", err)
			}
			if !tt.wantMismatch && isMismatch {
				t.Errorf("unexpected domain mismatch error: %v", err)
			}
		})
	}
}

func TestParseRegistryChallenge(t *testing.T) {
	tests := []struct {
		input                             string
		wantRealm, wantService, wantScope string
	}{
		{
			`Bearer realm="https://auth.example.com/token",service="registry",scope="repo:foo:pull"`,
			"https://auth.example.com/token", "registry", "repo:foo:pull",
		},
		{
			`Bearer realm="https://r.ollama.ai/v2/token",service="ollama",scope="-"`,
			"https://r.ollama.ai/v2/token", "ollama", "-",
		},
		{"", "", "", ""},
	}

	for _, tt := range tests {
		result := parseRegistryChallenge(tt.input)
		if result.Realm != tt.wantRealm || result.Service != tt.wantService || result.Scope != tt.wantScope {
			t.Errorf("parseRegistryChallenge(%q) = {%q, %q, %q}, want {%q, %q, %q}",
				tt.input, result.Realm, result.Service, result.Scope,
				tt.wantRealm, tt.wantService, tt.wantScope)
		}
	}
}

func TestRegistryChallengeURL(t *testing.T) {
	challenge := registryChallenge{
		Realm:   "https://auth.example.com/token",
		Service: "registry",
		Scope:   "repo:foo:pull repo:bar:push",
	}

	u, err := challenge.URL()
	if err != nil {
		t.Fatalf("URL() error: %v", err)
	}

	if u.Host != "auth.example.com" {
		t.Errorf("host = %q, want %q", u.Host, "auth.example.com")
	}
	if u.Path != "/token" {
		t.Errorf("path = %q, want %q", u.Path, "/token")
	}

	q := u.Query()
	if q.Get("service") != "registry" {
		t.Errorf("service = %q, want %q", q.Get("service"), "registry")
	}
	if scopes := q["scope"]; len(scopes) != 2 {
		t.Errorf("scope count = %d, want 2", len(scopes))
	}
	if q.Get("ts") == "" {
		t.Error("missing ts")
	}
	if q.Get("nonce") == "" {
		t.Error("missing nonce")
	}

	// Nonces should differ between calls
	u2, _ := challenge.URL()
	if q.Get("nonce") == u2.Query().Get("nonce") {
		t.Error("nonce should be unique per call")
	}
}

func TestRegistryChallengeURLInvalid(t *testing.T) {
	challenge := registryChallenge{Realm: "://invalid"}
	if _, err := challenge.URL(); err == nil {
		t.Error("expected error for invalid URL")
	}
}
