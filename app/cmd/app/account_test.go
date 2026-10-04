//go:build windows || darwin

package main

import (
	"context"
	"encoding/base64"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
)

// useTestAccountServer points account requests at handler and signs them
// with a fixed signature.
func useTestAccountServer(t *testing.T, handler http.HandlerFunc) {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)

	previousURL, previousSign, previousKey := ollamaDotCom, signOllamaData, ollamaPublicKey
	ollamaDotCom = server.URL
	signOllamaData = func(context.Context, []byte) (string, error) { return "test-signature", nil }
	ollamaPublicKey = func() (string, error) { return "ssh-ed25519 test-key", nil }
	t.Cleanup(func() {
		ollamaDotCom, signOllamaData, ollamaPublicKey = previousURL, previousSign, previousKey
	})
}

func TestAccountSignedIn(t *testing.T) {
	ts := newTestSettings(t)
	useTestAccountServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/api/me" {
			t.Errorf("request = %s %s", r.Method, r.URL.Path)
		}
		if r.Header.Get("Authorization") != "test-signature" || r.URL.Query().Get("ts") == "" {
			t.Errorf("request is not signed: %v", r.URL)
		}
		w.Write([]byte(`{"name":"jmorgan","email":"j@example.com","plan":"pro","avatarurl":"/avatars/j.png"}`))
	})

	account, err := ts.Account(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	want := accountState{
		SignedIn:  true,
		Name:      "jmorgan",
		Email:     "j@example.com",
		Plan:      "pro",
		AvatarURL: ollamaDotCom + "/avatars/j.png",
	}
	if account != want {
		t.Fatalf("account = %+v, want %+v", account, want)
	}
	if cached := ts.CachedAccount(); !cached.SignedIn || !cached.Cached || cached.Name != "jmorgan" {
		t.Fatalf("cached account = %+v", cached)
	}
}

func TestAccountSignedOutClearsCache(t *testing.T) {
	ts := newTestSettings(t)
	status := http.StatusOK
	useTestAccountServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(status)
		w.Write([]byte(`{"name":"jmorgan"}`))
	})
	if _, err := ts.Account(t.Context()); err != nil {
		t.Fatal(err)
	}

	status = http.StatusUnauthorized
	account, err := ts.Account(t.Context())
	if err != nil || account.SignedIn {
		t.Fatalf("account = %+v, error = %v", account, err)
	}
	if cached := ts.CachedAccount(); cached.SignedIn {
		t.Fatalf("signing out should clear the cached account: %+v", cached)
	}
}

func TestAccountUnavailableUsesCache(t *testing.T) {
	ts := newTestSettings(t)
	status := http.StatusOK
	useTestAccountServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(status)
		w.Write([]byte(`{"name":"jmorgan"}`))
	})
	if _, err := ts.Account(t.Context()); err != nil {
		t.Fatal(err)
	}

	status = http.StatusBadGateway
	account, err := ts.Account(t.Context())
	if err == nil {
		t.Fatal("expected an error while ollama.com is unavailable")
	}
	if !account.SignedIn || !account.Cached || account.Name != "jmorgan" {
		t.Fatalf("account = %+v, want the cached account", account)
	}
}

func TestSignOut(t *testing.T) {
	ts := newTestSettings(t)
	signedIn := true
	useTestAccountServer(t, func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/api/me":
			if !signedIn {
				w.WriteHeader(http.StatusUnauthorized)
				return
			}
			w.Write([]byte(`{"name":"jmorgan"}`))
		case r.Method == http.MethodDelete && strings.HasPrefix(r.URL.Path, "/api/user/keys/"):
			key, err := base64.RawURLEncoding.DecodeString(strings.TrimPrefix(r.URL.Path, "/api/user/keys/"))
			if err != nil || string(key) != "ssh-ed25519 test-key" {
				t.Errorf("signed out key %q, %v", key, err)
			}
			signedIn = false
		default:
			t.Errorf("unexpected request %s %s", r.Method, r.URL.Path)
		}
	})
	if _, err := ts.Account(t.Context()); err != nil {
		t.Fatal(err)
	}

	if err := ts.SignOut(t.Context()); err != nil {
		t.Fatal(err)
	}
	if signedIn {
		t.Fatal("sign out didn't disconnect the key")
	}
	if cached := ts.CachedAccount(); cached.SignedIn {
		t.Fatalf("cached account after sign out = %+v", cached)
	}
}

func TestSignInURL(t *testing.T) {
	ts := newTestSettings(t)
	useTestAccountServer(t, func(http.ResponseWriter, *http.Request) {})
	signIn, err := ts.SignInURL()
	if err != nil {
		t.Fatal(err)
	}
	u, err := url.Parse(signIn)
	if err != nil {
		t.Fatal(err)
	}
	key, err := base64.RawURLEncoding.DecodeString(u.Query().Get("key"))
	if err != nil || string(key) != "ssh-ed25519 test-key" {
		t.Fatalf("sign in URL key = %q, %v", key, err)
	}
	if u.Scheme+"://"+u.Host != ollamaDotCom || u.Path != "/connect" || u.Query().Get("launch") != "true" {
		t.Fatalf("sign in URL = %q", signIn)
	}
}
