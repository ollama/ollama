//go:build windows || darwin

package main

import (
	"net/http"
	"net/url"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/internal/onboarding"
)

// useTestOnboarding points onboarding at a new app database.
func useTestOnboarding(t *testing.T) *testSettings {
	t.Helper()
	ts := newTestSettings(t)
	previousStore, previousSettings := appStore, settings
	appStore, settings = ts.store, ts.settingsController
	t.Cleanup(func() { appStore, settings = previousStore, previousSettings })
	return ts
}

func TestCompleteOnboarding(t *testing.T) {
	ts := useTestOnboarding(t)
	if !onboardingNeeded() {
		t.Fatal("a new install doesn't need onboarding")
	}
	if err := completeOnboarding(); err != nil {
		t.Fatal(err)
	}
	if onboardingNeeded() {
		t.Fatal("onboarding is still needed after completing it")
	}
	// The CLI skips its welcome once the app completes onboarding.
	completed, err := onboarding.State{Dir: filepath.Dir(ts.store.DBPath)}.Completed()
	if err != nil || !completed {
		t.Fatalf("shared onboarding completed = %v, %v", completed, err)
	}
}

func TestOnboardingCompletedInCLI(t *testing.T) {
	ts := useTestOnboarding(t)
	if err := (onboarding.State{Dir: filepath.Dir(ts.store.DBPath)}).Complete(); err != nil {
		t.Fatal(err)
	}
	if onboardingNeeded() {
		t.Fatal("onboarding is needed after the CLI completed it")
	}
}

func TestOnboardingSignInURL(t *testing.T) {
	useTestOnboarding(t)
	useTestAccountServer(t, func(http.ResponseWriter, *http.Request) {})
	for _, signUp := range []bool{false, true} {
		signIn, err := onboardingSignInURL(signUp)
		if err != nil {
			t.Fatal(err)
		}
		u, err := url.Parse(signIn)
		if err != nil {
			t.Fatal(err)
		}
		query := u.Query()
		if u.Path != "/connect" || query.Get("key") == "" || query.Has("launch") {
			t.Errorf("sign in URL = %q", signIn)
		}
		if got := query.Get("signup") == "true"; got != signUp {
			t.Errorf("sign in URL %q starts with a new account = %v, want %v", signIn, got, signUp)
		}
	}
}
