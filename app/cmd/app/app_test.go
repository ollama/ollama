//go:build windows || darwin

package main

import (
	"errors"
	"testing"

	"github.com/ollama/ollama/app/store"
)

func TestShouldShowOnboarding(t *testing.T) {
	tests := []struct {
		name     string
		settings store.Settings
		err      error
		want     bool
	}{
		{
			name:     "fresh install",
			settings: store.Settings{OnboardingVersion: 0},
			want:     true,
		},
		{
			name:     "completed onboarding",
			settings: store.Settings{OnboardingVersion: store.CurrentOnboardingVersion},
			want:     false,
		},
		{
			name: "settings failure",
			err:  errors.New("settings unavailable"),
			want: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := shouldShowOnboarding(tt.settings, tt.err); got != tt.want {
				t.Fatalf("shouldShowOnboarding() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestDispatchURLSchemeRequest(t *testing.T) {
	tests := []struct {
		name        string
		request     string
		wantConnect bool
		wantOpen    bool
		wantApps    bool
		wantErr     bool
	}{
		{name: "bare URL opens app", request: "ollama://", wantOpen: true},
		{name: "root URL opens app", request: "ollama:///", wantOpen: true},
		{name: "apps URL opens Apps", request: "ollama://apps", wantApps: true},
		{name: "apps path opens Apps", request: "ollama:///apps", wantApps: true},
		{name: "apps trailing slash opens Apps", request: "ollama://apps/", wantApps: true},
		{name: "connect URL starts connection", request: "ollama://connect", wantConnect: true},
		{name: "connect path starts connection", request: "ollama:///connect", wantConnect: true},
		{name: "unsupported URL", request: "ollama://unsupported", wantErr: true},
		{name: "invalid URL", request: "ollama://%", wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			connected := false
			opened := false
			openedApps := false
			err := dispatchURLSchemeRequest(
				tt.request,
				func() { connected = true },
				func() { opened = true },
				func() { openedApps = true },
			)
			if (err != nil) != tt.wantErr {
				t.Fatalf("dispatchURLSchemeRequest() error = %v, wantErr %v", err, tt.wantErr)
			}
			if connected != tt.wantConnect {
				t.Errorf("connect called = %v, want %v", connected, tt.wantConnect)
			}
			if opened != tt.wantOpen {
				t.Errorf("open called = %v, want %v", opened, tt.wantOpen)
			}
			if openedApps != tt.wantApps {
				t.Errorf("open Apps called = %v, want %v", openedApps, tt.wantApps)
			}
		})
	}
}

func TestRunInitialWindowsUIWithBareURL(t *testing.T) {
	hiddenCalls := 0
	urlCalls := 0
	onboardingCalls := 0
	openCalls := 0

	runInitialWindowsUI(
		false,
		true,
		"ollama://",
		func() { hiddenCalls++ },
		func(request string) {
			urlCalls++
			err := dispatchURLSchemeRequest(request,
				func() { t.Fatal("unexpected sign-in") },
				func() { openCalls++ },
				func() { t.Fatal("unexpected Apps navigation") },
			)
			if err != nil {
				t.Fatalf("dispatchURLSchemeRequest() error = %v", err)
			}
		},
		func(path string) {
			onboardingCalls++
		},
	)

	if urlCalls != 1 {
		t.Fatalf("URL handled %d times, want 1", urlCalls)
	}
	if openCalls != 1 {
		t.Errorf("app opened %d times, want 1", openCalls)
	}
	if hiddenCalls != 0 {
		t.Errorf("hidden startup called %d times, want 0", hiddenCalls)
	}
	if onboardingCalls != 0 {
		t.Errorf("onboarding opened %d times, want 0", onboardingCalls)
	}
}

func TestRunInitialWindowsUIWithAppsURL(t *testing.T) {
	appsCalls := 0
	runInitialWindowsUI(
		false,
		true,
		"ollama://apps",
		func() { t.Fatal("unexpected hidden startup") },
		func(request string) {
			err := dispatchURLSchemeRequest(request,
				func() { t.Fatal("unexpected sign-in") },
				func() { t.Fatal("unexpected home navigation") },
				func() { appsCalls++ },
			)
			if err != nil {
				t.Fatalf("dispatchURLSchemeRequest() error = %v", err)
			}
		},
		func(string) { t.Fatal("unexpected onboarding") },
	)
	if appsCalls != 1 {
		t.Fatalf("Apps opened %d times, want 1", appsCalls)
	}
}

func TestRunInitialWindowsUIRoutesInteractiveLaunch(t *testing.T) {
	for _, tt := range []struct {
		name           string
		showOnboarding bool
		wantPath       string
	}{
		{name: "fresh install preserves onboarding", showOnboarding: true, wantPath: "/"},
		{name: "returning launch opens apps", wantPath: "/connect"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var gotPath string
			runInitialWindowsUI(
				false,
				tt.showOnboarding,
				"",
				func() { t.Fatal("unexpected hidden startup") },
				func(string) { t.Fatal("unexpected URL handling") },
				func(path string) { gotPath = path },
			)
			if gotPath != tt.wantPath {
				t.Fatalf("initial UI path = %q, want %q", gotPath, tt.wantPath)
			}
		})
	}
}
