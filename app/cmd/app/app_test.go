//go:build windows || darwin

package main

import "testing"

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

func TestRunInitialUI(t *testing.T) {
	tests := []struct {
		name         string
		startHidden  bool
		request      string
		wantHidden   bool
		wantURL      string
		wantSettings bool
	}{
		{name: "interactive launch opens settings", wantSettings: true},
		{name: "hidden launch shows nothing", startHidden: true, wantHidden: true},
		{name: "URL request wins over a hidden launch", startHidden: true, request: "ollama://apps", wantURL: "ollama://apps"},
		{name: "URL request wins over settings", request: "ollama://", wantURL: "ollama://"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hidden := false
			handled := ""
			settingsPanes := []settingsPane{}
			runInitialUI(
				tt.startHidden,
				tt.request,
				func() { hidden = true },
				func(request string) { handled = request },
				func(pane settingsPane) { settingsPanes = append(settingsPanes, pane) },
			)
			if hidden != tt.wantHidden {
				t.Errorf("hidden startup = %v, want %v", hidden, tt.wantHidden)
			}
			if handled != tt.wantURL {
				t.Errorf("handled URL = %q, want %q", handled, tt.wantURL)
			}
			if got := len(settingsPanes) == 1 && settingsPanes[0] == settingsPaneDefault; got != tt.wantSettings {
				t.Errorf("settings shown = %v, want %v", settingsPanes, tt.wantSettings)
			}
		})
	}
}
