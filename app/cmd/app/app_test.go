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
