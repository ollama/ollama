//go:build windows

package wintray

import "testing"

func TestUpdateNotificationMessage(t *testing.T) {
	for _, tt := range []struct {
		ver  string
		want string
	}{
		{"0.32.15", "Ollama version 0.32.15 is ready to install"},
		{"", "An Ollama update is ready to install"},
		{" \t", "An Ollama update is ready to install"},
	} {
		if got := updateNotificationMessage(tt.ver); got != tt.want {
			t.Fatalf("updateNotificationMessage(%q) = %q, want %q", tt.ver, got, tt.want)
		}
	}
}
