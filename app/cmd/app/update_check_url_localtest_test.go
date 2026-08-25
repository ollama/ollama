//go:build (windows || darwin) && updater_localtest

package main

import (
	"testing"

	"github.com/ollama/ollama/app/updater"
)

func TestConfigureLocalUpdateCheckURL(t *testing.T) {
	old := updater.UpdateCheckURLBase
	defer func() {
		updater.UpdateCheckURLBase = old
	}()

	if err := configureLocalUpdateCheckURL("http://127.0.0.1:8765/api/update"); err != nil {
		t.Fatalf("configureLocalUpdateCheckURL returned error: %v", err)
	}
	if got := updater.UpdateCheckURLBase; got != "http://127.0.0.1:8765/api/update" {
		t.Fatalf("UpdateCheckURLBase = %q", got)
	}
}

func TestConfigureLocalUpdateCheckURLRejectsNonLoopback(t *testing.T) {
	old := updater.UpdateCheckURLBase
	defer func() {
		updater.UpdateCheckURLBase = old
	}()

	for _, rawURL := range []string{
		"https://127.0.0.1:8765/api/update",
		"http://example.com/api/update",
		"http://127.0.0.1:8765/other",
		"http://127.0.0.1:8765/api/update?x=1",
	} {
		t.Run(rawURL, func(t *testing.T) {
			if err := configureLocalUpdateCheckURL(rawURL); err == nil {
				t.Fatal("expected error")
			}
		})
	}
}
