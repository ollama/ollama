//go:build darwin

package main

import "testing"

func TestBundleInApplications(t *testing.T) {
	const systemWidePath = "/Applications/Ollama.app"
	const homeDir = "/Users/example"

	tests := []struct {
		name       string
		bundlePath string
		want       bool
	}{
		{name: "system Applications root", bundlePath: systemWidePath, want: true},
		{name: "system Applications subdirectory", bundlePath: "/Applications/AI/Ollama.app", want: true},
		{name: "nested system Applications subdirectory", bundlePath: "/Applications/LLM/Local/Ollama.app", want: true},
		{name: "user Applications root", bundlePath: "/Users/example/Applications/Ollama.app", want: true},
		{name: "nested user Applications subdirectory", bundlePath: "/Users/example/Applications/AI/Ollama.app", want: true},
		{name: "Downloads", bundlePath: "/Users/example/Downloads/Ollama.app", want: false},
		{name: "mounted volume", bundlePath: "/Volumes/Ollama/Ollama.app", want: false},
		{name: "path sharing Applications prefix", bundlePath: "/ApplicationsBackup/Ollama.app", want: false},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := bundleInApplications(test.bundlePath, systemWidePath, homeDir); got != test.want {
				t.Fatalf("bundleInApplications(%q) = %t, want %t", test.bundlePath, got, test.want)
			}
		})
	}
}
