package cmd

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"os"
	"sync/atomic"
	"testing"
)

func TestCLIHelp(t *testing.T) {
	for _, tt := range []struct {
		name        string
		args        []string
		interactive bool
	}{
		{name: "noninteractive", args: []string{}},
		{name: "noninteractive help", args: []string{"--help"}},
		{name: "interactive help", args: []string{"--help"}, interactive: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			original := isInteractiveTerminal
			isInteractiveTerminal = func() bool { return tt.interactive }
			t.Cleanup(func() { isInteractiveTerminal = original })

			home := t.TempDir()
			setCmdTestHome(t, home)
			t.Setenv("LOCALAPPDATA", home)
			var requests atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requests.Add(1)
				w.WriteHeader(http.StatusServiceUnavailable)
			}))
			defer server.Close()
			t.Setenv("OLLAMA_HOST", server.URL)

			var want bytes.Buffer
			help := NewCLI()
			help.SetArgs([]string{"--help"})
			help.SetOut(&want)
			if err := help.Execute(); err != nil {
				t.Fatal(err)
			}

			var stdout, stderr bytes.Buffer
			cli := NewCLI()
			cli.SetArgs(tt.args)
			cli.SetOut(&stdout)
			cli.SetErr(&stderr)
			if err := cli.Execute(); err != nil {
				t.Fatal(err)
			}
			if stdout.String() != want.String() {
				t.Errorf("stdout = %q, want help %q", stdout.String(), want.String())
			}
			if stderr.Len() != 0 {
				t.Errorf("unexpected stderr: %s", &stderr)
			}
			if got := requests.Load(); got != 0 {
				t.Errorf("help made %d server requests", got)
			}
			entries, err := os.ReadDir(home)
			if err != nil {
				t.Fatal(err)
			}
			if len(entries) != 0 {
				t.Errorf("help wrote files in the user home: %v", entries)
			}
		})
	}
}
