package cmd

import (
	"os"
	"testing"
)

func TestGetRunServerParams(t *testing.T) {
	t.Run("default values", func(t *testing.T) {
		cmd := NewCLI()
		serveCmd, _, err := cmd.Find([]string{"serve"})
		if err != nil {
			t.Errorf("expected serve command, got %s", err)
		}
		host, port, extraOrigins, err := getRunServerParams(serveCmd)
		// assertions
		if err != nil {
			t.Errorf("unexpected error, got %s", err)
		}
		if host != "127.0.0.1" {
			t.Errorf("unexpected host, got %s", host)
		}
		if port != "11434" {
			t.Errorf("unexpected port, got %s", port)
		}
		if len(extraOrigins) != 0 {
			t.Errorf("unexpected origins, got %s", extraOrigins)
		}
	})
	t.Run("environment variables take precedence over default", func(t *testing.T) {
		cmd := NewCLI()
		serveCmd, _, err := cmd.Find([]string{"serve"})
		if err != nil {
			t.Errorf("expected serve command, got %s", err)
		}
		// setup environment variables
		err = os.Setenv("OLLAMA_HOST", "0.0.0.0")
		if err != nil {
			t.Errorf("could not set env var")
		}
		err = os.Setenv("OLLAMA_PORT", "9999")
		if err != nil {
			t.Errorf("could not set env var")
		}
		defer func() {
			os.Unsetenv("OLLAMA_HOST")
			os.Unsetenv("OLLAMA_PORT")
		}()

		host, port, extraOrigins, err := getRunServerParams(serveCmd)
		// assertions
		if err != nil {
			t.Errorf("unexpected error, got %s", err)
		}
		if host != "0.0.0.0" {
			t.Errorf("unexpected host, got %s", host)
		}
		if port != "9999" {
			t.Errorf("unexpected port, got %s", port)
		}
		if len(extraOrigins) != 0 {
			t.Errorf("unexpected origins, got %s", extraOrigins)
		}
	})
	t.Run("command line args take precedence over env vars", func(t *testing.T) {
		cmd := NewCLI()
		serveCmd, _, err := cmd.Find([]string{"serve"})
		if err != nil {
			t.Errorf("expected serve command, got %s", err)
		}
		// setup environment variables
		err = os.Setenv("OLLAMA_HOST", "0.0.0.0")
		if err != nil {
			t.Errorf("could not set env var")
		}
		err = os.Setenv("OLLAMA_PORT", "9999")
		if err != nil {
			t.Errorf("could not set env var")
		}
		defer func() {
			os.Unsetenv("OLLAMA_HOST")
			os.Unsetenv("OLLAMA_PORT")
		}()
		// now set command flags
		serveCmd.Flags().Set("host", "localhost")
		serveCmd.Flags().Set("port", "8888")
		serveCmd.Flags().Set("allowed-origins", "http://foo.example.com,http://192.168.1.1")

		host, port, extraOrigins, err := getRunServerParams(serveCmd)
		if err != nil {
			t.Errorf("unexpected error, got %s", err)
		}
		if host != "localhost" {
			t.Errorf("unexpected host, got %s", host)
		}
		if port != "8888" {
			t.Errorf("unexpected port, got %s", port)
		}
		if len(extraOrigins) != 2 {
			t.Errorf("expected two origins, got length %d", len(extraOrigins))
		}
	})
}
