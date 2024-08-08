package cmd

import (
	"errors"
	"io"
	"os"
	"testing"

	"github.com/spf13/cobra"
)

type testGetAuthInput struct {
	envAuth       string
	envAuthB64    string
	flagAuth      string
	flagAuthB64   string
	flagAuthStdin string
}
type testGetAuthExpect struct {
	username string
	password string
}

func TestGetAuth(t *testing.T) {
	tests := []struct {
		input       testGetAuthInput
		expect      testGetAuthExpect
		exceptError error
	}{
		{testGetAuthInput{"", "", "", "", ""}, testGetAuthExpect{"", ""}, nil},
		{testGetAuthInput{"user:pass", "", "", "", ""}, testGetAuthExpect{"user", "pass"}, nil},
		{testGetAuthInput{"", "dXNlcjpwYXNz", "", "", ""}, testGetAuthExpect{"user", "pass"}, nil},
		{testGetAuthInput{"", "", "user:pass", "", ""}, testGetAuthExpect{"user", "pass"}, nil},
		{testGetAuthInput{"", "", "", "dXNlcjpwYXNz", ""}, testGetAuthExpect{"user", "pass"}, nil},
		{testGetAuthInput{"user:pass", "dXNlcjpwYXNz", "", "", ""}, testGetAuthExpect{}, errors.New("cannot use both OLLAMA_AUTH and OLLAMA_AUTH_B64")},
		{testGetAuthInput{"", "", "user:pass", "dXNlcjpwYXNz", ""}, testGetAuthExpect{}, errors.New("cannot use --auth, --auth-stdin and --auth-b64 together")},
		// flag takes precedence over environment variables
		{testGetAuthInput{"userEnv:pass", "", "", "dXNlcjpwYXNz", ""}, testGetAuthExpect{"user", "pass"}, nil},
		{testGetAuthInput{"userEnv:pass", "", "user:pass", "", ""}, testGetAuthExpect{"user", "pass"}, nil},
		{testGetAuthInput{"", "dXNlcjpwYXNz", "userFlag:pass", "", ""}, testGetAuthExpect{"userFlag", "pass"}, nil},
		{testGetAuthInput{"", "dXNlcjpwYXNz", "", "dXNlckZsYWc6cGFzcw==", ""}, testGetAuthExpect{"userFlag", "pass"}, nil},
		// test stdin
		// {testGetAuthInput{"", "", "", "", "user:pass"}, testGetAuthExpect{"user", "pass"}, nil},
	}

	for _, test := range tests {
		os.Clearenv()
		cmd := &cobra.Command{}

		if test.input.envAuth != "" {
			os.Setenv("OLLAMA_AUTH", test.input.envAuth)
		}
		if test.input.envAuthB64 != "" {
			os.Setenv("OLLAMA_AUTH_B64", test.input.envAuthB64)
		}
		cmd.Flags().String("auth", test.input.flagAuth, "")
		cmd.Flags().String("auth-b64", test.input.flagAuthB64, "")
		cmd.Flags().Bool("auth-stdin", test.input.flagAuthStdin != "", "")

		// TODO: test flagAuthStdin
		if test.input.flagAuthStdin != "" {
		}

		username, password, err := getBasicAuth(cmd)
		if err != nil {
			if err.Error() != test.exceptError.Error() {
				t.Errorf("Unexpected error: %v", err)
			}
		}

		if username != test.expect.username || password != test.expect.password {
			t.Errorf("Expected (%s, %s), got (%s, %s)", test.expect.username, test.expect.password, username, password)
		}
	}
}

type stdinReader struct {
	io.Reader
}

func (r *stdinReader) Close() error {
	return nil
}
