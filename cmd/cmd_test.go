package cmd

import (
	"fmt"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestHostString(t *testing.T) {
	type TestHostPort struct {
		Test string
		Host string
		Port string
		Err  error
	}

	errInvalidIP := fmt.Errorf("invalid IP address specified in OLLAMA_HOST")
	errInvalidPort := fmt.Errorf("invalid port specified in OLLAMA_HOST")

	tests := []TestHostPort{
		{"", "127.0.0.1", "11434", nil},
		{"127.0.0.1", "127.0.0.1", "11434", nil},
		{"0.0.0.0", "0.0.0.0", "11434", nil},
		{"127.0.0.1:1337", "127.0.0.1", "1337", nil},
		{"0.0.0.0:1337", "0.0.0.0", "1337", nil},
		{":1337", "0.0.0.0", "1337", nil},

		{" 127.0.0.1 ", "127.0.0.1", "11434", nil},
		{"\"127.0.0.1\"", "127.0.0.1", "11434", nil},
		{" \"127.0.0.1\" ", "127.0.0.1", "11434", nil},
		{"\" 127.0.0.1 \" ", "", "", errInvalidIP},

		{"[::1]", "::1", "11434", nil}, // localhost
		{"[::]", "::", "11434", nil},   // open
		{"[::1]:1337", "::1", "1337", nil},
		{"[::]:1337", "::", "1337", nil},

		{"somehost", "", "", errInvalidIP},
		{"somehost:11434", "", "", errInvalidIP},
		{"127.0.0.0.1", "", "", errInvalidIP},
		{"127.0.0.0.1:1337", "", "", errInvalidIP},
		{"[:]", "", "", errInvalidIP},
		{"[:]:1337", "", "", errInvalidIP},

		{"127.0.0.1:66000", "", "", errInvalidPort},
		{"127.0.0.1:0", "", "", errInvalidPort},
		{"127.0.0.1:-1", "", "", errInvalidPort},
		{"[::1]:66000", "", "", errInvalidPort},
		{"[::1]:0", "", "", errInvalidPort},
		{"127.0.0.1:", "", "", errInvalidPort},
	}

	for _, thp := range tests {
		os.Setenv("OLLAMA_HOST", thp.Test)
		host, port, err := getHostIPAndPort()
		assert.Equal(t, thp.Err, err)
		assert.Equal(t, thp.Host, host)
		assert.Equal(t, thp.Port, port)
	}
}
