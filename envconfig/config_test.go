package envconfig

import (
	"fmt"
	"net"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestConfig(t *testing.T) {
	Debug = false // Reset whatever was loaded in init()
	t.Setenv("OLLAMA_DEBUG", "")
	LoadConfig()
	require.False(t, Debug)
	t.Setenv("OLLAMA_DEBUG", "false")
	LoadConfig()
	require.False(t, Debug)
	t.Setenv("OLLAMA_DEBUG", "1")
	LoadConfig()
	require.True(t, Debug)
	t.Setenv("OLLAMA_FLASH_ATTENTION", "1")
	LoadConfig()
	require.True(t, FlashAttention)
}

func TestClientFromEnvironment(t *testing.T) {
	type testCase struct {
		value  string
		expect string
		err    error
	}

	hostTestCases := map[string]*testCase{
		"empty":               {value: "", expect: "127.0.0.1:11434"},
		"only address":        {value: "1.2.3.4", expect: "1.2.3.4:11434"},
		"only port":           {value: ":1234", expect: ":1234"},
		"address and port":    {value: "1.2.3.4:1234", expect: "1.2.3.4:1234"},
		"hostname":            {value: "example.com", expect: "example.com:11434"},
		"hostname and port":   {value: "example.com:1234", expect: "example.com:1234"},
		"zero port":           {value: ":0", expect: ":0"},
		"too large port":      {value: ":66000", err: ErrInvalidHostPort},
		"too small port":      {value: ":-1", err: ErrInvalidHostPort},
		"ipv6 localhost":      {value: "[::1]", expect: "[::1]:11434"},
		"ipv6 world open":     {value: "[::]", expect: "[::]:11434"},
		"ipv6 no brackets":    {value: "::1", expect: "[::1]:11434"},
		"ipv6 + port":         {value: "[::1]:1337", expect: "[::1]:1337"},
		"extra space":         {value: " 1.2.3.4 ", expect: "1.2.3.4:11434"},
		"extra quotes":        {value: "\"1.2.3.4\"", expect: "1.2.3.4:11434"},
		"extra space+quotes":  {value: " \" 1.2.3.4 \" ", expect: "1.2.3.4:11434"},
		"extra single quotes": {value: "'1.2.3.4'", expect: "1.2.3.4:11434"},
	}

	for k, v := range hostTestCases {
		t.Run(k, func(t *testing.T) {
			t.Setenv("OLLAMA_HOST", v.value)
			LoadConfig()

			oh, err := getOllamaHost()
			if err != v.err {
				t.Fatalf("expected %s, got %s", v.err, err)
			}

			if err == nil {
				host := net.JoinHostPort(oh.Host, oh.Port)
				assert.Equal(t, v.expect, host, fmt.Sprintf("%s: expected %s, got %s", k, v.expect, host))
			}
		})
	}
}
