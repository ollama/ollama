package envconfig

import (
	"math"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestConfig(t *testing.T) {
	t.Setenv("OLLAMA_DEBUG", "")
	require.False(t, Debug())

	t.Setenv("OLLAMA_DEBUG", "false")
	require.False(t, Debug())

	t.Setenv("OLLAMA_DEBUG", "1")
	require.True(t, Debug())

	t.Setenv("OLLAMA_FLASH_ATTENTION", "1")
	LoadConfig()
	require.True(t, FlashAttention)
	t.Setenv("OLLAMA_KEEP_ALIVE", "")
	LoadConfig()
	require.Equal(t, 5*time.Minute, KeepAlive)
	t.Setenv("OLLAMA_KEEP_ALIVE", "3")
	LoadConfig()
	require.Equal(t, 3*time.Second, KeepAlive)
	t.Setenv("OLLAMA_KEEP_ALIVE", "1h")
	LoadConfig()
	require.Equal(t, 1*time.Hour, KeepAlive)
	t.Setenv("OLLAMA_KEEP_ALIVE", "-1s")
	LoadConfig()
	require.Equal(t, time.Duration(math.MaxInt64), KeepAlive)
	t.Setenv("OLLAMA_KEEP_ALIVE", "-1")
	LoadConfig()
	require.Equal(t, time.Duration(math.MaxInt64), KeepAlive)
}

func TestClientFromEnvironment(t *testing.T) {
	cases := map[string]struct {
		value  string
		expect string
	}{
		"empty":               {"", "127.0.0.1:11434"},
		"only address":        {"1.2.3.4", "1.2.3.4:11434"},
		"only port":           {":1234", ":1234"},
		"address and port":    {"1.2.3.4:1234", "1.2.3.4:1234"},
		"hostname":            {"example.com", "example.com:11434"},
		"hostname and port":   {"example.com:1234", "example.com:1234"},
		"zero port":           {":0", ":0"},
		"too large port":      {":66000", ":11434"},
		"too small port":      {":-1", ":11434"},
		"ipv6 localhost":      {"[::1]", "[::1]:11434"},
		"ipv6 world open":     {"[::]", "[::]:11434"},
		"ipv6 no brackets":    {"::1", "[::1]:11434"},
		"ipv6 + port":         {"[::1]:1337", "[::1]:1337"},
		"extra space":         {" 1.2.3.4 ", "1.2.3.4:11434"},
		"extra quotes":        {"\"1.2.3.4\"", "1.2.3.4:11434"},
		"extra space+quotes":  {" \" 1.2.3.4 \" ", "1.2.3.4:11434"},
		"extra single quotes": {"'1.2.3.4'", "1.2.3.4:11434"},
	}

	for name, tt := range cases {
		t.Run(name, func(t *testing.T) {
			t.Setenv("OLLAMA_HOST", tt.value)
			if host := Host(); host.Host != tt.expect {
				t.Errorf("%s: expected %s, got %s", name, tt.expect, host.Host)
			}
		})
	}
}
