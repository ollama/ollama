package envconfig

import (
	"math"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
)

func TestHost(t *testing.T) {
	cases := map[string]struct {
		value  string
		expect string
	}{
		"empty":               {"", "http://127.0.0.1:11434"},
		"only address":        {"1.2.3.4", "http://1.2.3.4:11434"},
		"only port":           {":1234", "http://:1234"},
		"address and port":    {"1.2.3.4:1234", "http://1.2.3.4:1234"},
		"hostname":            {"example.com", "http://example.com:11434"},
		"hostname and port":   {"example.com:1234", "http://example.com:1234"},
		"zero port":           {":0", "http://:0"},
		"too large port":      {":66000", "http://:11434"},
		"too small port":      {":-1", "http://:11434"},
		"ipv6 localhost":      {"[::1]", "http://[::1]:11434"},
		"ipv6 world open":     {"[::]", "http://[::]:11434"},
		"ipv6 no brackets":    {"::1", "http://[::1]:11434"},
		"ipv6 + port":         {"[::1]:1337", "http://[::1]:1337"},
		"extra space":         {" 1.2.3.4 ", "http://1.2.3.4:11434"},
		"extra quotes":        {"\"1.2.3.4\"", "http://1.2.3.4:11434"},
		"extra space+quotes":  {" \" 1.2.3.4 \" ", "http://1.2.3.4:11434"},
		"extra single quotes": {"'1.2.3.4'", "http://1.2.3.4:11434"},
		"http":                {"http://1.2.3.4", "http://1.2.3.4:80"},
		"http port":           {"http://1.2.3.4:4321", "http://1.2.3.4:4321"},
		"https":               {"https://1.2.3.4", "https://1.2.3.4:443"},
		"https port":          {"https://1.2.3.4:4321", "https://1.2.3.4:4321"},
		"proxy path":          {"https://example.com/ollama", "https://example.com:443/ollama"},
		"unix socket":         {"unix:///tmp/ollama.sock", "unix://%2Ftmp%2Follama.sock"},
	}

	for name, tt := range cases {
		t.Run(name, func(t *testing.T) {
			t.Setenv("OLLAMA_HOST", tt.value)
			if host := Host(); host.String() != tt.expect {
				t.Errorf("%s: expected %s, got %s", name, tt.expect, host.String())
			}
		})
	}
}

func TestOrigins(t *testing.T) {
	cases := []struct {
		value  string
		expect []string
	}{
		{"", []string{
			"http://localhost",
			"https://localhost",
			"http://localhost:*",
			"https://localhost:*",
			"http://127.0.0.1",
			"https://127.0.0.1",
			"http://127.0.0.1:*",
			"https://127.0.0.1:*",
			"http://0.0.0.0",
			"https://0.0.0.0",
			"http://0.0.0.0:*",
			"https://0.0.0.0:*",
			"app://*",
			"file://*",
			"tauri://*",
			"vscode-webview://*",
		}},
		{"http://10.0.0.1", []string{
			"http://10.0.0.1",
			"http://localhost",
			"https://localhost",
			"http://localhost:*",
			"https://localhost:*",
			"http://127.0.0.1",
			"https://127.0.0.1",
			"http://127.0.0.1:*",
			"https://127.0.0.1:*",
			"http://0.0.0.0",
			"https://0.0.0.0",
			"http://0.0.0.0:*",
			"https://0.0.0.0:*",
			"app://*",
			"file://*",
			"tauri://*",
			"vscode-webview://*",
		}},
		{"http://172.16.0.1,https://192.168.0.1", []string{
			"http://172.16.0.1",
			"https://192.168.0.1",
			"http://localhost",
			"https://localhost",
			"http://localhost:*",
			"https://localhost:*",
			"http://127.0.0.1",
			"https://127.0.0.1",
			"http://127.0.0.1:*",
			"https://127.0.0.1:*",
			"http://0.0.0.0",
			"https://0.0.0.0",
			"http://0.0.0.0:*",
			"https://0.0.0.0:*",
			"app://*",
			"file://*",
			"tauri://*",
			"vscode-webview://*",
		}},
		{"http://totally.safe,http://definitely.legit", []string{
			"http://totally.safe",
			"http://definitely.legit",
			"http://localhost",
			"https://localhost",
			"http://localhost:*",
			"https://localhost:*",
			"http://127.0.0.1",
			"https://127.0.0.1",
			"http://127.0.0.1:*",
			"https://127.0.0.1:*",
			"http://0.0.0.0",
			"https://0.0.0.0",
			"http://0.0.0.0:*",
			"https://0.0.0.0:*",
			"app://*",
			"file://*",
			"tauri://*",
			"vscode-webview://*",
		}},
	}
	for _, tt := range cases {
		t.Run(tt.value, func(t *testing.T) {
			t.Setenv("OLLAMA_ORIGINS", tt.value)

			if diff := cmp.Diff(Origins(), tt.expect); diff != "" {
				t.Errorf("%s: mismatch (-want +got):\n%s", tt.value, diff)
			}
		})
	}
}

func TestBool(t *testing.T) {
	cases := map[string]bool{
		"":      false,
		"true":  true,
		"false": false,
		"1":     true,
		"0":     false,
		// invalid values
		"random":    true,
		"something": true,
	}

	for k, v := range cases {
		t.Run(k, func(t *testing.T) {
			t.Setenv("OLLAMA_BOOL", k)
			if b := Bool("OLLAMA_BOOL")(); b != v {
				t.Errorf("%s: expected %t, got %t", k, v, b)
			}
		})
	}
}

func TestUint(t *testing.T) {
	cases := map[string]uint{
		"0":    0,
		"1":    1,
		"1337": 1337,
		// default values
		"":       11434,
		"-1":     11434,
		"0o10":   11434,
		"0x10":   11434,
		"string": 11434,
	}

	for k, v := range cases {
		t.Run(k, func(t *testing.T) {
			t.Setenv("OLLAMA_UINT", k)
			if i := Uint("OLLAMA_UINT", 11434)(); i != v {
				t.Errorf("%s: expected %d, got %d", k, v, i)
			}
		})
	}
}

func TestKeepAlive(t *testing.T) {
	cases := map[string]time.Duration{
		"":       5 * time.Minute,
		"1s":     time.Second,
		"1m":     time.Minute,
		"1h":     time.Hour,
		"5m0s":   5 * time.Minute,
		"1h2m3s": 1*time.Hour + 2*time.Minute + 3*time.Second,
		"0":      time.Duration(0),
		"60":     60 * time.Second,
		"120":    2 * time.Minute,
		"3600":   time.Hour,
		"-0":     time.Duration(0),
		"-1":     time.Duration(math.MaxInt64),
		"-1m":    time.Duration(math.MaxInt64),
		// invalid values
		" ":   5 * time.Minute,
		"???": 5 * time.Minute,
		"1d":  5 * time.Minute,
		"1y":  5 * time.Minute,
		"1w":  5 * time.Minute,
	}

	for tt, expect := range cases {
		t.Run(tt, func(t *testing.T) {
			t.Setenv("OLLAMA_KEEP_ALIVE", tt)
			if actual := KeepAlive(); actual != expect {
				t.Errorf("%s: expected %s, got %s", tt, expect, actual)
			}
		})
	}
}

func TestLoadTimeout(t *testing.T) {
	defaultTimeout := 5 * time.Minute
	cases := map[string]time.Duration{
		"":       defaultTimeout,
		"1s":     time.Second,
		"1m":     time.Minute,
		"1h":     time.Hour,
		"5m0s":   defaultTimeout,
		"1h2m3s": 1*time.Hour + 2*time.Minute + 3*time.Second,
		"0":      time.Duration(math.MaxInt64),
		"60":     60 * time.Second,
		"120":    2 * time.Minute,
		"3600":   time.Hour,
		"-0":     time.Duration(math.MaxInt64),
		"-1":     time.Duration(math.MaxInt64),
		"-1m":    time.Duration(math.MaxInt64),
		// invalid values
		" ":   defaultTimeout,
		"???": defaultTimeout,
		"1d":  defaultTimeout,
		"1y":  defaultTimeout,
		"1w":  defaultTimeout,
	}

	for tt, expect := range cases {
		t.Run(tt, func(t *testing.T) {
			t.Setenv("OLLAMA_LOAD_TIMEOUT", tt)
			if actual := LoadTimeout(); actual != expect {
				t.Errorf("%s: expected %s, got %s", tt, expect, actual)
			}
		})
	}
}

func TestVar(t *testing.T) {
	cases := map[string]string{
		"value":       "value",
		" value ":     "value",
		" 'value' ":   "value",
		` "value" `:   "value",
		" ' value ' ": " value ",
		` " value " `: " value ",
	}

	for k, v := range cases {
		t.Run(k, func(t *testing.T) {
			t.Setenv("OLLAMA_VAR", k)
			if s := Var("OLLAMA_VAR"); s != v {
				t.Errorf("%s: expected %q, got %q", k, v, s)
			}
		})
	}
}
