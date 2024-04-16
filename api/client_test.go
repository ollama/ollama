package api

import "testing"

func TestClientFromEnvironment(t *testing.T) {
	type testCase struct {
		value  string
		expect string
		err    error
	}

	testCases := map[string]*testCase{
		"empty":                      {value: "", expect: "http://127.0.0.1:11434"},
		"only address":               {value: "1.2.3.4", expect: "http://1.2.3.4:11434"},
		"only port":                  {value: ":1234", expect: "http://:1234"},
		"address and port":           {value: "1.2.3.4:1234", expect: "http://1.2.3.4:1234"},
		"scheme http and address":    {value: "http://1.2.3.4", expect: "http://1.2.3.4:80"},
		"scheme https and address":   {value: "https://1.2.3.4", expect: "https://1.2.3.4:443"},
		"scheme, address, and port":  {value: "https://1.2.3.4:1234", expect: "https://1.2.3.4:1234"},
		"hostname":                   {value: "example.com", expect: "http://example.com:11434"},
		"hostname and port":          {value: "example.com:1234", expect: "http://example.com:1234"},
		"scheme http and hostname":   {value: "http://example.com", expect: "http://example.com:80"},
		"scheme https and hostname":  {value: "https://example.com", expect: "https://example.com:443"},
		"scheme, hostname, and port": {value: "https://example.com:1234", expect: "https://example.com:1234"},
		"trailing slash":             {value: "example.com/", expect: "http://example.com:11434"},
		"trailing slash port":        {value: "example.com:1234/", expect: "http://example.com:1234"},
	}

	for k, v := range testCases {
		t.Run(k, func(t *testing.T) {
			t.Setenv("OLLAMA_HOST", v.value)

			client, err := ClientFromEnvironment()
			if err != v.err {
				t.Fatalf("expected %s, got %s", v.err, err)
			}

			if client.base.String() != v.expect {
				t.Fatalf("expected %s, got %s", v.expect, client.base.String())
			}
		})
	}

	hostTestCases := map[string]*testCase{
		"empty":             {value: "", expect: "127.0.0.1:11434"},
		"only address":      {value: "1.2.3.4", expect: "1.2.3.4:11434"},
		"only port":         {value: ":1234", expect: ":1234"},
		"address and port":  {value: "1.2.3.4:1234", expect: "1.2.3.4:1234"},
		"hostname":          {value: "example.com", expect: "example.com:11434"},
		"hostname and port": {value: "example.com:1234", expect: "example.com:1234"},
		"zero port":         {value: ":0", expect: ":0"},
		"too large port":    {value: ":66000", err: ErrInvalidHostPort},
		"too small port":    {value: ":-1", err: ErrInvalidHostPort},
		"ipv6 localhost":    {value: "[::1]", expect: "[::1]:11434"},
		"ipv6 world open":   {value: "[::]", expect: "[::]:11434"},
		"ipv6 no brackets":  {value: "::1", expect: "[::1]:11434"},
	}

	for k, v := range hostTestCases {
		t.Run(k, func(t *testing.T) {
			t.Setenv("OLLAMA_HOST", v.value)

			client, err := ClientFromEnvironment()
			if err != v.err {
				t.Fatalf("expected %s, got %s", v.err, err)
			}

			if client != nil && client.GetHost() != v.expect {
				t.Fatalf("expected %s, got %s", v.expect, client.GetHost())
			}
		})
	}
}
