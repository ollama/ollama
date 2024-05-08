package api

import (
	"fmt"
	"net"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

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

	// test ingnore _defaultApiClient
	for k, v := range testCases {
		t.Run(k, func(t *testing.T) {
			// set _defaultApiClient=nil for really new client in ClientFromEnvironment
			_defaultApiClient = nil

			t.Setenv("OLLAMA_HOST", v.value)
			client, err := ClientFromEnvironment()
			if err != v.err {
				t.Fatalf("case %s: expected %s, got %s", k, v.err, err)
			}

			if client.base.String() != v.expect {
				t.Fatalf("case %s: expected %s, got %s", k, v.expect, client.base.String())
			}
		})
	}

	// test with _defaultApiClient
	for k, v := range testCases {
		_defaultApiClient = nil
		t.Setenv("OLLAMA_HOST", v.value)
		client, err := ClientFromEnvironment()
		if err != v.err {
			t.Fatalf("case %s: expected %s, got %s", k, v.err, err)
		}
		if client.base.String() != v.expect {
			t.Fatalf("case %s: expected %s, got %s", k, v.expect, client.base.String())
		}
		require.Equal(t, _defaultApiClient, client)

		// call ClientFromEnvironment again, should return _defaultApiClient directly
		client2, err := ClientFromEnvironment()
		require.Nil(t, err)
		require.NotNil(t, client2)
		// address and fields of _defaultApiClient, client2 and client should equal
		require.Equal(t, client, client2, fmt.Sprintf("case %s: expected %T, got %T", k, client, client2))
		require.Equal(t, _defaultApiClient, client2, fmt.Sprintf("case %s: expected %T, got %T", k, _defaultApiClient, client2))
		require.Equal(t, client.base.String(), client2.base.String(), fmt.Sprintf("case %s: expected %s, got %s", k, client.base.String(), client2.base.String()))
		require.Equal(t, client.http, _defaultApiClient.http, fmt.Sprintf("case %s: expected %T, got %T", k, client.http, _defaultApiClient.http))
	}
}

func TestGetOllamaHost(t *testing.T) {
	hostTestCases := map[string]*struct {
		value  string
		expect string
		err    error
	}{
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

			oh, err := GetOllamaHost()
			if err != v.err {
				t.Fatalf("case %s: expected %s, got %s", k, v.err, err)
			}

			if err == nil {
				host := net.JoinHostPort(oh.Host, oh.Port)
				assert.Equal(t, v.expect, host, fmt.Sprintf("%s: expected %s, got %s", k, v.expect, host))
			}
		})
	}
}
