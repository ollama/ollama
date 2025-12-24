package server

import (
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGetBlobsPath(t *testing.T) {
	// GetBlobsPath expects an actual directory to exist
	tempDir := t.TempDir()

	tests := []struct {
		name     string
		digest   string
		expected string
		err      error
	}{
		{
			"empty digest",
			"",
			filepath.Join(tempDir, "blobs"),
			nil,
		},
		{
			"valid with colon",
			"sha256:456402914e838a953e0cf80caa6adbe75383d9e63584a964f504a7bbb8f7aad9",
			filepath.Join(tempDir, "blobs", "sha256-456402914e838a953e0cf80caa6adbe75383d9e63584a964f504a7bbb8f7aad9"),
			nil,
		},
		{
			"valid with dash",
			"sha256-456402914e838a953e0cf80caa6adbe75383d9e63584a964f504a7bbb8f7aad9",
			filepath.Join(tempDir, "blobs", "sha256-456402914e838a953e0cf80caa6adbe75383d9e63584a964f504a7bbb8f7aad9"),
			nil,
		},
		{
			"digest too short",
			"sha256-45640291",
			"",
			ErrInvalidDigestFormat,
		},
		{
			"digest too long",
			"sha256-456402914e838a953e0cf80caa6adbe75383d9e63584a964f504a7bbb8f7aad9aaaaaaaaaa",
			"",
			ErrInvalidDigestFormat,
		},
		{
			"digest invalid chars",
			"../sha256-456402914e838a953e0cf80caa6adbe75383d9e63584a964f504a7bbb8f7a",
			"",
			ErrInvalidDigestFormat,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("OLLAMA_MODELS", tempDir)

			got, err := GetBlobsPath(tc.digest)

			require.ErrorIs(t, tc.err, err, tc.name)
			assert.Equal(t, tc.expected, got, tc.name)
		})
	}
}

func TestParseModelPath(t *testing.T) {
	tests := []struct {
		name string
		arg  string
		want ModelPath
	}{
		{
			"full path https",
			"https://example.com/ns/repo:tag",
			ModelPath{
				ProtocolScheme: "https",
				Registry:       "example.com",
				Namespace:      "ns",
				Repository:     "repo",
				Tag:            "tag",
			},
		},
		{
			"full path http",
			"http://example.com/ns/repo:tag",
			ModelPath{
				ProtocolScheme: "http",
				Registry:       "example.com",
				Namespace:      "ns",
				Repository:     "repo",
				Tag:            "tag",
			},
		},
		{
			"no protocol",
			"example.com/ns/repo:tag",
			ModelPath{
				ProtocolScheme: "https",
				Registry:       "example.com",
				Namespace:      "ns",
				Repository:     "repo",
				Tag:            "tag",
			},
		},
		{
			"no registry",
			"ns/repo:tag",
			ModelPath{
				ProtocolScheme: "https",
				Registry:       DefaultRegistry,
				Namespace:      "ns",
				Repository:     "repo",
				Tag:            "tag",
			},
		},
		{
			"no namespace",
			"repo:tag",
			ModelPath{
				ProtocolScheme: "https",
				Registry:       DefaultRegistry,
				Namespace:      DefaultNamespace,
				Repository:     "repo",
				Tag:            "tag",
			},
		},
		{
			"no tag",
			"repo",
			ModelPath{
				ProtocolScheme: "https",
				Registry:       DefaultRegistry,
				Namespace:      DefaultNamespace,
				Repository:     "repo",
				Tag:            DefaultTag,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := ParseModelPath(tc.arg)

			if got != tc.want {
				t.Errorf("got: %q want: %q", got, tc.want)
			}
		})
	}
}

func FuzzParseModelPath(f *testing.F) {
	// Add seed corpus
	seeds := []string{
		"https://example.com/ns/repo:tag",
		"http://example.com/ns/repo:tag",
		"example.com/ns/repo:tag",
		"ns/repo:tag",
		"repo:tag",
		"repo",
		"",
		"://ns/repo:tag",
		"a/b/c/d/e",
		":::",
		"///",
	}
	for _, s := range seeds {
		f.Add(s)
	}

	f.Fuzz(func(t *testing.T, input string) {
		mp := ParseModelPath(input)

		// Test GetFullTagname doesn't panic
		fullTag := mp.GetFullTagname()
		if fullTag == "" {
			t.Errorf("GetFullTagname() returned empty string for input: %q", input)
		}

		// Test GetShortTagname doesn't panic
		shortTag := mp.GetShortTagname()
		if shortTag == "" {
			t.Errorf("GetShortTagname() returned empty string for input: %q", input)
		}

		// Test that parsing the full tagname produces consistent results
		mp2 := ParseModelPath(fullTag)
		if mp2.GetFullTagname() != fullTag {
			t.Errorf("Round-trip inconsistency: ParseModelPath(%q).GetFullTagname() = %q, but ParseModelPath(%q).GetFullTagname() = %q",
				input, fullTag, fullTag, mp2.GetFullTagname())
		}
	})
}
