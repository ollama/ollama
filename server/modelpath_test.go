package server

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGetBlobsPath(t *testing.T) {
	// GetBlobsPath expects an actual directory to exist
	dir, err := os.MkdirTemp("", "ollama-test")
	require.NoError(t, err)
	defer os.RemoveAll(dir)

	tests := []struct {
		name     string
		digest   string
		expected string
		err      error
	}{
		{
			"empty digest",
			"",
			filepath.Join(dir, "blobs"),
			nil,
		},
		{
			"valid with colon",
			"sha256:456402914e838a953e0cf80caa6adbe75383d9e63584a964f504a7bbb8f7aad9",
			filepath.Join(dir, "blobs", "sha256-456402914e838a953e0cf80caa6adbe75383d9e63584a964f504a7bbb8f7aad9"),
			nil,
		},
		{
			"valid with dash",
			"sha256-456402914e838a953e0cf80caa6adbe75383d9e63584a964f504a7bbb8f7aad9",
			filepath.Join(dir, "blobs", "sha256-456402914e838a953e0cf80caa6adbe75383d9e63584a964f504a7bbb8f7aad9"),
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
			t.Setenv("OLLAMA_MODELS", dir)

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
