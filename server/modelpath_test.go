package server

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/ollama/ollama/envconfig"
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
			envconfig.LoadConfig()

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

func TestMigrateRegistryDomain(t *testing.T) {
	p := t.TempDir()
	t.Setenv("OLLAMA_MODELS", p)
	envconfig.LoadConfig()

	manifests := []string{
		filepath.Join("registry.ollama.ai", "library", "llama3", "7b"),
		filepath.Join("registry.ollama.ai", "library", "mistral", "latest"),
		filepath.Join("registry.other.com", "library", "llama3", "13b"),
	}

	for _, manifest := range manifests {
		n := filepath.Join(p, "manifests", manifest)
		if err := os.MkdirAll(filepath.Dir(n), 0o750); err != nil {
			t.Fatal(err)
		}

		f, err := os.Create(n)
		if err != nil {
			t.Fatal(err)
		}

		if err := f.Close(); err != nil {
			t.Fatal(err)
		}
	}

	t.Run("migrate", func(t *testing.T) {
		if err := migrateRegistryDomain(); err != nil {
			t.Fatal(err)
		}

		checkFileExists(t, filepath.Join(p, "manifests", "*", "*", "*", "*"), []string{
			filepath.Join(p, "manifests", DefaultRegistry, "library", "llama3", "7b"),
			filepath.Join(p, "manifests", DefaultRegistry, "library", "mistral", "latest"),
			filepath.Join(p, "manifests", "registry.other.com", "library", "llama3", "13b"),
		})
	})

	t.Run("idempotent", func(t *testing.T) {
		// subsequent run should be a noop
		if err := migrateRegistryDomain(); err != nil {
			t.Fatal(err)
		}

		checkFileExists(t, filepath.Join(p, "manifests", "*", "*", "*", "*"), []string{
			filepath.Join(p, "manifests", DefaultRegistry, "library", "llama3", "7b"),
			filepath.Join(p, "manifests", DefaultRegistry, "library", "mistral", "latest"),
			filepath.Join(p, "manifests", "registry.other.com", "library", "llama3", "13b"),
		})
	})

	t.Run("no migration needed", func(t *testing.T) {
		n := filepath.Join(p, "manifests", "registry.ollama.ai", "library", "gemma", "7b")
		if err := os.MkdirAll(filepath.Dir(n), 0o750); err != nil {
			t.Fatal(err)
		}

		f, err := os.Create(n)
		if err != nil {
			t.Fatal(err)
		}

		if err := f.Close(); err != nil {
			t.Fatal(err)
		}

		if err := migrateRegistryDomain(); err != nil {
			t.Fatal(err)
		}

		checkFileExists(t, filepath.Join(p, "manifests", "*", "*", "*", "*"), []string{
			filepath.Join(p, "manifests", DefaultRegistry, "library", "llama3", "7b"),
			filepath.Join(p, "manifests", DefaultRegistry, "library", "mistral", "latest"),
			filepath.Join(p, "manifests", "registry.ollama.ai", "library", "gemma", "7b"),
			filepath.Join(p, "manifests", "registry.other.com", "library", "llama3", "13b"),
		})
	})

	t.Run("no migration source", func(t *testing.T) {
		// cleanup premigration directories
		if err := os.RemoveAll(filepath.Join(p, "manifests", "registry.ollama.ai")); err != nil {
			t.Fatal(err)
		}

		if err := migrateRegistryDomain(); err != nil {
			t.Fatal(err)
		}
	})
}
