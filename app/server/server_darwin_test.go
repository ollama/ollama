//go:build darwin

package server

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/app/store"
)

func TestServerCmdIncludesLaunchctlEnv(t *testing.T) {
	orig := launchctlGetenv
	t.Cleanup(func() {
		launchctlGetenv = orig
	})

	launchctlGetenv = func(key string) (string, bool) {
		switch key {
		case "GGML_METAL_TENSOR_DISABLE":
			return "1", true
		case "OLLAMA_KV_CACHE_TYPE":
			return "q8_0", true
		default:
			return "", false
		}
	}

	prevTensor, hadTensor := os.LookupEnv("GGML_METAL_TENSOR_DISABLE")
	prevKV, hadKV := os.LookupEnv("OLLAMA_KV_CACHE_TYPE")
	if err := os.Unsetenv("GGML_METAL_TENSOR_DISABLE"); err != nil {
		t.Fatalf("unset GGML_METAL_TENSOR_DISABLE: %v", err)
	}
	if err := os.Unsetenv("OLLAMA_KV_CACHE_TYPE"); err != nil {
		t.Fatalf("unset OLLAMA_KV_CACHE_TYPE: %v", err)
	}
	t.Cleanup(func() {
		if hadTensor {
			_ = os.Setenv("GGML_METAL_TENSOR_DISABLE", prevTensor)
		} else {
			_ = os.Unsetenv("GGML_METAL_TENSOR_DISABLE")
		}
		if hadKV {
			_ = os.Setenv("OLLAMA_KV_CACHE_TYPE", prevKV)
		} else {
			_ = os.Unsetenv("OLLAMA_KV_CACHE_TYPE")
		}
	})

	st := &store.Store{DBPath: filepath.Join(t.TempDir(), "db.sqlite")}
	defer st.Close()

	s := &Server{store: st}
	cmd, err := s.cmd(t.Context())
	if err != nil {
		t.Fatalf("s.cmd() error = %v", err)
	}

	want := map[string]string{
		"GGML_METAL_TENSOR_DISABLE": "1",
		"OLLAMA_KV_CACHE_TYPE":      "q8_0",
	}

	for key, value := range want {
		found := false
		for _, env := range cmd.Env {
			if env == key+"="+value {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("expected environment variable %s=%s in command env", key, value)
		}
	}
}

func TestMergeDarwinLaunchctlEnv(t *testing.T) {
	orig := launchctlGetenv
	t.Cleanup(func() {
		launchctlGetenv = orig
	})

	t.Run("fills missing allowlisted variables", func(t *testing.T) {
		launchctlGetenv = func(key string) (string, bool) {
			switch key {
			case "GGML_METAL_TENSOR_DISABLE":
				return "1", true
			case "OLLAMA_KV_CACHE_TYPE":
				return "q8_0", true
			case "OLLAMA_FLASH_ATTENTION":
				return "1", true
			default:
				return "", false
			}
		}

		env := map[string]string{}
		mergeDarwinLaunchctlEnv(env)

		if got := env["GGML_METAL_TENSOR_DISABLE"]; got != "1" {
			t.Fatalf("GGML_METAL_TENSOR_DISABLE = %q, want %q", got, "1")
		}
		if got := env["OLLAMA_KV_CACHE_TYPE"]; got != "q8_0" {
			t.Fatalf("OLLAMA_KV_CACHE_TYPE = %q, want %q", got, "q8_0")
		}
		if got := env["OLLAMA_FLASH_ATTENTION"]; got != "1" {
			t.Fatalf("OLLAMA_FLASH_ATTENTION = %q, want %q", got, "1")
		}
	})

	t.Run("does not overwrite explicit environment", func(t *testing.T) {
		launchctlGetenv = func(key string) (string, bool) {
			return "launchctl-value", true
		}

		env := map[string]string{
			"OLLAMA_KV_CACHE_TYPE": "f16",
		}
		mergeDarwinLaunchctlEnv(env)

		if got := env["OLLAMA_KV_CACHE_TYPE"]; got != "f16" {
			t.Fatalf("OLLAMA_KV_CACHE_TYPE = %q, want %q", got, "f16")
		}
	})

	t.Run("ignores empty launchctl values", func(t *testing.T) {
		launchctlGetenv = func(key string) (string, bool) {
			return "", false
		}

		env := map[string]string{}
		mergeDarwinLaunchctlEnv(env)

		if _, ok := env["OLLAMA_KV_CACHE_TYPE"]; ok {
			t.Fatal("expected OLLAMA_KV_CACHE_TYPE to be omitted")
		}
	})
}
