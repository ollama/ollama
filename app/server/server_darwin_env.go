//go:build darwin

package server

import (
	"bytes"
	"os/exec"
	"strings"
)

var launchctlGetenv = getenvFromLaunchctl

var darwinLaunchctlEnvAllowlist = []string{
	"GGML_METAL_TENSOR_DISABLE",
	"OLLAMA_DEBUG",
	"OLLAMA_DEBUG_LOG_REQUESTS",
	"OLLAMA_FLASH_ATTENTION",
	"OLLAMA_GPU_OVERHEAD",
	"OLLAMA_KEEP_ALIVE",
	"OLLAMA_KV_CACHE_TYPE",
	"OLLAMA_LLM_LIBRARY",
	"OLLAMA_LOAD_TIMEOUT",
	"OLLAMA_MAX_LOADED_MODELS",
	"OLLAMA_MAX_QUEUE",
	"OLLAMA_NUM_PARALLEL",
	"OLLAMA_SCHED_SPREAD",
	"OLLAMA_MULTIUSER_CACHE",
	"OLLAMA_NEW_ENGINE",
}

func mergeDarwinLaunchctlEnv(env map[string]string) {
	for _, key := range darwinLaunchctlEnvAllowlist {
		if _, ok := env[key]; ok {
			continue
		}
		if value, ok := launchctlGetenv(key); ok {
			env[key] = value
		}
	}
}

func getenvFromLaunchctl(key string) (string, bool) {
	cmd := exec.Command("launchctl", "getenv", key)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	out, err := cmd.Output()
	if err != nil {
		return "", false
	}

	value := strings.TrimSpace(string(out))
	if value == "" {
		return "", false
	}

	return value, true
}
